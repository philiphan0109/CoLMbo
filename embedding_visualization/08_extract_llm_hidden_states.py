#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    ensure_output_dir,
    load_config,
    resolve_device,
    save_embedding_artifact,
    setup_local_env,
    settle_modules_on_device,
)


POOLING_CHOICES = (
    "last_input",
    "last_nonpad_prompt",
    "prompt_mean",
    "audio_mean",
    "all_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract in-LM GPT hidden-state embeddings for EARS task examples. "
            "Each row is an (utterance, task) pair: ECAPA raw -> mapper prefix + task prompt -> GPT hidden state."
        )
    )
    parser.add_argument(
        "--analysis-csv",
        default="embedding_visualization/runs/ears_default/analysis/ecapa_prediction_analysis_utterance.csv",
        help="Merged ECAPA/prediction utterance table from 06_merge_ecapa_with_predictions.py",
    )
    parser.add_argument(
        "--ecapa-raw-metadata",
        default="embedding_visualization/runs/ears_default/data/ecapa/utterance_ecapa_raw_metadata.csv",
        help="Utterance-level ECAPA raw metadata CSV",
    )
    parser.add_argument(
        "--ecapa-raw-embeddings",
        default="embedding_visualization/runs/ears_default/data/ecapa/utterance_ecapa_raw_embeddings.npy",
        help="Utterance-level ECAPA raw embeddings NPY",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/runs/ears_default/llm_analysis",
        help="Output directory for in-LM hidden-state artifacts",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for GPT hidden-state extraction (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPT forward passes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on (utterance, task) rows to process (0 = all)",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Optional balanced cap per task, applied before --limit (0 = disabled)",
    )
    parser.add_argument(
        "--pooling",
        choices=list(POOLING_CHOICES),
        default="last_input",
        help=(
            "How to collapse GPT final-layer hidden states to one vector. "
            "`last_input` matches the position used for next-token generation."
        ),
    )
    return parser.parse_args()


def load_mapper(config):
    from mapper import get_sid_mapper

    wrapper_cfg = config["wrapper"]
    if wrapper_cfg.get("text_decoder") != "gpt2":
        raise ValueError("This extractor currently supports the GPT-2 decoder path.")

    gpt_embedding_size = 768
    mapper = get_sid_mapper(
        wrapper_cfg["map_type"],
        None,
        int(wrapper_cfg["prefix_size"]),
        gpt_embedding_size,
        int(wrapper_cfg["sid_prefix_length"]),
        int(wrapper_cfg["sid_prefix_length_clip"]),
        int(wrapper_cfg["num_layers"]),
    )

    mapper_path = Path(config["train"]["snapshot_path"]) / wrapper_cfg["mapper_ck_name"]
    snapshot = torch.load(mapper_path, map_location="cpu")
    state = {
        key.replace("module.", ""): value
        for key, value in snapshot["sid_mapper"].items()
    }
    mapper.load_state_dict(state)
    mapper.eval()
    return mapper


def load_gpt_and_tokenizer(config):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    decoder_name = config["wrapper"]["text_decoder"]
    gpt = GPT2LMHeadModel.from_pretrained(decoder_name)
    gpt.eval()
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({"pad_token": "!"})
    return gpt, tokenizer


def load_task_rows(analysis_csv: Path, limit: int, limit_per_task: int):
    df = pd.read_csv(analysis_csv)
    df = df[df["task"].isin(["gender", "age", "ethnicity"])].copy()
    df = df.sort_values(["task", "audio_path"]).reset_index(drop=True)
    if limit_per_task and limit_per_task > 0:
        df = (
            df.groupby("task", group_keys=False, sort=True)
            .head(limit_per_task)
            .reset_index(drop=True)
        )
    if limit and limit > 0:
        df = df.iloc[:limit].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No usable task rows found in {analysis_csv}")
    return df


def attach_raw_embeddings(task_df: pd.DataFrame, raw_metadata_path: Path, raw_embeddings_path: Path):
    raw_meta = pd.read_csv(raw_metadata_path)
    raw_embeddings = np.load(raw_embeddings_path).astype(np.float32)
    if len(raw_meta) != raw_embeddings.shape[0]:
        raise ValueError(
            f"Raw metadata/embedding length mismatch: {len(raw_meta)} vs {raw_embeddings.shape[0]}"
        )

    audio_to_idx = {
        audio_path: idx for idx, audio_path in enumerate(raw_meta["audio_path"].astype(str))
    }
    missing = [
        path
        for path in task_df["audio_path"].astype(str).unique()
        if path not in audio_to_idx
    ]
    if missing:
        raise RuntimeError(f"Missing ECAPA raw embeddings for {len(missing)} audio paths.")

    raw_indices = task_df["audio_path"].astype(str).map(audio_to_idx).to_numpy()
    return raw_embeddings[raw_indices]


def maybe_use_tqdm(total):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc="LLM hidden extraction", unit="row")
    except Exception:
        return None


def make_sid_prefix(raw_batch, mapper, config, device):
    wrapper_cfg = config["wrapper"]
    sid_prefix_length = int(wrapper_cfg["sid_prefix_length"])
    gpt_embedding_size = 768
    norm_sid_emb = bool(wrapper_cfg["norm_sid_emb"])

    sid_embeddings = torch.as_tensor(raw_batch, dtype=torch.float32, device=device)
    if norm_sid_emb:
        sid_embeddings = sid_embeddings / sid_embeddings.norm(2, -1, keepdim=True)

    return mapper(sid_embeddings).contiguous().view(
        -1, sid_prefix_length, gpt_embedding_size
    )


def tokenize_prompts(prompts, tokenizer, device):
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        max_length=10,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def pool_hidden(hidden, prompt_attention_mask, sid_prefix_length: int, pooling: str):
    if pooling == "last_input":
        return hidden[:, -1, :]
    if pooling == "audio_mean":
        return hidden[:, :sid_prefix_length, :].mean(dim=1)
    if pooling == "prompt_mean":
        return hidden[:, sid_prefix_length:, :].mean(dim=1)
    if pooling == "all_mean":
        return hidden.mean(dim=1)
    if pooling == "last_nonpad_prompt":
        prompt_lengths = prompt_attention_mask.sum(dim=1).clamp(min=1)
        indices = sid_prefix_length + prompt_lengths - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, indices, :]
    raise ValueError(f"Unknown pooling: {pooling}")


def main():
    setup_local_env()
    args = parse_args()
    requested_device = resolve_device(args.device)
    output_dir = ensure_output_dir(Path(args.output_dir))
    config = load_config(Path(args.config))

    task_df = load_task_rows(Path(args.analysis_csv), args.limit, args.limit_per_task)
    raw_embeddings = attach_raw_embeddings(
        task_df,
        Path(args.ecapa_raw_metadata),
        Path(args.ecapa_raw_embeddings),
    )

    mapper = load_mapper(config)
    gpt, tokenizer = load_gpt_and_tokenizer(config)
    device = settle_modules_on_device(
        [("ECAPA mapper", mapper), ("GPT-2", gpt)],
        requested_device,
    )

    sid_prefix_length = int(config["wrapper"]["sid_prefix_length"])
    vectors = []
    pbar = maybe_use_tqdm(len(task_df))

    with torch.no_grad():
        for start in range(0, len(task_df), args.batch_size):
            end = min(start + args.batch_size, len(task_df))
            batch_df = task_df.iloc[start:end]
            raw_batch = raw_embeddings[start:end]

            sid_prefix = make_sid_prefix(raw_batch, mapper, config, device)
            prompt_tokens = tokenize_prompts(batch_df["prompt"], tokenizer, device)
            prompt_embeds = gpt.transformer.wte(prompt_tokens["input_ids"])
            inputs_embeds = torch.cat([sid_prefix, prompt_embeds], dim=1)

            outputs = gpt(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            final_hidden = outputs.hidden_states[-1]
            pooled = pool_hidden(
                final_hidden,
                prompt_tokens["attention_mask"],
                sid_prefix_length,
                args.pooling,
            )
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

            if pbar is not None:
                pbar.update(end - start)
            else:
                print(f"[{end}/{len(task_df)}] processed")

    if pbar is not None:
        pbar.close()

    metadata_df = task_df.copy()
    metadata_df["embedding_family"] = "gpt2_in_lm_hidden"
    metadata_df["hidden_layer"] = "final"
    metadata_df["pooling"] = args.pooling
    metadata_df["text_decoder"] = config["wrapper"]["text_decoder"]
    metadata_df["sid_prefix_length"] = sid_prefix_length
    metadata_df["prompt_token_length"] = 10
    metadata_df["device_used"] = str(device)

    hidden_vectors = np.vstack(vectors)
    prefix = f"utterance_task_llm_hidden_{args.pooling}"
    save_embedding_artifact(output_dir, prefix, metadata_df, hidden_vectors)

    print(f"Wrote LLM hidden-state artifacts to: {output_dir}")
    print(f"Rows: {len(metadata_df)}")
    print(f"Utterances: {metadata_df['audio_path'].nunique()}")
    print(f"Speakers: {metadata_df['speaker_id'].nunique()}")
    print(f"Tasks: {', '.join(sorted(metadata_df['task'].unique()))}")
    print(f"Embedding shape: {hidden_vectors.shape}")
    print(f"Pooling: {args.pooling}")
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
