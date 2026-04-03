#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from common import (
    LABEL_COLUMNS,
    aggregate_by_speaker,
    build_ears_index,
    ensure_output_dir,
    load_config,
    maybe_prefix_model_keys,
    resolve_device,
    save_embedding_artifact,
    settle_modules_on_device,
    setup_local_env,
    EARSWaveformResolver,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract ECAPA raw embeddings and ECAPA->mapper embeddings for EARS audio. "
            "Outputs utterance-level and speaker-level artifacts."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="TEARS manifest JSONL path",
    )
    parser.add_argument(
        "--ears-root",
        required=True,
        help="Root directory with EARS raw speaker audio",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/data/ecapa/default",
        help="Output directory for ECAPA artifacts",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for extraction (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of utterances to process (0 = all)",
    )
    parser.add_argument(
        "--mapper-pool",
        choices=["mean", "flatten"],
        default="mean",
        help="How to convert mapper prefix sequence into a plotting vector",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N utterances when tqdm is unavailable",
    )
    return parser.parse_args()


def load_sid_model(config_path: Path):
    from encoder.encoder import Model

    config = load_config(config_path)
    sid_ck_name = config["sid_model"]["sid_ck_name"]
    snapshot_path = Path(config["train"]["snapshot_path"])
    sid_path = snapshot_path / sid_ck_name

    model = Model(n_mels=80, embedding_dim=192, channel=1024)
    snapshot = torch.load(sid_path, map_location="cpu")
    model.load_state_dict(maybe_prefix_model_keys(snapshot["sid_model"]))
    model.eval()
    return model, config


def load_sid_mapper(config):
    from mapper import get_sid_mapper

    wrapper_cfg = config["wrapper"]
    map_type = wrapper_cfg["map_type"]
    prefix_size = int(wrapper_cfg["prefix_size"])
    sid_prefix_length = int(wrapper_cfg["sid_prefix_length"])
    sid_prefix_length_clip = int(wrapper_cfg["sid_prefix_length_clip"])
    num_layers = int(wrapper_cfg["num_layers"])
    norm_sid_emb = bool(wrapper_cfg["norm_sid_emb"])

    # GPT-2 base is fixed in this repo config, so this avoids loading the full LLM.
    if wrapper_cfg.get("text_decoder") == "gpt2":
        gpt_embedding_size = 768
    else:
        raise ValueError(
            "Only GPT-2 based mapper loading is currently supported for visualization."
        )

    mapper = get_sid_mapper(
        map_type,
        None,
        prefix_size,
        gpt_embedding_size,
        sid_prefix_length,
        sid_prefix_length_clip,
        num_layers,
    )

    mapper_path = Path(config["train"]["snapshot_path"]) / wrapper_cfg["mapper_ck_name"]
    snapshot = torch.load(mapper_path, map_location="cpu")
    state = snapshot["sid_mapper"]
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "")] = value
    mapper.load_state_dict(cleaned)
    mapper.eval()
    return mapper, sid_prefix_length, gpt_embedding_size, norm_sid_emb


def build_extractor(sample_rate):
    from load_data.extract_fbanks import Mel_Spectrogram

    return Mel_Spectrogram(sample_rate=sample_rate)


def maybe_use_tqdm(total):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc="ECAPA extraction", unit="utt")
    except Exception:
        return None


def main():
    setup_local_env()
    args = parse_args()
    requested_device = resolve_device(args.device)
    out_dir = ensure_output_dir(Path(args.output_dir))

    sid_model, config = load_sid_model(Path(args.config))
    mapper, sid_prefix_length, gpt_embedding_size, norm_sid_emb = load_sid_mapper(config)
    device = settle_modules_on_device(
        [("ECAPA sid_model", sid_model), ("ECAPA mapper", mapper)],
        requested_device,
    )
    sample_rate = int(config["data"].get("sample_rate", 16000))
    extractor = build_extractor(sample_rate)
    resolver = EARSWaveformResolver(Path(args.ears_root))

    index_df = build_ears_index(Path(args.manifest))
    if args.limit and args.limit > 0:
        index_df = index_df.iloc[: args.limit].reset_index(drop=True)
    if index_df.empty:
        raise RuntimeError("No EARS rows found in the provided manifest.")

    raw_vectors = []
    mapper_vectors = []
    kept_rows = []

    pbar = maybe_use_tqdm(len(index_df))
    with torch.no_grad():
        for i, row in index_df.iterrows():
            waveform, sr = resolver.resolve(row["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

            processed = extractor(waveform).to(device)
            sid_emb = sid_model(processed)
            raw_vec = sid_emb.squeeze(0).detach().cpu().numpy()

            mapper_input = sid_emb
            if norm_sid_emb:
                mapper_input = mapper_input / mapper_input.norm(2, -1, keepdim=True)
            mapper_seq = mapper(mapper_input).contiguous().view(
                -1, sid_prefix_length, gpt_embedding_size
            )
            if args.mapper_pool == "mean":
                mapper_vec = mapper_seq.mean(dim=1).squeeze(0).detach().cpu().numpy()
            else:
                mapper_vec = mapper_seq.reshape(mapper_seq.shape[0], -1).squeeze(0).detach().cpu().numpy()

            raw_vectors.append(raw_vec.astype(np.float32))
            mapper_vectors.append(mapper_vec.astype(np.float32))
            kept_rows.append(row.to_dict())

            if pbar is not None:
                pbar.update(1)
            elif (i + 1) % max(1, args.progress_every) == 0 or (i + 1) == len(index_df):
                print(f"[{i + 1}/{len(index_df)}] processed")

    if pbar is not None:
        pbar.close()

    utterance_df = pd.DataFrame(kept_rows)
    utterance_df["embedding_family"] = "ecapa"

    raw_vectors = np.vstack(raw_vectors)
    mapper_vectors = np.vstack(mapper_vectors)

    save_embedding_artifact(out_dir, "utterance_ecapa_raw", utterance_df, raw_vectors)
    save_embedding_artifact(
        out_dir, "utterance_ecapa_mapper", utterance_df, mapper_vectors
    )

    speaker_raw_df, speaker_raw_vectors = aggregate_by_speaker(
        utterance_df, raw_vectors, extra_columns=["embedding_family"]
    )
    speaker_mapper_df, speaker_mapper_vectors = aggregate_by_speaker(
        utterance_df, mapper_vectors, extra_columns=["embedding_family"]
    )

    save_embedding_artifact(out_dir, "speaker_ecapa_raw", speaker_raw_df, speaker_raw_vectors)
    save_embedding_artifact(
        out_dir, "speaker_ecapa_mapper", speaker_mapper_df, speaker_mapper_vectors
    )

    print(f"Wrote ECAPA artifacts to: {out_dir}")
    print(f"Utterances: {len(utterance_df)}")
    print(f"Speakers:   {utterance_df['speaker_id'].nunique()}")
    print(f"Device used: {device}")
    print(f"Mapper pool: {args.mapper_pool}")


if __name__ == "__main__":
    main()
