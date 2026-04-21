#!/usr/bin/env python3
"""Training loop for frozen-GPT CoLMbo mapper fine-tuning from cached ECAPA embeddings."""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (
    add_wandb_args,
    init_wandb,
    load_config,
    load_mapper_checkpoint,
    maybe_tqdm,
    resolve_device,
    setup_local_env,
    wandb_finish,
    wandb_log,
)


def build_arg_parser(default_sampling="uniform"):
    parser = argparse.ArgumentParser(
        description=(
            "Train the CoLMbo audio mapper while keeping ECAPA and GPT-2 frozen. "
            "Use 02_train_mapper_baseline.py for uniform sampling and "
            "03_train_mapper_weighted.py for task+label weighted sampling."
        )
    )
    parser.add_argument(
        "--examples-csv",
        default="training_scripts/data/ears_timit_ecapa_cache/train_examples_with_embeddings.csv",
        help="Training CSV from 01_precompute_ecapa_embeddings.py",
    )
    parser.add_argument(
        "--embeddings-npy",
        default="training_scripts/data/ears_timit_ecapa_cache/audio_embeddings.npy",
        help="ECAPA embeddings .npy from 01_precompute_ecapa_embeddings.py",
    )
    parser.add_argument("--config", default="config.yaml", help="Project config path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Defaults depend on sampling mode.",
    )
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = config value)")
    parser.add_argument("--lr", type=float, default=0.0, help="Learning rate (0 = config value)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=-1.0,
        help="AdamW weight decay (<0 = config value)",
    )
    parser.add_argument(
        "--sampling",
        choices=["uniform", "weighted"],
        default=default_sampling,
        help="Uniform baseline or smoothed task+label weighted sampling",
    )
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=0.5,
        help="Exponent for class balancing: (total / count(task,label)) ** alpha",
    )
    parser.add_argument(
        "--hard-predictions",
        default=None,
        help="Optional predictions.csv for hard-example weighting by (audio_path, task)",
    )
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=2.0,
        help="Multiplier for examples marked incorrect in --hard-predictions",
    )
    parser.add_argument(
        "--init-mapper",
        default=None,
        help="Initial mapper checkpoint. Defaults to config train snapshot mapper.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Do not load an initial mapper checkpoint; train the mapper from random init.",
    )
    parser.add_argument(
        "--prompt-max-length",
        type=int,
        default=10,
        help="Prompt token length. 10 matches wrapper.py baseline inference.",
    )
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=0,
        help="Target response token length (0 = config wrapper.tok_len)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--log-every", type=int, default=50, help="Step logging interval")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many batches",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Optional step cap for smoke tests (0 = full epoch)",
    )
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=0,
        help="Optional cap on examples loaded from CSV for smoke tests (0 = all)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision when training on CUDA.",
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="When --wandb is enabled, upload final and best mapper checkpoints as W&B artifacts.",
    )
    add_wandb_args(parser, default_job_type=f"train_mapper_{default_sampling}")
    return parser


class EmbeddingTextDataset(Dataset):
    def __init__(self, examples_csv, embeddings_npy, limit_examples=0):
        self.examples_csv = str(examples_csv)
        self.rows = []
        with Path(examples_csv).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("embedding_id", "") == "":
                    continue
                self.rows.append(row)
                if limit_examples and len(self.rows) >= limit_examples:
                    break
        if not self.rows:
            raise RuntimeError(f"No embedding-indexed rows found in {examples_csv}")
        self.embeddings = np.load(embeddings_npy, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        embedding = np.asarray(self.embeddings[int(row["embedding_id"])], dtype=np.float32)
        return {
            "sid_embedding": torch.from_numpy(embedding.copy()),
            "prompt": row["prompt"],
            "response": row["response"],
            "task": row["task"],
            "label": row["label"],
            "audio_path": row["audio_path"],
            "example_id": row.get("example_id", idx),
        }


def collate_examples(features):
    return {
        "sid_embeddings": torch.stack([x["sid_embedding"] for x in features], dim=0),
        "prompts": [x["prompt"] for x in features],
        "responses": [x["response"] for x in features],
        "tasks": [x["task"] for x in features],
        "labels": [x["label"] for x in features],
        "audio_paths": [x["audio_path"] for x in features],
        "example_ids": [x["example_id"] for x in features],
    }


def load_hard_prediction_map(path):
    hard = {}
    if not path:
        return hard
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            audio_path = row.get("audio_path")
            task = row.get("task")
            correct = row.get("is_value_correct")
            if not audio_path or not task or correct == "":
                continue
            try:
                hard[(audio_path, task)] = int(correct) == 1
            except ValueError:
                continue
    return hard


def compute_sample_weights(rows, alpha, hard_predictions=None, hard_weight=2.0):
    counts = Counter((row["task"], row["label"]) for row in rows)
    total = float(len(rows))
    hard_map = load_hard_prediction_map(hard_predictions)
    weights = []
    for row in rows:
        key = (row["task"], row["label"])
        weight = (total / float(counts[key])) ** alpha
        hard_correct = hard_map.get((row["audio_path"], row["task"]))
        if hard_correct is False:
            weight *= hard_weight
        weights.append(weight)
    mean_weight = float(np.mean(weights)) if weights else 1.0
    weights = [w / mean_weight for w in weights]
    return weights, counts


def write_weight_audit(path, rows, weights):
    fields = ["example_id", "audio_path", "task", "label", "weight"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, weight in zip(rows, weights):
            writer.writerow(
                {
                    "example_id": row.get("example_id", ""),
                    "audio_path": row["audio_path"],
                    "task": row["task"],
                    "label": row["label"],
                    "weight": f"{weight:.8f}",
                }
            )


def build_models(config, device, init_mapper=None, random_init=False):
    from mapper import get_sid_mapper
    from transformers import AutoTokenizer, GPT2LMHeadModel

    wrapper_cfg = config["wrapper"]
    tokenizer = AutoTokenizer.from_pretrained(wrapper_cfg["text_decoder"])
    tokenizer.add_special_tokens({"pad_token": "!"})

    gpt = GPT2LMHeadModel.from_pretrained(wrapper_cfg["text_decoder"])
    if len(tokenizer) > gpt.get_input_embeddings().num_embeddings:
        gpt.resize_token_embeddings(len(tokenizer))
    gpt.to(device)
    gpt.eval()
    for param in gpt.parameters():
        param.requires_grad = False

    gpt_embedding_size = gpt.transformer.wte.weight.shape[1]
    mapper = get_sid_mapper(
        wrapper_cfg["map_type"],
        None,
        int(wrapper_cfg["prefix_size"]),
        gpt_embedding_size,
        int(wrapper_cfg["sid_prefix_length"]),
        int(wrapper_cfg["sid_prefix_length_clip"]),
        int(wrapper_cfg["num_layers"]),
    )
    if not random_init:
        if init_mapper is None:
            init_mapper = Path(config["train"]["snapshot_path"]) / wrapper_cfg["mapper_ck_name"]
        load_mapper_checkpoint(mapper, init_mapper, device)
        print(f"Loaded initial mapper: {init_mapper}")
    else:
        print("Training mapper from random initialization.")
    mapper.to(device)
    mapper.train()
    return tokenizer, gpt, mapper


def tokenize_batch(tokenizer, prompts, responses, prompt_max_length, target_max_length, device):
    prompt_tokens = tokenizer(
        prompts,
        add_special_tokens=True,
        max_length=prompt_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    target_texts = [text + " <|endoftext|>" for text in responses]
    target_tokens = tokenizer(
        target_texts,
        add_special_tokens=True,
        max_length=target_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return (
        {key: value.to(device) for key, value in prompt_tokens.items()},
        {key: value.to(device) for key, value in target_tokens.items()},
    )


def mapper_lm_loss(
    gpt,
    mapper,
    tokenizer,
    sid_embeddings,
    prompts,
    responses,
    norm_sid_emb,
    sid_prefix_length,
    gpt_embedding_size,
    prompt_max_length,
    target_max_length,
    device,
):
    sid_embeddings = sid_embeddings.to(device)
    if norm_sid_emb:
        sid_embeddings = sid_embeddings / sid_embeddings.norm(2, dim=-1, keepdim=True).clamp_min(1e-8)

    prompt_tokens, target_tokens = tokenize_batch(
        tokenizer,
        prompts,
        responses,
        prompt_max_length,
        target_max_length,
        device,
    )
    sid_prefix = mapper(sid_embeddings).contiguous().view(
        -1,
        sid_prefix_length,
        gpt_embedding_size,
    )
    with torch.no_grad():
        prompt_embeds = gpt.transformer.wte(prompt_tokens["input_ids"])
        target_embeds = gpt.transformer.wte(target_tokens["input_ids"])

    inputs_embeds = torch.cat([sid_prefix, prompt_embeds, target_embeds], dim=1)
    target_labels = target_tokens["input_ids"].clone()
    target_labels[target_tokens["attention_mask"] == 0] = -100
    labels = torch.full(
        (
            sid_embeddings.shape[0],
            sid_prefix.shape[1] + prompt_tokens["input_ids"].shape[1] + target_labels.shape[1],
        ),
        -100,
        dtype=torch.long,
        device=device,
    )
    labels[:, sid_prefix.shape[1] + prompt_tokens["input_ids"].shape[1] :] = target_labels

    outputs = gpt(inputs_embeds=inputs_embeds, labels=labels)
    return outputs.loss


def save_checkpoint(path, mapper, epoch, global_step, train_loss, args, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sid_mapper": mapper.state_dict(),
            "epochs_run": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "sampling": args.sampling,
            "args": vars(args),
            "config": config,
        },
        path,
    )


def main(default_sampling="uniform"):
    setup_local_env()
    parser = build_arg_parser(default_sampling)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    train_cfg = config["train"]
    wrapper_cfg = config["wrapper"]
    batch_size = args.batch_size if args.batch_size > 0 else int(train_cfg["batch_size"])
    lr = args.lr if args.lr > 0 else float(train_cfg["lr"])
    weight_decay = args.weight_decay if args.weight_decay >= 0 else float(train_cfg["weight_decay"])
    target_max_length = args.target_max_length if args.target_max_length > 0 else int(wrapper_cfg["tok_len"])
    output_dir = Path(args.output_dir or f"training_scripts/runs/mapper_{args.sampling}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dataset = EmbeddingTextDataset(args.examples_csv, args.embeddings_npy, args.limit_examples)

    sampler = None
    shuffle = True
    weight_counts = None
    if args.sampling == "weighted":
        weights, weight_counts = compute_sample_weights(
            dataset.rows,
            args.weight_alpha,
            args.hard_predictions,
            args.hard_weight,
        )
        write_weight_audit(output_dir / "sampling_weights.csv", dataset.rows, weights)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_examples,
    )

    tokenizer, gpt, mapper = build_models(
        config,
        device,
        init_mapper=args.init_mapper,
        random_init=args.random_init,
    )
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_summary = {
        "examples_csv": str(args.examples_csv),
        "embeddings_npy": str(args.embeddings_npy),
        "output_dir": str(output_dir),
        "n_examples": len(dataset),
        "sampling": args.sampling,
        "weight_alpha": args.weight_alpha,
        "hard_predictions": args.hard_predictions or "",
        "hard_weight": args.hard_weight,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "prompt_max_length": args.prompt_max_length,
        "target_max_length": target_max_length,
        "device": str(device),
    }
    if weight_counts is not None:
        run_summary["task_label_counts"] = {
            f"{task}::{label}": count for (task, label), count in sorted(weight_counts.items())
        }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=True)
    run = init_wandb(args, config, run_config=run_summary)

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "step", "global_step", "loss", "avg_epoch_loss", "elapsed_sec"],
        )
        writer.writeheader()

    try:
        global_step = 0
        best_loss = None
        print(f"Training examples: {len(dataset)}")
        print(f"Output dir:        {output_dir}")
        print(f"Device:            {device}")
        print(f"Sampling:          {args.sampling}")

        for epoch in range(1, args.epochs + 1):
            mapper.train()
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            optimizer.zero_grad(set_to_none=True)
            pbar = maybe_tqdm(len(dataloader), desc=f"epoch {epoch}", unit="batch")
            started = time.time()

            for step, batch in enumerate(dataloader, start=1):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = mapper_lm_loss(
                        gpt,
                        mapper,
                        tokenizer,
                        batch["sid_embeddings"],
                        batch["prompts"],
                        batch["responses"],
                        bool(wrapper_cfg["norm_sid_emb"]),
                        int(wrapper_cfg["sid_prefix_length"]),
                        gpt.transformer.wte.weight.shape[1],
                        args.prompt_max_length,
                        target_max_length,
                        device,
                    )
                    scaled_loss = loss / max(1, args.gradient_accumulation_steps)

                scaler.scale(scaled_loss).backward()

                if step % max(1, args.gradient_accumulation_steps) == 0:
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(mapper.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value
                epoch_loss_count += 1
                avg_loss = epoch_loss_sum / epoch_loss_count

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"loss={loss_value:.4f} avg={avg_loss:.4f}")
                elif step % max(1, args.log_every) == 0:
                    print(
                        f"epoch={epoch} step={step}/{len(dataloader)} "
                        f"loss={loss_value:.4f} avg={avg_loss:.4f}"
                    )

                if step % max(1, args.log_every) == 0 or step == len(dataloader):
                    elapsed = time.time() - started
                    with log_path.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "epoch",
                                "step",
                                "global_step",
                                "loss",
                                "avg_epoch_loss",
                                "elapsed_sec",
                            ],
                        )
                        writer.writerow(
                            {
                                "epoch": epoch,
                                "step": step,
                                "global_step": global_step,
                                "loss": f"{loss_value:.8f}",
                                "avg_epoch_loss": f"{avg_loss:.8f}",
                                "elapsed_sec": f"{elapsed:.2f}",
                            }
                        )
                    wandb_log(
                        run,
                        {
                            "train/loss": loss_value,
                            "train/avg_epoch_loss_so_far": avg_loss,
                            "train/epoch": epoch,
                            "train/step_in_epoch": step,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/examples_seen": global_step * batch_size,
                            "train/batches_per_sec": step / max(1e-6, elapsed),
                        },
                        step=global_step,
                    )

                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break

            if pbar is not None:
                pbar.close()

            epoch_avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"mapper_epoch_{epoch:03d}.pt",
                    mapper,
                    epoch,
                    global_step,
                    epoch_avg_loss,
                    args,
                    config,
                )
            final_mapper_path = output_dir / "mapper_ce_llm.pt"
            best_mapper_path = output_dir / "best_mapper_ce_llm.pt"
            save_checkpoint(
                final_mapper_path,
                mapper,
                epoch,
                global_step,
                epoch_avg_loss,
                args,
                config,
            )
            if best_loss is None or epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                save_checkpoint(
                    best_mapper_path,
                    mapper,
                    epoch,
                    global_step,
                    epoch_avg_loss,
                    args,
                    config,
                )
            wandb_log(
                run,
                {
                    "train/epoch_avg_loss": epoch_avg_loss,
                    "train/best_loss": best_loss,
                    "train/completed_epoch": epoch,
                },
                step=global_step,
            )
            if run is not None:
                run.summary["best_loss"] = best_loss
                run.summary["final_loss"] = epoch_avg_loss
                run.summary["global_step"] = global_step
            print(f"epoch={epoch} avg_loss={epoch_avg_loss:.6f} best_loss={best_loss:.6f}")

        final_mapper_path = output_dir / "mapper_ce_llm.pt"
        best_mapper_path = output_dir / "best_mapper_ce_llm.pt"
        if run is not None and args.wandb_log_model:
            import wandb

            artifact = wandb.Artifact(
                name=f"colmbo-mapper-{args.sampling}-{run.id}",
                type="model",
                metadata={
                    "sampling": args.sampling,
                    "best_loss": best_loss,
                    "epochs": args.epochs,
                    "global_step": global_step,
                },
            )
            artifact.add_file(str(final_mapper_path), name="mapper_ce_llm.pt")
            artifact.add_file(str(best_mapper_path), name="best_mapper_ce_llm.pt")
            run.log_artifact(artifact)

        print(f"Final mapper: {final_mapper_path}")
        print(f"Best mapper:  {best_mapper_path}")
        print(f"Train log:    {log_path}")
    finally:
        wandb_finish(run)


if __name__ == "__main__":
    main()
