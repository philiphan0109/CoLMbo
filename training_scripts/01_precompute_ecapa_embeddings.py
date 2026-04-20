#!/usr/bin/env python3
"""Precompute frozen ECAPA embeddings for expanded CoLMbo training examples."""

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from common import (
    load_config,
    load_sid_model_from_config,
    load_waveform_for_manifest,
    maybe_tqdm,
    resolve_device,
    setup_local_env,
)


EXAMPLE_FIELD_EXTRA = "embedding_id"
INDEX_FIELDS = [
    "embedding_id",
    "source_prefix",
    "audio_path",
    "resolved_audio_path",
    "resolution_method",
    "n_examples",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute one frozen ECAPA vector per unique audio_path in an expanded "
            "task-example manifest, then write an embedding-indexed training CSV."
        )
    )
    parser.add_argument(
        "--examples-csv",
        default="training_scripts/data/train_examples_ears_timit.csv",
        help="Expanded CSV from 00_build_train_manifest.py",
    )
    parser.add_argument(
        "--output-dir",
        default="training_scripts/data/ears_timit_ecapa_cache",
        help="Directory for embeddings.npy, index CSV, and training CSV",
    )
    parser.add_argument("--config", default="config.yaml", help="Project config path")
    parser.add_argument("--ears-root", default="tears_audio", help="Local EARS raw audio root")
    parser.add_argument("--timit-root", default="timit_root", help="Local TIMIT root")
    parser.add_argument("--device", default="cuda", help="Device for ECAPA extraction")
    parser.add_argument(
        "--sid-checkpoint",
        default=None,
        help="Optional sid_model.pt override. Defaults to config train snapshot.",
    )
    parser.add_argument(
        "--limit-audio",
        type=int,
        default=0,
        help="Optional cap on unique audio items for smoke tests (0 = all)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Fallback progress print interval if tqdm is unavailable",
    )
    return parser.parse_args()


def read_examples(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_unique_audio(rows):
    unique = OrderedDict()
    for row in rows:
        audio_path = row["audio_path"]
        if audio_path not in unique:
            unique[audio_path] = {
                "source_prefix": row["source_prefix"],
                "audio_path": audio_path,
                "n_examples": 0,
            }
        unique[audio_path]["n_examples"] += 1
    return unique


def main():
    setup_local_env()
    args = parse_args()
    examples = read_examples(args.examples_csv)
    if not examples:
        raise RuntimeError(f"No examples found in {args.examples_csv}")

    config = load_config(args.config)
    sample_rate = int(config["data"].get("sample_rate", 16000))
    device = resolve_device(args.device)
    roots = {
        "ears_dataset_processed": Path(args.ears_root) if args.ears_root else None,
        "timit_dataset": Path(args.timit_root) if args.timit_root else None,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_audio = build_unique_audio(examples)
    audio_items = list(unique_audio.values())
    if args.limit_audio and args.limit_audio > 0:
        audio_items = audio_items[: args.limit_audio]

    from load_data.extract_fbanks import Mel_Spectrogram

    extractor = Mel_Spectrogram(sample_rate=sample_rate).to(device)
    sid_model = load_sid_model_from_config(args.config, device, args.sid_checkpoint)

    embedding_rows = []
    vectors = []
    errors = []
    embedding_id_by_audio = {}
    pbar = maybe_tqdm(len(audio_items), desc="ECAPA precompute", unit="audio")
    started = time.time()

    with torch.no_grad():
        for idx, item in enumerate(audio_items):
            audio_path = item["audio_path"]
            try:
                waveform, _, resolved_path, method = load_waveform_for_manifest(
                    audio_path,
                    roots,
                    sample_rate=sample_rate,
                )
                waveform = waveform.to(device)
                features = extractor(waveform)
                embedding = sid_model(features)
                vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
                embedding_id = len(vectors)
                vectors.append(vector)
                embedding_id_by_audio[audio_path] = embedding_id
                embedding_rows.append(
                    {
                        "embedding_id": embedding_id,
                        "source_prefix": item["source_prefix"],
                        "audio_path": audio_path,
                        "resolved_audio_path": resolved_path,
                        "resolution_method": method,
                        "n_examples": item["n_examples"],
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "source_prefix": item["source_prefix"],
                        "audio_path": audio_path,
                        "error": str(exc),
                    }
                )

            if pbar is not None:
                pbar.update(1)
            elif (idx + 1) % max(1, args.progress_every) == 0 or idx + 1 == len(audio_items):
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"[{idx + 1}/{len(audio_items)}] "
                    f"ok={len(vectors)} errors={len(errors)} rate={(idx + 1) / elapsed:.2f}/s"
                )

    if pbar is not None:
        pbar.close()

    if not vectors:
        raise RuntimeError("No ECAPA embeddings were computed. Check roots and audio resolution errors.")

    embeddings_path = output_dir / "audio_embeddings.npy"
    index_path = output_dir / "audio_embedding_index.csv"
    train_csv = output_dir / "train_examples_with_embeddings.csv"
    errors_path = output_dir / "audio_embedding_errors.csv"
    summary_path = output_dir / "summary.json"

    np.save(embeddings_path, np.vstack(vectors).astype(np.float32))
    write_csv(index_path, embedding_rows, INDEX_FIELDS)

    train_rows = []
    fieldnames = list(examples[0].keys())
    if EXAMPLE_FIELD_EXTRA not in fieldnames:
        fieldnames.append(EXAMPLE_FIELD_EXTRA)
    for row in examples:
        embedding_id = embedding_id_by_audio.get(row["audio_path"])
        if embedding_id is None:
            continue
        out = dict(row)
        out[EXAMPLE_FIELD_EXTRA] = embedding_id
        train_rows.append(out)
    write_csv(train_csv, train_rows, fieldnames)

    if errors:
        write_csv(errors_path, errors, ["source_prefix", "audio_path", "error"])

    summary = {
        "examples_csv": str(args.examples_csv),
        "output_dir": str(output_dir),
        "n_input_examples": len(examples),
        "n_unique_audio_requested": len(audio_items),
        "n_unique_audio_embedded": len(vectors),
        "n_audio_errors": len(errors),
        "n_training_examples_with_embeddings": len(train_rows),
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "train_csv": str(train_csv),
        "errors_path": str(errors_path) if errors else "",
        "device": str(device),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Embeddings:     {embeddings_path}")
    print(f"Index CSV:      {index_path}")
    print(f"Training CSV:   {train_csv}")
    print(f"Embedded audio: {len(vectors)} / {len(audio_items)}")
    print(f"Training rows:  {len(train_rows)} / {len(examples)}")
    if errors:
        print(f"Errors CSV:     {errors_path}")
    print(f"Device used:    {device}")


if __name__ == "__main__":
    main()
