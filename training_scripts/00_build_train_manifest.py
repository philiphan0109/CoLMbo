#!/usr/bin/env python3
"""Build an expanded all-task training manifest from TEARS metadata."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import (
    SOURCE_TASKS,
    TASKS,
    add_wandb_args,
    collect_task_values,
    init_wandb,
    iter_expanded_examples,
    iter_manifest,
    resolve_audio_reference,
    source_internal_split,
    setup_local_env,
    wandb_finish,
    wandb_log,
)


FIELDNAMES = [
    "example_id",
    "row_index",
    "source_prefix",
    "task",
    "label",
    "speaker_id",
    "audio_path",
    "resolved_audio_path",
    "resolution_method",
    "prompt",
    "response",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand TEARS rows into one row per supported task/prompt/response pair. "
            "This is the manifest consumed by the ECAPA precompute and mapper training scripts."
        )
    )
    parser.add_argument(
        "--raw-manifest",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Local TEARS JSONL manifest. Use --download if it does not exist.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the TEARS split from HuggingFace if --raw-manifest is missing.",
    )
    parser.add_argument("--hf-dataset", default="cmu-mlsp/TEARS", help="HuggingFace dataset id")
    parser.add_argument("--split", default="train", help="TEARS split to download")
    parser.add_argument(
        "--output",
        default="training_scripts/data/train_examples_ears_timit.csv",
        help="Expanded task-example CSV output path",
    )
    parser.add_argument("--ears-root", default="tears_audio", help="Local EARS raw audio root")
    parser.add_argument("--timit-root", default="timit_root", help="Local TIMIT root")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to include when supported by the source dataset",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_TASKS.keys()),
        choices=list(SOURCE_TASKS.keys()),
        help="TEARS audio_path source prefixes to include",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["train"],
        help=(
            "Internal source dataset splits to include from audio_path. "
            "Default is train only, which avoids leaking EARS val/test or TIMIT test "
            "audio into mapper training."
        ),
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Skip examples whose local audio cannot be resolved.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on raw TEARS rows to scan after source filtering (0 = all).",
    )
    add_wandb_args(parser, default_job_type="build_train_manifest")
    return parser.parse_args()


def maybe_download_manifest(raw_manifest, dataset_id, split):
    if raw_manifest.exists():
        return
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            f"{raw_manifest} does not exist and datasets is unavailable. "
            "Install the project requirements or run baseline_scripts/00_fetch_tears_manifest.py first."
        ) from exc

    raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, split=split)
    prefix_counts = Counter()
    with raw_manifest.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            audio_path = row.get("audio_path")
            prefix = str(audio_path).split("/")[0] if audio_path else "UNKNOWN"
            prefix_counts[prefix] += 1
            payload = {
                "index": idx,
                "audio_path": audio_path,
                "speaker": row.get("speaker"),
                "prompts": row.get("prompts"),
                "responses": row.get("responses"),
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    summary_path = raw_manifest.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_id,
                "split": split,
                "rows_exported": sum(prefix_counts.values()),
                "audio_path_prefix_counts": dict(prefix_counts),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )


def main():
    setup_local_env()
    args = parse_args()
    run = init_wandb(
        args,
        run_config={
            "stage": "build_train_manifest",
            "raw_manifest": args.raw_manifest,
            "output": args.output,
            "tasks": args.tasks,
            "sources": args.sources,
            "source_splits": args.source_splits,
            "check_audio": args.check_audio,
        },
    )
    raw_manifest = Path(args.raw_manifest)
    try:
        if args.download:
            maybe_download_manifest(raw_manifest, args.hf_dataset, args.split)
        if not raw_manifest.exists():
            raise FileNotFoundError(
                f"Missing raw manifest: {raw_manifest}. "
                "Pass --download or create it with baseline_scripts/00_fetch_tears_manifest.py."
            )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        roots = {
            "ears_dataset_processed": Path(args.ears_root) if args.ears_root else None,
            "timit_dataset": Path(args.timit_root) if args.timit_root else None,
        }
        wanted_sources = set(args.sources)
        wanted_source_splits = {str(split).lower() for split in args.source_splits}
        task_values = collect_task_values(raw_manifest)

        rows_scanned = 0
        examples_written = 0
        skipped_unresolved = Counter()
        source_counts = Counter()
        task_counts = Counter()
        label_counts = Counter()

        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            for raw_row in iter_manifest(raw_manifest):
                audio_path = raw_row.get("audio_path")
                source = str(audio_path).split("/")[0] if audio_path else ""
                if source not in wanted_sources:
                    continue
                internal_split = source_internal_split(audio_path)
                if wanted_source_splits and internal_split not in wanted_source_splits:
                    continue
                if args.max_rows and rows_scanned >= args.max_rows:
                    break
                rows_scanned += 1

                resolved_path = ""
                resolution_method = ""
                if args.check_audio:
                    resolved, method = resolve_audio_reference(audio_path, roots)
                    if resolved is None:
                        skipped_unresolved[(source, method)] += 1
                        continue
                    resolved_path = str(resolved)
                    resolution_method = method

                for example in iter_expanded_examples(raw_row, args.tasks, task_values):
                    example["example_id"] = examples_written
                    example["resolved_audio_path"] = resolved_path
                    example["resolution_method"] = resolution_method
                    writer.writerow({field: example.get(field, "") for field in FIELDNAMES})
                    examples_written += 1
                    source_counts[example["source_prefix"]] += 1
                    task_counts[example["task"]] += 1
                    label_counts[(example["task"], example["label"])] += 1

        summary = {
            "raw_manifest": str(raw_manifest),
            "output": str(output),
            "source_splits": sorted(wanted_source_splits),
            "rows_scanned": rows_scanned,
            "examples_written": examples_written,
            "source_counts": dict(source_counts),
            "task_counts": dict(task_counts),
            "task_label_counts": {
                f"{task}::{label}": count for (task, label), count in sorted(label_counts.items())
            },
            "skipped_unresolved_audio": {
                f"{source}::{reason}": count for (source, reason), count in sorted(skipped_unresolved.items())
            },
        }
        summary_path = output.with_suffix(".summary.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

        wandb_log(
            run,
            {
                "manifest/rows_scanned": rows_scanned,
                "manifest/examples_written": examples_written,
                "manifest/skipped_unresolved_audio": sum(skipped_unresolved.values()),
                **{f"manifest/task_count/{task}": count for task, count in task_counts.items()},
                **{f"manifest/source_count/{src}": count for src, count in source_counts.items()},
            },
        )
        if run is not None:
            run.summary.update(summary)

        print(f"Wrote expanded manifest: {output}")
        print(f"Wrote summary:           {summary_path}")
        print(f"Raw rows scanned:        {rows_scanned}")
        print(f"Examples written:        {examples_written}")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")
        if skipped_unresolved:
            print("Skipped unresolved audio:")
            for key, count in sorted(skipped_unresolved.items()):
                print(f"  {key}: {count}")
    finally:
        wandb_finish(run)


if __name__ == "__main__":
    main()
