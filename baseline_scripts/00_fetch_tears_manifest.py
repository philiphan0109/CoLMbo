#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch TEARS metadata from Hugging Face and write a local JSONL manifest."
    )
    parser.add_argument("--dataset", default="cmu-mlsp/TEARS", help="HF dataset repo id")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--output",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to export",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds) if args.limit is None else min(args.limit, len(ds))

    prefix_counts = {}
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            if idx >= total:
                break
            audio_path = row.get("audio_path")
            prefix = str(audio_path).split("/")[0] if audio_path else "UNKNOWN"
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

            payload = {
                "index": idx,
                "audio_path": audio_path,
                "speaker": row.get("speaker"),
                "prompts": row.get("prompts"),
                "responses": row.get("responses"),
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "split": args.split,
                "rows_exported": total,
                "audio_path_prefix_counts": prefix_counts,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Wrote manifest: {out_path}")
    print(f"Wrote summary:  {summary_path}")
    print(f"Rows exported:  {total}")
    print("audio_path prefixes:")
    for k in sorted(prefix_counts):
        print(f"  {k}: {prefix_counts[k]}")


if __name__ == "__main__":
    main()
