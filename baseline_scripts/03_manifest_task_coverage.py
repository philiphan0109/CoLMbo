#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


TASKS = ("gender", "age", "dialect")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report task prompt coverage by source prefix in a TEARS manifest."
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Manifest JSONL from 00_fetch_tears_manifest.py",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to scan.",
    )
    return parser.parse_args()


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def has_task_prompt(prompts, task):
    prompts = prompts or []
    key = task.lower()
    return any(key in str(p).lower() for p in prompts)


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    stats = defaultdict(lambda: {"rows": 0, "gender": 0, "age": 0, "dialect": 0})
    scanned = 0

    for row in iter_manifest(manifest):
        scanned += 1
        if args.max_rows is not None and scanned > args.max_rows:
            break

        audio_path = row.get("audio_path")
        prefix = str(audio_path).split("/")[0] if audio_path else "UNKNOWN"
        stats[prefix]["rows"] += 1

        prompts = row.get("prompts")
        for task in TASKS:
            if has_task_prompt(prompts, task):
                stats[prefix][task] += 1

    print(f"Scanned rows: {scanned}")
    print("")
    print("source_prefix,rows,rows_with_gender,rows_with_age,rows_with_dialect")
    for prefix in sorted(stats):
        s = stats[prefix]
        print(f"{prefix},{s['rows']},{s['gender']},{s['age']},{s['dialect']}")


if __name__ == "__main__":
    main()
