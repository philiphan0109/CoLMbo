#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import build_ears_index, ensure_output_dir, setup_local_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a clean EARS-only index from the TEARS manifest."
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="TEARS manifest JSONL path",
    )
    parser.add_argument(
        "--output",
        default="embedding_visualization/data/index/ears_test_index.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main():
    setup_local_env()
    args = parse_args()
    output_path = Path(args.output)
    ensure_output_dir(output_path.parent)

    df = build_ears_index(Path(args.manifest))
    df.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(f"Speakers: {df['speaker_id'].nunique()}")
        print(
            "Labels present: "
            f"gender={df['gender'].notna().sum()}, "
            f"age={df['age'].notna().sum()}, "
            f"ethnicity={df['ethnicity'].notna().sum()}"
        )


if __name__ == "__main__":
    main()
