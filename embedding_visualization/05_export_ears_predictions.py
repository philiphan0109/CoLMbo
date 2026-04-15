#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import (
    LABEL_COLUMNS,
    ensure_output_dir,
    extract_speaker_id,
    parse_ears_segment,
    setup_local_env,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a clean EARS-only baseline prediction table from existing batch-eval "
            "artifacts. Prefers a pre-aggregated predictions CSV and falls back to "
            "concatenating shard predictions."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        default="baseline_scripts/data/ears_full_sharded/predictions_all.csv",
        help="Preferred full predictions CSV path",
    )
    parser.add_argument(
        "--chunk-root",
        default="baseline_scripts/data/ears_full_sharded",
        help="Shard root used if --predictions-csv does not exist",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/runs/ears_default/analysis",
        help="Directory for exported EARS prediction tables",
    )
    return parser.parse_args()


def load_predictions(predictions_csv: Path, chunk_root: Path):
    if predictions_csv.exists():
        return pd.read_csv(predictions_csv)

    chunk_files = sorted(chunk_root.glob("chunk_*/predictions.csv"))
    if not chunk_files:
        raise FileNotFoundError(
            f"Could not find {predictions_csv} or any chunk predictions under {chunk_root}"
        )
    return pd.concat((pd.read_csv(path) for path in chunk_files), ignore_index=True)


def build_clean_table(df: pd.DataFrame):
    df = df.copy()
    df = df[df["source_prefix"] == "ears_dataset_processed"]
    df = df[df["task"].isin(LABEL_COLUMNS)]
    df = df[df["status"] == "ok"]
    df = df.drop_duplicates(subset=["task", "audio_path"]).reset_index(drop=True)

    df["speaker_id"] = df["audio_path"].map(extract_speaker_id)
    df["split"] = df["audio_path"].map(
        lambda path: (parse_ears_segment(path) or {}).get("split", "")
    )
    df["gold_value"] = df["true_value"].fillna("").astype(str).str.strip().str.lower()
    df["predicted_value"] = df["pred_value"].fillna("").astype(str).str.strip().str.lower()
    df["is_exact_match"] = pd.to_numeric(
        df["is_exact_match"], errors="coerce"
    ).fillna(0).astype(int)
    df["is_value_correct"] = pd.to_numeric(
        df["is_value_correct"], errors="coerce"
    ).fillna(0).astype(int)
    df["is_incorrect"] = 1 - df["is_value_correct"]

    keep_columns = [
        "task",
        "audio_path",
        "speaker_id",
        "split",
        "prompt",
        "gold_response",
        "prediction",
        "gold_value",
        "predicted_value",
        "is_exact_match",
        "is_value_correct",
        "is_incorrect",
    ]
    return df[keep_columns].sort_values(["task", "audio_path"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame):
    rows = []
    for task, group in df.groupby("task", sort=True):
        n_rows = len(group)
        n_correct = int(group["is_value_correct"].sum())
        rows.append(
            {
                "task": task,
                "n_eval": n_rows,
                "n_correct": n_correct,
                "n_incorrect": n_rows - n_correct,
                "value_accuracy": (n_correct / n_rows) if n_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    setup_local_env()
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))

    raw_df = load_predictions(Path(args.predictions_csv), Path(args.chunk_root))
    clean_df = build_clean_table(raw_df)
    summary_df = build_summary(clean_df)

    long_path = output_dir / "ears_predictions_long.csv"
    summary_path = output_dir / "ears_predictions_summary.csv"

    clean_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote: {long_path}")
    print(f"Wrote: {summary_path}")
    print(f"Rows: {len(clean_df)}")
    print(f"Utterances: {clean_df['audio_path'].nunique()}")
    print(f"Speakers: {clean_df['speaker_id'].nunique()}")
    for row in summary_df.to_dict(orient="records"):
        print(
            f"{row['task']}: n={row['n_eval']} "
            f"acc={100.0 * row['value_accuracy']:.2f}%"
        )


if __name__ == "__main__":
    main()
