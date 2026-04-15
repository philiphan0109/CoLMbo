#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import ensure_output_dir, setup_local_env


COORD_SPECS = (
    ("pca", "ecapa_raw"),
    ("tsne", "ecapa_raw"),
    ("pca", "ecapa_mapper"),
    ("tsne", "ecapa_mapper"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge EARS baseline predictions with saved ECAPA coordinate files. "
            "Writes an utterance-level analysis table and a speaker-level summary table."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        default="embedding_visualization/runs/ears_default/analysis/ears_predictions_long.csv",
        help="EARS prediction table from 05_export_ears_predictions.py",
    )
    parser.add_argument(
        "--run-root",
        default="embedding_visualization/runs/ears_default",
        help="Embedding visualization run root containing data/ and plots/",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/runs/ears_default/analysis",
        help="Directory for merged analysis tables",
    )
    return parser.parse_args()


def load_coord_frame(run_root: Path, reducer: str, level: str, space_name: str):
    path = run_root / "plots" / f"coords_{reducer}_{level}_{space_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing coordinate CSV: {path}")
    df = pd.read_csv(path)
    prefix = f"{space_name}_{reducer}"
    return df.rename(columns={"x": f"{prefix}_x", "y": f"{prefix}_y"})


def merge_utterance_coords(pred_df: pd.DataFrame, run_root: Path):
    merged = pred_df.copy()
    for reducer, space_name in COORD_SPECS:
        coord_df = load_coord_frame(run_root, reducer, "utterance", space_name)
        keep_cols = [
            "audio_path",
            f"{space_name}_{reducer}_x",
            f"{space_name}_{reducer}_y",
            "speaker_id",
            "split",
            "gender",
            "age",
            "ethnicity",
        ]
        coord_df = coord_df[keep_cols]
        merged = merged.merge(
            coord_df,
            on=["audio_path", "speaker_id", "split"],
            how="left",
            suffixes=("", "_coord"),
        )
        for label_col in ("gender", "age", "ethnicity"):
            coord_name = f"{label_col}_coord"
            if coord_name in merged.columns:
                merged[label_col] = merged[label_col].fillna(merged[coord_name])
                merged = merged.drop(columns=[coord_name])

    def resolve_gold_value(row):
        task = row["task"]
        metadata_value = str(row.get(task, "")).strip().lower()
        if metadata_value:
            return metadata_value
        return str(row.get("gold_value", "")).strip().lower()

    merged["gold_value_resolved"] = merged.apply(resolve_gold_value, axis=1)
    merged["is_value_correct_resolved"] = (
        (merged["gold_value_resolved"] != "")
        & (merged["predicted_value"].fillna("").astype(str).str.strip().str.lower() == merged["gold_value_resolved"])
    ).astype(int)
    merged["is_incorrect_resolved"] = 1 - merged["is_value_correct_resolved"]
    return merged


def build_speaker_summary(utterance_df: pd.DataFrame, run_root: Path):
    summary = (
        utterance_df.groupby(["speaker_id", "task"], sort=True)
        .agg(
            split=("split", "first"),
            gold_value=("gold_value", "first"),
            gender=("gender", "first"),
            age=("age", "first"),
            ethnicity=("ethnicity", "first"),
            n_utterances=("audio_path", "count"),
            n_correct=("is_value_correct", "sum"),
            n_incorrect=("is_incorrect", "sum"),
            n_correct_resolved=("is_value_correct_resolved", "sum"),
            n_incorrect_resolved=("is_incorrect_resolved", "sum"),
            exact_match_rate=("is_exact_match", "mean"),
            value_accuracy=("is_value_correct", "mean"),
            value_accuracy_resolved=("is_value_correct_resolved", "mean"),
        )
        .reset_index()
    )
    summary["error_rate"] = 1.0 - summary["value_accuracy"]
    summary["error_rate_resolved"] = 1.0 - summary["value_accuracy_resolved"]

    for reducer, space_name in COORD_SPECS:
        coord_df = load_coord_frame(run_root, reducer, "speaker", space_name)
        keep_cols = [
            "speaker_id",
            f"{space_name}_{reducer}_x",
            f"{space_name}_{reducer}_y",
        ]
        summary = summary.merge(coord_df[keep_cols], on="speaker_id", how="left")
    return summary


def build_merge_summary(utterance_df: pd.DataFrame):
    rows = []
    for task, group in utterance_df.groupby("task", sort=True):
        n_rows = len(group)
        n_correct = int(group["is_value_correct"].sum())
        n_correct_resolved = int(group["is_value_correct_resolved"].sum())
        n_missing_raw = int(group["ecapa_raw_tsne_x"].isna().sum())
        n_missing_mapper = int(group["ecapa_mapper_tsne_x"].isna().sum())
        rows.append(
            {
                "task": task,
                "n_eval": n_rows,
                "n_correct": n_correct,
                "n_incorrect": n_rows - n_correct,
                "value_accuracy": (n_correct / n_rows) if n_rows else 0.0,
                "n_correct_resolved": n_correct_resolved,
                "n_incorrect_resolved": n_rows - n_correct_resolved,
                "value_accuracy_resolved": (n_correct_resolved / n_rows) if n_rows else 0.0,
                "missing_raw_coords": n_missing_raw,
                "missing_mapper_coords": n_missing_mapper,
            }
        )
    return pd.DataFrame(rows)


def main():
    setup_local_env()
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = ensure_output_dir(Path(args.output_dir))

    pred_df = pd.read_csv(Path(args.predictions_csv))
    utterance_df = merge_utterance_coords(pred_df, run_root)
    speaker_df = build_speaker_summary(utterance_df, run_root)
    summary_df = build_merge_summary(utterance_df)

    utterance_path = output_dir / "ecapa_prediction_analysis_utterance.csv"
    speaker_path = output_dir / "ecapa_prediction_analysis_speaker.csv"
    summary_path = output_dir / "ecapa_prediction_analysis_summary.csv"

    utterance_df.to_csv(utterance_path, index=False)
    speaker_df.to_csv(speaker_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote: {utterance_path}")
    print(f"Wrote: {speaker_path}")
    print(f"Wrote: {summary_path}")
    print(f"Utterance rows: {len(utterance_df)}")
    print(f"Speaker-task rows: {len(speaker_df)}")
    for row in summary_df.to_dict(orient="records"):
        print(
            f"{row['task']}: n={row['n_eval']} "
            f"acc(metadata)={100.0 * row['value_accuracy_resolved']:.2f}% "
            f"missing(raw={row['missing_raw_coords']}, mapper={row['missing_mapper_coords']})"
        )


if __name__ == "__main__":
    main()
