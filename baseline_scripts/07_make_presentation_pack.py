#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a presentation-ready summary pack from baseline outputs."
    )
    parser.add_argument(
        "--results-summary-csv",
        default="baseline_scripts/data/batch_eval/summary.csv",
        help="Summary CSV with real inference metrics (from 05_batch_eval_baseline.py).",
    )
    parser.add_argument(
        "--results-failures-csv",
        default="baseline_scripts/data/batch_eval/failures.csv",
        help="Failures CSV from 05_batch_eval_baseline.py.",
    )
    parser.add_argument(
        "--coverage-summary-csv",
        default="baseline_scripts/data/batch_eval_coverage_now/summary.csv",
        help="Dry-run coverage summary CSV (from 05_batch_eval_baseline.py --dry-run).",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_scripts/data/presentation_pack",
        help="Directory to write presentation artifacts.",
    )
    parser.add_argument(
        "--top-failures",
        type=int,
        default=20,
        help="How many failure samples to export.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def pick_row(rows, task, source_prefix):
    for r in rows:
        if r.get("task") == task and r.get("source_prefix") == source_prefix:
            return r
    return None


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(Path(args.results_summary_csv))
    failures_rows = read_csv(Path(args.results_failures_csv))
    coverage_rows = read_csv(Path(args.coverage_summary_csv))

    # Extract key rows for quick presentation stats.
    age_ears = pick_row(summary_rows, "age", "ears_dataset_processed")
    gender_ears = pick_row(summary_rows, "gender", "ears_dataset_processed")
    dialect_timit = pick_row(summary_rows, "dialect", "timit_dataset")
    age_coverage = pick_row(coverage_rows, "age", "ears_dataset_processed")
    gender_coverage = pick_row(coverage_rows, "gender", "ears_dataset_processed")

    n_age_eval = to_int(age_ears["n_eval"]) if age_ears else 0
    n_gender_eval = to_int(gender_ears["n_eval"]) if gender_ears else 0
    age_acc = to_float(age_ears["value_accuracy"]) if age_ears else 0.0
    gender_acc = to_float(gender_ears["value_accuracy"]) if gender_ears else 0.0
    age_exact = to_float(age_ears["exact_match_accuracy"]) if age_ears else 0.0
    gender_exact = to_float(gender_ears["exact_match_accuracy"]) if gender_ears else 0.0
    dialect_eval = to_int(dialect_timit["n_eval"]) if dialect_timit else 0

    age_cov = to_int(age_coverage["n_selected"]) if age_coverage else 0
    gender_cov = to_int(gender_coverage["n_selected"]) if gender_coverage else 0

    # Keep failure export concise and focused.
    trimmed_failures = []
    for r in failures_rows[: args.top_failures]:
        trimmed_failures.append(
            {
                "task": r.get("task", ""),
                "source_prefix": r.get("source_prefix", ""),
                "true_value": r.get("true_value", ""),
                "pred_value": r.get("pred_value", ""),
                "prompt": r.get("prompt", ""),
                "prediction": r.get("prediction", ""),
                "audio_path": r.get("audio_path", ""),
            }
        )

    fail_csv = out_dir / "02_top_failures.csv"
    write_csv(
        fail_csv,
        trimmed_failures,
        ["task", "source_prefix", "true_value", "pred_value", "prompt", "prediction", "audio_path"],
    )

    # Compact table for slides.
    metrics_table = [
        {
            "task": "gender",
            "dataset": "ears_dataset_processed",
            "n_eval": n_gender_eval,
            "value_accuracy": f"{gender_acc:.4f}",
            "exact_match_accuracy": f"{gender_exact:.4f}",
        },
        {
            "task": "age",
            "dataset": "ears_dataset_processed",
            "n_eval": n_age_eval,
            "value_accuracy": f"{age_acc:.4f}",
            "exact_match_accuracy": f"{age_exact:.4f}",
        },
        {
            "task": "dialect",
            "dataset": "timit_dataset",
            "n_eval": dialect_eval,
            "value_accuracy": "N/A",
            "exact_match_accuracy": "N/A",
        },
    ]
    metrics_csv = out_dir / "01_metrics_table.csv"
    write_csv(
        metrics_csv,
        metrics_table,
        ["task", "dataset", "n_eval", "value_accuracy", "exact_match_accuracy"],
    )

    summary_md = out_dir / "00_exec_summary.md"
    summary_text = f"""# CoLMbo Baseline: Interim Results (for Presentation)

## What Was Evaluated
- Split: `TEARS test`
- Available audio now: `EARS` subset downloaded locally (30 speakers: p001-p030)
- Evaluated tasks in current run: `gender`, `age` (EARS only)
- `dialect` is blocked pending TIMIT local access

## Raw Performance (Current Run)
- Gender (EARS): value accuracy = **{gender_acc:.2%}** on **{n_gender_eval}** samples
- Age (EARS): value accuracy = **{age_acc:.2%}** on **{n_age_eval}** samples
- Exact-match accuracy is lower (expected) due paraphrasing:
  - Gender exact match = **{gender_exact:.2%}**
  - Age exact match = **{age_exact:.2%}**

## Coverage With Current Local Data (Dry-Run)
- Potential evaluable EARS rows now:
  - Gender: **{gender_cov}**
  - Age: **{age_cov}**
- Dialect: requires TIMIT root; currently **not evaluated**

## Key Findings
- Model is strong on gender for current EARS subset.
- Model makes age-range confusions (`36-45` -> `18-25` / `26-35`) in failure samples.
- Exact-match metric alone underestimates quality for generative answers; value-level scoring is more appropriate.

## Blocking Item
- TIMIT (LDC93S1) path is required for dialect baseline.
"""
    summary_md.write_text(summary_text, encoding="utf-8")

    talking_md = out_dir / "03_talking_points.md"
    talking_text = """# Talking Points (2-3 min)

- I built a reproducible baseline pipeline around TEARS test metadata and external audio sources.
- I validated and loaded EARS audio locally; current run evaluates gender and age.
- Current raw value-level results:
  - Gender is near-perfect on the evaluated subset.
  - Age shows meaningful errors, mainly underestimating older speakers.
- Exact string match is not ideal for this generative model; value-level correctness is a better baseline metric.
- Dialect evaluation is blocked only by missing TIMIT local access (licensed LDC dataset), not by pipeline readiness.
- Next immediate step: add TIMIT root, rerun the same pipeline, and report full 3-task baseline.
"""
    talking_md.write_text(talking_text, encoding="utf-8")

    print(f"Wrote: {summary_md}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {fail_csv}")
    print(f"Wrote: {talking_md}")


if __name__ == "__main__":
    main()
