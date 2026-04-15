#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import LABEL_COLUMNS, ensure_output_dir, setup_local_env

setup_local_env()

import matplotlib.pyplot as plt
import pandas as pd

SPACE_ORDER = ("ecapa_raw", "ecapa_mapper")
SPACE_TITLES = {
    "ecapa_raw": "ECAPA Raw",
    "ecapa_mapper": "ECAPA Mapper",
}
REDUCER_CHOICES = ("tsne", "pca")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create ECAPA prediction-error overlay plots. Points are colored by gold label "
            "and incorrect baseline predictions are highlighted."
        )
    )
    parser.add_argument(
        "--utterance-csv",
        default="embedding_visualization/runs/ears_default/analysis/ecapa_prediction_analysis_utterance.csv",
        help="Merged utterance-level analysis CSV from 06_merge_ecapa_with_predictions.py",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/runs/ears_default/error_plots",
        help="Directory for error-overlay figures and summaries",
    )
    parser.add_argument(
        "--reducers",
        nargs="+",
        default=["tsne"],
        choices=list(REDUCER_CHOICES),
        help="Reducers to visualize",
    )
    return parser.parse_args()


def build_palette(values):
    labels = [x for x in sorted(pd.Series(values).dropna().astype(str).unique()) if x != ""]
    cmap_name = "tab10" if len(labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
    return {label: cmap(i) for i, label in enumerate(labels)}


def make_grid(df: pd.DataFrame, reducer: str, output_dir: Path):
    fig, axes = plt.subplots(
        len(SPACE_ORDER),
        len(LABEL_COLUMNS),
        figsize=(5.0 * len(LABEL_COLUMNS), 4.5 * len(SPACE_ORDER)),
        squeeze=False,
    )

    task_summary_rows = []
    for col_idx, task in enumerate(LABEL_COLUMNS):
        task_df = df[df["task"] == task].copy()
        n_rows = len(task_df)
        n_incorrect = int(task_df["is_incorrect_resolved"].sum())
        accuracy = 1.0 - (n_incorrect / n_rows if n_rows else 0.0)
        task_summary_rows.append(
            {
                "task": task,
                "n_eval": n_rows,
                "n_incorrect": n_incorrect,
                "value_accuracy": accuracy,
            }
        )

        palette = build_palette(task_df["gold_value_resolved"])

        for row_idx, space_name in enumerate(SPACE_ORDER):
            ax = axes[row_idx][col_idx]
            x_col = f"{space_name}_{reducer}_x"
            y_col = f"{space_name}_{reducer}_y"
            plot_df = task_df.dropna(subset=[x_col, y_col]).copy()

            for label_value, color in palette.items():
                mask = plot_df["gold_value_resolved"] == label_value
                ax.scatter(
                    plot_df.loc[mask, x_col],
                    plot_df.loc[mask, y_col],
                    s=7,
                    c=[color],
                    alpha=0.65,
                    edgecolors="none",
                    label=label_value,
                )

            incorrect_df = plot_df[plot_df["is_incorrect_resolved"] == 1]
            if not incorrect_df.empty:
                ax.scatter(
                    incorrect_df[x_col],
                    incorrect_df[y_col],
                    s=28,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.9,
                )

            if row_idx == 0:
                ax.set_title(
                    f"{task.capitalize()}\n"
                    f"n={n_rows}, acc={100.0 * accuracy:.1f}%, err={n_incorrect}"
                )
            if col_idx == 0:
                ax.set_ylabel(SPACE_TITLES[space_name])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="best", fontsize=7, frameon=True)

    fig.suptitle(
        f"{reducer.upper()} Utterance-Level ECAPA Error Overlay\n"
        "Colors = gold label, black outlines = incorrect baseline predictions",
        fontsize=14,
    )
    fig.tight_layout()

    out_path = output_dir / f"{reducer}_utterance_error_grid.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    summary_path = output_dir / f"{reducer}_utterance_error_summary.csv"
    pd.DataFrame(task_summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


def main():
    setup_local_env()
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))

    plt.style.use("seaborn-v0_8-whitegrid")
    df = pd.read_csv(Path(args.utterance_csv))

    for reducer in args.reducers:
        make_grid(df, reducer, output_dir)


if __name__ == "__main__":
    main()
