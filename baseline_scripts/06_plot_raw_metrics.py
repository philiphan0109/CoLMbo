#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot raw baseline metrics from 05_batch_eval_baseline outputs."
    )
    parser.add_argument(
        "--summary-csv",
        default="baseline_scripts/data/batch_eval/summary.csv",
        help="Summary CSV from 05_batch_eval_baseline.py",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_scripts/data/batch_eval/figures",
        help="Directory for output plots",
    )
    return parser.parse_args()


def save_heatmap(df, out_path):
    plot_df = df[df["source_prefix"] != "OVERALL"].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="task", columns="source_prefix", values="value_accuracy").fillna(0.0)
    tasks = list(pivot.index)
    sources = list(pivot.columns)
    values = pivot.values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Value Accuracy Heatmap (Task x Dataset)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value Accuracy")

    for i, task in enumerate(tasks):
        for j, src in enumerate(sources):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_grouped_accuracy(df, out_path):
    plot_df = df[df["source_prefix"] != "OVERALL"].copy()
    if plot_df.empty:
        return

    tasks = sorted(plot_df["task"].unique())
    sources = sorted(plot_df["source_prefix"].unique())
    width = 0.22
    x = list(range(len(tasks)))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, src in enumerate(sources):
        vals = []
        for t in tasks:
            hit = plot_df[(plot_df["task"] == t) & (plot_df["source_prefix"] == src)]
            vals.append(float(hit["value_accuracy"].iloc[0]) if not hit.empty else 0.0)
        xpos = [v + (idx - (len(sources) - 1) / 2) * width for v in x]
        ax.bar(xpos, vals, width=width, label=src)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Value Accuracy")
    ax.set_title("Value Accuracy by Task and Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_support_plot(df, out_path):
    plot_df = df[df["source_prefix"] != "OVERALL"].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(["task", "source_prefix"])
    labels = [f"{t}\n{s}" for t, s in zip(plot_df["task"], plot_df["source_prefix"])]
    counts = plot_df["n_eval"].astype(int).tolist()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Evaluated Samples")
    ax.set_title("Support per (Task, Dataset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    for col in ("value_accuracy", "exact_match_accuracy", "value_extraction_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    heatmap_path = out_dir / "01_value_accuracy_heatmap.png"
    grouped_path = out_dir / "02_value_accuracy_grouped.png"
    support_path = out_dir / "03_support_counts.png"

    save_heatmap(df, heatmap_path)
    save_grouped_accuracy(df, grouped_path)
    save_support_plot(df, support_path)

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {grouped_path}")
    print(f"Saved: {support_path}")


if __name__ == "__main__":
    main()
