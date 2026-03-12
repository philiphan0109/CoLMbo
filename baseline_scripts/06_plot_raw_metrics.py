#!/usr/bin/env python3
import argparse
import math
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
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output resolution for PNG figures",
    )
    return parser.parse_args()


def prep_df(df):
    out = df[df["source_prefix"] != "OVERALL"].copy()
    for col in (
        "value_accuracy",
        "exact_match_accuracy",
        "value_extraction_rate",
        "n_eval",
        "n_value_available",
        "n_value_correct",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    delta = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    low = max(0.0, center - delta)
    high = min(1.0, center + delta)
    return low, high


def save_value_heatmap(df, out_path, dpi):
    if df.empty:
        return

    acc = (
        df.pivot(index="task", columns="source_prefix", values="value_accuracy")
        .fillna(0.0)
        .sort_index()
    )
    support = (
        df.pivot(index="task", columns="source_prefix", values="n_value_available")
        .fillna(0.0)
        .astype(int)
        .reindex(index=acc.index, columns=acc.columns)
    )

    tasks = list(acc.index)
    sources = list(acc.columns)
    values = acc.values

    fig, ax = plt.subplots(figsize=(10, max(4.8, 1.2 * len(tasks))))
    im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Value Accuracy by Task and Dataset")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Value Accuracy")

    for i in range(len(tasks)):
        for j in range(len(sources)):
            val = values[i, j]
            n = support.iat[i, j]
            color = "white" if val >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}\n(n={n})",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_grouped_value_with_ci(df, out_path, dpi):
    if df.empty:
        return

    tasks = sorted(df["task"].unique())
    sources = sorted(df["source_prefix"].unique())
    width = min(0.78 / max(len(sources), 1), 0.26)
    x = list(range(len(tasks)))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for idx, src in enumerate(sources):
        vals = []
        err_lo = []
        err_hi = []
        for task in tasks:
            hit = df[(df["task"] == task) & (df["source_prefix"] == src)]
            if hit.empty:
                vals.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            row = hit.iloc[0]
            val = float(row["value_accuracy"])
            n = int(row["n_value_available"])
            k = int(row["n_value_correct"])
            low, high = wilson_interval(k, n)
            vals.append(val)
            err_lo.append(max(0.0, val - low))
            err_hi.append(max(0.0, high - val))

        xpos = [v + (idx - (len(sources) - 1) / 2.0) * width for v in x]
        ax.bar(
            xpos,
            vals,
            width=width,
            label=src,
            yerr=[err_lo, err_hi],
            capsize=3,
            linewidth=0.6,
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Value Accuracy")
    ax.set_title("Value Accuracy with 95% Wilson Confidence Intervals")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_exact_vs_value(df, out_path, dpi):
    if df.empty:
        return

    plot_df = df.sort_values(["task", "source_prefix"]).reset_index(drop=True)
    labels = [f"{t}\n{s}" for t, s in zip(plot_df["task"], plot_df["source_prefix"])]
    exact = plot_df["exact_match_accuracy"].astype(float).tolist()
    value = plot_df["value_accuracy"].astype(float).tolist()
    x = list(range(len(labels)))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(9.5, 0.75 * len(labels)), 5.0))
    ax.bar([i - width / 2 for i in x], exact, width=width, label="Exact Match", color="#577590")
    ax.bar([i + width / 2 for i in x], value, width=width, label="Value Accuracy", color="#43aa8b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exact Match vs Value Accuracy")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_support_and_extraction(df, out_path, dpi):
    if df.empty:
        return

    plot_df = df.sort_values(["task", "source_prefix"]).reset_index(drop=True)
    labels = [f"{t}\n{s}" for t, s in zip(plot_df["task"], plot_df["source_prefix"])]
    support = plot_df["n_eval"].astype(int).tolist()
    extraction = plot_df["value_extraction_rate"].astype(float).tolist()
    x = list(range(len(labels)))

    fig, ax1 = plt.subplots(figsize=(max(9.5, 0.72 * len(labels)), 5.1))
    bars = ax1.bar(x, support, color="#4d908e", alpha=0.85, label="n_eval")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_ylabel("Evaluated Samples")
    ax1.set_title("Support and Value Extraction Rate")

    ax2 = ax1.twinx()
    ax2.plot(x, extraction, color="#f3722c", marker="o", linewidth=2.0, label="Value Extraction Rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Extraction Rate")

    handles = [bars, ax2.lines[0]]
    labels_legend = ["n_eval", "Value Extraction Rate"]
    ax1.legend(handles, labels_legend, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    df = prep_df(pd.read_csv(summary_path))

    fig_paths = [
        out_dir / "01_value_accuracy_heatmap.png",
        out_dir / "02_value_accuracy_with_ci.png",
        out_dir / "03_exact_vs_value_accuracy.png",
        out_dir / "04_support_and_extraction_rate.png",
    ]

    save_value_heatmap(df, fig_paths[0], args.dpi)
    save_grouped_value_with_ci(df, fig_paths[1], args.dpi)
    save_exact_vs_value(df, fig_paths[2], args.dpi)
    save_support_and_extraction(df, fig_paths[3], args.dpi)

    for p in fig_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
