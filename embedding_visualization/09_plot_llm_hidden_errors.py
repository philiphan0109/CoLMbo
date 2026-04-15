#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import LABEL_COLUMNS, ensure_output_dir, load_artifact_pair, setup_local_env

setup_local_env()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reduce and plot in-LM hidden-state embeddings. Each task is reduced separately "
            "so the task prompt does not dominate the visualization."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        default="embedding_visualization/runs/ears_default/llm_analysis/utterance_task_llm_hidden_last_input_metadata.csv",
        help="LLM hidden-state metadata CSV from 08_extract_llm_hidden_states.py",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/runs/ears_default/llm_analysis/plots",
        help="Output directory for LLM hidden-state plots",
    )
    parser.add_argument(
        "--reducers",
        nargs="+",
        default=["tsne", "pca"],
        choices=["tsne", "pca"],
        help="Reducers to run",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Base t-SNE perplexity, clipped for small subsets",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reducers",
    )
    parser.add_argument(
        "--hide-errors",
        action="store_true",
        help="Do not draw black outlines around incorrect predictions",
    )
    return parser.parse_args()


def reduce_embeddings(embeddings, reducer_name: str, perplexity: float, random_state: int):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 rows to reduce embeddings.")

    scaled = StandardScaler().fit_transform(embeddings)
    if reducer_name == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(scaled)

    n_rows = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(1.0, (n_rows - 1) / 3.0), n_rows - 1e-3)
    pca_components = min(50, scaled.shape[1], max(2, n_rows - 1))
    pca_input = PCA(n_components=pca_components, random_state=random_state).fit_transform(scaled)
    return TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity,
        max_iter=1000,
    ).fit_transform(pca_input)


def build_palette(values):
    labels = [x for x in sorted(pd.Series(values).dropna().astype(str).unique()) if x != ""]
    cmap_name = "tab10" if len(labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
    return {label: cmap(i) for i, label in enumerate(labels)}


def plot_grid(
    meta_df,
    embeddings,
    reducer_name: str,
    output_dir: Path,
    perplexity: float,
    random_state: int,
    show_errors: bool,
):
    fig, axes = plt.subplots(
        1,
        len(LABEL_COLUMNS),
        figsize=(5.2 * len(LABEL_COLUMNS), 4.7),
        squeeze=False,
    )

    summary_rows = []
    for col_idx, task in enumerate(LABEL_COLUMNS):
        ax = axes[0][col_idx]
        mask = meta_df["task"] == task
        task_df = meta_df[mask].reset_index(drop=True)
        if len(task_df) < 2:
            ax.set_title(f"{task.capitalize()}\nnot enough rows")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        task_embeddings = embeddings[mask.to_numpy()]
        coords = reduce_embeddings(task_embeddings, reducer_name, perplexity, random_state)

        coord_df = task_df.copy()
        coord_df["x"] = coords[:, 0]
        coord_df["y"] = coords[:, 1]
        coord_path = output_dir / f"coords_{reducer_name}_llm_hidden_{task}.csv"
        coord_df.to_csv(coord_path, index=False)

        n_rows = len(task_df)
        n_incorrect = int(task_df["is_incorrect_resolved"].sum())
        accuracy = 1.0 - (n_incorrect / n_rows if n_rows else 0.0)
        summary_rows.append(
            {
                "task": task,
                "n_eval": n_rows,
                "n_incorrect": n_incorrect,
                "value_accuracy": accuracy,
            }
        )

        palette = build_palette(task_df["gold_value_resolved"])
        for label_value, color in palette.items():
            label_mask = task_df["gold_value_resolved"] == label_value
            ax.scatter(
                coords[label_mask.to_numpy(), 0],
                coords[label_mask.to_numpy(), 1],
                s=7,
                c=[color],
                alpha=0.65,
                edgecolors="none",
                label=label_value,
            )

        incorrect_df = coord_df[coord_df["is_incorrect_resolved"] == 1]
        if show_errors and not incorrect_df.empty:
            ax.scatter(
                incorrect_df["x"],
                incorrect_df["y"],
                s=28,
                facecolors="none",
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )

        ax.set_title(
            f"{task.capitalize()}\n"
            f"n={n_rows}, acc={100.0 * accuracy:.1f}%, err={n_incorrect}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, frameon=True)

    pooling = str(meta_df["pooling"].iloc[0]) if "pooling" in meta_df.columns else "unknown"
    fig.suptitle(
        f"{reducer_name.upper()} In-LM GPT Hidden-State Error Overlay ({pooling})\n"
        +
        (
            "Colors = gold label, black outlines = incorrect baseline predictions"
            if show_errors
            else "Colors = gold label"
        ),
        fontsize=14,
    )
    fig.tight_layout()

    suffix = "error_grid" if show_errors else "label_grid"
    out_path = output_dir / f"{reducer_name}_llm_hidden_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    summary_path = output_dir / f"{reducer_name}_llm_hidden_error_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


def main():
    setup_local_env()
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))
    plt.style.use("seaborn-v0_8-whitegrid")

    meta_df, embeddings = load_artifact_pair(Path(args.metadata_csv))
    for reducer in args.reducers:
        plot_grid(
            meta_df,
            embeddings,
            reducer,
            output_dir,
            args.tsne_perplexity,
            args.random_state,
            show_errors=not args.hide_errors,
        )


if __name__ == "__main__":
    main()
