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

SPACE_ORDER = ("pdaf_raw", "ecapa_raw", "ecapa_mapper")
SPACE_TITLES = {
    "pdaf_raw": "PDAF Raw",
    "ecapa_raw": "ECAPA Raw",
    "ecapa_mapper": "ECAPA Mapper",
}
LEVEL_ORDER = ("utterance", "speaker")
REDUCER_ORDER = ("pca", "tsne")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create EARS embedding visualization grids from saved artifacts. "
            "Produces a small final review set: PCA/t-SNE x utterance/speaker."
        )
    )
    parser.add_argument(
        "--ecapa-dir",
        default="embedding_visualization/data/ecapa",
        help="Directory containing ECAPA artifact subdirectories",
    )
    parser.add_argument(
        "--pdaf-dir",
        default="embedding_visualization/data/pdaf",
        help="Directory containing PDAF artifact subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/output",
        help="Directory for final plots and coordinate CSVs",
    )
    parser.add_argument(
        "--reducers",
        nargs="+",
        default=["pca", "tsne"],
        choices=list(REDUCER_ORDER),
        help="Reducers to run",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Base t-SNE perplexity (will be clipped for small datasets)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reducers",
    )
    return parser.parse_args()


def collect_pairs(root: Path):
    pairs = {}
    if not root.exists():
        return pairs
    for metadata_path in root.rglob("*_metadata.csv"):
        name = metadata_path.name.replace("_metadata.csv", "")
        pairs.setdefault(name, []).append(metadata_path)
    return pairs


def load_concat(root: Path, artifact_name: str):
    pair_paths = sorted(collect_pairs(root).get(artifact_name, []))
    if not pair_paths:
        return None, None
    frames = []
    arrays = []
    for path in pair_paths:
        meta_df, emb = load_artifact_pair(path)
        frames.append(meta_df)
        arrays.append(emb)
    return pd.concat(frames, ignore_index=True), np.vstack(arrays)


def load_all_spaces(ecapa_root: Path, pdaf_root: Path):
    spaces = {}
    for level in LEVEL_ORDER:
        pdaf_meta, pdaf_emb = load_concat(pdaf_root, f"{level}_pdaf_raw")
        if pdaf_meta is not None:
            spaces[(level, "pdaf_raw")] = (pdaf_meta, pdaf_emb)

        ecapa_meta, ecapa_emb = load_concat(ecapa_root, f"{level}_ecapa_raw")
        if ecapa_meta is not None:
            spaces[(level, "ecapa_raw")] = (ecapa_meta, ecapa_emb)

        mapper_meta, mapper_emb = load_concat(ecapa_root, f"{level}_ecapa_mapper")
        if mapper_meta is not None:
            spaces[(level, "ecapa_mapper")] = (mapper_meta, mapper_emb)
    return spaces


def reduce_embeddings(embeddings, reducer_name: str, perplexity: float, random_state: int):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 rows to reduce embeddings.")

    scaled = StandardScaler().fit_transform(embeddings)
    if reducer_name == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(scaled)

    n_rows = embeddings.shape[0]
    effective_perplexity = min(
        perplexity,
        max(1.0, (n_rows - 1) / 3.0),
        n_rows - 1e-3,
    )
    pca_components = min(50, scaled.shape[1], max(2, n_rows - 1))
    pca_input = PCA(n_components=pca_components, random_state=random_state).fit_transform(scaled)
    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity,
        max_iter=1000,
    )
    return reducer.fit_transform(pca_input)


def label_palette(values):
    uniques = [x for x in sorted(pd.Series(values).dropna().astype(str).unique()) if x != ""]
    colors = plt.get_cmap("tab10", max(len(uniques), 1))
    return {label: colors(i) for i, label in enumerate(uniques)}


def plot_grid(spaces, level: str, reducer_name: str, output_dir: Path, coords_cache):
    present_spaces = [
        space
        for space in SPACE_ORDER
        if (level, space) in spaces and (level, space, reducer_name) in coords_cache
    ]
    if not present_spaces:
        return

    fig, axes = plt.subplots(
        len(present_spaces),
        len(LABEL_COLUMNS),
        figsize=(4.5 * len(LABEL_COLUMNS), 4.0 * len(present_spaces)),
        squeeze=False,
    )

    for row_idx, space_name in enumerate(present_spaces):
        meta_df, embeddings = spaces[(level, space_name)]
        coords = coords_cache[(level, space_name, reducer_name)]

        for col_idx, label_name in enumerate(LABEL_COLUMNS):
            ax = axes[row_idx][col_idx]
            palette = label_palette(meta_df[label_name].fillna("").astype(str))
            point_size = 7 if level == "utterance" else 36

            for label_value, color in palette.items():
                mask = meta_df[label_name].fillna("").astype(str) == label_value
                ax.scatter(
                    coords[mask.to_numpy(), 0],
                    coords[mask.to_numpy(), 1],
                    s=point_size,
                    c=[color],
                    alpha=0.75 if level == "utterance" else 0.9,
                    edgecolors="none",
                    label=label_value,
                )

            if row_idx == 0:
                ax.set_title(label_name.capitalize())
            if col_idx == 0:
                ax.set_ylabel(SPACE_TITLES[space_name])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="best", fontsize=7, frameon=True)

    fig.suptitle(f"{reducer_name.upper()} - {level.capitalize()} Level", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / f"{reducer_name}_{level}_grid.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_coords_csv(output_dir: Path, level: str, space_name: str, reducer_name: str, meta_df, coords):
    out_df = meta_df.copy()
    out_df["x"] = coords[:, 0]
    out_df["y"] = coords[:, 1]
    out_path = output_dir / f"coords_{reducer_name}_{level}_{space_name}.csv"
    out_df.to_csv(out_path, index=False)


def main():
    setup_local_env()
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))

    plt.style.use("seaborn-v0_8-whitegrid")
    spaces = load_all_spaces(Path(args.ecapa_dir), Path(args.pdaf_dir))
    if not spaces:
        raise RuntimeError(
            "No embedding artifacts found. Run the extraction scripts first."
        )

    coords_cache = {}
    for level, space_name in spaces:
        meta_df, embeddings = spaces[(level, space_name)]
        if len(meta_df) < 2 or embeddings.shape[0] < 2:
            print(
                f"Skipping {space_name} ({level}) for reduction: "
                f"need at least 2 rows, found {len(meta_df)}."
            )
            continue
        for reducer_name in args.reducers:
            coords = reduce_embeddings(
                embeddings,
                reducer_name,
                args.tsne_perplexity,
                args.random_state,
            )
            coords_cache[(level, space_name, reducer_name)] = coords
            save_coords_csv(output_dir, level, space_name, reducer_name, meta_df, coords)

    for reducer_name in args.reducers:
        for level in LEVEL_ORDER:
            plot_grid(spaces, level, reducer_name, output_dir, coords_cache)


if __name__ == "__main__":
    main()
