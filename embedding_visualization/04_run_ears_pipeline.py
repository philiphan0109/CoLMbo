#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from common import ensure_output_dir, setup_local_env


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full EARS embedding visualization pipeline: "
            "build index, extract ECAPA spaces, extract PDAF raw space, and plot grids."
        )
    )
    parser.add_argument(
        "--ears-root",
        required=True,
        help="Root directory with EARS raw speaker audio",
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="TEARS manifest JSONL path",
    )
    parser.add_argument(
        "--output-root",
        default="embedding_visualization/runs/ears_default",
        help="Root directory for all intermediate artifacts and final plots",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for embedding extraction (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on utterances to process (0 = all)",
    )
    parser.add_argument(
        "--mapper-pool",
        choices=["mean", "flatten"],
        default="mean",
        help="How to convert mapper prefix sequence into a plotting vector",
    )
    parser.add_argument(
        "--reducers",
        nargs="+",
        default=["pca", "tsne"],
        choices=["pca", "tsne"],
        help="Dimensionality reducers for the final plots",
    )
    parser.add_argument(
        "--skip-pdaf",
        action="store_true",
        help="Skip PDAF extraction if you only want ECAPA-based plots",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse previously generated artifacts when the expected outputs already exist",
    )
    return parser.parse_args()


def run_step(cmd):
    print("")
    print("[run] " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, check=True)


def main():
    setup_local_env()
    args = parse_args()

    ears_root = Path(args.ears_root)
    if not ears_root.exists():
        raise FileNotFoundError(
            "\n".join(
                [
                    f"EARS root does not exist: {ears_root}",
                    "",
                    "This pipeline expects a local EARS audio root containing speaker folders/files.",
                    "In this repo, that is usually a directory like `tears_audio/` created by",
                    "`baseline_scripts/01_download_ears_speakers.py`.",
                    "",
                    "If you previously ran the storage-light chunked EARS workflow, `tears_audio/`",
                    "may have been deleted between chunks on purpose.",
                    "",
                    "Next step:",
                    "  1. Restore or re-download the EARS audio root",
                    "  2. Re-run this command with `--ears-root <restored_path>`",
                ]
            )
        )

    output_root = ensure_output_dir(Path(args.output_root))
    data_root = ensure_output_dir(output_root / "data")
    plot_root = ensure_output_dir(output_root / "plots")
    ecapa_root = ensure_output_dir(data_root / "ecapa")
    pdaf_root = ensure_output_dir(data_root / "pdaf")
    index_out = data_root / "index" / "ears_test_index.csv"

    base = [sys.executable]

    if not (args.skip_existing and index_out.exists()):
        run_step(
            base
            + [
                "embedding_visualization/00_build_ears_index.py",
                "--manifest",
                str(args.manifest),
                "--output",
                str(index_out),
            ]
        )
    else:
        print(f"[skip] index exists: {index_out}")

    ecapa_expected = ecapa_root / "utterance_ecapa_raw_metadata.csv"
    if not (args.skip_existing and ecapa_expected.exists()):
        ecapa_cmd = base + [
            "embedding_visualization/01_extract_ecapa_spaces.py",
            "--manifest",
            str(args.manifest),
            "--ears-root",
            str(ears_root),
            "--output-dir",
            str(ecapa_root),
            "--config",
            str(args.config),
            "--device",
            str(args.device),
            "--mapper-pool",
            args.mapper_pool,
        ]
        if args.limit > 0:
            ecapa_cmd += ["--limit", str(args.limit)]
        run_step(ecapa_cmd)
    else:
        print(f"[skip] ECAPA artifacts exist: {ecapa_expected}")

    if not args.skip_pdaf:
        pdaf_expected = pdaf_root / "utterance_pdaf_raw_metadata.csv"
        if not (args.skip_existing and pdaf_expected.exists()):
            pdaf_cmd = base + [
                "embedding_visualization/02_extract_pdaf_raw.py",
                "--manifest",
                str(args.manifest),
                "--ears-root",
                str(ears_root),
                "--output-dir",
                str(pdaf_root),
                "--device",
                str(args.device),
            ]
            if args.limit > 0:
                pdaf_cmd += ["--limit", str(args.limit)]
            run_step(pdaf_cmd)
        else:
            print(f"[skip] PDAF artifacts exist: {pdaf_expected}")

    run_step(
        base
        + [
            "embedding_visualization/03_plot_embedding_spaces.py",
            "--ecapa-dir",
            str(ecapa_root),
            "--pdaf-dir",
            str(pdaf_root),
            "--output-dir",
            str(plot_root),
            "--reducers",
            *args.reducers,
        ]
    )

    print("")
    print("Pipeline complete.")
    print(f"Index: {index_out}")
    print(f"ECAPA artifacts: {ecapa_root}")
    if args.skip_pdaf:
        print("PDAF artifacts: skipped")
    else:
        print(f"PDAF artifacts: {pdaf_root}")
    print(f"Plots: {plot_root}")


if __name__ == "__main__":
    main()
