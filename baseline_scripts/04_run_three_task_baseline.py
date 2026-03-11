#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


TASKS = ("gender", "age", "dialect")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a three-task single-sample baseline for CoLMbo.\n"
            "For each task: materialize sample wav -> run inference -> save report."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Manifest JSONL from 00_fetch_tears_manifest.py",
    )
    parser.add_argument("--config", default="config.yaml", help="Base config path")
    parser.add_argument(
        "--ears-root",
        default="tears_audio",
        help="Root with downloaded raw EARS audio (for gender/age tasks)",
    )
    parser.add_argument(
        "--timit-root",
        default=None,
        help=(
            "Root containing TIMIT audio for strict dialect task. "
            "Expected to resolve paths like timit_dataset/train/... "
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_scripts/data/three_task_baseline",
        help="Directory to store per-task wav/config/meta and final report",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to use for subprocess steps",
    )
    return parser.parse_args()


def run_cmd(cmd, cwd):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def pick_source_and_roots(task, ears_root, timit_root):
    if task in ("gender", "age"):
        return "ears_dataset_processed", [ears_root], None
    if task == "dialect":
        if timit_root is None:
            return None, [], "dialect requires --timit-root (strict mode)"
        return "timit_dataset", [timit_root], None
    return None, [], f"unknown task: {task}"


def build_task_config(base_config, waveform_path, prompt_text):
    cfg = json.loads(json.dumps(base_config))  # deep copy via JSON-safe structure
    cfg["data"]["waveform"] = str(waveform_path)
    cfg["data"]["prompt"] = str(prompt_text)
    return cfg


def main():
    args = parse_args()
    repo_root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    results = []
    for task in TASKS:
        task_row = {
            "task": task,
            "status": "pending",
            "source_prefix": "",
            "waveform": "",
            "prompt": "",
            "gold_answer": "",
            "prediction": "",
            "error": "",
        }

        source_prefix, roots, err = pick_source_and_roots(
            task, args.ears_root, args.timit_root
        )
        if err is not None:
            task_row["status"] = "skipped"
            task_row["error"] = err
            results.append(task_row)
            continue

        task_row["source_prefix"] = source_prefix
        wav_out = output_dir / f"{task}.wav"
        meta_out = output_dir / f"{task}_meta.json"
        cfg_out = output_dir / f"{task}_config.yaml"

        make_sample_cmd = [
            args.python_bin,
            "baseline_scripts/02_make_sample_wav.py",
            "--manifest",
            str(manifest),
            "--task",
            task,
            "--source-prefix",
            source_prefix,
            "--output-wav",
            str(wav_out),
            "--output-meta",
            str(meta_out),
        ]
        for root in roots:
            make_sample_cmd.extend(["--audio-root", str(root)])

        rc, out = run_cmd(make_sample_cmd, repo_root)
        if rc != 0:
            task_row["status"] = "sample_failed"
            task_row["error"] = out.strip()
            results.append(task_row)
            continue

        with meta_out.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        prompt = meta.get("selected_prompt")
        gold = meta.get("selected_answer")
        task_row["waveform"] = str(wav_out)
        task_row["prompt"] = str(prompt) if prompt is not None else ""
        task_row["gold_answer"] = str(gold) if gold is not None else ""

        if not prompt:
            task_row["status"] = "sample_failed"
            task_row["error"] = f"No prompt found for task={task}"
            results.append(task_row)
            continue

        task_config = build_task_config(base_config, wav_out, prompt)
        with cfg_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(task_config, f, sort_keys=False)

        infer_cmd = [
            args.python_bin,
            "run_single_example.py",
            "--config",
            str(cfg_out),
        ]
        rc, out = run_cmd(infer_cmd, repo_root)
        if rc != 0:
            task_row["status"] = "inference_failed"
            task_row["error"] = out.strip()
            results.append(task_row)
            continue

        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        prediction = lines[-1] if lines else ""
        task_row["prediction"] = prediction
        task_row["status"] = "ok"
        results.append(task_row)

    json_report = output_dir / "baseline_report.json"
    csv_report = output_dir / "baseline_report.csv"
    with json_report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    with csv_report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "status",
                "source_prefix",
                "waveform",
                "prompt",
                "gold_answer",
                "prediction",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("Three-task baseline complete.")
    print(f"JSON report: {json_report}")
    print(f"CSV report:  {csv_report}")
    for row in results:
        print(f"- {row['task']}: {row['status']}")


if __name__ == "__main__":
    main()
