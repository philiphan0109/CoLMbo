#!/usr/bin/env python3
"""Evaluate a baseline or fine-tuned CoLMbo mapper with value and balanced accuracy."""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    SOURCE_TASKS,
    TASKS,
    add_wandb_args,
    canonical_value,
    collect_task_values,
    init_wandb,
    iter_expanded_examples,
    iter_manifest,
    load_config,
    load_mapper_checkpoint,
    load_sid_model_from_config,
    load_waveform_for_manifest,
    maybe_tqdm,
    normalize_text,
    resolve_audio_reference,
    resolve_device,
    setup_local_env,
    wandb_finish,
    wandb_log,
)


PREDICTION_FIELDS = [
    "task",
    "source_prefix",
    "audio_path",
    "resolved_audio_path",
    "resolution_method",
    "prompt",
    "gold_response",
    "true_value",
    "prediction",
    "pred_value",
    "is_exact_match",
    "is_value_correct",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "task",
    "source_prefix",
    "n_selected",
    "n_eval",
    "n_inference_failed",
    "n_audio_unresolved",
    "n_value_available",
    "n_value_pred",
    "n_value_correct",
    "exact_match_accuracy",
    "value_accuracy",
    "balanced_value_accuracy",
    "value_extraction_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a mapper checkpoint on TEARS-style manifests. This mirrors the "
            "baseline value extraction while adding balanced per-label accuracy."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="Raw TEARS JSONL manifest to expand for evaluation",
    )
    parser.add_argument(
        "--examples-csv",
        default=None,
        help="Optional expanded examples CSV. If provided, --manifest is used only for known labels.",
    )
    parser.add_argument("--config", default="config.yaml", help="Project config path")
    parser.add_argument(
        "--mapper-checkpoint",
        default=None,
        help="Mapper checkpoint to evaluate. Defaults to config train snapshot mapper.",
    )
    parser.add_argument("--ears-root", default="tears_audio", help="Local EARS raw audio root")
    parser.add_argument("--timit-root", default="timit_root", help="Local TIMIT root")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_TASKS.keys()),
        choices=list(SOURCE_TASKS.keys()),
        help="Source prefixes to evaluate",
    )
    parser.add_argument(
        "--max-samples-per-group",
        type=int,
        default=200,
        help="Max rows per (task, source_prefix); use <=0 for all",
    )
    parser.add_argument(
        "--allow-unresolved-audio",
        action="store_true",
        help="Keep unresolved rows and let inference fail instead of filtering during selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select/resolve rows; skip model loading and inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_scripts/runs/eval_mapper",
        help="Output directory for predictions and summaries",
    )
    parser.add_argument("--progress-every", type=int, default=25, help="Fallback progress interval")
    parser.add_argument(
        "--wandb-log-predictions",
        action="store_true",
        help="When --wandb is enabled, log a W&B table with prediction rows.",
    )
    parser.add_argument(
        "--wandb-prediction-table-limit",
        type=int,
        default=2000,
        help="Max prediction rows to send to W&B when --wandb-log-predictions is used.",
    )
    add_wandb_args(parser, default_job_type="eval_mapper")
    return parser.parse_args()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def row_from_example(example, resolved_path="", resolution_method=""):
    return {
        "task": example["task"],
        "source_prefix": example["source_prefix"],
        "audio_path": example["audio_path"],
        "resolved_audio_path": resolved_path,
        "resolution_method": resolution_method,
        "prompt": example["prompt"],
        "gold_response": example.get("response", example.get("gold_response", "")),
        "true_value": example.get("label", example.get("true_value", "")),
        "prediction": "",
        "pred_value": "",
        "is_exact_match": "",
        "is_value_correct": "",
        "status": "pending",
        "error": "",
    }


def maybe_add_row(rows, unresolved_counts, per_group, args, example, roots):
    group_key = (example["task"], example["source_prefix"])
    max_per_group = args.max_samples_per_group
    if max_per_group is not None and max_per_group <= 0:
        max_per_group = None
    if max_per_group is not None and per_group[group_key] >= max_per_group:
        return

    resolved_path = ""
    method = ""
    if not args.allow_unresolved_audio:
        resolved, method = resolve_audio_reference(example["audio_path"], roots)
        if resolved is None:
            unresolved_counts[group_key] += 1
            return
        resolved_path = str(resolved)

    rows.append(row_from_example(example, resolved_path, method))
    per_group[group_key] += 1


def build_rows_from_manifest(args, roots, task_values):
    rows = []
    unresolved_counts = Counter()
    per_group = Counter()
    wanted_sources = set(args.sources)

    for raw_row in iter_manifest(args.manifest):
        audio_path = raw_row.get("audio_path")
        source = str(audio_path).split("/")[0] if audio_path else ""
        if source not in wanted_sources:
            continue
        for example in iter_expanded_examples(raw_row, args.tasks, task_values):
            maybe_add_row(rows, unresolved_counts, per_group, args, example, roots)
    return rows, unresolved_counts


def build_rows_from_examples_csv(args, roots):
    rows = []
    unresolved_counts = Counter()
    per_group = Counter()
    wanted_sources = set(args.sources)
    wanted_tasks = set(args.tasks)
    with Path(args.examples_csv).open("r", encoding="utf-8", newline="") as f:
        for example in csv.DictReader(f):
            if example["source_prefix"] not in wanted_sources or example["task"] not in wanted_tasks:
                continue
            maybe_add_row(rows, unresolved_counts, per_group, args, example, roots)
    return rows, unresolved_counts


class MapperInferencer:
    def __init__(self, config_path, mapper_checkpoint, device_arg):
        import torch
        from load_data.extract_fbanks import Mel_Spectrogram
        from wrapper import ExpWrapper

        self.torch = torch
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        wrapper_cfg = dict(self.config["wrapper"])
        self.sample_rate = int(self.config["data"].get("sample_rate", 16000))
        self.device = resolve_device(device_arg)

        if self.device.type == "cpu":
            wrapper_cfg["gpu_id"] = "cpu"
            wrapper_cfg["device"] = "cpu"
        else:
            wrapper_cfg["gpu_id"] = self.device.index if self.device.index is not None else 0
            wrapper_cfg["device"] = f"cuda:{wrapper_cfg['gpu_id']}"

        self.extractor = Mel_Spectrogram(sample_rate=self.sample_rate).to(self.device)
        self.sid_model = load_sid_model_from_config(self.config_path, self.device)
        self.exp = ExpWrapper(wrapper_cfg, wrapper_cfg["gpu_id"])

        if mapper_checkpoint is None:
            mapper_checkpoint = Path(self.config["train"]["snapshot_path"]) / wrapper_cfg["mapper_ck_name"]
        load_mapper_checkpoint(self.exp.sid_mapper, mapper_checkpoint, self.device)
        self.exp.sid_mapper = self.exp.sid_mapper.to(self.device)

        self.sid_model.eval()
        self.exp.gpt.eval()
        self.exp.sid_mapper.eval()
        self.mapper_checkpoint = str(mapper_checkpoint)

    def predict(self, row, roots):
        waveform, _, resolved_path, method = load_waveform_for_manifest(
            row["audio_path"],
            roots,
            sample_rate=self.sample_rate,
        )
        waveform = waveform.to(self.device)
        with self.torch.no_grad():
            features = self.extractor(waveform)
            sid_emb = self.sid_model(features)
            sid_prefix = self.exp.get_sid_prefix(sid_emb)
            prompt_prefix, _ = self.exp.get_prompt_prefix_single(row["prompt"])
            prefix = self.torch.cat((sid_prefix, prompt_prefix), dim=1)
            texts = self.exp.generate_beam(sids_prefix=prefix)
        row["resolved_audio_path"] = resolved_path
        row["resolution_method"] = method
        return texts[0] if texts else ""


def add_metrics(rows, task_values):
    for row in rows:
        if row["status"] != "ok":
            continue
        pred = row.get("prediction", "")
        gold = row.get("gold_response", "")
        true_value = row.get("true_value")
        pred_value = canonical_value(row["task"], pred, task_values)
        row["pred_value"] = "" if pred_value is None else str(pred_value)
        row["is_exact_match"] = int(normalize_text(pred) == normalize_text(gold))
        row["is_value_correct"] = int(
            pred_value is not None
            and true_value not in ("", None)
            and normalize_text(pred_value) == normalize_text(true_value)
        )


def summarize_group(task, source_prefix, group_rows, n_audio_unresolved=0):
    ok_rows = [row for row in group_rows if row["status"] == "ok"]
    n_eval = len(ok_rows)
    n_exact = sum(int(row.get("is_exact_match", 0) or 0) for row in ok_rows)
    n_infer_failed = len([row for row in group_rows if row["status"] == "inference_failed"])
    value_rows = [row for row in ok_rows if row.get("true_value") not in ("", None)]
    n_value_available = len(value_rows)
    n_value_pred = sum(1 for row in value_rows if row.get("pred_value") not in ("", None))
    n_value_correct = sum(int(row.get("is_value_correct", 0) or 0) for row in value_rows)

    label_totals = defaultdict(lambda: [0, 0])
    for row in value_rows:
        label = normalize_text(row.get("true_value"))
        label_totals[label][1] += 1
        label_totals[label][0] += int(row.get("is_value_correct", 0) or 0)
    per_label_acc = [
        correct / total for correct, total in label_totals.values() if total > 0
    ]

    return {
        "task": task,
        "source_prefix": source_prefix,
        "n_selected": len(group_rows),
        "n_eval": n_eval,
        "n_inference_failed": n_infer_failed,
        "n_audio_unresolved": int(n_audio_unresolved),
        "n_value_available": n_value_available,
        "n_value_pred": n_value_pred,
        "n_value_correct": n_value_correct,
        "exact_match_accuracy": (n_exact / n_eval) if n_eval else 0.0,
        "value_accuracy": (n_value_correct / n_value_available) if n_value_available else 0.0,
        "balanced_value_accuracy": (
            sum(per_label_acc) / len(per_label_acc) if per_label_acc else 0.0
        ),
        "value_extraction_rate": (
            n_value_pred / n_value_available if n_value_available else 0.0
        ),
    }


def summarize(rows, unresolved_counts):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["source_prefix"])].append(row)

    summary = []
    all_keys = set(groups.keys()) | set(unresolved_counts.keys())
    for task, source in sorted(all_keys):
        summary.append(
            summarize_group(
                task,
                source,
                groups.get((task, source), []),
                unresolved_counts.get((task, source), 0),
            )
        )

    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_overall = []
    for task, task_rows in sorted(by_task.items()):
        unresolved = sum(
            count for (unresolved_task, _), count in unresolved_counts.items() if unresolved_task == task
        )
        overall = summarize_group(task, "OVERALL", task_rows, unresolved)
        task_overall.append(overall)
        summary.append(overall)

    if task_overall:
        summary.append(
            {
                "task": "MACRO_AVG",
                "source_prefix": "OVERALL",
                "n_selected": sum(row["n_selected"] for row in task_overall),
                "n_eval": sum(row["n_eval"] for row in task_overall),
                "n_inference_failed": sum(row["n_inference_failed"] for row in task_overall),
                "n_audio_unresolved": sum(row["n_audio_unresolved"] for row in task_overall),
                "n_value_available": sum(row["n_value_available"] for row in task_overall),
                "n_value_pred": sum(row["n_value_pred"] for row in task_overall),
                "n_value_correct": sum(row["n_value_correct"] for row in task_overall),
                "exact_match_accuracy": sum(row["exact_match_accuracy"] for row in task_overall)
                / len(task_overall),
                "value_accuracy": sum(row["value_accuracy"] for row in task_overall)
                / len(task_overall),
                "balanced_value_accuracy": sum(
                    row["balanced_value_accuracy"] for row in task_overall
                )
                / len(task_overall),
                "value_extraction_rate": sum(row["value_extraction_rate"] for row in task_overall)
                / len(task_overall),
            }
        )
    return summary


def main():
    setup_local_env()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config) if args.wandb else None
    run = init_wandb(
        args,
        config,
        run_config={
            "stage": "eval_mapper",
            "manifest": args.manifest,
            "examples_csv": args.examples_csv or "",
            "mapper_checkpoint": args.mapper_checkpoint or "",
            "output_dir": str(out_dir),
            "tasks": args.tasks,
            "sources": args.sources,
            "max_samples_per_group": args.max_samples_per_group,
            "dry_run": args.dry_run,
        },
    )
    roots = {
        "ears_dataset_processed": Path(args.ears_root) if args.ears_root else None,
        "timit_dataset": Path(args.timit_root) if args.timit_root else None,
    }
    try:
        task_values = collect_task_values(args.manifest)

        if args.examples_csv:
            rows, unresolved_counts = build_rows_from_examples_csv(args, roots)
        else:
            rows, unresolved_counts = build_rows_from_manifest(args, roots, task_values)
        print(f"Selected rows for evaluation: {len(rows)}")
        wandb_log(run, {"eval/n_selected": len(rows)})

        if args.dry_run:
            for row in rows:
                row["status"] = "dry_run"
        elif rows:
            inferencer = MapperInferencer(args.config, args.mapper_checkpoint, args.device)
            pbar = maybe_tqdm(len(rows), desc="Inference", unit="sample")
            started = time.time()
            for idx, row in enumerate(rows, start=1):
                try:
                    row["prediction"] = inferencer.predict(row, roots)
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "inference_failed"
                    row["error"] = str(exc)

                elapsed = max(1e-6, time.time() - started)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"elapsed={format_duration(elapsed)} rate={idx / elapsed:.2f}/s"
                    )
                elif idx % max(1, args.progress_every) == 0 or idx == len(rows):
                    print(
                        f"[{idx}/{len(rows)}] elapsed={format_duration(elapsed)} "
                        f"rate={idx / elapsed:.2f}/s"
                    )
                if idx % max(1, args.progress_every) == 0 or idx == len(rows):
                    ok_count = sum(1 for row_item in rows[:idx] if row_item["status"] == "ok")
                    failed_count = sum(
                        1 for row_item in rows[:idx] if row_item["status"] == "inference_failed"
                    )
                    wandb_log(
                        run,
                        {
                            "eval/inference_processed": idx,
                            "eval/inference_ok": ok_count,
                            "eval/inference_failed": failed_count,
                            "eval/inference_rate_samples_per_sec": idx / elapsed,
                        },
                        step=idx,
                    )
            if pbar is not None:
                pbar.close()

        add_metrics(rows, task_values)
        summary = summarize(rows, unresolved_counts)

        predictions_csv = out_dir / "predictions.csv"
        summary_csv = out_dir / "summary.csv"
        summary_json = out_dir / "summary.json"
        failures_csv = out_dir / "failures.csv"
        write_csv(predictions_csv, rows, PREDICTION_FIELDS)
        write_csv(summary_csv, summary, SUMMARY_FIELDS)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        failures = [row for row in rows if row.get("is_value_correct") == 0]
        write_csv(failures_csv, failures, PREDICTION_FIELDS)

        if run is not None:
            import wandb

            summary_table = wandb.Table(columns=SUMMARY_FIELDS)
            for row in summary:
                summary_table.add_data(*[row.get(field, "") for field in SUMMARY_FIELDS])
            run.log({"eval/summary_table": summary_table})

            for row in summary:
                prefix = f"eval/{row['task']}/{row['source_prefix']}"
                wandb_log(
                    run,
                    {
                        f"{prefix}/value_accuracy": row["value_accuracy"],
                        f"{prefix}/balanced_value_accuracy": row["balanced_value_accuracy"],
                        f"{prefix}/exact_match_accuracy": row["exact_match_accuracy"],
                        f"{prefix}/value_extraction_rate": row["value_extraction_rate"],
                        f"{prefix}/n_eval": row["n_eval"],
                    },
                )

            if args.wandb_log_predictions:
                prediction_table = wandb.Table(columns=PREDICTION_FIELDS)
                for row in rows[: max(0, args.wandb_prediction_table_limit)]:
                    prediction_table.add_data(*[row.get(field, "") for field in PREDICTION_FIELDS])
                run.log({"eval/predictions_table": prediction_table})

            artifact = wandb.Artifact(
                name=f"colmbo-eval-{run.id}",
                type="evaluation",
                metadata={
                    "n_rows": len(rows),
                    "n_summary_rows": len(summary),
                    "n_failures": len(failures),
                },
            )
            artifact.add_file(str(summary_csv), name="summary.csv")
            artifact.add_file(str(summary_json), name="summary.json")
            artifact.add_file(str(failures_csv), name="failures.csv")
            run.log_artifact(artifact)
            macro = next(
                (
                    row
                    for row in summary
                    if row["task"] == "MACRO_AVG" and row["source_prefix"] == "OVERALL"
                ),
                None,
            )
            if macro:
                run.summary["macro_value_accuracy"] = macro["value_accuracy"]
                run.summary["macro_balanced_value_accuracy"] = macro["balanced_value_accuracy"]

        print(f"Predictions CSV: {predictions_csv}")
        print(f"Summary CSV:     {summary_csv}")
        print(f"Summary JSON:    {summary_json}")
        print(f"Failures CSV:    {failures_csv}")
    finally:
        wandb_finish(run)


if __name__ == "__main__":
    main()
