#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
import yaml

# Allow importing project modules when script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep caches local to workspace to avoid permission issues.
os.environ.setdefault("HF_HOME", ".hf_home")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", ".hf_home/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", ".hf_home/transformers")
os.environ.setdefault("NUMBA_CACHE_DIR", ".numba_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

from encoder.encoder import Model
from load_data.extract_fbanks import Mel_Spectrogram
from wrapper import ExpWrapper


TASKS = ("gender", "age", "dialect")
TASK_TO_SPEAKER_KEY = {
    "gender": "gender",
    "age": "age",
    "dialect": "dialect_region",
}
DEFAULT_TASK_PREFIXES = {
    "gender": ("ears_dataset_processed", "timit_dataset"),
    "age": ("ears_dataset_processed", "timit_dataset", "voxceleb2_dataset"),
    "dialect": ("timit_dataset",),
}
PROMPT_KEYWORDS = {
    "gender": "gender",
    "age": "age",
    "dialect": "dialect",
}
EARS_SEGMENT_RE = re.compile(
    r"^ears_dataset_processed/(?P<split>[^/]+)/(?P<speaker>p\d{3})/(?P<stem>.+)_(?P<start>\d+)_(?P<end>\d+)\.wav$",
    flags=re.IGNORECASE,
)
AGE_RANGE_RE = re.compile(r"(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch baseline evaluation for CoLMbo across TEARS sources/tasks.\n"
            "Outputs per-sample predictions and grouped raw metrics."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="Manifest JSONL from 00_fetch_tears_manifest.py",
    )
    parser.add_argument("--config", default="config.yaml", help="Model config path")
    parser.add_argument(
        "--ears-root",
        default=None,
        help="Root with raw EARS audio (required for ears_dataset_processed)",
    )
    parser.add_argument(
        "--timit-root",
        default=None,
        help="Root used to resolve timit_dataset/<...> paths",
    )
    parser.add_argument(
        "--voxceleb-root",
        default=None,
        help="Root used to resolve voxceleb2_dataset/<...> paths",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--max-samples-per-group",
        type=int,
        default=200,
        help=(
            "Max samples per (task, source_prefix) group. "
            "Use <=0 to evaluate all available rows."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_scripts/data/batch_eval",
        help="Output folder for predictions + metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model inference, only resolve/evaluate pipeline selection logic.",
    )
    return parser.parse_args()


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_gender(text):
    text = normalize_text(text)
    if re.search(r"\bfemale\b", text):
        return "female"
    if re.search(r"\bmale\b", text):
        return "male"
    return None


def canonical_age(text):
    if text is None:
        return None
    text_norm = normalize_text(text)
    direct = re.search(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", text_norm)
    if direct:
        a, b = int(direct.group(1)), int(direct.group(2))
        return f"{min(a,b)}-{max(a,b)}"
    for match in AGE_RANGE_RE.finditer(text_norm):
        a, b = int(match.group(1)), int(match.group(2))
        return f"{min(a,b)}-{max(a,b)}"
    return None


def canonical_dialect(text, known_dialects):
    text_norm = normalize_text(text)
    if not text_norm:
        return None
    best = None
    best_len = -1
    for d in known_dialects:
        dn = normalize_text(d)
        if dn and dn in text_norm and len(dn) > best_len:
            best = d
            best_len = len(dn)
    return best


def canonical_value(task, text, known_dialects):
    if task == "gender":
        return canonical_gender(text)
    if task == "age":
        return canonical_age(text)
    if task == "dialect":
        return canonical_dialect(text, known_dialects)
    return None


def choose_prompt_answer(row, task):
    prompts = row.get("prompts") or []
    responses = row.get("responses") or []
    if len(prompts) != len(responses):
        return None, None
    key = PROMPT_KEYWORDS[task]
    for p, r in zip(prompts, responses):
        if key in str(p).lower():
            return p, r
    return None, None


def parse_ears_segment(audio_path):
    match = EARS_SEGMENT_RE.match(str(audio_path))
    if not match:
        return None
    info = match.groupdict()
    info["speaker"] = info["speaker"].lower()
    info["start"] = int(info["start"])
    info["end"] = int(info["end"])
    return info


def resolve_ears_audio(audio_path, ears_root, cache_root):
    direct = ears_root / audio_path
    if direct.exists():
        return direct, "direct_manifest_path"

    parsed = parse_ears_segment(audio_path)
    if parsed is None:
        return None, "unparsed_ears_path"

    raw_path = ears_root / parsed["speaker"] / f"{parsed['stem']}.wav"
    if not raw_path.exists():
        return None, "missing_ears_raw_file"

    seg_out = cache_root / audio_path
    if seg_out.exists():
        return seg_out, "materialized_segment_cached"

    wav, sr = torchaudio.load(str(raw_path))
    start = max(0, parsed["start"])
    end = min(max(start + 1, parsed["end"]), wav.shape[1])
    if start >= wav.shape[1] or end <= start:
        return None, "invalid_segment_bounds"
    seg = wav[:, start:end]
    if seg.shape[1] == 0:
        return None, "empty_segment"

    seg_out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(seg_out), seg, sr)
    return seg_out, "materialized_segment_created"


def resolve_audio(audio_path, source_prefix, roots, cache_root):
    if source_prefix == "ears_dataset_processed":
        ears_root = roots.get("ears_dataset_processed")
        if ears_root is None:
            return None, "missing_ears_root"
        return resolve_ears_audio(audio_path, ears_root, cache_root)

    root = roots.get(source_prefix)
    if root is None:
        return None, f"missing_root_{source_prefix}"
    candidate = root / audio_path
    if candidate.exists():
        return candidate, "direct_manifest_path"
    return None, "missing_audio_file"


def speaker_true_value(task, speaker_obj):
    if not isinstance(speaker_obj, dict):
        return None
    raw = speaker_obj.get(TASK_TO_SPEAKER_KEY[task])
    if raw is None:
        return None
    if task == "gender":
        return canonical_gender(raw)
    if task == "age":
        return canonical_age(raw)
    if task == "dialect":
        return str(raw).strip()
    return None


class Inferencer:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config_train = config["train"]
        config_sid_model = config["sid_model"]
        config_wrapper = dict(config["wrapper"])
        data_cfg = config["data"]
        sample_rate = int(data_cfg.get("sample_rate", 16000))

        self.sample_rate = sample_rate
        self.device = self.resolve_device(config_wrapper.get("device", "cuda"))
        if self.device.type == "cpu":
            config_wrapper["gpu_id"] = "cpu"
            config_wrapper["device"] = "cpu"
        else:
            config_wrapper["gpu_id"] = self.device.index if self.device.index is not None else 0
            config_wrapper["device"] = f"cuda:{config_wrapper['gpu_id']}"

        self.extractor = Mel_Spectrogram(sample_rate=sample_rate)
        self.exp = ExpWrapper(config_wrapper, config_wrapper["gpu_id"])
        self.sid_model = Model(n_mels=80, embedding_dim=192, channel=1024)

        ecapa_ckpt = torch.load("./pretrained_sid/ecapa.ckpt", map_location=self.device)
        if isinstance(ecapa_ckpt, dict) and "state_dict" in ecapa_ckpt:
            ecapa_ckpt = ecapa_ckpt["state_dict"]
        if not any(k.startswith("model.") for k in ecapa_ckpt.keys()):
            ecapa_ckpt = {f"model.{k}": v for k, v in ecapa_ckpt.items()}
        self.sid_model.load_state_dict(ecapa_ckpt)

        self.exp.load_sid_model(self.sid_model, config_train["snapshot_path"], config_sid_model["sid_ck_name"])
        self.exp.load_mapper(config_train["snapshot_path"], config_wrapper["mapper_ck_name"])

        self.sid_model = self.sid_model.to(self.device)
        self.exp.sid_mapper = self.exp.sid_mapper.to(self.device)
        self.sid_model.eval()
        self.exp.gpt.eval()
        self.exp.sid_mapper.eval()

    @staticmethod
    def resolve_device(config_device: str):
        if config_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        if config_device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(config_device)
        return torch.device("cpu")

    def predict(self, audio_path: Path, prompt: str):
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        with torch.no_grad():
            proc = self.extractor(wav).to(self.device)
            sid_emb = self.sid_model(proc)
            sid_prefix = self.exp.get_sid_prefix(sid_emb)
            prompt_prefix, _ = self.exp.get_prompt_prefix_single(prompt)
            prefix = torch.cat((sid_prefix, prompt_prefix), dim=1)
            texts = self.exp.generate_beam(sids_prefix=prefix)
        return texts[0] if texts else ""


def build_rows(args, roots, known_dialects):
    rows = []
    per_group_selected = defaultdict(int)
    unresolved_counts = defaultdict(int)
    max_per_group = args.max_samples_per_group
    if max_per_group is not None and max_per_group <= 0:
        max_per_group = None

    task_prefixes = {t: DEFAULT_TASK_PREFIXES[t] for t in args.tasks}
    available_prefixes = {k for k, v in roots.items() if v is not None}
    cache_root = Path(args.output_dir) / "resolved_audio"

    for row in iter_manifest(Path(args.manifest)):
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        source_prefix = str(audio_path).split("/")[0]
        for task in args.tasks:
            if source_prefix not in task_prefixes[task]:
                continue
            if source_prefix not in available_prefixes:
                continue
            group_key = (task, source_prefix)
            if max_per_group is not None and per_group_selected[group_key] >= max_per_group:
                continue

            prompt, gold_response = choose_prompt_answer(row, task)
            if prompt is None:
                continue

            resolved_audio, resolution_method = resolve_audio(
                audio_path,
                source_prefix,
                roots,
                cache_root,
            )
            if resolved_audio is None:
                unresolved_counts[group_key] += 1
                continue

            rows.append(
                {
                    "task": task,
                    "source_prefix": source_prefix,
                    "audio_path": audio_path,
                    "resolved_audio_path": str(resolved_audio),
                    "resolution_method": resolution_method,
                    "prompt": prompt,
                    "gold_response": gold_response,
                    "true_value": speaker_true_value(task, row.get("speaker")),
                    "prediction": "",
                    "pred_value": "",
                    "status": "pending",
                    "error": "",
                }
            )
            per_group_selected[group_key] += 1
    return rows, unresolved_counts


def add_metrics(rows, known_dialects):
    for r in rows:
        if r["status"] != "ok":
            r["is_exact_match"] = ""
            r["is_value_correct"] = ""
            continue
        pred = r.get("prediction", "")
        gold = r.get("gold_response", "")
        true_value = r.get("true_value")
        pred_value = canonical_value(r["task"], pred, known_dialects)
        r["pred_value"] = "" if pred_value is None else str(pred_value)
        exact = normalize_text(pred) == normalize_text(gold)
        value_ok = (
            pred_value is not None
            and true_value is not None
            and normalize_text(pred_value) == normalize_text(true_value)
        )
        r["is_exact_match"] = int(exact)
        r["is_value_correct"] = int(value_ok)


def summarize(rows, unresolved_counts):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["source_prefix"])].append(r)
    all_group_keys = set(groups.keys()) | set(unresolved_counts.keys())

    summary = []
    for (task, src) in sorted(all_group_keys):
        grp = groups.get((task, src), [])
        ok_rows = [x for x in grp if x["status"] == "ok"]
        n_selected = len([x for x in grp if x["status"] in ("ok", "dry_run")])
        n_infer_errors = len([x for x in grp if x["status"] == "inference_failed"])
        n_audio_unresolved = int(unresolved_counts.get((task, src), 0))
        n_eval = len(ok_rows)
        n_exact = sum(int(x.get("is_exact_match", 0) or 0) for x in ok_rows)

        value_available = [x for x in ok_rows if x.get("true_value") not in ("", None)]
        n_value_available = len(value_available)
        n_value_pred = sum(1 for x in value_available if x.get("pred_value") not in ("", None))
        n_value_correct = sum(int(x.get("is_value_correct", 0) or 0) for x in value_available)

        summary.append(
            {
                "task": task,
                "source_prefix": src,
                "n_selected": n_selected,
                "n_eval": n_eval,
                "n_inference_failed": n_infer_errors,
                "n_audio_unresolved": n_audio_unresolved,
                "n_value_available": n_value_available,
                "n_value_pred": n_value_pred,
                "n_value_correct": n_value_correct,
                "exact_match_accuracy": (n_exact / n_eval) if n_eval else 0.0,
                "value_accuracy": (n_value_correct / n_value_available) if n_value_available else 0.0,
                "value_extraction_rate": (n_value_pred / n_value_available) if n_value_available else 0.0,
            }
        )

    # Task-level overall rollups.
    by_task = defaultdict(list)
    for row in summary:
        by_task[row["task"]].append(row)
    for task, parts in sorted(by_task.items()):
        out = {
            "task": task,
            "source_prefix": "OVERALL",
            "n_selected": sum(p["n_selected"] for p in parts),
            "n_eval": sum(p["n_eval"] for p in parts),
            "n_inference_failed": sum(p["n_inference_failed"] for p in parts),
            "n_audio_unresolved": sum(p["n_audio_unresolved"] for p in parts),
            "n_value_available": sum(p["n_value_available"] for p in parts),
            "n_value_pred": sum(p["n_value_pred"] for p in parts),
            "n_value_correct": sum(p["n_value_correct"] for p in parts),
        }
        out["exact_match_accuracy"] = (
            out["n_eval"] and sum(p["exact_match_accuracy"] * p["n_eval"] for p in parts) / out["n_eval"]
        ) or 0.0
        out["value_accuracy"] = (
            out["n_value_available"]
            and out["n_value_correct"] / out["n_value_available"]
        ) or 0.0
        out["value_extraction_rate"] = (
            out["n_value_available"]
            and out["n_value_pred"] / out["n_value_available"]
        ) or 0.0
        summary.append(out)

    return summary


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "ears_dataset_processed": Path(args.ears_root) if args.ears_root else None,
        "timit_dataset": Path(args.timit_root) if args.timit_root else None,
        "voxceleb2_dataset": Path(args.voxceleb_root) if args.voxceleb_root else None,
    }

    # Build known dialect set from manifest speaker metadata for robust extraction.
    known_dialects = set()
    for row in iter_manifest(Path(args.manifest)):
        sp = row.get("speaker")
        if isinstance(sp, dict) and sp.get("dialect_region"):
            known_dialects.add(str(sp["dialect_region"]))

    rows, unresolved_counts = build_rows(args, roots, known_dialects)
    pending = [r for r in rows if r["status"] == "pending"]
    print(f"Selected rows for evaluation: {len(pending)}")
    if args.dry_run:
        print("Dry-run mode: skipping model inference.")
        for r in rows:
            if r["status"] == "pending":
                r["status"] = "dry_run"
    else:
        infer = Inferencer(Path(args.config))
        for i, r in enumerate(rows, start=1):
            if r["status"] != "pending":
                continue
            try:
                pred = infer.predict(Path(r["resolved_audio_path"]), r["prompt"])
                r["prediction"] = pred
                r["status"] = "ok"
            except Exception as e:
                r["status"] = "inference_failed"
                r["error"] = str(e)
            if i % 50 == 0:
                print(f"Processed {i}/{len(rows)} rows...")

    add_metrics(rows, known_dialects)
    summary = summarize(rows, unresolved_counts)

    predictions_csv = out_dir / "predictions.csv"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    failures_csv = out_dir / "failures.csv"

    pred_fields = [
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
    write_csv(predictions_csv, rows, pred_fields)

    summary_fields = [
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
        "value_extraction_rate",
    ]
    write_csv(summary_csv, summary, summary_fields)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    failures = [
        r
        for r in rows
        if r["status"] == "ok" and r.get("is_value_correct", "") == 0
    ]
    write_csv(failures_csv, failures, pred_fields)

    print(f"Predictions CSV: {predictions_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"Summary JSON:    {summary_json}")
    print(f"Failures CSV:    {failures_csv}")


if __name__ == "__main__":
    main()
