#!/usr/bin/env python3
"""Shared helpers for CoLMbo mapper fine-tuning experiments."""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASKS = ("gender", "age", "ethnicity", "dialect")
SOURCE_TASKS = {
    "ears_dataset_processed": ("gender", "age", "ethnicity"),
    "timit_dataset": ("gender", "age", "dialect"),
}
TASK_TO_SPEAKER_KEY = {
    "gender": "gender",
    "age": "age",
    "dialect": "dialect_region",
    "ethnicity": "ethnicity",
}
PROMPT_KEYWORDS = {
    "gender": "gender",
    "age": "age",
    "dialect": "dialect",
    "ethnicity": "ethnicity",
}
MISSING_VALUE_TOKENS = {"", "unknown", "none", "null", "nan", "n/a", "na"}
AGE_RANGE_RE = re.compile(r"(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})")
EARS_SEGMENT_RE = re.compile(
    r"^ears_dataset_processed/(?P<split>[^/]+)/(?P<speaker>p\d{3})/"
    r"(?P<stem>.+)_(?P<start>\d+)_(?P<end>\d+)\.wav$",
    flags=re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^p\d{3}$", flags=re.IGNORECASE)
ETHNICITY_CANONICAL = {
    "white": "White",
    "caucasian": "White",
    "black": "Black or African American",
    "african american": "Black or African American",
    "asian": "Asian",
    "hispanic": "Hispanic or Latino",
    "latino": "Hispanic or Latino",
    "latina": "Hispanic or Latino",
    "middle eastern": "Middle Eastern or North African",
    "native american": "Native American or Alaska Native",
    "american indian": "Native American or Alaska Native",
    "pacific islander": "Native Hawaiian or Other Pacific Islander",
    "multiracial": "Multiracial",
    "mixed": "Multiracial",
}


def setup_local_env():
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".hf_home" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".hf_home" / "transformers"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("WANDB_DIR", str(REPO_ROOT / ".wandb"))
    os.environ.setdefault("WANDB_DATA_DIR", str(REPO_ROOT / ".wandb" / "data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(REPO_ROOT / ".wandb" / "cache"))
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def add_wandb_args(parser, default_job_type=None):
    group = parser.add_argument_group("wandb")
    group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this run.",
    )
    group.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project. Defaults to config wandb.project or 'explainability'.",
    )
    group.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    group.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    group.add_argument("--wandb-group", default=None, help="Optional W&B run group.")
    group.add_argument(
        "--wandb-job-type",
        default=default_job_type,
        help="Optional W&B job type.",
    )
    group.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional W&B tags separated by spaces.",
    )
    group.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional W&B mode. Use offline if the machine cannot reach wandb.ai.",
    )
    return parser


def _wandb_project_from_config(config):
    if isinstance(config, dict):
        wandb_cfg = config.get("wandb") or {}
        if isinstance(wandb_cfg, dict) and wandb_cfg.get("project"):
            return wandb_cfg["project"]
    return "explainability"


def init_wandb(args, config=None, run_config=None):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested with --wandb, but wandb is not installed. "
            "Install it in the active environment with: pip install wandb"
        ) from exc

    project = getattr(args, "wandb_project", None) or _wandb_project_from_config(config)
    merged_config = {}
    if run_config:
        merged_config.update(run_config)
    merged_config["cli_args"] = vars(args)

    init_kwargs = {
        "project": project,
        "config": merged_config,
    }
    optional_kwargs = {
        "entity": getattr(args, "wandb_entity", None),
        "name": getattr(args, "wandb_run_name", None),
        "group": getattr(args, "wandb_group", None),
        "job_type": getattr(args, "wandb_job_type", None),
        "tags": getattr(args, "wandb_tags", None),
        "mode": getattr(args, "wandb_mode", None),
    }
    init_kwargs.update({key: value for key, value in optional_kwargs.items() if value})
    return wandb.init(**init_kwargs)


def wandb_log(run, data, step=None):
    if run is not None:
        run.log(data, step=step)


def wandb_finish(run):
    if run is not None:
        run.finish()


def load_config(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_manifest(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_missing_value(value):
    if value is None:
        return True
    return normalize_text(value) in MISSING_VALUE_TOKENS


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
        return f"{min(a, b)}-{max(a, b)}"
    for match in AGE_RANGE_RE.finditer(text_norm):
        a, b = int(match.group(1)), int(match.group(2))
        return f"{min(a, b)}-{max(a, b)}"
    return None


def longest_known_match(text_norm, known_values):
    best = None
    best_len = -1
    for value in sorted(known_values or []):
        val_norm = normalize_text(value)
        if val_norm and val_norm in text_norm and len(val_norm) > best_len:
            best = value
            best_len = len(val_norm)
    return best


def canonical_dialect(text, known_dialects=None):
    text_norm = normalize_text(text)
    if not text_norm:
        return None
    known = longest_known_match(text_norm, known_dialects)
    return known if known is not None else str(text).strip()


def canonical_ethnicity(text, known_ethnicities=None):
    text_norm = normalize_text(text)
    if not text_norm:
        return None
    known = longest_known_match(text_norm, known_ethnicities)
    if known is not None:
        return known
    for key, canonical in ETHNICITY_CANONICAL.items():
        if key in text_norm:
            for known_value in known_ethnicities or []:
                if normalize_text(known_value) == normalize_text(canonical):
                    return known_value
            return canonical
    return str(text).strip()


def canonical_value(task, text, task_values=None):
    task_values = task_values or defaultdict(set)
    if task == "gender":
        return canonical_gender(text)
    if task == "age":
        return canonical_age(text)
    if task == "dialect":
        return canonical_dialect(text, task_values.get("dialect", set()))
    if task == "ethnicity":
        return canonical_ethnicity(text, task_values.get("ethnicity", set()))
    return None


def collect_task_values(manifest_path):
    values = {"dialect": set(), "ethnicity": set()}
    for row in iter_manifest(manifest_path):
        speaker = row.get("speaker") or {}
        if not isinstance(speaker, dict):
            continue
        dialect = speaker.get("dialect_region")
        ethnicity = speaker.get("ethnicity")
        if not is_missing_value(dialect):
            values["dialect"].add(str(dialect).strip())
        if not is_missing_value(ethnicity):
            values["ethnicity"].add(str(ethnicity).strip())
    return values


def source_prefix(audio_path):
    return str(audio_path).split("/")[0] if audio_path else ""


def extract_speaker_id(audio_path):
    for part in str(audio_path).split("/"):
        if SPEAKER_RE.match(part):
            return part.lower()
    return None


def choose_prompt_response(row, task):
    prompts = row.get("prompts") or []
    responses = row.get("responses") or []
    if len(prompts) != len(responses):
        return None, None
    needle = PROMPT_KEYWORDS[task]
    for prompt, response in zip(prompts, responses):
        if needle in str(prompt).lower():
            return str(prompt), str(response).replace("\n", " ").strip()
    return None, None


def speaker_label(row, task, task_values=None):
    speaker = row.get("speaker") or {}
    if not isinstance(speaker, dict):
        return None
    raw = speaker.get(TASK_TO_SPEAKER_KEY[task])
    if is_missing_value(raw):
        return None
    return canonical_value(task, raw, task_values)


def iter_expanded_examples(row, tasks=None, task_values=None):
    tasks = tuple(tasks or TASKS)
    audio_path = row.get("audio_path")
    src = source_prefix(audio_path)
    supported = SOURCE_TASKS.get(src, ())
    if not audio_path or not supported:
        return

    speaker = row.get("speaker") or {}
    speaker_id = speaker.get("id") if isinstance(speaker, dict) else None
    speaker_id = speaker_id or extract_speaker_id(audio_path)

    for task in tasks:
        if task not in supported:
            continue
        prompt, response = choose_prompt_response(row, task)
        if prompt is None or response is None:
            continue
        label = speaker_label(row, task, task_values)
        if label is None:
            continue
        yield {
            "row_index": row.get("index", ""),
            "source_prefix": src,
            "task": task,
            "label": str(label),
            "prompt": prompt,
            "response": response,
            "speaker_id": "" if speaker_id is None else str(speaker_id),
            "audio_path": audio_path,
        }


def parse_ears_segment(audio_path):
    match = EARS_SEGMENT_RE.match(str(audio_path))
    if not match:
        return None
    info = match.groupdict()
    info["speaker"] = info["speaker"].lower()
    info["start"] = int(info["start"])
    info["end"] = int(info["end"])
    return info


def _candidate_with_wav_suffix(path):
    candidates = [path]
    if str(path).upper().endswith(".WAV") and not str(path).endswith(".wav"):
        candidates.append(Path(str(path) + ".wav"))
    return candidates


def resolve_audio_reference(audio_path, roots):
    src = source_prefix(audio_path)
    root = roots.get(src)
    if root is None:
        return None, f"missing_root_{src}"
    root = Path(root)

    if src == "ears_dataset_processed":
        direct = root / audio_path
        if direct.exists():
            return direct, "direct_manifest_path"
        parsed = parse_ears_segment(audio_path)
        if parsed is None:
            return None, "unparsed_ears_path"
        raw_path = root / parsed["speaker"] / f"{parsed['stem']}.wav"
        if raw_path.exists():
            return raw_path, "ears_raw_segment"
        return None, "missing_ears_raw_file"

    candidates = []
    direct = root / audio_path
    candidates.extend(_candidate_with_wav_suffix(direct))

    parts = str(audio_path).split("/")
    if src == "timit_dataset" and len(parts) >= 3:
        split = parts[1].upper()
        rest = Path(*parts[2:])
        candidates.extend(_candidate_with_wav_suffix(root / "data" / split / rest))
        candidates.extend(_candidate_with_wav_suffix(root / split / rest))

    for candidate in candidates:
        if candidate.exists():
            method = "direct_manifest_path" if candidate == direct else "fallback_local_path"
            return candidate, method
    return None, "missing_audio_file"


def load_waveform_for_manifest(audio_path, roots, sample_rate=None):
    import torch
    import torchaudio

    resolved, method = resolve_audio_reference(audio_path, roots)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve {audio_path}: {method}")

    parsed = parse_ears_segment(audio_path)
    if source_prefix(audio_path) == "ears_dataset_processed" and method == "ears_raw_segment":
        waveform, sr = torchaudio.load(str(resolved))
        start = max(0, parsed["start"])
        end = min(max(start + 1, parsed["end"]), waveform.shape[1])
        if start >= waveform.shape[1] or end <= start:
            raise ValueError(
                f"Invalid EARS segment bounds for {audio_path}: "
                f"start={start}, end={end}, length={waveform.shape[1]}"
            )
        waveform = waveform[:, start:end].clone()
    else:
        waveform, sr = torchaudio.load(str(resolved))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate is not None and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    if waveform.numel() == 0:
        raise ValueError(f"Resolved empty waveform for {audio_path}")
    waveform = waveform.to(dtype=torch.float32)
    return waveform, sr, str(resolved), method


def resolve_device(device_arg):
    import torch

    device_arg = str(device_arg)
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
    return torch.device("cpu")


def maybe_prefix_model_keys(state_dict):
    if any(k.startswith("model.") for k in state_dict.keys()):
        return state_dict
    return {f"model.{k}": value for key, value in state_dict.items()}


def clean_module_state_dict(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def load_sid_model_from_config(config_path, device, sid_checkpoint=None):
    import torch
    from encoder.encoder import Model

    config = load_config(config_path)
    if sid_checkpoint is None:
        sid_checkpoint = Path(config["train"]["snapshot_path"]) / config["sid_model"]["sid_ck_name"]
    sid_checkpoint = Path(sid_checkpoint)

    model = Model(n_mels=80, embedding_dim=192, channel=1024)
    snapshot = torch.load(sid_checkpoint, map_location=device)
    if isinstance(snapshot, dict) and "sid_model" in snapshot:
        state = snapshot["sid_model"]
    elif isinstance(snapshot, dict) and "state_dict" in snapshot:
        state = snapshot["state_dict"]
    else:
        state = snapshot
    model.load_state_dict(maybe_prefix_model_keys(state))
    model.to(device)
    model.eval()
    return model


def load_mapper_checkpoint(mapper, mapper_checkpoint, device):
    import torch

    snapshot = torch.load(mapper_checkpoint, map_location=device)
    state = snapshot["sid_mapper"] if isinstance(snapshot, dict) and "sid_mapper" in snapshot else snapshot
    mapper.load_state_dict(clean_module_state_dict(state))
    return snapshot


def maybe_tqdm(total, desc, unit):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit=unit)
    except Exception:
        return None
