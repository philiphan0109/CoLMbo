#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EARS_SEGMENT_RE = re.compile(
    r"^ears_dataset_processed/(?P<split>[^/]+)/(?P<speaker>p\d{3})/(?P<stem>.+)_(?P<start>\d+)_(?P<end>\d+)\.wav$",
    flags=re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^p\d{3}$", flags=re.IGNORECASE)
LABEL_COLUMNS = ("gender", "age", "ethnicity")


def setup_local_env():
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".hf_home" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".hf_home" / "transformers"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(REPO_ROOT / "embedding_visualization" / ".mplconfig")
    )
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_ears_segment(audio_path: str):
    match = EARS_SEGMENT_RE.match(str(audio_path))
    if not match:
        return None
    info = match.groupdict()
    info["speaker"] = info["speaker"].lower()
    info["start"] = int(info["start"])
    info["end"] = int(info["end"])
    return info


def extract_speaker_id(audio_path: str):
    for part in str(audio_path).split("/"):
        if SPEAKER_RE.match(part):
            return part.lower()
    return None


def build_ears_index(manifest_path: Path):
    rows = []
    for row in iter_manifest(manifest_path):
        audio_path = row.get("audio_path", "")
        if not str(audio_path).startswith("ears_dataset_processed/"):
            continue
        speaker = row.get("speaker") or {}
        speaker_id = extract_speaker_id(audio_path)
        split = ""
        parsed = parse_ears_segment(audio_path)
        if parsed is not None:
            split = parsed["split"]
        rows.append(
            {
                "row_index": row.get("index"),
                "audio_path": audio_path,
                "speaker_id": speaker_id,
                "split": split,
                "gender": speaker.get("gender"),
                "age": speaker.get("age"),
                "ethnicity": speaker.get("ethnicity"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    return df


class EARSWaveformResolver:
    def __init__(self, ears_root: Path):
        self.ears_root = Path(ears_root)
        self._cached_raw_path = None
        self._cached_waveform = None
        self._cached_sr = None

    def _load_raw(self, raw_path: Path):
        if self._cached_raw_path == raw_path:
            return self._cached_waveform, self._cached_sr
        waveform, sample_rate = torchaudio.load(str(raw_path))
        self._cached_raw_path = raw_path
        self._cached_waveform = waveform
        self._cached_sr = sample_rate
        return waveform, sample_rate

    def resolve(self, audio_path: str):
        direct = self.ears_root / audio_path
        if direct.exists():
            return torchaudio.load(str(direct))

        parsed = parse_ears_segment(audio_path)
        if parsed is None:
            raise FileNotFoundError(f"Could not parse EARS path: {audio_path}")

        raw_path = self.ears_root / parsed["speaker"] / f"{parsed['stem']}.wav"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw EARS file: {raw_path}")

        waveform, sample_rate = self._load_raw(raw_path)
        start = max(0, parsed["start"])
        end = min(max(start + 1, parsed["end"]), waveform.shape[1])
        if start >= waveform.shape[1] or end <= start:
            raise ValueError(
                f"Invalid segment bounds for {audio_path}: start={start}, end={end}, "
                f"length={waveform.shape[1]}"
            )
        return waveform[:, start:end].clone(), sample_rate


def maybe_prefix_model_keys(state_dict):
    if any(k.startswith("model.") for k in state_dict.keys()):
        return state_dict
    return {f"model.{k}": v for k, v in state_dict.items()}


def resolve_device(device_arg: str):
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
    return torch.device("cpu")


def settle_modules_on_device(named_modules, requested_device: torch.device):
    requested_device = torch.device(requested_device)

    if requested_device.type != "cuda":
        for _, module in named_modules:
            module.to(requested_device)
        return requested_device

    try:
        for _, module in named_modules:
            module.to(requested_device)
        return requested_device
    except torch.cuda.OutOfMemoryError as exc:
        fallback = torch.device("cpu")
        failing_name = "module"
        if named_modules:
            failing_name = named_modules[-1][0]
        print(
            f"CUDA OOM while moving {failing_name} to {requested_device}. "
            "Falling back to CPU for this run."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _, module in named_modules:
            module.to(fallback)
        return fallback


def load_config(config_path: Path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_embedding_artifact(out_dir: Path, prefix: str, metadata_df: pd.DataFrame, embeddings):
    out_dir = ensure_output_dir(out_dir)
    metadata_path = out_dir / f"{prefix}_metadata.csv"
    embedding_path = out_dir / f"{prefix}_embeddings.npy"
    metadata_df.to_csv(metadata_path, index=False)
    np.save(embedding_path, np.asarray(embeddings, dtype=np.float32))
    return metadata_path, embedding_path


def aggregate_by_speaker(metadata_df: pd.DataFrame, embeddings, extra_columns=None):
    if metadata_df.empty:
        return metadata_df.copy(), np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    if extra_columns is None:
        extra_columns = []
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rows = []
    speaker_vectors = []

    grouped = metadata_df.groupby("speaker_id", sort=True)
    for speaker_id, idx in grouped.indices.items():
        idx_list = list(idx)
        speaker_meta = metadata_df.iloc[idx_list[0]].to_dict()
        speaker_meta["n_utterances"] = len(idx_list)
        rows.append(
            {
                "speaker_id": speaker_id,
                "gender": speaker_meta.get("gender"),
                "age": speaker_meta.get("age"),
                "ethnicity": speaker_meta.get("ethnicity"),
                "n_utterances": len(idx_list),
                **{col: speaker_meta.get(col) for col in extra_columns},
            }
        )
        speaker_vectors.append(embeddings[idx_list].mean(axis=0))

    return pd.DataFrame(rows), np.vstack(speaker_vectors)


def load_artifact_pair(metadata_path: Path):
    metadata_path = Path(metadata_path)
    if not metadata_path.name.endswith("_metadata.csv"):
        raise ValueError(f"Expected metadata path, got: {metadata_path}")
    embedding_path = metadata_path.with_name(
        metadata_path.name.replace("_metadata.csv", "_embeddings.npy")
    )
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding file for {metadata_path}")
    return pd.read_csv(metadata_path), np.load(embedding_path)
