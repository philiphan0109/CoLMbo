#!/usr/bin/env python3
import argparse
import os

import torch
import torchaudio
import yaml

# Keep all caches local to this workspace to avoid permission issues.
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


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config_device: str):
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if config_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(config_device)
    if config_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
    return torch.device("cpu")


def maybe_prefix_model_keys(state_dict):
    # ECAPA checkpoint may be stored without the wrapper prefix.
    if any(k.startswith("model.") for k in state_dict.keys()):
        return state_dict
    return {f"model.{k}": v for k, v in state_dict.items()}


def main(config_path: str):
    config = load_config(config_path)
    config_train = config["train"]
    config_sid_model = config["sid_model"]
    config_data = config["data"]
    config_wrapper = dict(config["wrapper"])

    snapshot_path = config_train["snapshot_path"]
    device = resolve_device(config_wrapper.get("device", "cuda"))
    sample_rate = int(config_data.get("sample_rate", 16000))

    # Keep wrapper compatible with map_location logic in wrapper.py.
    if device.type == "cpu":
        config_wrapper["gpu_id"] = "cpu"
        config_wrapper["device"] = "cpu"
    else:
        config_wrapper["gpu_id"] = device.index if device.index is not None else 0
        config_wrapper["device"] = f"cuda:{config_wrapper['gpu_id']}"

    extractor = Mel_Spectrogram(sample_rate=sample_rate)
    exp = ExpWrapper(config_wrapper, config_wrapper["gpu_id"])

    sid_model = Model(n_mels=80, embedding_dim=192, channel=1024)
    ecapa_ckpt = torch.load("./pretrained_sid/ecapa.ckpt", map_location=device)
    if isinstance(ecapa_ckpt, dict) and "state_dict" in ecapa_ckpt:
        ecapa_ckpt = ecapa_ckpt["state_dict"]
    sid_model.load_state_dict(maybe_prefix_model_keys(ecapa_ckpt))

    exp.load_sid_model(sid_model, snapshot_path, config_sid_model["sid_ck_name"])
    exp.load_mapper(snapshot_path, config_wrapper["mapper_ck_name"])

    sid_model = sid_model.to(device)
    exp.sid_mapper = exp.sid_mapper.to(device)

    sid_model.eval()
    exp.gpt.eval()
    exp.sid_mapper.eval()

    waveform_audio, sr = torchaudio.load(config_data["waveform"])
    if sr != sample_rate:
        waveform_audio = torchaudio.functional.resample(waveform_audio, sr, sample_rate)

    with torch.no_grad():
        processed_waveform = extractor(waveform_audio).to(device)
        sid_emb = sid_model(processed_waveform)
        sids_prefix = exp.get_sid_prefix(sid_emb)
        prompt_prefix, _ = exp.get_prompt_prefix_single(config_data["prompt"])
        prefix_emb = torch.cat((sids_prefix, prompt_prefix), dim=1)
        generated_texts = exp.generate_beam(sids_prefix=prefix_emb)

    print(generated_texts[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single CoLMbo inference example")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config.yaml file",
    )
    args = parser.parse_args()
    main(args.config)
