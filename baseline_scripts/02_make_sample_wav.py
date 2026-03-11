#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import re
from pathlib import Path

import torch
import torchaudio


PROMPT_KEYWORDS = {
    "gender": "gender",
    "age": "age",
    "dialect": "dialect",
}

EARS_SEGMENT_RE = re.compile(
    r"^ears_dataset_processed/(?P<split>[^/]+)/(?P<speaker>p\d{3})/(?P<stem>.+)_(?P<start>\d+)_(?P<end>\d+)\.wav$",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create sample.wav from a TEARS manifest row.\n"
            "Supports direct paths and TEARS EARS-segment paths by materializing\n"
            "segments from raw EARS audio when needed."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Manifest JSONL from 00_fetch_tears_manifest.py",
    )
    parser.add_argument(
        "--audio-root",
        action="append",
        required=True,
        help=(
            "Root directory containing audio files. Can be passed multiple times. "
            "Each root is joined with manifest audio_path."
        ),
    )
    parser.add_argument("--task", choices=["gender", "age", "dialect"], default="gender")
    parser.add_argument(
        "--allow-prompt-fallback",
        action="store_true",
        help=(
            "If no task-specific prompt exists for a row, fallback to that row's first "
            "prompt/response pair. Disabled by default for strict task labels."
        ),
    )
    parser.add_argument(
        "--source-prefix",
        default="ears_dataset_processed",
        help=(
            "Filter manifest rows by audio_path prefix "
            "(default: ears_dataset_processed). Use 'any' to disable filtering."
        ),
    )
    parser.add_argument(
        "--cache-root",
        default="baseline_scripts/data/resolved_audio",
        help=(
            "Where to materialize TEARS segment files when source is "
            "ears_dataset_processed/*_start_end.wav."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of manifest rows to scan.",
    )
    parser.add_argument("--output-wav", default="sample.wav", help="Output wav path")
    parser.add_argument(
        "--output-meta",
        default="baseline_scripts/data/sample_meta.json",
        help="Output sidecar JSON path with selected prompt/answer/source path",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    return parser.parse_args()


def choose_prompt_answer(prompts, responses, task, allow_prompt_fallback=False):
    prompts = prompts or []
    responses = responses or []
    keyword = PROMPT_KEYWORDS[task]
    if len(prompts) != len(responses):
        return None, None

    for p, r in zip(prompts, responses):
        if keyword in str(p).lower():
            return p, r

    if allow_prompt_fallback and prompts:
        return prompts[0], responses[0]
    return None, None


def iter_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_direct_path(audio_path, roots):
    for root in roots:
        candidate = root / audio_path
        if candidate.exists():
            return candidate, "direct_manifest_path"
    return None, None


def parse_ears_segment_path(audio_path):
    match = EARS_SEGMENT_RE.match(str(audio_path))
    if not match:
        return None
    parsed = match.groupdict()
    parsed["speaker"] = parsed["speaker"].lower()
    parsed["start"] = int(parsed["start"])
    parsed["end"] = int(parsed["end"])
    return parsed


def find_ears_raw_file(parsed, roots):
    # TEARS processed: ears_dataset_processed/<split>/p001/<stem>_<start>_<end>.wav
    # EARS raw zip:    <root>/p001/<stem>.wav
    rel_raw = Path(parsed["speaker"]) / f"{parsed['stem']}.wav"
    for root in roots:
        candidate = root / rel_raw
        if candidate.exists():
            return candidate
    return None


def materialize_ears_segment(audio_path, parsed, raw_file, cache_root):
    out_path = cache_root / audio_path
    if out_path.exists():
        return out_path, "materialized_segment_cached"

    waveform, sr = torchaudio.load(str(raw_file))
    start = max(0, parsed["start"])
    end = max(start + 1, parsed["end"])
    if waveform.shape[1] == 0:
        return None, "raw_audio_empty"
    if start >= waveform.shape[1]:
        return None, "segment_start_out_of_bounds"
    end = min(end, waveform.shape[1])
    segment = waveform[:, start:end]
    if segment.shape[1] == 0:
        return None, "segment_empty_after_slice"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), segment, sr)
    return out_path, "materialized_segment_created"


def resolve_audio_path(audio_path, roots, cache_root):
    direct, method = resolve_direct_path(audio_path, roots)
    if direct is not None:
        return direct, method

    parsed = parse_ears_segment_path(audio_path)
    if parsed is None:
        prefix = str(audio_path).split("/")[0] if audio_path else "UNKNOWN"
        return None, f"unresolved_{prefix}"

    raw_file = find_ears_raw_file(parsed, roots)
    if raw_file is None:
        return None, f"unresolved_missing_ears_raw_{parsed['speaker']}"

    return materialize_ears_segment(audio_path, parsed, raw_file, cache_root)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    audio_roots = [Path(r) for r in args.audio_root]
    cache_root = Path(args.cache_root)
    for root in audio_roots:
        if not root.exists():
            print(f"[warn] audio root does not exist yet: {root}")

    selected_row = None
    selected_audio = None
    selected_resolution_method = None
    reasons = Counter()
    scanned = 0

    for row in iter_manifest(manifest_path):
        scanned += 1
        if args.max_rows is not None and scanned > args.max_rows:
            break

        audio_path = row.get("audio_path")
        if not audio_path:
            reasons["missing_audio_path"] += 1
            continue
        prefix = str(audio_path).split("/")[0]
        if args.source_prefix != "any" and prefix != args.source_prefix:
            reasons[f"filtered_prefix_{prefix}"] += 1
            continue

        prompt, answer = choose_prompt_answer(
            row.get("prompts"),
            row.get("responses"),
            args.task,
            args.allow_prompt_fallback,
        )
        if prompt is None:
            reasons["missing_prompt_response_for_task"] += 1
            continue

        resolved, method = resolve_audio_path(audio_path, audio_roots, cache_root)
        if resolved is None:
            reasons[method] += 1
            continue

        selected_row = row
        selected_audio = resolved
        selected_resolution_method = method
        selected_prompt = prompt
        selected_answer = answer
        break

    if selected_row is None:
        roots_str = ", ".join(str(r) for r in audio_roots)
        reason_lines = "\n".join(
            f"  {k}: {v}" for k, v in reasons.most_common(12)
        ) or "  (no reasons recorded)"
        raise RuntimeError(
            "Could not resolve any audio file from manifest with provided roots: "
            f"{roots_str}\n"
            f"Scanned rows: {scanned}\n"
            f"Top reasons:\n{reason_lines}"
        )

    waveform, sr = torchaudio.load(str(selected_audio))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != args.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)
        sr = args.sample_rate

    # Ensure float32 waveform for consistent save behavior.
    waveform = waveform.to(torch.float32)

    output_wav = Path(args.output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_wav), waveform, sr)

    meta = {
        "task": args.task,
        "source_audio_path_manifest": selected_row.get("audio_path"),
        "source_audio_path_resolved": str(selected_audio),
        "resolution_method": selected_resolution_method,
        "source_prefix": str(selected_row.get("audio_path")).split("/")[0],
        "speaker": selected_row.get("speaker"),
        "selected_prompt": selected_prompt,
        "selected_answer": selected_answer,
        "output_wav": str(output_wav),
        "sample_rate": sr,
        "num_samples": int(waveform.shape[-1]),
    }

    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    print(f"Created wav:  {output_wav}")
    print(f"Created meta: {output_meta}")
    print(f"Resolved via: {selected_resolution_method}")
    if selected_prompt:
        print(f"Prompt ({args.task}): {selected_prompt}")
    if selected_answer:
        print(f"Gold answer: {selected_answer}")


if __name__ == "__main__":
    main()
