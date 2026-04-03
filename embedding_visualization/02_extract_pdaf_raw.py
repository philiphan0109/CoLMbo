#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from common import (
    aggregate_by_speaker,
    build_ears_index,
    ensure_output_dir,
    resolve_device,
    save_embedding_artifact,
    settle_modules_on_device,
    setup_local_env,
    EARSWaveformResolver,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract PDAF raw embeddings for EARS audio. "
            "This uses a mel-spectrogram frontend and neutral attention controls "
            "(lambda=0, zero prob_phn, all-ones mask) because the repo does not "
            "currently expose the original phoneme-aware inference path."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_test_manifest.jsonl",
        help="TEARS manifest JSONL path",
    )
    parser.add_argument(
        "--ears-root",
        required=True,
        help="Root directory with EARS raw speaker audio",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_visualization/data/pdaf/default",
        help="Output directory for PDAF artifacts",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for extraction (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of utterances to process (0 = all)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N utterances when tqdm is unavailable",
    )
    return parser.parse_args()


def load_pdaf_model():
    from encoder.self_attn import TransformerSelfAttention

    model = TransformerSelfAttention(
        input_dim=128,
        num_heads=8,
        dim_feedforward=128,
        number_Of_spks=106,
        dropout=0.0,
    )
    snapshot = torch.load("checkpoints/pdaf.pt", map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()
    return model


def build_extractor(sample_rate):
    from load_data.extract_fbanks import Mel_Spectrogram

    return Mel_Spectrogram(sample_rate=sample_rate, n_mels=128)


def maybe_use_tqdm(total):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc="PDAF extraction", unit="utt")
    except Exception:
        return None


def main():
    setup_local_env()
    args = parse_args()
    requested_device = resolve_device(args.device)
    out_dir = ensure_output_dir(Path(args.output_dir))

    model = load_pdaf_model()
    device = settle_modules_on_device([("PDAF model", model)], requested_device)
    extractor = build_extractor(args.sample_rate)
    resolver = EARSWaveformResolver(Path(args.ears_root))

    index_df = build_ears_index(Path(args.manifest))
    if args.limit and args.limit > 0:
        index_df = index_df.iloc[: args.limit].reset_index(drop=True)
    if index_df.empty:
        raise RuntimeError("No EARS rows found in the provided manifest.")

    vectors = []
    kept_rows = []
    pbar = maybe_use_tqdm(len(index_df))

    with torch.no_grad():
        for i, row in index_df.iterrows():
            waveform, sr = resolver.resolve(row["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != args.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)

            features = extractor(waveform).squeeze(1).to(device)
            seq_len = features.shape[1]
            prob_phn = torch.zeros(
                (features.shape[0], seq_len, seq_len), dtype=features.dtype, device=device
            )
            mask = torch.ones(
                (features.shape[0], seq_len, seq_len), dtype=features.dtype, device=device
            )

            _, emb = model(features, prob_phn=prob_phn, mask=mask, lambda_val=0.0)
            vectors.append(emb.squeeze(0).detach().cpu().numpy().astype(np.float32))

            row_dict = row.to_dict()
            row_dict["embedding_family"] = "pdaf"
            row_dict["pdaf_mode"] = "neutral_mel128"
            kept_rows.append(row_dict)

            if pbar is not None:
                pbar.update(1)
            elif (i + 1) % max(1, args.progress_every) == 0 or (i + 1) == len(index_df):
                print(f"[{i + 1}/{len(index_df)}] processed")

    if pbar is not None:
        pbar.close()

    utterance_df = pd.DataFrame(kept_rows)
    vectors = np.vstack(vectors)

    save_embedding_artifact(out_dir, "utterance_pdaf_raw", utterance_df, vectors)
    speaker_df, speaker_vectors = aggregate_by_speaker(
        utterance_df, vectors, extra_columns=["embedding_family", "pdaf_mode"]
    )
    save_embedding_artifact(out_dir, "speaker_pdaf_raw", speaker_df, speaker_vectors)

    print(f"Wrote PDAF artifacts to: {out_dir}")
    print(f"Utterances: {len(utterance_df)}")
    print(f"Speakers:   {utterance_df['speaker_id'].nunique()}")
    print(f"Device used: {device}")
    print("PDAF mode: neutral_mel128")


if __name__ == "__main__":
    main()
