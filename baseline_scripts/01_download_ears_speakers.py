#!/usr/bin/env python3
import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path


SPEAKER_RE = re.compile(r"^p\d{3}$", flags=re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download EARS speaker zip files referenced by a TEARS manifest."
    )
    parser.add_argument(
        "--manifest",
        default="baseline_scripts/data/tears_train_manifest.jsonl",
        help="Manifest JSONL from 00_fetch_tears_manifest.py",
    )
    parser.add_argument(
        "--output-root",
        default="tears_audio",
        help="Directory where EARS audio should be extracted",
    )
    parser.add_argument(
        "--cache-dir",
        default="baseline_scripts/data/ears_zip_cache",
        help="Directory to cache downloaded speaker zip files",
    )
    parser.add_argument(
        "--url-template",
        default="https://github.com/facebookresearch/ears_dataset/releases/download/dataset/{speaker}.zip",
        help="Speaker zip URL template",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=1,
        help="Max number of speaker zips to download (default: 1 for quick setup)",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download even if zip already exists in cache",
    )
    return parser.parse_args()


def iter_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_speakers_from_path(audio_path: str):
    speakers = set()
    for part in str(audio_path).split("/"):
        if SPEAKER_RE.match(part):
            speakers.add(part.lower())
    return speakers


def download_file(url: str, dst: Path):
    with urllib.request.urlopen(url) as src, dst.open("wb") as out:
        out.write(src.read())


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    output_root = Path(args.output_root)
    cache_dir = Path(args.cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_speakers = set()
    total_rows = 0
    for row in iter_manifest_rows(manifest_path):
        total_rows += 1
        all_speakers.update(extract_speakers_from_path(row.get("audio_path", "")))

    speakers = sorted(all_speakers)
    if not speakers:
        print("No EARS speaker ids (p###) found in manifest audio_path values.")
        print("This usually means you only exported TIMIT-path samples.")
        return

    selected = speakers[: args.max_speakers]
    print(f"Rows scanned: {total_rows}")
    print(f"EARS speakers found: {len(speakers)}")
    print(f"Downloading {len(selected)} speaker zip(s): {', '.join(selected)}")

    downloaded = 0
    for speaker in selected:
        zip_name = f"{speaker}.zip"
        zip_path = cache_dir / zip_name
        url = args.url_template.format(speaker=speaker)

        if zip_path.exists() and not args.force_redownload:
            print(f"[skip] cached zip exists: {zip_path}")
        else:
            print(f"[get ] {url}")
            download_file(url, zip_path)
            downloaded += 1

        print(f"[unzip] {zip_path} -> {output_root}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_root)

    print(f"Completed. Downloaded new zips: {downloaded}")
    print(f"Audio root ready at: {output_root}")


if __name__ == "__main__":
    main()
