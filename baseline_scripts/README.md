# Baseline Scripts

Numbered scripts for quickly bootstrapping TEARS-based baseline runs.

## 00_fetch_tears_manifest.py
Fetches `cmu-mlsp/TEARS` metadata from Hugging Face and writes a local JSONL manifest.

Example:
```bash
python baseline_scripts/00_fetch_tears_manifest.py --split train
```

## 01_download_ears_speakers.py
Downloads EARS speaker zip files referenced in the manifest (only `p###` speakers).

Example:
```bash
python baseline_scripts/01_download_ears_speakers.py --max-speakers 1
```

## 02_make_sample_wav.py
Builds `sample.wav` from the first resolvable manifest row for a task.
It supports:
- direct file resolution (`<audio_root>/<audio_path>`)
- TEARS EARS processed-path reconstruction by slicing from raw EARS audio
  and caching materialized segments under `baseline_scripts/data/resolved_audio`.

Also writes `baseline_scripts/data/sample_meta.json` with selected prompt/answer.

Example (EARS audio root):
```bash
python baseline_scripts/02_make_sample_wav.py \
  --audio-root tears_audio \
  --task gender \
  --source-prefix ears_dataset_processed \
  --output-wav sample.wav
```

Note:
- `dialect` may be unavailable in some source subsets. By default, the script is strict and skips rows without a matching task prompt.
- If you only need a runnable sample (not strict task matching), add `--allow-prompt-fallback`.

## 03_manifest_task_coverage.py
Reports prompt coverage (`gender`, `age`, `dialect`) by source prefix from the manifest.

Example:
```bash
python baseline_scripts/03_manifest_task_coverage.py
```

## 04_run_three_task_baseline.py
Runs all three tasks with strict source selection:
- `gender`, `age` from `ears_dataset_processed` (needs `--ears-root`)
- `dialect` from `timit_dataset` (needs `--timit-root`)

Outputs:
- `baseline_scripts/data/three_task_baseline/baseline_report.json`
- `baseline_scripts/data/three_task_baseline/baseline_report.csv`

Example:
```bash
python baseline_scripts/04_run_three_task_baseline.py \
  --ears-root tears_audio \
  --timit-root /path/to/timit_root
```

## 05_batch_eval_baseline.py
Runs batched evaluation and writes:
- per-sample predictions: `predictions.csv`
- grouped metrics: `summary.csv` / `summary.json`
- failure subset: `failures.csv`

By default (for speed), caps evaluation at `200` samples per `(task, source_prefix)` group.
Set `--max-samples-per-group 0` to evaluate all available rows.

Example:
```bash
python baseline_scripts/05_batch_eval_baseline.py \
  --manifest baseline_scripts/data/tears_test_manifest.jsonl \
  --ears-root tears_audio \
  --timit-root /path/to/timit_root \
  --voxceleb-root /path/to/voxceleb2_root \
  --output-dir baseline_scripts/data/batch_eval
```

## 06_plot_raw_metrics.py
Creates figures from `summary.csv`:
- task x dataset heatmap (`value_accuracy`)
- grouped accuracy bars
- support counts plot

Example:
```bash
python baseline_scripts/06_plot_raw_metrics.py \
  --summary-csv baseline_scripts/data/batch_eval/summary.csv \
  --output-dir baseline_scripts/data/batch_eval/figures
```

## 07_make_presentation_pack.py
Builds a compact presentation bundle from existing outputs:
- executive summary markdown
- slide-friendly metrics table CSV
- top failure examples CSV
- talking points markdown

Example:
```bash
python baseline_scripts/07_make_presentation_pack.py \
  --results-summary-csv baseline_scripts/data/batch_eval/summary.csv \
  --results-failures-csv baseline_scripts/data/batch_eval/failures.csv \
  --coverage-summary-csv baseline_scripts/data/batch_eval_coverage_now/summary.csv \
  --output-dir baseline_scripts/data/presentation_pack
```

Example (TIMIT licensed root):
```bash
python baseline_scripts/02_make_sample_wav.py \
  --audio-root /path/to/timit_root \
  --task dialect \
  --output-wav sample.wav
```
