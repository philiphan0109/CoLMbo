# Mapper Training Scripts

This folder is for the current all-task CoLMbo baseline and weighted-sampling comparison.

The intended experiment is:

1. Expand TEARS metadata into one row per supported task.
2. Precompute frozen ECAPA embeddings once per unique audio segment.
3. Train the mapper with uniform sampling.
4. Train the mapper with task+label weighted sampling.
5. Evaluate both mapper checkpoints with value accuracy and balanced value accuracy.

## Environment Check

```bash
find tears_audio -maxdepth 1 -mindepth 1 -type d | wc -l
ls checkpoints checkpoint_mlp_mapper
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, pandas, numpy, torchaudio; print('ok')"
```

## Build Training Examples

If the train split has not been exported yet:

```bash
python training_scripts/00_build_train_manifest.py \
  --download \
  --raw-manifest baseline_scripts/data/tears_train_manifest.jsonl \
  --output training_scripts/data/train_examples_ears_timit.csv \
  --ears-root tears_audio \
  --timit-root timit_root \
  --check-audio
```

If `baseline_scripts/data/tears_train_manifest.jsonl` already exists, omit `--download`.

## Cache ECAPA Embeddings

```bash
python training_scripts/01_precompute_ecapa_embeddings.py \
  --examples-csv training_scripts/data/train_examples_ears_timit.csv \
  --output-dir training_scripts/data/ears_timit_ecapa_cache \
  --device cuda
```

Smoke test:

```bash
python training_scripts/01_precompute_ecapa_embeddings.py \
  --examples-csv training_scripts/data/train_examples_ears_timit.csv \
  --output-dir training_scripts/data/smoke_ecapa_cache \
  --device cuda \
  --limit-audio 32
```

## Train Uniform Baseline

```bash
python training_scripts/02_train_mapper_baseline.py \
  --examples-csv training_scripts/data/ears_timit_ecapa_cache/train_examples_with_embeddings.csv \
  --embeddings-npy training_scripts/data/ears_timit_ecapa_cache/audio_embeddings.npy \
  --output-dir training_scripts/runs/mapper_uniform_20ep \
  --epochs 20 \
  --batch-size 64 \
  --device cuda
```

For a fast smoke run:

```bash
python training_scripts/02_train_mapper_baseline.py \
  --examples-csv training_scripts/data/ears_timit_ecapa_cache/train_examples_with_embeddings.csv \
  --embeddings-npy training_scripts/data/ears_timit_ecapa_cache/audio_embeddings.npy \
  --output-dir training_scripts/runs/mapper_uniform_smoke \
  --epochs 1 \
  --limit-examples 256 \
  --max-steps-per-epoch 10 \
  --batch-size 16 \
  --device cuda
```

## Train Weighted Variant

```bash
python training_scripts/03_train_mapper_weighted.py \
  --examples-csv training_scripts/data/ears_timit_ecapa_cache/train_examples_with_embeddings.csv \
  --embeddings-npy training_scripts/data/ears_timit_ecapa_cache/audio_embeddings.npy \
  --output-dir training_scripts/runs/mapper_weighted_20ep \
  --epochs 20 \
  --batch-size 64 \
  --weight-alpha 0.5 \
  --device cuda
```

Optional hard-example multiplier from existing EARS baseline predictions:

```bash
python training_scripts/03_train_mapper_weighted.py \
  --examples-csv training_scripts/data/ears_timit_ecapa_cache/train_examples_with_embeddings.csv \
  --embeddings-npy training_scripts/data/ears_timit_ecapa_cache/audio_embeddings.npy \
  --output-dir training_scripts/runs/mapper_weighted_hard_20ep \
  --epochs 20 \
  --batch-size 64 \
  --weight-alpha 0.5 \
  --hard-predictions baseline_scripts/data/ears_full_sharded/predictions_all.csv \
  --hard-weight 2.0 \
  --device cuda
```

## Evaluate

Evaluate the original pretrained mapper:

```bash
python training_scripts/04_eval_trained_mapper.py \
  --manifest baseline_scripts/data/tears_test_manifest.jsonl \
  --ears-root tears_audio \
  --timit-root timit_root \
  --mapper-checkpoint checkpoint_mlp_mapper/mapper_ce_llm.pt \
  --tasks gender age ethnicity dialect \
  --max-samples-per-group 0 \
  --output-dir training_scripts/runs/eval_pretrained_ears_timit
```

Evaluate a fine-tuned mapper:

```bash
python training_scripts/04_eval_trained_mapper.py \
  --manifest baseline_scripts/data/tears_test_manifest.jsonl \
  --ears-root tears_audio \
  --timit-root timit_root \
  --mapper-checkpoint training_scripts/runs/mapper_weighted_20ep/best_mapper_ce_llm.pt \
  --tasks gender age ethnicity dialect \
  --max-samples-per-group 0 \
  --output-dir training_scripts/runs/eval_weighted_20ep
```

The main comparison rows are `source_prefix == OVERALL` for each task and the `MACRO_AVG` row in each `summary.csv`.
