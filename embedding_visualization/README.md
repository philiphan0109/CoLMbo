# Embedding Visualization

Modular EARS embedding analysis for:

- `PDAF raw` embeddings
- `ECAPA raw` embeddings
- `ECAPA -> mapper` learned embeddings

For each representation, the pipeline supports:

- utterance-level analysis
- speaker-level analysis

The intended final review set is small:

- `pca_utterance_grid.png`
- `pca_speaker_grid.png`
- `tsne_utterance_grid.png`
- `tsne_speaker_grid.png`

## Files

- `00_build_ears_index.py`
  Builds a clean EARS-only TEARS index.
- `01_extract_ecapa_spaces.py`
  Extracts raw ECAPA embeddings and ECAPA->mapper embeddings.
- `02_extract_pdaf_raw.py`
  Extracts raw PDAF embeddings.
- `03_plot_embedding_spaces.py`
  Reduces embeddings to 2D and generates the visualization grids.
- `04_run_ears_pipeline.py`
  One-command runner that executes the full EARS workflow end-to-end.
- `05_export_ears_predictions.py`
  Exports a clean EARS-only baseline prediction table from existing batch-eval outputs.
- `06_merge_ecapa_with_predictions.py`
  Merges ECAPA coordinate files with baseline predictions and writes utterance/speaker analysis tables.
- `07_plot_ecapa_prediction_errors.py`
  Creates ECAPA error-overlay plots: points are colored by gold label and incorrect predictions are highlighted.
- `08_extract_llm_hidden_states.py`
  Extracts in-LM GPT hidden-state embeddings from `ECAPA raw -> mapper prefix + task prompt`.
- `09_plot_llm_hidden_errors.py`
  Reduces and plots the in-LM hidden-state embeddings with prediction errors highlighted.
- `common.py`
  Shared data/model helpers.

## One-Command Run

Once `tears_audio/` is available again, the simplest entry point is:

```bash
python embedding_visualization/04_run_ears_pipeline.py \
  --ears-root tears_audio \
  --output-root embedding_visualization/runs/ears_default
```

This will:

- build the EARS index
- extract ECAPA raw embeddings
- extract ECAPA mapper embeddings
- extract PDAF raw embeddings
- generate the final PCA and t-SNE grids

If you want to reuse existing artifacts:

```bash
python embedding_visualization/04_run_ears_pipeline.py \
  --ears-root tears_audio \
  --output-root embedding_visualization/runs/ears_default \
  --skip-existing
```

## Typical Workflow

Build the EARS index:

```bash
python embedding_visualization/00_build_ears_index.py
```

Extract ECAPA raw + mapper spaces:

```bash
python embedding_visualization/01_extract_ecapa_spaces.py \
  --ears-root tears_audio \
  --output-dir embedding_visualization/data/ecapa/default
```

Extract PDAF raw space:

```bash
python embedding_visualization/02_extract_pdaf_raw.py \
  --ears-root tears_audio \
  --output-dir embedding_visualization/data/pdaf/default
```

Generate plots:

```bash
python embedding_visualization/03_plot_embedding_spaces.py \
  --ecapa-dir embedding_visualization/data/ecapa \
  --pdaf-dir embedding_visualization/data/pdaf \
  --output-dir embedding_visualization/output
```

## Baseline Error Analysis (ECAPA)

If you already have a completed embedding run, you can layer the baseline predictions on top:

```bash
python embedding_visualization/05_export_ears_predictions.py

python embedding_visualization/06_merge_ecapa_with_predictions.py \
  --run-root embedding_visualization/runs/ears_default

python embedding_visualization/07_plot_ecapa_prediction_errors.py \
  --utterance-csv embedding_visualization/runs/ears_default/analysis/ecapa_prediction_analysis_utterance.csv \
  --output-dir embedding_visualization/runs/ears_default/error_plots \
  --reducers tsne pca
```

Main outputs:

- `embedding_visualization/runs/ears_default/analysis/ears_predictions_long.csv`
- `embedding_visualization/runs/ears_default/analysis/ecapa_prediction_analysis_utterance.csv`
- `embedding_visualization/runs/ears_default/analysis/ecapa_prediction_analysis_speaker.csv`
- `embedding_visualization/runs/ears_default/error_plots/tsne_utterance_error_grid.png`
- `embedding_visualization/runs/ears_default/error_plots/pca_utterance_error_grid.png`

## In-LM Hidden-State Analysis

This stage extracts GPT hidden states after the language model has processed:

```text
ECAPA raw embedding -> audio mapper -> GPT-compatible audio prefix + task prompt
```

Each row is an `(utterance, task)` example. By default, the pooled vector is the
final-layer hidden state at the last input position (`last_input`), which is the
position GPT uses for next-token generation.

Smoke test:

```bash
python embedding_visualization/08_extract_llm_hidden_states.py \
  --device cpu \
  --batch-size 8 \
  --limit-per-task 10 \
  --output-dir embedding_visualization/runs/ears_default/llm_analysis_smoketest

python embedding_visualization/09_plot_llm_hidden_errors.py \
  --metadata-csv embedding_visualization/runs/ears_default/llm_analysis_smoketest/utterance_task_llm_hidden_last_input_metadata.csv \
  --output-dir embedding_visualization/runs/ears_default/llm_analysis_smoketest/plots \
  --reducers pca tsne
```

Full run:

```bash
python embedding_visualization/08_extract_llm_hidden_states.py \
  --device cuda \
  --batch-size 32 \
  --output-dir embedding_visualization/runs/ears_default/llm_analysis

python embedding_visualization/09_plot_llm_hidden_errors.py \
  --metadata-csv embedding_visualization/runs/ears_default/llm_analysis/utterance_task_llm_hidden_last_input_metadata.csv \
  --output-dir embedding_visualization/runs/ears_default/llm_analysis/plots \
  --reducers tsne pca
```

Main outputs:

- `embedding_visualization/runs/ears_default/llm_analysis/utterance_task_llm_hidden_last_input_metadata.csv`
- `embedding_visualization/runs/ears_default/llm_analysis/utterance_task_llm_hidden_last_input_embeddings.npy`
- `embedding_visualization/runs/ears_default/llm_analysis/plots/tsne_llm_hidden_error_grid.png`
- `embedding_visualization/runs/ears_default/llm_analysis/plots/pca_llm_hidden_error_grid.png`

## Notes

- The ECAPA branch follows the current CoLMbo inference path in this repo.
- The PDAF branch is implemented as a best-effort raw embedding extractor using a 128-mel frontend and neutral attention controls (`lambda=0`, zero `prob_phn`, all-ones mask), because the repo does not expose the original phoneme-aware inference preprocessing as a standalone script.
- Extraction does not materialize cached segment WAV files; it slices EARS segments in memory from raw speaker audio to stay storage-light.
