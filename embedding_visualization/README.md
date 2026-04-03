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

## Notes

- The ECAPA branch follows the current CoLMbo inference path in this repo.
- The PDAF branch is implemented as a best-effort raw embedding extractor using a 128-mel frontend and neutral attention controls (`lambda=0`, zero `prob_phn`, all-ones mask), because the repo does not expose the original phoneme-aware inference preprocessing as a standalone script.
- Extraction does not materialize cached segment WAV files; it slices EARS segments in memory from raw speaker audio to stay storage-light.
