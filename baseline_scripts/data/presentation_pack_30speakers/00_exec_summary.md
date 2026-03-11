# CoLMbo Baseline: Interim Results (for Presentation)

## What Was Evaluated
- Split: `TEARS test`
- Available audio now: `EARS` subset downloaded locally (30 speakers: p001-p030)
- Evaluated tasks in current run: `gender`, `age` (EARS only)
- `dialect` is blocked pending TIMIT local access

## Raw Performance (Current Run)
- Gender (EARS): value accuracy = **92.22%** on **2032** samples
- Age (EARS): value accuracy = **41.98%** on **2032** samples
- Exact-match accuracy is lower (expected) due paraphrasing:
  - Gender exact match = **6.30%**
  - Age exact match = **15.90%**

## Coverage With Current Local Data (Dry-Run)
- Potential evaluable EARS rows now:
  - Gender: **2032**
  - Age: **2032**
- Dialect: requires TIMIT root; currently **not evaluated**

## Key Findings
- Model is strong on gender for current EARS subset.
- Model makes age-range confusions (`36-45` -> `18-25` / `26-35`) in failure samples.
- Exact-match metric alone underestimates quality for generative answers; value-level scoring is more appropriate.

## Blocking Item
- TIMIT (LDC93S1) path is required for dialect baseline.
