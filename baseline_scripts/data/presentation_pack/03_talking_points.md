# Talking Points (2-3 min)

- I built a reproducible baseline pipeline around TEARS test metadata and external audio sources.
- I validated and loaded EARS audio locally; current run evaluates gender and age.
- Current raw value-level results:
  - Gender is near-perfect on the evaluated subset.
  - Age shows meaningful errors, mainly underestimating older speakers.
- Exact string match is not ideal for this generative model; value-level correctness is a better baseline metric.
- Dialect evaluation is blocked only by missing TIMIT local access (licensed LDC dataset), not by pipeline readiness.
- Next immediate step: add TIMIT root, rerun the same pipeline, and report full 3-task baseline.
