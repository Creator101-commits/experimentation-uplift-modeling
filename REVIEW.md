# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts, and README figures.

Fixed cumulative-curve endpoints, causal feature validation, rare-outcome cross-fitting, invalid allocation checks, two-sided power under the null, and common nuisance estimates for policy comparisons. Later fixes: gzip header detection before CSV parsing, kernel push directory derived from metadata, smoke manifest SHA verification, raw-float64 duplicate hashing, and paired bootstrap differences retained for every reference model.

## Verified locally

- 52 tests passed with `uv run pytest` on Python 3.13.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed (strict mode).
- The README image was rendered from `results/` by `scripts/render_figure.py` and checked for text overlap programmatically (zero overlapping labels across all five panels).

## Verified on Kaggle

- Smoke kernel `experimentation-uplift-modeling-smoke` on git SHA `dbca790`: sampled exactly 200,000 rows; prepare 52.5s, benchmark 38.9s, total 109s; `manifest.git_sha` matched and `ordering_ok` was true.
- Full kernel `experimentation-uplift-modeling-benchmark` on git SHA `41bad9b5ca1b0be30793e58062631eb7a646c6f5`: protocol `uplift-benchmark-v1`, `ordering_ok` true, XGBoost on CUDA, total runtime about 60 minutes. Outputs curated unedited into `results/`.

## Limits

- CLI demo data are synthetic rather than Criteo; individual counterfactuals are not observed; power calculations use normal approximations.
- The published `metrics.json` `paired_diffs` retains only differences against the logistic T-learner reference because of an overwrite defect fixed in `39499e6` after the run; per-model bootstrap intervals are complete.
- Per-model Qini curves and calibration bins were not persisted, so the figure shows scalar curve summaries with confidence intervals.
- One frozen split of one dataset version; duplicate source rows retained per the source release.
- New CI workflows are configuration changes; their hosted execution must be checked after publication.
