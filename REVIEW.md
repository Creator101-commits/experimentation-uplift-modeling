# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts, and README figures.

Fixed cumulative-curve endpoints, causal feature validation, rare-outcome cross-fitting, invalid allocation checks, two-sided power under the null, and common nuisance estimates for policy comparisons. Later fixes: gzip header detection before CSV parsing, kernel push directory derived from metadata, smoke manifest SHA verification, raw-float64 duplicate hashing, paired bootstrap differences retained for every reference model, and tie-aware ranking evaluation.

## Tie-handling correction (this rerun)

The first published run sorted scores with a stable row-order argsort, so tied scores inherited source-file order. Evidence: the constant-score baselines scored Qini 1008.7 / AUUC 0.00094 while random scored -69, which is impossible for constant scores. The evaluation was rewritten around distinct-score tie blocks: curves are computed only at block boundaries and connected linearly (scikit-uplift convention), and uplift@k plus policy values interpolate cumulative weighted sums inside a straddling tie block, the expected value under random tie-breaking. The identical logic runs in the Poisson-weighted bootstrap path, and unit weights reproduce the plain path exactly. Tests cover constant scores (Qini ~0, AUUC at the random-line value), invariance to within-tie permutations, a hand-computed tied fixture, and bootstrap/plain equality. Constant policies (treat-none, treat-all) are no longer treated as rankings; their policy values are the control/treated means under the known propensity, and they are excluded from ranking metrics and calibration.

## Verified locally

- 57 tests passed with `uv run pytest` on Python 3.13.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed (strict mode, 22 files).
- The README image was rendered from `results/` by `scripts/render_figure.py` (2x3 grid: Qini curves with the random line, uplift@k, calibration, policy gain vs treat-all, A/B effect, per-decile uplift) and checked for text overlap programmatically (zero overlapping labels across all six panels).

## Verified on Kaggle

- Smoke kernel `experimentation-uplift-modeling-smoke` on git SHA `cc1ff94`: sampled exactly 200,000 rows; download 5.3s, sample-prepare 51.5s, benchmark 61.1s, total 138.7s; `manifest.git_sha` matched and `ordering_ok` was true; all new artifact sections (curves, calibration bins, seed summaries, failure analysis, policy gains) present.
- Full kernel `experimentation-uplift-modeling-benchmark` on git SHA `cc1ff94cf0d666858790f279b68e09372e95e836`: protocol `uplift-benchmark-v1`, `ordering_ok` true, XGBoost on CUDA, total runtime about 110 minutes (validation 1431s, test eval 3319s, bootstrap 1650s). Outputs curated unedited into `results/`.

## Limits

- CLI demo data are synthetic rather than Criteo; individual counterfactuals are not observed; power calculations use normal approximations.
- One frozen split of one dataset version; duplicate source rows retained per the source release.
- Policy-value gains versus treat-all are negative for every ranking policy because the average treatment effect is positive; the metric measures total conversions, not incremental targeting benefit.
- New CI workflows are configuration changes; their hosted execution must be checked after publication.
