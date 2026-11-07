# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Fixed cumulative-curve endpoints, causal feature validation, rare-outcome cross-fitting, invalid allocation checks, two-sided power under the null, and common nuisance estimates for policy comparisons.

## Verified locally

- 37 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

CLI data are synthetic rather than Criteo. Individual counterfactuals are not observed. Power calculations use normal approximations.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.
