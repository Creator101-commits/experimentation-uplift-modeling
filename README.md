![Banner](./image.png)

# Experimentation Uplift Modeling

> A reproducible study of randomized experiments, uplift models, and treatment policies.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/experimentation-uplift-modeling?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/experimentation-uplift-modeling)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/experimentation-uplift-modeling)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/experimentation-uplift-modeling)
![GitHub](https://img.shields.io/github/license/Creator101-commits/experimentation-uplift-modeling)

This repository implements power calculations, randomized A/B analysis, S- and T-learners, gradient-boosted variants, and a cross-fitted doubly robust learner. The frozen benchmark ran on the full public Criteo uplift dataset on Kaggle; curated outputs live in `results/`. The CLI demo commands still use seeded synthetic observations. Conditional effects are estimates, not observed individual counterfactuals.

## Table of Contents

- [Experimentation Uplift Modeling](#experimentation-uplift-modeling)
- [Quickstart / Demo](#quickstart--demo)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation and Methodology](#evaluation-and-methodology)
- [Results](#results)
- [Findings](#findings)
- [Development](#development)
- [Contributing](#contributing)
- [Release History](#release-history)
- [License](#license)
- [Meta](#meta)
- [Provenance](#provenance)

## Quickstart / Demo
[(Back to top)](#table-of-contents)

Run power analysis, evaluate randomized experiments, and benchmark uplift learners:

```sh
uv sync
uv run uplift power --baseline-rate 0.04 --mde 0.005
uv run uplift ab-test
uv run uplift compare
uv run uplift report
```

## Installation
[(Back to top)](#table-of-contents)

Requirements:

- macOS or Linux
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```sh
git clone https://github.com/Creator101-commits/experimentation-uplift-modeling.git
cd experimentation-uplift-modeling
uv sync
```

## Usage
[(Back to top)](#table-of-contents)

The package CLI provides tools for causal inference and uplift modeling:

```sh
# Calculate required sample size and minimum detectable effects
uv run uplift power --baseline-rate 0.04 --mde 0.005 --target-power 0.80

# Analyze randomized trial outcome rates, confidence intervals, and covariate balance
uv run uplift ab-test --samples 5000

# Benchmark S-Learners, T-Learners, and Doubly Robust estimators
uv run uplift compare --samples 5000

# Generate full research report
uv run uplift report -o artifacts/report.md
```

Python library usage:

```python
from uplift import ABTestAnalyzer, PowerCalculator, SLearner, TLearner, DoublyRobustLearner

# Prospective power calculation
power_res = PowerCalculator.calculate_sample_size(baseline_rate=0.04, mde_absolute=0.005)

# Classical A/B test analysis
ab_res = ABTestAnalyzer.analyze(treatment=W, outcome=y)

# Fit S-Learner or Doubly Robust CATE estimator
dr = DoublyRobustLearner(known_propensity=0.85)
dr.fit(X_train, W_train, y_train)
cate_pred = dr.predict_uplift(X_test)
```

The real-data pipeline is also exposed through the CLI:

```sh
# Download, validate, and partition the Criteo uplift dataset
uv run uplift prepare-data --out data/processed

# Run the frozen benchmark (selection on validation, one test read)
uv run uplift benchmark --data data/processed --out artifacts/benchmark
```

## Evaluation and Methodology

[(Back to top)](#table-of-contents)

The benchmark follows a frozen protocol (`uplift-benchmark-v1`, see
`results/analysis_plan.json`):

- Only pre-treatment features `f0` through `f11` are model inputs;
  `exposure` and `visit` are post-treatment fields and are excluded.
- A deterministic 60/20/20 train/validation/test partition is stratified
  jointly by treatment and conversion (seed 42). Imputers, scalers, and
  all selections use training and validation data only; the test
  partition is read once after the analysis plan and configurations are
  frozen (`card_write_epoch < first_test_read_epoch`, verified in
  `results/manifest.json`).
- Policy value uses the known randomized propensity
  (0.8500001490267658), verified against observed assignment rates per
  split within 0.001.
- Primary metrics are the Qini coefficient, normalized AUUC, and
  uplift@10/20/30 targeting gains. Uncertainty is a paired Poisson(1)
  weighted bootstrap (500 resamples, seed 0, percentile 95% intervals).
- Uplift calibration compares predicted-uplift deciles against observed
  treated-minus-control rates; the reported `calibration_mean_gap` is the
  mean absolute gap across the 10 equal-count bins.
- Duplicate source rows (1,259,545) are counted in the data manifest and
  retained, matching the source release.

## Results

[(Back to top)](#table-of-contents)

Frozen Kaggle run `uplift-benchmark-v1` at git SHA
`41bad9b5ca1b0be30793e58062631eb7a646c6f5` (ordering guard passed;
`xgboost_device=cuda`; total runtime about 60 minutes). Curated
artifacts are in `results/`; every number below traces to those files.

Data validation (`results/data-manifest.json`): 13,979,592 rows; 16-column
schema; zero nulls; observed treatment rate 0.8500, conversion rate
0.00292, visit rate 0.0470, exposure rate 0.0306; 1,259,545 duplicate
rows reported and retained; partitions 8,387,755 / 2,795,919 / 2,795,918
with per-split treatment rates within 0.001 of the known propensity.

A/B effect on the full dataset (`results/metrics.json` -> `ab_report`):
control conversion 0.1938% (n=2,096,937), treated 0.3089%
(n=11,882,655); absolute lift 0.115 percentage points, 95% CI
[0.108, 0.122]; relative lift 59.4%; number needed to treat about 868;
the interval excludes zero and the post-hoc power at the observed
effect is 1.0.

Model selection used validation AUUC only: the XGBoost grid selected
`s_learner_xgb_d4_lr0.1` (validation AUUC 0.001039) and
`t_learner_xgb_d4_lr0.05` (0.000959); fixed baselines and learners are
listed in `results/selection/`.

Frozen test metrics (`results/metrics.json` -> `test_metrics`, 95%
bootstrap CIs):

| model | AUUC | Qini | uplift@20 |
| --- | --- | --- | --- |
| S-XGB d4 lr0.1 | 0.00105 [0.00091, 0.00120] | 1142.6 [975.7, 1317.3] | 0.00517 [0.00446, 0.00587] |
| T-logistic | 0.00103 [0.00089, 0.00117] | 1097.1 [931.1, 1260.9] | 0.00488 [0.00417, 0.00561] |
| outcome only | 0.00103 [0.00088, 0.00117] | 1079.5 [910.1, 1243.7] | 0.00490 [0.00418, 0.00561] |
| S-logistic | 0.00103 [0.00088, 0.00117] | 1079.7 [910.4, 1244.0] | 0.00490 [0.00419, 0.00562] |
| DR-logistic | 0.00103 [0.00088, 0.00117] | 1077.1 [912.7, 1241.8] | 0.00483 [0.00411, 0.00554] |
| DR-XGBoost | 0.00103 [0.00088, 0.00117] | 1077.1 [913.6, 1243.7] | 0.00480 [0.00408, 0.00552] |
| treat none / all | 0.00094 [0.00085, 0.00102] | 1008.7 [934.6, 1079.5] | 0.00000 |
| T-XGB d4 lr0.05 | 0.00093 [0.00081, 0.00107] | 858.7 [689.8, 1022.8] | 0.00451 [0.00385, 0.00517] |
| random | 0.00055 [0.00046, 0.00064] | -69.3 [-166.3, 37.7] | 0.00108 [0.00077, 0.00138] |

## Findings

[(Back to top)](#table-of-contents)

- The randomized treatment has a small but statistically clear average
  effect: 0.115 percentage points absolute (0.194% to 0.309% conversion),
  59.4% relative, NNT about 868.
- The gradient-boosted S-learner is nominally best on every ranking
  metric (AUUC 0.00105, Qini 1142.6, uplift@20 0.00517), but its paired
  difference against the logistic T-learner includes zero on both AUUC
  (+0.000019, CI [-0.000006, +0.000047]) and Qini (+44.2, CI [-16.0,
  +110.0]). No learner is significantly better than the logistic
  T-learner.
- The outcome-only baseline, which ranks by predicted conversion
  probability and ignores treatment, is statistically
  indistinguishable from the logistic T-learner on AUUC (-0.000007,
  CI [-0.000017, +0.000003]) and Qini (-17.4, CI [-41.2, +5.8]); the
  same holds for the S-logistic, DR-logistic, DR-XGBoost, and S-XGBoost
  learners. Only the T-learner XGBoost sits significantly below the
  reference. On this dataset, modeling the treatment effect adds no
  statistically detectable ranking gain over outcome propensity alone.
- The T-learner XGBoost variant is significantly worse than the
  logistic T-learner (AUUC -0.00010, CI [-0.00015, -0.00005]; Qini
  -244.7, CI [-369.9, -121.5]) despite being the best T-learner XGBoost
  grid cell on validation.
- Random targeting is significantly worse than the learners (AUUC
  -0.00049, CI [-0.00057, -0.00041]; Qini -1169.0, CI [-1375.0,
  -990.0]) and its own Qini interval includes zero.
- Learned rankings beat the degenerate constant policies: the logistic
  T-learner exceeds treat-all on AUUC by +0.00010 (CI [0.00003,
  0.00017]).
- Calibration is good for the learners: mean absolute bin gaps are
  0.00006 to 0.0006. Random ranking has a gap of 0.499 and treat-none
  0.00115, on the order of the average lift it fails to model.
- Absolute targeting gains are small: even the best model's top-20%
  cohort shows a treated-minus-control conversion difference of about
  0.52 percentage points, and learner intervals overlap heavily.

## Development
[(Back to top)](#table-of-contents)

Run the testing and linting suite:

```sh
uv run pytest
uv run ruff check .
uv run mypy src
```

## Contributing
[(Back to top)](#table-of-contents)

Contributions are welcome. To propose a change:

1. Fork it (<https://github.com/Creator101-commits/experimentation-uplift-modeling/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Open a new Pull Request

Please make sure tests pass and the code is formatted before opening a PR.

## Release History
[(Back to top)](#table-of-contents)

No releases have been published yet.

## License
[(Back to top)](#table-of-contents)

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Meta
[(Back to top)](#table-of-contents)

Sreeharsha Kannegundla

Project link: [https://github.com/Creator101-commits/experimentation-uplift-modeling](https://github.com/Creator101-commits/experimentation-uplift-modeling)

## Evaluation limits

[(Back to top)](#table-of-contents)

- Predicted uplift values are estimates of conditional average effects,
  not observed individual counterfactuals.
- Results describe one frozen 60/20/20 split of one dataset version;
  they do not establish effects on other populations or time periods.
- The published `results/metrics.json` retains paired bootstrap
  differences only against the logistic T-learner reference (and the
  T-learner against treat-all) because of a bookkeeping defect fixed in
  commit `39499e6` after the run; per-model confidence intervals are
  complete.
- The benchmark did not persist per-model Qini curves or calibration
  bins, so `image.png` shows the scalar curve summaries (Qini
  coefficient, AUUC) with confidence intervals rather than full curves.
- 1,259,545 duplicate source rows were counted and retained, matching
  the source release; they may inflate effective sample size slightly.
- The CLI demo commands (`power`, `ab-test`, `compare`, `report`) still
  run on seeded synthetic data, not Criteo.
- The A/B p-value underflowed to 0.0 in float64; the confidence interval
  is the reported evidence.

## Provenance

[(Back to top)](#table-of-contents)

- Dataset: Criteo uplift dataset via the Hugging Face mirror
  `criteo/criteo-uplift`, pinned revision
  `2424920019e49d52d72c13ac1143ec5d53af276b`, file
  `criteo-research-uplift-v2.1.csv.gz` (311,422,618 bytes, SHA-256
  `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`).
- License: CC BY-NC-SA 4.0 (non-commercial). Raw data are not
  redistributed; `results/` contains metrics, manifests, and selection
  tables only.
- Original dataset description:
  [Criteo uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
- Benchmark run: private Kaggle kernel at git SHA
  `41bad9b5ca1b0be30793e58062631eb7a646c6f5`, protocol
  `uplift-benchmark-v1`; `results/manifest.json` records the epoch
  ordering guard, timings, and partition sizes.

## Reproducing the figure

`image.png` contains only Matplotlib graphs rendered from `results/`.
Run `uv run --with matplotlib python scripts/render_figure.py` from the
repository root; the script rewrites `scripts/figure-data.json` with the
plotted values and saves `image.png`. Matplotlib is intentionally not a
project dependency, so it is provided ephemerally. No GPU performance is
inferred from CPU demonstrations.
