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
- Ranking metrics are tie-block aware: cumulative curves are evaluated
  only at distinct-score boundaries and connected linearly (the
  scikit-uplift convention), and budget cutoffs that fall inside a tie
  block interpolate the cumulative weighted sums, matching expected
  random tie-breaking. The same logic runs in the weighted bootstrap.
- `treat_none` and `treat_all` are constant policies, not rankings:
  their policy values are the control and treated means estimated with
  the known propensity (IPW and DR), and they are excluded from AUUC,
  Qini, uplift@k, and calibration tables (reported as N/A, not zero).
  `outcome_only` is a valid ranking baseline ranked by predicted
  conversion probability but has no uplift estimate, so it is excluded
  from uplift calibration. `random` remains the ranking reference.
- Policy gain is the policy value at each budget minus the `treat_all`
  constant value and minus the `random` policy value, for both IPW and
  DR estimators.
- Primary metrics are the Qini coefficient (unnormalized excess area
  above the random-targeting line), the AUUC (trapezoidal area under
  the cumulative uplift-gain curve), and uplift@10/20/30 targeting
  gains. Uncertainty is a paired Poisson(1) weighted bootstrap (500
  resamples, seed 0, percentile 95% intervals) shared across models.
- Stochastic learners (random, DR-logistic, DR-XGBoost, and the
  selected XGBoost S/T learners) run at seeds 0, 1, and 2; per-seed
  rows are in `test_metrics.csv` and `seed_summary` reports the mean
  and across-seed range. Each row also records model size in bytes,
  runtime, and outcome-model log loss and Brier score for `mu0`/`mu1`
  where defined.
- Uplift calibration compares predicted-uplift deciles against observed
  treated-minus-control rates in 10 equal-count bins; the reported
  `calibration_mean_gap` is the mean absolute gap across bins.
- `metrics.json` persists the full evidence: per-model per-seed Qini
  curves (101 points) plus the random line, calibration bins, uplift
  deciles, quantile slices of the top-3 T-XGBoost gain features, policy
  gains, and paired bootstrap differences against the outcome-only and
  logistic T-learner references.
- Duplicate source rows (1,259,545) are counted in the data manifest and
  retained, matching the source release.

## Results

[(Back to top)](#table-of-contents)

Frozen Kaggle run `uplift-benchmark-v1` at git SHA
`cc1ff94cf0d666858790f279b68e09372e95e836` (ordering guard passed;
`xgboost_device=cuda`; total runtime about 110 minutes). This rerun
replaced the first publication after a tie-handling defect was found:
tied scores previously inherited source-file order, which let constant
baselines score as if they ranked the file. Curated artifacts are in
`results/`; every number below traces to those files.

![Uplift benchmark results: Qini and cumulative gain curves, uplift at
budget, calibration, policy gains, the A/B effect and decile uplift](image.png)

### What the figure shows

- (a) Qini curves: every learner's curve clears the dashed random
  line through the mid-range of the targeting axis before the curves
  converge near 100%; thin lines are the three stochastic seeds and
  thick lines their mean. The XGBoost S-learner ends highest, test
  AUUC 0.00105 versus 0.00055 for random.
- (b) Cumulative uplift gain: targeting the whole frozen test
  population accumulates about 0.115 percentage points of conversion
  gain, matching the A/B absolute lift; the random model's curve stays
  near zero.
- (c) Uplift at budget: at a 10% targeting budget the XGBoost
  S-learner shows a treated-minus-control gap of 0.843 pp
  [0.698, 0.991] versus 0.099 pp for random, and the ordering holds at
  the 20% and 30% budgets.
- (d) Calibration: for the top three learners by AUUC, mean predicted
  uplift tracks observed uplift along the diagonal at the top decile,
  while the logistic learners underestimate the top bin and the middle
  bins are noisy, including negative observed uplift.
- (e) DR policy gain vs random: every budgeted policy beats the random
  policy by roughly 0.08-0.10 pp at all three budgets, yet all stay at
  or below treat-all because the average treatment effect is positive.
- (f) A/B effect on the full dataset: treated conversion is 0.309%
  versus 0.194% for control, an absolute lift of 0.115 pp
  [0.108, 0.122], relative lift 59.4%, p = 0.000 and NNT about 868.
- (g) Observed uplift by predicted decile for the best learner: the
  top predicted decile shows 0.843 pp observed uplift versus 0.006 pp
  in the lowest decile, with a noisy middle.

The frozen test shows a small but statistically certain treatment
effect and learners that isolate a high-uplift top decile, yet no
learner is significantly better than the simpler references on AUUC
or Qini.

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
`s_learner_xgb_d4_lr0.05` (validation AUUC 0.001048) and
`t_learner_xgb_d4_lr0.05` (0.000951); fixed baselines and learners are
listed in `results/selection/`.

Frozen test metrics (`results/metrics.json` -> `bootstrap.models`,
point estimates with 95% paired Poisson bootstrap CIs; stochastic
learners aggregate seeds 0-2):

| model | AUUC | Qini | uplift@20 |
| --- | --- | --- | --- |
| S-XGB d4 lr0.05 | 0.00105 [0.00090, 0.00119] | 1124.3 [955.3, 1294.6] | 0.00506 [0.00435, 0.00581] |
| T-logistic | 0.00103 [0.00089, 0.00117] | 1097.7 [931.9, 1261.7] | 0.00488 [0.00417, 0.00561] |
| S-logistic | 0.00103 [0.00088, 0.00117] | 1080.7 [910.9, 1245.0] | 0.00490 [0.00419, 0.00562] |
| outcome-only | 0.00103 [0.00088, 0.00117] | 1080.3 [910.4, 1244.6] | 0.00490 [0.00418, 0.00561] |
| DR-XGBoost | 0.00103 [0.00088, 0.00117] | 1079.9 [916.0, 1246.4] | 0.00481 [0.00408, 0.00553] |
| DR-logistic | 0.00103 [0.00088, 0.00117] | 1078.3 [913.9, 1243.1] | 0.00482 [0.00410, 0.00554] |
| T-XGB d4 lr0.05 | 0.00091 [0.00077, 0.00104] | 802.8 [614.2, 972.1] | 0.00443 [0.00372, 0.00515] |
| random | 0.00055 [0.00046, 0.00064] | -69.3 [-165.7, 37.5] | 0.00108 [0.00077, 0.00138] |
| treat-none / treat-all | n/a | n/a | n/a |

Constant-policy values (`bootstrap.models.policy_treat_*`, IPW with the
known propensity; DR agrees to ~1e-9): treat-none 0.1936%
[0.1813, 0.2072] and treat-all 0.3089% [0.3022, 0.3159], matching the
raw arm rates.

## Findings

[(Back to top)](#table-of-contents)

- The randomized treatment has a small but statistically clear average
  effect: 0.115 percentage points absolute (0.194% to 0.309% conversion),
  59.4% relative, NNT about 868.
- The gradient-boosted S-learner is nominally best on every ranking
  metric (AUUC 0.00105, Qini 1124.3, uplift@20 0.00506), but no paired
  difference against the references excludes zero: versus the logistic
  T-learner AUUC +0.000011 (CI [-0.000010, +0.000032]) and Qini +24.6
  (CI [-25.8, +74.0]); versus outcome-only AUUC +0.000018 (CI
  [-0.000002, +0.000037]) and Qini +41.9 (CI [-4.7, +88.2]). No learner
  is significantly better than the logistic T-learner.
- The outcome-only baseline, which ranks by predicted conversion
  probability and ignores treatment, is statistically
  indistinguishable from the logistic T-learner on AUUC (-0.000007,
  CI [-0.000017, +0.000003]) and Qini (-17.2, CI [-41.5, +5.9]); the
  same holds for the S-logistic, DR-logistic, DR-XGBoost, and S-XGBoost
  learners. Only the T-learner XGBoost sits significantly below the
  references. On this dataset, modeling the treatment effect adds no
  statistically detectable ranking gain over outcome propensity alone.
- The T-learner XGBoost variant is significantly worse than both
  references (versus outcome-only AUUC -0.000118, CI [-0.000180,
  -0.000057], Qini -282.7, CI [-429.4, -135.4]; versus the logistic
  T-learner AUUC -0.000125, CI [-0.000187, -0.000063]) despite being
  the best T-learner XGBoost grid cell on validation.
- Random targeting is significantly worse than every learner and
  baseline (versus the logistic T-learner AUUC -0.000488, CI
  [-0.000574, -0.000412]; Qini -1169.6, CI [-1376.3, -990.7]) and its
  own Qini interval includes zero, as a random ranking should.
- Constant policies are reported as policy values, not rankings:
  treat-none is the control mean 0.1936% and treat-all the treated mean
  0.3089% under IPW/DR with the known propensity. Because the average
  treatment effect is positive, treat-all maximizes total conversions:
  every ranking policy's DR policy gain versus treat-all is negative at
  all three budgets with intervals below zero (least-bad: DR-XGB
  -0.0131 pp at 10%, S-XGB -0.0079 pp at 20% and -0.0063 pp at 30%).
  Targeting concentrates the effect but cannot beat treating everyone
  on total policy value; versus random, every learned policy gains
  +0.09 to +0.10 pp at the 10% budget with intervals excluding zero.
- Calibration is good for the learners: mean absolute bin gaps run from
  0.00006 (S-XGB, the best) to 0.0006 (T-XGB 0.00056, DR-XGBoost
  0.00059); the logistic learners sit in between. `outcome_only` has no
  uplift estimate and is excluded from calibration by design.
- Absolute targeting gains are small: even the best model's top-20%
  cohort shows a treated-minus-control conversion difference of about
  0.51 percentage points, and learner intervals overlap heavily.
- Stochastic learners are stable across seeds 0-2: AUUC ranges span at
  most ~3e-5 (e.g., S-XGB 0.001046-0.001056), far below the bootstrap
  interval widths, so seed noise does not drive the comparisons.

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
- This rerun replaced an earlier publication that mis-ordered tied
  scores by source row; all ranking metrics are now tie-block aware and
  invariant to within-tie permutations (see REVIEW.md).
- Policy-value gains versus treat-all are negative for every ranking
  policy because the average treatment effect is positive; they measure
  total conversions, not incremental targeting benefit.
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
  `cc1ff94cf0d666858790f279b68e09372e95e836`, protocol
  `uplift-benchmark-v1`; `results/manifest.json` records the epoch
  ordering guard, timings, and partition sizes.

## Reproducing the figure

`image.png` contains only Matplotlib graphs rendered from `results/`.
Run `uv run --with matplotlib python scripts/render_figure.py` from the
repository root to regenerate it. Matplotlib is intentionally not a
project dependency, so it is provided ephemerally. No GPU performance is
inferred from CPU demonstrations.
