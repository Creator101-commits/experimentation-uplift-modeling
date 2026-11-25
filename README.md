# Experimentation Uplift Modeling

> Marketing uplift modeling

Does a marketing treatment actually work, and can a model find who
benefits most? This project benchmarks uplift learners on the full
Criteo randomized trial: S- and T-learners, gradient-boosted variants
and a doubly robust estimator, under a frozen protocol where the test
set is read once. The honest headline: the treatment effect is real but
tiny, and no learner beats the simpler baselines.

![Uplift benchmark results: Qini and cumulative gain curves, uplift at budget, calibration, policy gains, the A/B effect and decile uplift](image.png)

The figure follows the evidence: (a) Qini curves per learner against
the random line, (b) cumulative conversion gain across the population,
(c) treated-minus-control uplift at each targeting budget, (d)
predicted versus observed uplift by decile, (e) policy gain versus
random, (f) the raw A/B effect, and (g) observed uplift by predicted
decile for the best learner.

## Results

| model | AUUC [95% CI] | Qini [95% CI] |
| --- | --- | --- |
| S-XGB d4 lr0.05 | 0.00105 [0.00090, 0.00119] | 1124.3 [955.3, 1294.6] |
| T-logistic | 0.00103 [0.00089, 0.00117] | 1097.7 [931.9, 1261.7] |
| S-logistic | 0.00103 [0.00088, 0.00117] | 1080.7 [910.9, 1245.0] |
| outcome-only | 0.00103 [0.00088, 0.00117] | 1080.3 [910.4, 1244.6] |
| DR-XGBoost | 0.00103 [0.00088, 0.00117] | 1079.9 [916.0, 1246.4] |
| DR-logistic | 0.00103 [0.00088, 0.00117] | 1078.3 [913.9, 1243.1] |
| T-XGB d4 lr0.05 | 0.00091 [0.00077, 0.00104] | 802.8 [614.2, 972.1] |
| random | 0.00055 [0.00046, 0.00064] | -69.3 [-165.7, 37.5] |
| treat-none / treat-all | n/a | n/a |

- **The treatment effect is small but certain.** Treated conversion is
  0.309% versus 0.194% for control: absolute lift 0.115 pp [0.108,
  0.122], relative lift 59.4%, NNT about 868.
- **The best learner is only nominally best.** S-XGB tops AUUC and Qini,
  but no paired difference against the references excludes zero (vs
  T-logistic AUUC +0.000011 [-0.000010, +0.000032]).
- **Modeling the treatment adds no detectable gain.** outcome-only,
  which ignores treatment entirely, ties the logistic T-learner on AUUC
  (-0.000007 [-0.000017, +0.000003]) and Qini.
- **Two learners are significantly worse.** T-XGB sits below the
  references (AUUC -0.000118 [-0.000180, -0.000057] vs outcome-only)
  and random below every learner (AUUC -0.000488 [-0.000574, -0.000412]).
- **Targeting concentrates but cannot beat treat-all.** Every ranking
  policy's gain versus treat-all is negative at all budgets because the
  average effect is positive; versus random, every learned policy gains
  +0.09 to +0.10 pp at the 10% budget.
- **Calibration and seeds are stable.** Mean absolute calibration gaps
  run 0.00006 to 0.0006, and across seeds 0-2 AUUC ranges span at most
  ~3e-5, far below the bootstrap widths.

## How it works

```text
download + sha256 verify -> 60/20/20 split stratified by treatment+conversion
-> impute/scale on train+val only -> validation AUUC selects configs
-> freeze analysis plan -> read test once -> Qini/AUUC/uplift@k + bootstrap
-> IPW/DR policy values at budgets -> calibration deciles
```

Only pre-treatment features `f0`-`f11` are model inputs; `exposure` and
`visit` are post-treatment and excluded. Policy value uses the known
randomized propensity (0.8500001490267658). Predicted uplift is an
estimated average effect, never an observed individual counterfactual.
This rerun replaced an earlier publication after a tie-handling defect
(tied scores had inherited source-file order). Criteo data is
CC BY-NC-SA, non-commercial.

## Run it

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src

# real-data pipeline
uv run uplift prepare-data --out data/processed
uv run uplift benchmark --data data/processed --out artifacts/benchmark
scripts/kaggle_run.sh   # Kaggle reproduction of the frozen run
```
