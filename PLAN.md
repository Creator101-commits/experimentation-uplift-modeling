# Build Plan: Experimentation and Uplift Modeling

## Goal

Build a reproducible Python study that moves from experiment design and average treatment-effect estimation to individualized treatment targeting. The project will compare multiple uplift estimators under one frozen protocol and determine whether their targeting policies create measurable incremental value beyond treating everyone, treating nobody, or using an outcome-only model.

## Scope

### Included in v1

- Acquire and validate the public Criteo uplift-modeling dataset.
- Perform prospective power and minimum-detectable-effect calculations for a binary outcome.
- Produce a conventional treatment-versus-control analysis with absolute and relative effects, confidence intervals, hypothesis tests, and practical interpretation.
- Audit treatment balance and outcome frequency without using post-treatment fields as predictors.
- Create deterministic train, validation, and frozen test partitions stratified by treatment and outcome.
- Implement policy baselines: treat none, treat all, random targeting, and outcome-propensity targeting.
- Implement S-learner and T-learner estimators using logistic regression and gradient-boosted trees.
- Implement one cross-fitted doubly robust learner or policy-value estimator.
- Evaluate rankings with uplift curves, Qini curves, AUUC, uplift@K, and gain by treatment-budget fraction.
- Evaluate treatment policies with inverse-propensity-weighted and doubly robust value estimates.
- Assess uplift calibration by comparing predicted-uplift bins with observed randomized differences.
- Report uncertainty with bootstrap intervals and repeated model seeds.
- Track data hashes, assumptions, configurations, models, metrics, and figures in MLflow.
- Produce a command-line workflow and a complete research report.

### Explicitly outside v1

- Claims that individual counterfactual outcomes are observed.
- Deployment of real marketing offers or contact with users.
- Observational causal inference with unknown treatment assignment.
- Instrumental variables, regression discontinuity, synthetic controls, mediation analysis, and time-varying treatment.
- Deep neural uplift models.
- Causal claims based on post-treatment variables such as exposure or visits.
- Automated hyperparameter searches without a fixed compute budget.
- Financial impact claims unsupported by declared treatment and conversion values.

## Research questions

1. Does the randomized treatment produce a statistically and practically meaningful average conversion effect?
2. Can uplift models rank treatment-responsive users better than random targeting and outcome-only targeting?
3. Which estimator provides the best policy value after accounting for uncertainty and treatment budget?
4. How stable are model rankings across seeds, bootstrap samples, and targeting fractions?
5. Do complex learners improve targeting enough to justify their runtime and calibration cost?

## Stack

- **Language:** Python 3.12
- **Packaging:** `uv`, `pyproject.toml`, and a lockfile
- **Data:** pandas, NumPy, PyArrow, and Parquet
- **Statistics:** SciPy and statsmodels
- **Models:** scikit-learn, XGBoost, and scikit-uplift
- **Tracking:** MLflow
- **Visualization:** Matplotlib and seaborn
- **CLI:** Typer
- **Quality:** pytest, pytest-cov, Ruff, mypy, and GitHub Actions

## Architecture

```text
Criteo archive
      |
      v
download + hash + schema validation
      |
      v
pre-treatment features + treatment + outcome
      |
      +--------------------------+
      |                          |
      v                          v
power/A-B analysis       train/validation/test split
                                 |
            +--------------------+---------------------+
            |                    |                     |
        baselines           S/T learners       doubly robust learner
            |                    |                     |
            +--------------------+---------------------+
                                 |
                                 v
                 uplift ranking + policy evaluation
                                 |
                                 v
             uncertainty + calibration + slice analysis
                                 |
                                 v
                    MLflow + figures + report
```

### Components

#### `src/uplift/data/`

Downloads the source, verifies integrity, assigns stable row identifiers, validates treatment and outcome values, removes forbidden post-treatment predictors, creates partitions, and writes Parquet tables plus a manifest.

#### `src/uplift/experiments/`

Runs power calculations, treatment-balance checks, difference-in-proportions estimation, confidence intervals, and classical hypothesis tests. It stores the planned estimand and target effect before the model-comparison stage.

#### `src/uplift/models/`

Defines a shared `fit`, `predict_outcomes`, and `predict_uplift` interface. Implementations include simple policies, S-learners, T-learners, and a cross-fitted doubly robust approach.

#### `src/uplift/evaluation/`

Computes uplift curves, Qini coefficients, AUUC, uplift by targeting fraction, policy values, bootstrap intervals, and calibration tables. Evaluation functions operate on immutable arrays containing treatment, observed outcome, known propensity, and predicted uplift.

#### `src/uplift/reporting/`

Builds power curves, A/B-test summaries, treatment-balance plots, uplift and Qini curves, policy-value charts, calibration plots, runtime tables, and model comparison tables from saved evaluation artifacts.

#### `src/uplift/cli.py`

Exposes thin commands for download, preparation, experiment analysis, training, comparison, and report generation.

## Data

### Primary dataset

Use the public **Criteo Uplift Prediction Dataset**, which contains randomized treatment assignment, anonymized pre-treatment features, and several response fields. The primary v1 estimand will use `conversion` as the outcome and `treatment` as the intervention indicator.

### Expected source schema

- `f0` through `f11`: anonymized pre-treatment features
- `treatment`: randomized assignment indicator
- `exposure`: whether the advertisement was observed
- `visit`: visit response
- `conversion`: conversion response

Only `f0` through `f11` may be model features in the primary experiment. `exposure` and `visit` are potential post-treatment or intermediate variables and will not be used to predict conversion uplift.

### Acquisition and validation

- Download from the documented Criteo source or an integrity-verifiable mirror permitted by the dataset terms.
- Record source URL, acquisition date, license or use terms, file size, and SHA-256 hash.
- Validate exact column names, binary treatment/outcome domains, null frequencies, row count, and duplicate rows.
- Preserve the source file unchanged under `data/raw/` and keep it outside Git.
- Convert validated data to typed Parquet for downstream work.

### Processed tables

- `observations.parquet`: stable row ID, pre-treatment features, treatment, and outcomes
- `train.parquet`, `validation.parquet`, and `test.parquet`: deterministic partitions
- `manifest.json`: hashes, schema, row counts, split seed, treatment propensity, exclusions, and transformation version
- `analysis_plan.yaml`: primary outcome, estimands, target fractions, metric definitions, and final comparison rules

### Split protocol

- Create a deterministic 60/20/20 train/validation/test partition.
- Stratify jointly by treatment and conversion because conversion is rare.
- Fit imputers, scalers, encoders, and all models on training data only.
- Use validation data for model selection, targeting-threshold selection, and fusion of any ensemble.
- Open the test partition only after the analysis plan and final configurations are frozen.
- Use the known randomized treatment probability from the prepared dataset in policy estimators; verify it against observed assignment rates.

## Estimands and evaluation

### Classical experiment estimands

- Control conversion probability
- Treatment conversion probability
- Absolute risk difference
- Relative risk and relative lift
- Number needed to treat when meaningful
- 95% confidence intervals for primary effects

The phrase “2% lift” must always specify whether it means two percentage points of absolute lift or two percent relative lift.

### Uplift outputs

For each observation, models may estimate `mu1(x)`, the outcome probability under treatment, and `mu0(x)`, the outcome probability under control. Predicted uplift is `tau(x) = mu1(x) - mu0(x)`. This is an estimated conditional average effect, not an observed individual effect.

### Primary model metrics

- AUUC under one documented implementation
- Qini coefficient
- Uplift@10%, @20%, and @30% targeting fractions
- Doubly robust estimated policy value at the same treatment budgets

### Secondary metrics

- Outcome-model log loss and Brier score for `mu0` and `mu1`
- Uplift calibration error across predeclared quantile bins
- Policy gain over treat-all and random policies
- Runtime, peak memory when practical, and serialized model size

### Uncertainty

- Bootstrap evaluation rows while preserving each row's treatment and outcome pair.
- Report percentile or bias-corrected 95% intervals using one predeclared method.
- Use paired bootstrap comparisons because models score the same test observations.
- Report variability across at least three seeds for stochastic learners.

## Milestones

### 1. Reproducible skeleton

Create the package layout, `pyproject.toml`, locked environment, Typer CLI, test configuration, linting, typing, and GitHub Actions. Ship a help command, a passing smoke test, and a clean-install verification workflow.

### 2. Power-analysis module

Implement sample-size, power, and minimum-detectable-effect calculations for two proportions. Test calculations against trusted examples. Ship plots showing required sample size across baseline rates, absolute effects, relative effects, power, and allocation ratios.

### 3. Data acquisition and causal schema audit

Implement dataset download, hash validation, schema checks, stable row IDs, forbidden-feature checks, typed Parquet output, deterministic partitions, and the manifest. Ship a data card documenting treatment, outcomes, post-treatment variables, limitations, and intended estimand.

### 4. Classical A/B-test analysis

Implement treatment balance, conversion summaries, risk difference, relative risk, confidence intervals, hypothesis tests, and practical-effect reporting. Ship a standalone report section that remains valid without any uplift model.

### 5. Policy and learner baselines

Implement treat-none, treat-all, random, outcome-only, S-learner, and T-learner baselines using logistic regression. Add tiny synthetic tests where treatment effect direction is known. Ship the first shared uplift and policy-value comparison.

### 6. Nonlinear and doubly robust models

Add XGBoost S- and T-learners plus a cross-fitted doubly robust estimator. Tune only a bounded parameter set on validation data. Ship model artifacts, MLflow runs, and tests proving fold isolation.

### 7. Evaluation and calibration

Implement verified Qini, AUUC, targeting-fraction, policy-value, bootstrap, and calibration routines. Cross-check critical metrics with a reference library on fixed fixtures. Ship confidence-banded curves and calibration tables.

### 8. Frozen comparison and failure analysis

Freeze the analysis plan, select configurations using validation results, evaluate once on test data, then inspect failures by feature and predicted-uplift ranges without retuning. Ship paired model-difference intervals, runtime comparisons, and documented negative or neutral findings.

### 9. Final research artifact

Write the final report with abstract, assumptions, power design, data, estimands, methods, results, uncertainty, limitations, policy interpretation, and reproducibility instructions. Verify every README command from a clean environment and tag a v1 release after CI passes.

## Open questions

1. **Primary target effect:** Define the smallest practically useful effect in both absolute percentage points and relative lift before examining final outcomes.
2. **Outcome choice:** Keep conversion primary. Decide whether visit is a secondary endpoint with multiplicity-aware interpretation or excluded entirely.
3. **Treatment propensity:** Confirm the randomization probability and whether it is constant across all rows before policy evaluation.
4. **Dataset scale:** Decide whether laptop development uses a fixed representative subset while final results use the full data. Any subset must be created without selecting on treatment effect.
5. **Doubly robust implementation:** Choose between a small transparent internal implementation and a vetted causal-inference library after compatibility testing.
6. **Calibration bins:** Fix the number of bins and minimum observations per treatment arm before test evaluation.
7. **Targeting cost:** Decide whether policy value uses conversion alone or a declared conversion value and treatment cost. Monetary values must be labeled assumptions.
8. **Missing values:** Determine whether XGBoost native missing-value handling or train-fitted imputation is the primary protocol and keep it consistent across learners where possible.
9. **Multiplicity:** Predeclare one primary uplift metric and one primary treatment fraction so secondary comparisons are not mistaken for confirmatory evidence.
