# Uplift Benchmark Report

- git_sha: `41bad9b5ca1b0be30793e58062631eb7a646c6f5`
- protocol: `uplift-benchmark-v1`
- ordering_ok: `True`
- known_propensity: `0.8500`

## A/B report (full dataset)

- control rate: 0.00194 (n=2096937)
- treated rate: 0.00309 (n=11882655)
- absolute lift: 0.00115 CI [0.00108, 0.00122]
- relative lift: 59.45%
- p-value: 0.000e+00

## Test metrics (frozen, single read)

| model | AUUC | Qini | uplift@20 |
| --- | --- | --- | --- |
| treat_none | 0.0009 | 1008.72 | 0.0000 |
| treat_all | 0.0009 | 1008.72 | 0.0000 |
| random | 0.0005 | -69.30 | 0.0011 |
| outcome_only | 0.0010 | 1079.45 | 0.0049 |
| s_learner_logistic | 0.0010 | 1079.67 | 0.0049 |
| t_learner_logistic | 0.0010 | 1097.14 | 0.0049 |
| dr_logistic | 0.0010 | 1077.11 | 0.0048 |
| s_learner_xgb_d4_lr0.1 | 0.0011 | 1142.61 | 0.0052 |
| t_learner_xgb_d4_lr0.05 | 0.0009 | 858.66 | 0.0045 |
| dr_xgboost | 0.0010 | 1077.12 | 0.0048 |
