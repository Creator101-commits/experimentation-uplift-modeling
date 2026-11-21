"""Tests for uplift models and meta-learners."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.models.baselines import (
    OutcomeOnlyLearner,
    RandomPolicy,
    TreatAllPolicy,
    TreatNonePolicy,
)
from uplift.models.doubly_robust import DoublyRobustLearner
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner


def test_unfitted_model_raises_runtime_error() -> None:
    s_learner = SLearner()
    X = np.array([[1.0, 2.0]])
    with pytest.raises(RuntimeError, match="is not fitted"):
        s_learner.predict_uplift(X)


def test_treat_none_and_treat_all() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    W = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 1])

    none_policy = TreatNonePolicy().fit(X, W, y)
    all_policy = TreatAllPolicy().fit(X, W, y)

    tau_none = none_policy.predict_uplift(X)
    tau_all = all_policy.predict_uplift(X)

    assert np.allclose(tau_none, 0.0)
    assert np.allclose(tau_all, 1.0)


def test_s_learner_logistic_and_boosting() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    W = rng.binomial(n=1, p=0.5, size=200)
    y = rng.binomial(n=1, p=0.1, size=200)

    for est_type in ["logistic", "gradient_boosting"]:
        s_model = SLearner(estimator_type=est_type, random_state=42)  # type: ignore[arg-type]
        s_model.fit(X, W, y)

        mu0, mu1 = s_model.predict_outcomes(X[:10])
        tau = s_model.predict_uplift(X[:10])

        assert len(mu0) == 10
        assert len(mu1) == 10
        assert np.allclose(tau, mu1 - mu0)


def test_t_learner() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    W = rng.binomial(n=1, p=0.5, size=200)
    y = rng.binomial(n=1, p=0.1, size=200)

    t_model = TLearner(estimator_type="logistic", random_state=42)
    t_model.fit(X, W, y)

    mu0, mu1 = t_model.predict_outcomes(X[:10])
    tau = t_model.predict_uplift(X[:10])

    assert len(mu0) == 10
    assert len(mu1) == 10
    assert np.allclose(tau, mu1 - mu0)


def test_doubly_robust_learner() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 4))
    W = rng.binomial(n=1, p=0.7, size=300)
    y = rng.binomial(n=1, p=0.1, size=300)

    dr_model = DoublyRobustLearner(n_splits=2, known_propensity=0.7, random_state=42)
    dr_model.fit(X, W, y)

    mu0, mu1 = dr_model.predict_outcomes(X[:15])
    tau = dr_model.predict_uplift(X[:15])

    assert len(tau) == 15
    assert not np.isnan(tau).any()


def test_xgboost_learners_rank_positive_effect() -> None:
    """XGB S/T/DR learners recover a known positive effect better than random."""
    from uplift.evaluation.uplift_metrics import auuc_score

    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4))
    # Positive effect concentrated where f0 > 0.
    tau_true = 0.35 * (X[:, 0] > 0.0)
    mu0 = np.clip(0.05 + 0.02 * X[:, 1], 0.0, 1.0)
    mu1 = np.clip(mu0 + tau_true, 0.0, 1.0)
    W = rng.binomial(1, 0.8, size=n)
    y = rng.binomial(1, np.where(W == 1, mu1, mu0))

    n_tr = int(0.7 * n)
    X_tr, X_te = X[:n_tr], X[n_tr:]
    W_tr, W_te = W[:n_tr], W[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    fast = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.15}
    learners = {
        "s_xgb": SLearner(estimator_type="xgboost", random_state=0, xgb_params=fast),
        "t_xgb": TLearner(estimator_type="xgboost", random_state=0, xgb_params=fast),
        "dr_xgb": DoublyRobustLearner(
            n_splits=2, known_propensity=0.8, random_state=0,
            outcome_estimator="xgboost", xgb_params=fast,
        ),
    }
    rand = RandomPolicy(random_state=0).fit(X_tr, W_tr, y_tr)
    tau_rand = rand.predict_uplift(X_te)
    auuc_rand = auuc_score(y_te, W_te, tau_rand)

    for name, learner in learners.items():
        learner.fit(X_tr, W_tr, y_tr)
        tau = learner.predict_uplift(X_te)
        assert not np.isnan(tau).any(), name
        auuc = auuc_score(y_te, W_te, tau)
        assert auuc > auuc_rand, f"{name} AUUC {auuc} not above random {auuc_rand}"
        # The model should assign higher uplift to the f0>0 (persuadable) half.
        assert float(np.mean(tau[X_te[:, 0] > 0])) > float(np.mean(tau[X_te[:, 0] <= 0]))


def test_random_and_outcome_only() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    W = rng.binomial(n=1, p=0.5, size=100)
    y = rng.binomial(n=1, p=0.2, size=100)

    rand_policy = RandomPolicy().fit(X, W, y)
    out_model = OutcomeOnlyLearner().fit(X, W, y)

    tau_rand = rand_policy.predict_uplift(X)
    tau_out = out_model.predict_uplift(X)

    assert len(tau_rand) == 100
    assert len(tau_out) == 100
