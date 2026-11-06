"""Tests for uplift ranking curves, policy value, and calibration."""

from __future__ import annotations

import numpy as np

from uplift.evaluation.bootstrap import compute_bootstrap_ci
from uplift.evaluation.calibration import compute_uplift_calibration
from uplift.evaluation.policy_value import (
    doubly_robust_policy_value,
    ipw_policy_value,
)
from uplift.evaluation.uplift_metrics import (
    auuc_score,
    compute_uplift_curve,
    qini_score,
    uplift_at_k,
)


def test_uplift_curves_and_scores() -> None:
    # Synthetic ground truth
    rng = np.random.default_rng(42)
    n = 500
    W = rng.binomial(1, 0.8, n)
    # Give higher conversion to those with higher score
    scores = rng.normal(0, 1, n)
    p = 0.05 + 0.1 * (scores > 0.5) * W
    y = rng.binomial(1, np.clip(p, 0, 1))

    fracs, gain, qini = compute_uplift_curve(y, W, scores, n_bins=20)
    assert len(fracs) == 21
    assert len(gain) == 21
    assert len(qini) == 21

    auuc = auuc_score(y, W, scores)
    q = qini_score(y, W, scores)
    u10 = uplift_at_k(y, W, scores, k=0.10)

    assert isinstance(auuc, float)
    assert isinstance(q, float)
    assert isinstance(u10, float)


def test_policy_value_ipw_and_dr() -> None:
    y = np.array([1, 0, 1, 0, 0, 1])
    W = np.array([1, 1, 0, 0, 1, 0])
    actions = np.array([1, 1, 0, 0, 1, 0])
    mu0 = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.1])
    mu1 = np.array([0.5, 0.5, 0.3, 0.3, 0.5, 0.3])

    v_ipw = ipw_policy_value(y, W, actions, propensity=0.5)
    v_dr = doubly_robust_policy_value(y, W, actions, mu0, mu1, propensity=0.5)

    assert v_ipw >= 0.0
    assert v_dr >= 0.0


def test_uplift_calibration() -> None:
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1] * 20)
    W = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0] * 20)
    scores = np.linspace(-0.1, 0.2, len(y))

    bins = compute_uplift_calibration(y, W, scores, n_bins=4)
    assert len(bins) > 0
    assert all(b.n_samples > 0 for b in bins)


def test_bootstrap_ci() -> None:
    rng = np.random.default_rng(42)
    n = 100
    y = rng.binomial(1, 0.1, n)
    W = rng.binomial(1, 0.5, n)
    scores = rng.normal(0, 1, n)

    ci_dict = compute_bootstrap_ci(y, W, scores, n_bootstrap=20, random_state=42)
    assert "auuc" in ci_dict
    assert "qini" in ci_dict
    assert "uplift@20%" in ci_dict
    assert ci_dict["auuc"].ci_lower <= ci_dict["auuc"].ci_upper
