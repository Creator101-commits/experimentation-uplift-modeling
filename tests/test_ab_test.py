"""Tests for A/B testing statistical estimation and balance checks."""

from __future__ import annotations

import numpy as np
import pytest

from uplift.experimentation.ab_test import ABTestAnalyzer


def test_ab_test_hand_calculated() -> None:
    # Control: 1000 samples, 40 conversions (p0 = 0.04)
    # Treated: 1000 samples, 60 conversions (p1 = 0.06)
    W = np.array([0] * 1000 + [1] * 1000)
    y = np.array([1] * 40 + [0] * 960 + [1] * 60 + [0] * 940)

    res = ABTestAnalyzer.analyze(W, y)
    assert np.isclose(res.control_rate, 0.04)
    assert np.isclose(res.treated_rate, 0.06)
    assert np.isclose(res.absolute_lift, 0.02)
    assert np.isclose(res.relative_lift, 0.50)
    assert np.isclose(res.relative_risk, 1.50)
    assert res.p_value < 0.05
    assert res.ci_lower < res.absolute_lift < res.ci_upper
    assert res.number_needed_to_treat is not None
    assert np.isclose(res.number_needed_to_treat, 50.0)


def test_ab_test_empty_groups_error() -> None:
    W = np.array([1, 1, 1])
    y = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="Both treatment and control groups must contain samples"):
        ABTestAnalyzer.analyze(W, y)


def test_check_covariate_balance() -> None:
    rng = np.random.default_rng(42)
    # Randomized balanced covariates
    X = rng.normal(size=(2000, 4))
    W = rng.binomial(n=1, p=0.5, size=2000)

    smds = ABTestAnalyzer.check_covariate_balance(X, W)
    for col, smd in smds.items():
        assert smd < 0.10, f"Covariate {col} has large SMD: {smd}"
