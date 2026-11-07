"""Review regressions for causal inputs, rare outcomes, and cumulative metrics."""

import numpy as np
import pandas as pd
import pytest

from uplift.evaluation.policy_value import compute_policy_assignment, ipw_policy_value
from uplift.evaluation.uplift_metrics import compute_uplift_curve
from uplift.experimentation.power import PowerCalculator
from uplift.models import doubly_robust
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner


def test_curve_covers_entire_population_with_correct_endpoint():
    y = np.array([1, 0, 1, 0])
    w = np.array([1, 0, 1, 0])
    score = np.array([4, 3, 2, 1])
    x, gain, qini = compute_uplift_curve(y, w, score)
    assert x[0] == 0 and x[-1] == 1
    assert gain[-1] == pytest.approx(1.0)
    assert qini[-1] == pytest.approx(2.0)
    assert len(x) == len(np.unique(x))


@pytest.mark.parametrize("column", ["visit", "exposure", "treatment", "conversion", "unexpected"])
def test_direct_fit_rejects_features_outside_causal_allowlist(column):
    with pytest.raises(ValueError):
        SLearner().fit(
            pd.DataFrame({column: [0.0, 1.0, 2.0, 3.0]}),
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 0, 1]),
        )


@pytest.mark.parametrize("cls", [SLearner, TLearner, doubly_robust.DoublyRobustLearner])
def test_single_outcome_class_in_arm_does_not_crash(cls):
    x = np.arange(48, dtype=float).reshape(24, 2)
    w = np.array([0, 1] * 12)
    y = np.zeros(24, dtype=int)
    model = cls().fit(x, w, y)
    assert np.all(np.isfinite(model.predict_uplift(x)))


def test_budget_and_propensity_validation_and_stable_ties():
    np.testing.assert_array_equal(compute_policy_assignment(np.ones(4), 0.5), [1, 1, 0, 0])
    with pytest.raises(ValueError):
        compute_policy_assignment(np.ones(4), 1.1)
    with pytest.raises(ValueError):
        ipw_policy_value(np.ones(2), np.array([0, 1]), np.ones(2), 0.0)


def test_two_sided_power_under_null_equals_alpha():
    assert PowerCalculator.calculate_power(1000, 1000, 0.1, 0.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        PowerCalculator.calculate_sample_size(0.1, 0.0)
