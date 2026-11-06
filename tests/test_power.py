"""Tests for power analysis and sample size calculations."""

from __future__ import annotations

import pytest

from uplift.experimentation.power import PowerCalculator


def test_calculate_sample_size_monotonicity() -> None:
    # Smaller MDE requires larger sample size
    res_large_mde = PowerCalculator.calculate_sample_size(
        baseline_rate=0.05, mde_absolute=0.01, power=0.80
    )
    res_small_mde = PowerCalculator.calculate_sample_size(
        baseline_rate=0.05, mde_absolute=0.002, power=0.80
    )

    assert res_small_mde.total_sample_size > res_large_mde.total_sample_size
    assert res_large_mde.n_treated > res_large_mde.n_control


def test_calculate_power() -> None:
    # Large sample size achieves high power
    power_high = PowerCalculator.calculate_power(
        n_control=10000, n_treated=50000, baseline_rate=0.04, mde_absolute=0.01
    )
    power_low = PowerCalculator.calculate_power(
        n_control=100, n_treated=500, baseline_rate=0.04, mde_absolute=0.01
    )

    assert 0.90 < power_high <= 1.0
    assert power_low < power_high


def test_calculate_mde() -> None:
    mde = PowerCalculator.calculate_mde(
        n_control=10000, n_treated=50000, baseline_rate=0.04, power=0.80
    )
    assert 0.001 < mde < 0.01


def test_invalid_parameters_raise_error() -> None:
    with pytest.raises(ValueError, match="strictly within"):
        PowerCalculator.calculate_sample_size(baseline_rate=1.2, mde_absolute=0.01)

    with pytest.raises(ValueError, match="positive integers"):
        PowerCalculator.calculate_power(
            n_control=-10, n_treated=100, baseline_rate=0.04, mde_absolute=0.01
        )
