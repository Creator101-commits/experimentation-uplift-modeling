"""Statistical power analysis and minimum detectable effect calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class PowerResult:
    """Result of power or sample size calculation."""

    baseline_rate: float
    target_rate: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    n_control: int
    n_treated: int
    total_sample_size: int


class PowerCalculator:
    """Calculations for prospective experiment power and sample size."""

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        mde_absolute: float,
        alpha: float = 0.05,
        power: float = 0.80,
        treatment_ratio: float = 0.85,
    ) -> PowerResult:
        """Calculate required sample size for unequal allocation.

        Parameters
        ----------
        baseline_rate : float
            Control group conversion rate (e.g. 0.04).
        mde_absolute : float
            Minimum absolute risk difference to detect (e.g. 0.005).
        alpha : float, default=0.05
            Significance level (two-sided).
        power : float, default=0.80
            Statistical power (1 - beta).
        treatment_ratio : float, default=0.85
            Fraction of total traffic assigned to treatment.
        """
        p0 = baseline_rate
        p1 = baseline_rate + mde_absolute
        if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
            raise ValueError("Baseline and target conversion rates must be strictly within (0, 1).")

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))

        # Allocation ratio r = n1 / n0
        r = treatment_ratio / (1.0 - treatment_ratio)
        p_bar = (p0 + r * p1) / (1.0 + r)
        v_null = (1.0 + 1.0 / r) * p_bar * (1.0 - p_bar)
        v_alt = p0 * (1.0 - p0) + (p1 * (1.0 - p1)) / r

        delta = abs(p1 - p0)
        n0 = ((z_alpha * math.sqrt(v_null) + z_beta * math.sqrt(v_alt)) / delta) ** 2
        n_control = math.ceil(n0)
        n_treated = math.ceil(n_control * r)

        return PowerResult(
            baseline_rate=p0,
            target_rate=p1,
            mde_absolute=mde_absolute,
            mde_relative=mde_absolute / p0,
            alpha=alpha,
            power=power,
            n_control=n_control,
            n_treated=n_treated,
            total_sample_size=n_control + n_treated,
        )

    @staticmethod
    def calculate_power(
        n_control: int,
        n_treated: int,
        baseline_rate: float,
        mde_absolute: float,
        alpha: float = 0.05,
    ) -> float:
        """Calculate achieved statistical power given fixed sample sizes."""
        if n_control <= 0 or n_treated <= 0:
            raise ValueError("Sample sizes must be positive integers.")

        p0 = baseline_rate
        p1 = baseline_rate + mde_absolute
        r = n_treated / n_control
        p_bar = (n_control * p0 + n_treated * p1) / (n_control + n_treated)

        v_null = (1.0 + 1.0 / r) * p_bar * (1.0 - p_bar)
        v_alt = p0 * (1.0 - p0) + (p1 * (1.0 - p1)) / r

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        delta = abs(p1 - p0)

        z_val = (delta * math.sqrt(n_control) - z_alpha * math.sqrt(v_null)) / math.sqrt(v_alt)
        return float(stats.norm.cdf(z_val))

    @staticmethod
    def calculate_mde(
        n_control: int,
        n_treated: int,
        baseline_rate: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> float:
        """Calculate minimum detectable effect (MDE) given fixed sample sizes."""
        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))
        r = n_treated / n_control

        # Approximate variance under baseline
        var_approx = baseline_rate * (1.0 - baseline_rate) * (1.0 + 1.0 / r)
        se = math.sqrt(var_approx / n_control)
        return float((z_alpha + z_beta) * se)
