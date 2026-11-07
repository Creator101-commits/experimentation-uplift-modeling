"""Classical treatment versus control A/B test analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ABTestResult:
    """Statistical summary of randomized A/B trial."""

    n_control: int
    n_treated: int
    control_conversions: int
    treated_conversions: int
    control_rate: float
    treated_rate: float
    absolute_lift: float
    relative_lift: float
    relative_risk: float
    se_diff: float
    z_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    number_needed_to_treat: float | None


class ABTestAnalyzer:
    """Analyzer for randomized experiment outcomes."""

    @staticmethod
    def analyze(
        treatment: np.ndarray | pd.Series,
        outcome: np.ndarray | pd.Series,
        alpha: float = 0.05,
    ) -> ABTestResult:
        """Run two-proportion hypothesis test and confidence interval estimation."""
        W = np.asarray(treatment)
        y = np.asarray(outcome)
        if W.ndim != 1 or y.shape != W.shape:
            raise ValueError("Treatment and outcome must be aligned vectors.")
        if not np.all(np.isin(W, [0, 1])) or not np.all(np.isin(y, [0, 1])):
            raise ValueError("Treatment and outcome must be binary.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")

        n0 = int(np.sum(W == 0))
        n1 = int(np.sum(W == 1))
        if n0 == 0 or n1 == 0:
            raise ValueError("Both treatment and control groups must contain samples.")

        c0 = int(np.sum(y[W == 0]))
        c1 = int(np.sum(y[W == 1]))

        p0 = c0 / n0
        p1 = c1 / n1

        abs_lift = p1 - p0
        rel_lift = (p1 - p0) / p0 if p0 > 0 else 0.0
        rel_risk = p1 / p0 if p0 > 0 else 0.0

        # Pooled standard error under H0
        p_pool = (c0 + c1) / (n0 + n1)
        se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n0 + 1.0 / n1))

        # Unpooled standard error for confidence interval
        se_unpooled = math.sqrt((p0 * (1.0 - p0) / n0) + (p1 * (1.0 - p1) / n1))

        z_stat = (abs_lift / se_pool) if se_pool > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = abs_lift - z_crit * se_unpooled
        ci_upper = abs_lift + z_crit * se_unpooled

        nnt = (1.0 / abs_lift) if abs_lift > 0 else None

        return ABTestResult(
            n_control=n0,
            n_treated=n1,
            control_conversions=c0,
            treated_conversions=c1,
            control_rate=p0,
            treated_rate=p1,
            absolute_lift=abs_lift,
            relative_lift=rel_lift,
            relative_risk=rel_risk,
            se_diff=se_unpooled,
            z_stat=z_stat,
            p_value=p_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            number_needed_to_treat=nnt,
        )

    @staticmethod
    def check_covariate_balance(
        X: pd.DataFrame | np.ndarray,
        treatment: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Compute standardized mean differences (SMD) across covariates to verify randomization."""
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            col_names = [f"f{i}" for i in range(X_arr.shape[1])]

        W = np.asarray(treatment, dtype=int)
        smds: dict[str, float] = {}

        for i, col in enumerate(col_names):
            c_vals = X_arr[W == 0, i]
            t_vals = X_arr[W == 1, i]

            m0, m1 = float(np.mean(c_vals)), float(np.mean(t_vals))
            v0, v1 = float(np.var(c_vals, ddof=1)), float(np.var(t_vals, ddof=1))
            pooled_std = math.sqrt((v0 + v1) / 2.0)
            smd = (
                abs(m1 - m0) / pooled_std if pooled_std > 0 else (float("inf") if m1 != m0 else 0.0)
            )
            smds[col] = smd

        return smds
