"""Targeting policy evaluation using IPW and Doubly Robust estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift.evaluation.uplift_metrics import (
    SortedPrefix,
    build_sorted_prefix,
    cutoff_policy_value,
)


def compute_policy_assignment(
    uplift_scores: np.ndarray,
    budget_fraction: float = 0.20,
) -> np.ndarray:
    """Return binary policy decisions pi(x) in {0, 1} for a given treatment budget."""
    n = len(uplift_scores)
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be within [0, 1].")
    if np.asarray(uplift_scores).ndim != 1 or not np.all(np.isfinite(uplift_scores)):
        raise ValueError("uplift_scores must be a finite vector.")
    k_cutoff = int(np.ceil(budget_fraction * n))
    actions = np.zeros(n, dtype=int)
    actions[np.argsort(-np.asarray(uplift_scores), kind="stable")[:k_cutoff]] = 1
    return actions


def ipw_policy_value(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    policy_actions: np.ndarray,
    propensity: float = 0.85,
) -> float:
    """Compute Inverse Propensity Weighted (Horvitz-Thompson) policy value.

    V(pi) = (1/N) * sum [ (W*pi/e + (1-W)*(1-pi)/(1-e)) * Y ]
    """
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must be strictly within (0, 1).")
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=int)
    pi = np.asarray(policy_actions, dtype=int)

    weight = (W * pi) / propensity + ((1 - W) * (1 - pi)) / (1.0 - propensity)
    return float(np.mean(weight * y))


def doubly_robust_policy_value(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    policy_actions: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float = 0.85,
) -> float:
    """Compute Doubly Robust policy value.

    Combines outcome estimates with inverse-propensity residuals.
    """
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must be strictly within (0, 1).")
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=int)
    pi = np.asarray(policy_actions, dtype=int)

    # Plug-in value
    v_plugin = pi * mu1 + (1 - pi) * mu0

    # Correction terms
    res_treated = (W * pi / propensity) * (y - mu1)
    res_control = (((1 - W) * (1 - pi)) / (1.0 - propensity)) * (y - mu0)

    dr_value = v_plugin + res_treated + res_control
    return float(np.mean(dr_value))


def budget_policy_values(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    budgets: tuple[float, ...] | list[float],
    propensity: float,
    mu0: np.ndarray | None = None,
    mu1: np.ndarray | None = None,
    prefix: SortedPrefix | None = None,
) -> dict[str, float]:
    """Tie-aware IPW and DR policy values at each targeting budget.

    The policy treats the top ``k`` weight fraction by score. A tie block
    straddling the cutoff contributes proportionally through linear
    interpolation of the cumulative sums, the expected value under random
    tie-breaking. DR values use the AIPW form with the known propensity.
    """
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must be strictly within (0, 1).")
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=np.float64)
    if prefix is None:
        sp = build_sorted_prefix(y, W, np.asarray(uplift_scores, dtype=np.float64))
    else:
        sp = prefix
    order = sp.order
    e = float(propensity)
    a = (W * y / e)[order]
    b = ((1.0 - W) * y / (1.0 - e))[order]
    cum_a = np.r_[0.0, np.cumsum(a)]
    cum_b = np.r_[0.0, np.cumsum(b)]
    out: dict[str, float] = {}
    for k in budgets:
        tag = f"{int(round(k * 100))}"
        out[f"ipw_policy_value_{tag}"] = cutoff_policy_value(sp, cum_a, cum_b, k)
    if mu0 is not None and mu1 is not None:
        m0 = np.asarray(mu0, dtype=np.float64)
        m1 = np.asarray(mu1, dtype=np.float64)
        u_dr = (m1 + W * (y - m1) / e)[order]
        v_dr = (m0 + (1.0 - W) * (y - m0) / (1.0 - e))[order]
        cum_u = np.r_[0.0, np.cumsum(u_dr)]
        cum_v = np.r_[0.0, np.cumsum(v_dr)]
        for k in budgets:
            tag = f"{int(round(k * 100))}"
            out[f"dr_policy_value_{tag}"] = cutoff_policy_value(sp, cum_u, cum_v, k)
    return out


def constant_policy_values(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    propensity: float,
    treated_ate: float,
) -> dict[str, dict[str, float]]:
    """IPW and DR values for the degenerate treat-none/treat-all policies.

    treat-none is the control mean and treat-all the treated mean, both
    estimated with the known propensity; the DR variant uses the constant
    outcome predictions (0, treated_ate) of the fitted policies.
    """
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must be strictly within (0, 1).")
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=np.float64)
    e = float(propensity)
    a = W * y / e
    b = (1.0 - W) * y / (1.0 - e)
    ta_dr = treated_ate + W * (y - treated_ate) / e
    return {
        "treat_none": {"ipw": float(np.mean(b)), "dr": float(np.mean(b))},
        "treat_all": {"ipw": float(np.mean(a)), "dr": float(np.mean(ta_dr))},
    }
