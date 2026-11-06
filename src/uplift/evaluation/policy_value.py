"""Targeting policy evaluation using IPW and Doubly Robust estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_policy_assignment(
    uplift_scores: np.ndarray,
    budget_fraction: float = 0.20,
) -> np.ndarray:
    """Return binary policy decisions pi(x) in {0, 1} for a given treatment budget."""
    n = len(uplift_scores)
    k_cutoff = int(np.ceil(budget_fraction * n))
    ranks = np.argsort(np.argsort(-uplift_scores))
    return (ranks < k_cutoff).astype(int)


def ipw_policy_value(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    policy_actions: np.ndarray,
    propensity: float = 0.85,
) -> float:
    """Compute Inverse Propensity Weighted (Horvitz-Thompson) policy value.

    V(pi) = (1/N) * sum [ (W*pi/e + (1-W)*(1-pi)/(1-e)) * Y ]
    """
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
