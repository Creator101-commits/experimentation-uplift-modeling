"""Uplift ranking metrics: Qini curve, AUUC, and Uplift@K."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_uplift_curve(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative uplift and Qini curves across population percentiles.

    Returns
    -------
    percentiles : np.ndarray
        Population fractions from 0.0 to 1.0.
    uplift_curve : np.ndarray
        Cumulative incremental conversion gains.
    qini_curve : np.ndarray
        Qini curve values: n_t_conv - n_c_conv * (n_t / n_c).
    """
    y = np.asarray(y_true, dtype=int)
    W = np.asarray(treatment, dtype=int)
    scores = np.asarray(uplift_scores, dtype=np.float64)

    # Sort descending by predicted uplift
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if y.ndim != 1 or W.shape != y.shape or scores.shape != y.shape or not len(y):
        raise ValueError("Outcomes, treatment, and scores must be aligned nonempty vectors.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(W, [0, 1])):
        raise ValueError("Scores must be finite and treatment must be binary.")
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    W_sorted = W[order]

    n = len(y)
    cum_t = np.r_[0, np.cumsum(W_sorted)]
    cum_c = np.r_[0, np.cumsum(1 - W_sorted)]

    cum_y_t = np.r_[0, np.cumsum(y_sorted * W_sorted)]
    cum_y_c = np.r_[0, np.cumsum(y_sorted * (1 - W_sorted))]

    # Downsample to n_bins points
    indices = np.unique(np.linspace(0, n, min(n_bins, n) + 1, dtype=int))

    fracs = indices / n
    # Scaled Qini: cum_y_t - cum_y_c * (cum_t / cum_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = np.where(cum_c[indices] > 0, cum_t[indices] / cum_c[indices], 0.0)
        qini = cum_y_t[indices] - cum_y_c[indices] * scaling

        # Normalized uplift curve (gain per population)
        rate_t = np.where(cum_t[indices] > 0, cum_y_t[indices] / cum_t[indices], 0.0)
        rate_c = np.where(cum_c[indices] > 0, cum_y_c[indices] / cum_c[indices], 0.0)
        uplift_gain = (rate_t - rate_c) * fracs

    qini[0] = 0.0
    uplift_gain[0] = 0.0

    return fracs, uplift_gain, qini


def auuc_score(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
) -> float:
    """Compute Area Under the Uplift Curve (AUUC) via trapezoidal integration."""
    fracs, uplift_gain, _ = compute_uplift_curve(y_true, treatment, uplift_scores)
    return float(np.trapezoid(uplift_gain, fracs))


def qini_score(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
) -> float:
    """Compute unnormalized excess Qini area above the random targeting baseline."""
    fracs, _, qini = compute_uplift_curve(y_true, treatment, uplift_scores)
    # Area under model Qini
    area_model = np.trapezoid(qini, fracs)

    # Random baseline is a straight line from (0, 0) to (1, qini[-1])
    area_random = 0.5 * qini[-1]

    # Qini score is normalized excess area
    excess_area = area_model - area_random
    return float(excess_area)


def uplift_at_k(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    k: float = 0.20,
) -> float:
    """Compute empirical uplift difference (p_treated - p_control) in the top k fraction.

    Parameters
    ----------
    k : float, default=0.20
        Top fraction of population targeted (e.g. 0.1, 0.2, 0.3).
    """
    if not (0.0 < k <= 1.0):
        raise ValueError("Targeting fraction k must be strictly in (0.0, 1.0].")

    y = np.asarray(y_true, dtype=int)
    W = np.asarray(treatment, dtype=int)
    scores = np.asarray(uplift_scores, dtype=np.float64)

    cutoff = int(np.ceil(k * len(y)))
    top_indices = np.argsort(-scores)[:cutoff]

    y_top = y[top_indices]
    W_top = W[top_indices]

    t_mask = W_top == 1
    c_mask = W_top == 0

    if np.sum(t_mask) == 0 or np.sum(c_mask) == 0:
        return 0.0

    p_treated = float(np.mean(y_top[t_mask]))
    p_control = float(np.mean(y_top[c_mask]))

    return p_treated - p_control
