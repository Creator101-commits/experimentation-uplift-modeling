"""Uplift ranking metrics: Qini curve, AUUC, and Uplift@K.

All ranking metrics are tie-aware: the cumulative curve is evaluated only
at distinct-score block boundaries and boundary points are connected
linearly (the scikit-uplift convention), so constant or heavily tied
scores reproduce the random-targeting line instead of inheriting source
file order. Point metrics at a targeting budget interpolate the
cumulative weighted sums linearly inside a straddling tie block, which is
the expected value under random tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def weighted_prefix_stats(
    y_sorted: np.ndarray,
    W_sorted: np.ndarray,
    w_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted cumulative sums over score-sorted rows.

    Returns ``(cum_w, cum_t, cum_c, cum_y_t, cum_y_c)`` each with a leading
    zero, where ``cum_*[j]`` aggregates the first ``j`` sorted rows. Unit
    weights reproduce the unweighted cumulative counts exactly.
    """
    cum_w = np.r_[0.0, np.cumsum(w_sorted)]
    cum_t = np.r_[0.0, np.cumsum(w_sorted * W_sorted)]
    cum_c = np.r_[0.0, np.cumsum(w_sorted * (1.0 - W_sorted))]
    cum_y_t = np.r_[0.0, np.cumsum(w_sorted * y_sorted * W_sorted)]
    cum_y_c = np.r_[0.0, np.cumsum(w_sorted * y_sorted * (1.0 - W_sorted))]
    return cum_w, cum_t, cum_c, cum_y_t, cum_y_c


def tie_block_bounds(scores_sorted: np.ndarray) -> np.ndarray:
    """Row positions (into the sorted arrays) where the score value changes.

    Returns ``B + 1`` boundaries for ``B`` tie blocks; block ``j`` spans
    rows ``[bounds[j], bounds[j + 1])`` and shares one score value.
    """
    n = len(scores_sorted)
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    change = np.flatnonzero(scores_sorted[1:] != scores_sorted[:-1]) + 1
    return np.concatenate([[0], change.astype(np.int64), [n]])


@dataclass
class SortedPrefix:
    """Score-sorted view of one ranking with tie-block bookkeeping."""

    order: np.ndarray
    y_sorted: np.ndarray
    W_sorted: np.ndarray
    w_sorted: np.ndarray
    bounds: np.ndarray
    scores_sorted: np.ndarray
    cum_w: np.ndarray
    cum_t: np.ndarray
    cum_c: np.ndarray
    cum_y_t: np.ndarray
    cum_y_c: np.ndarray

    @property
    def total_w(self) -> float:
        return float(self.cum_w[-1])


def build_sorted_prefix(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> SortedPrefix:
    """Sort rows by descending score and precompute prefix statistics."""
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=np.float64)
    scores = np.asarray(uplift_scores, dtype=np.float64)

    if y.ndim != 1 or W.shape != y.shape or scores.shape != y.shape or not len(y):
        raise ValueError("Outcomes, treatment, and scores must be aligned nonempty vectors.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(W, [0, 1])):
        raise ValueError("Scores must be finite and treatment must be binary.")
    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape or not np.all(w >= 0):
            raise ValueError("weights must be nonnegative and aligned with y.")

    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    prefix = weighted_prefix_stats(y[order], W[order], w[order])
    return SortedPrefix(
        order=order,
        y_sorted=y[order],
        W_sorted=W[order],
        w_sorted=w[order],
        bounds=tie_block_bounds(scores_sorted),
        scores_sorted=scores_sorted,
        cum_w=prefix[0],
        cum_t=prefix[1],
        cum_c=prefix[2],
        cum_y_t=prefix[3],
        cum_y_c=prefix[4],
    )


def _boundary_metrics(sp: SortedPrefix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, uplift_gain, qini) evaluated at every tie-block boundary."""
    b = sp.bounds
    total_w = sp.total_w
    cum_w_b = sp.cum_w[b]
    fracs = cum_w_b / total_w if total_w > 0 else b / max(len(sp.y_sorted), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_t = sp.cum_t[b]
        cum_c = sp.cum_c[b]
        cum_y_t = sp.cum_y_t[b]
        cum_y_c = sp.cum_y_c[b]
        scaling = np.where(cum_c > 0, cum_t / cum_c, 0.0)
        qini = cum_y_t - cum_y_c * scaling
        rate_t = np.where(cum_t > 0, cum_y_t / cum_t, 0.0)
        rate_c = np.where(cum_c > 0, cum_y_c / cum_c, 0.0)
        uplift_gain = (rate_t - rate_c) * fracs
    qini[0] = 0.0
    uplift_gain[0] = 0.0
    return fracs, uplift_gain, qini


def boundary_areas(sp: SortedPrefix) -> tuple[float, float]:
    """AUUC and Qini score as exact trapezoids over tie-block boundaries."""
    fracs, uplift_gain, qini = _boundary_metrics(sp)
    auuc = float(np.trapezoid(uplift_gain, fracs))
    qini_score = float(np.trapezoid(qini, fracs) - 0.5 * qini[-1])
    return auuc, qini_score


def _dedupe_x(xs: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop repeated x positions (zero-weight spans share identical metrics)."""
    keep = np.ones(len(xs), dtype=bool)
    keep[:-1] = np.diff(xs) > 0
    return (xs[keep], *(a[keep] for a in arrays))


def resampled_curve(
    sp: SortedPrefix, n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tie-aware curve resampled at ``n_bins + 1`` uniform population fractions."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    fracs, uplift_gain, qini = _boundary_metrics(sp)
    xs, ug, qq = _dedupe_x(fracs, uplift_gain, qini)
    grid = np.linspace(0.0, 1.0, min(n_bins, len(sp.y_sorted)) + 1)
    return grid, np.interp(grid, xs, ug), np.interp(grid, xs, qq)


def _cutoff_block(sp: SortedPrefix, target_w: float) -> tuple[int, float]:
    """Tie block and within-block weight fraction containing ``target_w``."""
    cw_bounds = sp.cum_w[sp.bounds]
    j = int(np.searchsorted(cw_bounds, target_w, side="right")) - 1
    j = min(max(j, 0), len(sp.bounds) - 2)
    start = float(cw_bounds[j])
    end = float(cw_bounds[j + 1])
    phi = 0.0 if end <= start else min(max((target_w - start) / (end - start), 0.0), 1.0)
    return j, phi


def _interp_at_block(
    cum: np.ndarray, bounds: np.ndarray, j: int, phi: float
) -> float:
    """Linearly interpolate one prefix-sum array inside block ``j``."""
    lo = float(cum[bounds[j]])
    hi = float(cum[bounds[j + 1]])
    return lo + phi * (hi - lo)


def uplift_at_cutoff(sp: SortedPrefix, k: float) -> float:
    """Treated-minus-control rate in the top ``k`` weight fraction.

    A tie block straddling the cutoff contributes proportionally through
    linear interpolation of the cumulative weighted sums.
    """
    if not (0.0 < k <= 1.0):
        raise ValueError("Targeting fraction k must be strictly in (0.0, 1.0].")
    total_w = sp.total_w
    if total_w <= 0:
        return 0.0
    j, phi = _cutoff_block(sp, k * total_w)
    b = sp.bounds
    cum_t = _interp_at_block(sp.cum_t, b, j, phi)
    cum_c = _interp_at_block(sp.cum_c, b, j, phi)
    cum_y_t = _interp_at_block(sp.cum_y_t, b, j, phi)
    cum_y_c = _interp_at_block(sp.cum_y_c, b, j, phi)
    if cum_t <= 0 or cum_c <= 0:
        return 0.0
    return float(cum_y_t / cum_t - cum_y_c / cum_c)


def cutoff_policy_value(
    sp: SortedPrefix,
    cum_u: np.ndarray,
    cum_v: np.ndarray,
    k: float,
) -> float:
    """Self-normalized weighted mean of ``v + pi * (u - v)`` at top-k.

    ``u`` and ``v`` are per-row value vectors already sorted into
    ``sp.order``; ``cum_u``/``cum_v`` are their weighted prefix sums with a
    leading zero (unit weights for the plain path). Rows above the cutoff
    contribute ``u``, rows below contribute ``v``, and a tie block
    straddling the cutoff contributes the weight-fraction mix, the
    expected value under random tie-breaking.
    """
    total_w = sp.total_w
    if total_w <= 0 or not (0.0 < k <= 1.0):
        return 0.0
    j, phi = _cutoff_block(sp, k * total_w)
    b = sp.bounds
    d_lo = float(cum_u[b[j]] - cum_v[b[j]])
    d_hi = float(cum_u[b[j + 1]] - cum_v[b[j + 1]])
    num = float(cum_v[-1]) + d_lo + phi * (d_hi - d_lo)
    return num / total_w


def curve_from_prefix(
    cum_w: np.ndarray,
    cum_t: np.ndarray,
    cum_c: np.ndarray,
    cum_y_t: np.ndarray,
    cum_y_c: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy row-index downsampling kept for callers without tie blocks."""
    n = len(cum_w) - 1
    total_w = float(cum_w[-1])
    indices = np.unique(np.linspace(0, n, min(n_bins, n) + 1, dtype=int))
    fracs = cum_w[indices] / total_w if total_w > 0 else indices / n
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = np.where(cum_c[indices] > 0, cum_t[indices] / cum_c[indices], 0.0)
        qini = cum_y_t[indices] - cum_y_c[indices] * scaling
        rate_t = np.where(cum_t[indices] > 0, cum_y_t[indices] / cum_t[indices], 0.0)
        rate_c = np.where(cum_c[indices] > 0, cum_y_c[indices] / cum_c[indices], 0.0)
        uplift_gain = (rate_t - rate_c) * fracs
    qini[0] = 0.0
    uplift_gain[0] = 0.0
    return fracs, uplift_gain, qini


def _curve_from_sorted(
    y_sorted: np.ndarray,
    W_sorted: np.ndarray,
    w_sorted: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative uplift/Qini curves from score-sorted arrays and weights."""
    prefix = weighted_prefix_stats(y_sorted, W_sorted, w_sorted)
    return curve_from_prefix(*prefix, n_bins)


def compute_uplift_curve(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    n_bins: int = 100,
    weights: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative uplift and Qini curves across population percentiles.

    Returns
    -------
    percentiles : np.ndarray
        Population fractions from 0.0 to 1.0 (``n_bins + 1`` points).
    uplift_curve : np.ndarray
        Cumulative incremental conversion gains.
    qini_curve : np.ndarray
        Qini curve values: n_t_conv - n_c_conv * (n_t / n_c).
    """
    sp = build_sorted_prefix(y_true, treatment, uplift_scores, weights)
    return resampled_curve(sp, n_bins)


def auuc_score(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
) -> float:
    """Compute Area Under the Uplift Curve (AUUC) via trapezoidal integration."""
    sp = build_sorted_prefix(y_true, treatment, uplift_scores)
    auuc, _ = boundary_areas(sp)
    return auuc


def qini_score(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
) -> float:
    """Compute unnormalized excess Qini area above the random targeting baseline."""
    sp = build_sorted_prefix(y_true, treatment, uplift_scores)
    _, qini = boundary_areas(sp)
    return qini


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
    sp = build_sorted_prefix(y_true, treatment, uplift_scores)
    return uplift_at_cutoff(sp, k)
