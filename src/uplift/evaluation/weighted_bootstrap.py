"""Paired Poisson-weighted bootstrap for uplift metrics.

Each model is sorted once by score; every resample applies shared
Poisson(1) weights through weighted cumulative sums, so resamples are
paired across models. Ranking metrics are evaluated at tie-block
boundaries with linear interpolation inside straddling blocks, matching
the plain (unit-weight) path exactly. Policy values use the same
cutoff interpolation, and optional constant-policy series share the same
weight draws so gains against them stay paired.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from uplift.evaluation.uplift_metrics import (
    SortedPrefix,
    boundary_areas,
    build_sorted_prefix,
    cutoff_policy_value,
    uplift_at_cutoff,
    weighted_prefix_stats,
)

BUDGETS = (0.10, 0.20, 0.30)
BUDGET_TAGS = ("10", "20", "30")


@dataclass
class BootstrapResult:
    """Paired Poisson bootstrap samples and percentile intervals."""

    point: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, list[float]]]
    n_boot: int
    seed: int

    def ci(self, metric: str, name: str, level: float = 0.95) -> tuple[float, float]:
        """Percentile interval for one model/metric."""
        samples = np.asarray(self.metrics[name][metric], dtype=np.float64)
        alpha = 1.0 - level
        return (
            float(np.quantile(samples, alpha / 2.0)),
            float(np.quantile(samples, 1.0 - alpha / 2.0)),
        )

    def paired_diff(
        self, metric: str, name_a: str, name_b: str, level: float = 0.95
    ) -> tuple[float, float, float]:
        """Mean and percentile interval of the paired per-resample difference."""
        a = np.asarray(self.metrics[name_a][metric], dtype=np.float64)
        b = np.asarray(self.metrics[name_b][metric], dtype=np.float64)
        d = a - b
        alpha = 1.0 - level
        return (
            float(d.mean()),
            float(np.quantile(d, alpha / 2.0)),
            float(np.quantile(d, 1.0 - alpha / 2.0)),
        )


class SortedScorer:
    """Score-sorted prefix for one ranking; reweighted per bootstrap draw."""

    def __init__(
        self,
        sp: SortedPrefix,
        policies: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.sp = sp
        # policy name -> (u_sorted, v_sorted); row value is v + pi * (u - v)
        self.policies = policies or {}

    @classmethod
    def from_scores(
        cls,
        y: np.ndarray,
        W: np.ndarray,
        scores: np.ndarray,
        policies: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        n_bins: int = 100,
    ) -> SortedScorer:
        """Build a scorer; policy vectors are given in original row order.

        ``n_bins`` is accepted for backward compatibility and ignored:
        metrics are computed exactly at tie-block boundaries.
        """
        del n_bins
        sp = build_sorted_prefix(y, W, scores)
        sorted_policies = {
            name: (np.asarray(u, dtype=np.float64)[sp.order],
                  np.asarray(v, dtype=np.float64)[sp.order])
            for name, (u, v) in (policies or {}).items()
        }
        return cls(sp, sorted_policies)

    def _resample_prefix(self, w: np.ndarray) -> SortedPrefix:
        sp = self.sp
        w_sorted = w[sp.order]
        cum_w, cum_t, cum_c, cum_y_t, cum_y_c = weighted_prefix_stats(
            sp.y_sorted, sp.W_sorted, w_sorted
        )
        return replace(
            sp,
            w_sorted=w_sorted,
            cum_w=cum_w,
            cum_t=cum_t,
            cum_c=cum_c,
            cum_y_t=cum_y_t,
            cum_y_c=cum_y_c,
        )

    def metrics(self, w: np.ndarray) -> dict[str, float]:
        """All metrics for one weight vector (unit weights = point estimate)."""
        sp = self._resample_prefix(w)
        auuc, qini = boundary_areas(sp)
        out = {"auuc": auuc, "qini": qini}
        for k, tag in zip(BUDGETS, BUDGET_TAGS, strict=True):
            out[f"uplift_at_{tag}"] = uplift_at_cutoff(sp, k)
        w_sorted = sp.w_sorted
        for name, (u_s, v_s) in self.policies.items():
            cum_u = np.r_[0.0, np.cumsum(w_sorted * u_s)]
            cum_v = np.r_[0.0, np.cumsum(w_sorted * v_s)]
            for k, tag in zip(BUDGETS, BUDGET_TAGS, strict=True):
                out[f"{name}_policy_value_{tag}"] = cutoff_policy_value(
                    sp, cum_u, cum_v, k
                )
        return out


def paired_poisson_bootstrap(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    n_boot: int = 500,
    seed: int = 0,
    policy_vectors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    const_series: dict[str, dict[str, np.ndarray]] | None = None,
) -> BootstrapResult:
    """Paired Poisson(1) bootstrap over per-model score rankings.

    Parameters
    ----------
    policy_vectors
        ``model name -> policy name -> (u, v)`` per-row value vectors in
        original row order; each yields ``<policy>_policy_value_<k>``
        metrics through the same cutoff interpolation as the plain path.
    const_series
        ``pseudo name -> metric name -> per-row value vector`` evaluated as
        a weighted mean under each draw; used for constant policies that
        have no ranking, keeping them on the same resamples.
    """
    y = np.asarray(y_true, dtype=np.float64)
    W = np.asarray(treatment, dtype=np.float64)
    rng = np.random.default_rng(seed)
    scorers = {
        name: SortedScorer.from_scores(
            y,
            W,
            np.asarray(scores, dtype=np.float64),
            policies=(policy_vectors or {}).get(name),
        )
        for name, scores in scores_by_model.items()
    }

    ones = np.ones(len(y), dtype=np.float64)
    point: dict[str, dict[str, float]] = {}
    samples: dict[str, dict[str, list[float]]] = {}
    for name, scorer in scorers.items():
        m = scorer.metrics(ones)
        point[name] = m
        samples[name] = {metric: [] for metric in m}
    for name, vecs in (const_series or {}).items():
        point[name] = {
            metric: float(np.mean(np.asarray(vec, dtype=np.float64)))
            for metric, vec in vecs.items()
        }
        samples[name] = {metric: [] for metric in vecs}

    for _ in range(n_boot):
        w = rng.poisson(1.0, size=len(y)).astype(np.float64)
        w_sum = float(w.sum())
        for name, scorer in scorers.items():
            m = scorer.metrics(w)
            for metric, value in m.items():
                samples[name][metric].append(value)
        for name, vecs in (const_series or {}).items():
            for metric, vec in vecs.items():
                samples[name][metric].append(float(np.dot(w, vec) / w_sum))

    return BootstrapResult(point=point, metrics=samples, n_boot=n_boot, seed=seed)
