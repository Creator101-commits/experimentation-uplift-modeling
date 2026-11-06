"""Paired bootstrap confidence intervals for uplift evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift.evaluation.uplift_metrics import auuc_score, qini_score, uplift_at_k


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """Estimated metric point estimate and 95% percentile bootstrap interval."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float


def compute_bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_scores: np.ndarray | pd.Series,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, BootstrapConfidenceInterval]:
    """Compute 95% bootstrap confidence intervals for AUUC, Qini, and Uplift@20%."""
    y = np.asarray(y_true, dtype=int)
    W = np.asarray(treatment, dtype=int)
    scores = np.asarray(uplift_scores, dtype=np.float64)

    n = len(y)
    rng = np.random.default_rng(random_state)

    auuc_samples = np.zeros(n_bootstrap)
    qini_samples = np.zeros(n_bootstrap)
    u20_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]
        W_b = W[idx]
        s_b = scores[idx]

        auuc_samples[b] = auuc_score(y_b, W_b, s_b)
        qini_samples[b] = qini_score(y_b, W_b, s_b)
        u20_samples[b] = uplift_at_k(y_b, W_b, s_b, k=0.20)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    out: dict[str, BootstrapConfidenceInterval] = {
        "auuc": BootstrapConfidenceInterval(
            metric_name="auuc",
            point_estimate=auuc_score(y, W, scores),
            ci_lower=float(np.percentile(auuc_samples, lower_pct)),
            ci_upper=float(np.percentile(auuc_samples, upper_pct)),
        ),
        "qini": BootstrapConfidenceInterval(
            metric_name="qini",
            point_estimate=qini_score(y, W, scores),
            ci_lower=float(np.percentile(qini_samples, lower_pct)),
            ci_upper=float(np.percentile(qini_samples, upper_pct)),
        ),
        "uplift@20%": BootstrapConfidenceInterval(
            metric_name="uplift@20%",
            point_estimate=uplift_at_k(y, W, scores, k=0.20),
            ci_lower=float(np.percentile(u20_samples, lower_pct)),
            ci_upper=float(np.percentile(u20_samples, upper_pct)),
        ),
    }

    return out
