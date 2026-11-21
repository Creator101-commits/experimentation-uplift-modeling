"""Uplift calibration across quantile bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UpliftCalibrationBin:
    """Statistics for one quantile calibration bin."""

    bin_index: int
    n_samples: int
    n_treated: int
    n_control: int
    mean_predicted_uplift: float
    observed_uplift: float
    treated_rate: float
    control_rate: float
    calibration_gap: float


def compute_uplift_calibration(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    predicted_uplift: np.ndarray | pd.Series,
    n_bins: int = 5,
) -> list[UpliftCalibrationBin]:
    """Evaluate alignment between predicted CATE and observed randomized lift across bins."""
    y = np.asarray(y_true, dtype=int)
    W = np.asarray(treatment, dtype=int)
    tau = np.asarray(predicted_uplift, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(tau, quantiles)

    # Ensure unique edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    results: list[UpliftCalibrationBin] = []

    for i in range(n_bins):
        in_bin = (tau > bin_edges[i]) & (tau <= bin_edges[i + 1])
        n_bin = int(np.sum(in_bin))
        if n_bin == 0:
            continue

        y_b = y[in_bin]
        W_b = W[in_bin]
        n_t = int(np.sum(W_b == 1))
        n_c = int(np.sum(W_b == 0))

        p_t = float(np.mean(y_b[W_b == 1])) if n_t > 0 else 0.0
        p_c = float(np.mean(y_b[W_b == 0])) if n_c > 0 else 0.0
        obs_lift = p_t - p_c

        pred_mean = float(np.mean(tau[in_bin]))
        gap = abs(pred_mean - obs_lift)

        results.append(
            UpliftCalibrationBin(
                bin_index=i + 1,
                n_samples=n_bin,
                n_treated=n_t,
                n_control=n_c,
                mean_predicted_uplift=pred_mean,
                observed_uplift=obs_lift,
                treated_rate=p_t,
                control_rate=p_c,
                calibration_gap=gap,
            )
        )

    return results
