"""Synthetic Criteo-style uplift dataset generator for experiments and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def generate_synthetic_uplift_data(
    n_samples: int = 10000,
    treatment_propensity: float = 0.85,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic uplift data with known causal effects.

    Matches Criteo schema: 12 pre-treatment features f0..f11, randomized treatment,
    post-treatment exposure/visit, and conversion outcome.
    """
    rng = np.random.default_rng(random_state)

    # 12 continuous pre-treatment features
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 12))

    W = rng.binomial(n=1, p=treatment_propensity, size=n_samples)

    # Base propensity of conversion (control group baseline: ~4%)
    baseline_logits = -3.2 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.15 * X[:, 2]
    mu0 = expit(baseline_logits)

    # Heterogeneous treatment effect (uplift):
    # Persuadables (positive uplift) correlated with f0 and f3
    # Sleeping dogs (negative uplift) when f4 is high
    tau = 0.05 * (X[:, 0] > 0.5) + 0.04 * (X[:, 3] > 0.0) - 0.06 * (X[:, 4] > 1.5)
    # Clip to keep probabilities valid
    mu1 = np.clip(mu0 + tau, 0.001, 0.95)

    # Observed outcome under assigned treatment
    prob_observed = np.where(W == 1, mu1, mu0)
    conversion = rng.binomial(n=1, p=prob_observed)

    # Intermediate post-treatment signals (forbidden from model features)
    exposure = np.where(W == 1, rng.binomial(n=1, p=0.9, size=n_samples), 0)
    visit = np.where(conversion == 1, 1, rng.binomial(n=1, p=0.08, size=n_samples))

    data: dict[str, np.ndarray] = {f"f{i}": X[:, i] for i in range(12)}
    data["treatment"] = W
    data["exposure"] = exposure
    data["visit"] = visit
    data["conversion"] = conversion

    return pd.DataFrame(data)
