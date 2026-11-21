"""Frozen analysis plan for the uplift benchmark.

The plan is written to ``analysis_plan.json`` before any test read and
its SHA-256 is recorded in the run manifest, mirroring the frozen-protocol
pattern used by the recommender benchmark.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

# Declared constants that define the frozen analysis. These are fixed by
# the protocol, not tuned on data.
PRIMARY_OUTCOME = "conversion"
ESTIMANDS = ["risk_difference", "relative_lift"]
TARGETING_BUDGETS = [0.10, 0.20, 0.30]
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 0
BOOTSTRAP_LEVEL = 0.95
CALIBRATION_BINS = 10
SEEDS = [0, 1, 2]
XGB_GRID: dict[str, Any] = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators_max": 2000,
    "early_stopping_rounds": 100,
}


def build_analysis_plan(
    *,
    selected_configs: dict[str, Any],
    known_propensity: float,
    data_manifest_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the frozen plan payload."""
    plan: dict[str, Any] = {
        "protocol_version": "uplift-benchmark-v1",
        "primary_outcome": PRIMARY_OUTCOME,
        "estimands": ESTIMANDS,
        "estimand_note": (
            "risk_difference is in percentage points; relative_lift is a "
            "relative percent change vs control. They are not interchangeable."
        ),
        "targeting_budgets": TARGETING_BUDGETS,
        "metric_definitions": {
            "auuc": "normalized cumulative gain (area under uplift curve)",
            "qini": "unnormalized excess area (Qini coefficient)",
            "uplift_at_k": "treated-minus-control conversion in top-k by score",
            "policy_value": "IPW and DR policy value using the known propensity",
            "calibration": "equal-count bins of predicted uplift vs observed "
            "treated-minus-control rate",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": BOOTSTRAP_LEVEL,
            "method": "paired Poisson(1) weights, percentile intervals",
        },
        "calibration_bins": CALIBRATION_BINS,
        "seeds": SEEDS,
        "xgboost_grid": XGB_GRID,
        "known_propensity": known_propensity,
        "selected_configs": selected_configs,
        "data_manifest_sha256": data_manifest_sha256,
    }
    if extra:
        plan.update(extra)
    return plan


def plan_sha256(plan: dict[str, Any]) -> str:
    """Stable SHA-256 of the canonical plan JSON."""
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def write_analysis_plan(plan: dict[str, Any], path: Path) -> str:
    """Write the plan JSON and return its sha256 plus write epoch (ns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_sha256(plan)


def now_epoch_ns() -> int:
    """Current time in nanoseconds for strict ordering checks."""
    return time.time_ns()
