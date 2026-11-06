"""Causal contract and feature allowlist schema."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS: list[str] = [f"f{i}" for i in range(12)]
FORBIDDEN_POST_TREATMENT_COLUMNS: set[str] = {"exposure", "visit"}
TREATMENT_COLUMN: str = "treatment"
OUTCOME_COLUMN: str = "conversion"


def validate_feature_columns(columns: list[str]) -> None:
    """Validate that features contain only pre-treatment columns."""
    col_set = set(columns)
    forbidden = col_set.intersection(FORBIDDEN_POST_TREATMENT_COLUMNS)
    if forbidden:
        raise ValueError(
            f"Causal contract violation: post-treatment columns {forbidden} "
            f"cannot be used as model features."
        )


def extract_features_and_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract pre-treatment feature matrix X, treatment indicator W, and outcome y."""
    if TREATMENT_COLUMN not in df.columns:
        raise ValueError(f"Missing required treatment column: {TREATMENT_COLUMN}")
    if OUTCOME_COLUMN not in df.columns:
        raise ValueError(f"Missing required outcome column: {OUTCOME_COLUMN}")

    # Select only allowed feature columns present in dataframe
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        raise ValueError("No valid pre-treatment feature columns found in DataFrame.")

    # Validate that selected features do not contain any forbidden post-treatment columns
    validate_feature_columns(available_features)

    X = df[available_features].copy()
    W = df[TREATMENT_COLUMN].astype(int).copy()
    y = df[OUTCOME_COLUMN].astype(int).copy()

    return X, W, y
