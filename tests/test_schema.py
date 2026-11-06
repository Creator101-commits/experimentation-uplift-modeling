"""Tests for schema, causal contract, and data generation."""

from __future__ import annotations

import pandas as pd
import pytest

from uplift.data.schema import extract_features_and_targets, validate_feature_columns
from uplift.data.split import train_val_test_split_uplift
from uplift.data.synthetic import generate_synthetic_uplift_data


def test_validate_feature_columns_rejects_post_treatment() -> None:
    # Valid pre-treatment features
    validate_feature_columns(["f0", "f1", "f2"])

    # Forbidden post-treatment
    with pytest.raises(ValueError, match="Causal contract violation"):
        validate_feature_columns(["f0", "exposure"])

    with pytest.raises(ValueError, match="Causal contract violation"):
        validate_feature_columns(["f1", "visit"])


def test_extract_features_and_targets() -> None:
    df = generate_synthetic_uplift_data(n_samples=200, random_state=42)
    assert "exposure" in df.columns
    assert "visit" in df.columns

    # Extracting features must drop exposure and visit
    X, W, y = extract_features_and_targets(df)
    assert "exposure" not in X.columns
    assert "visit" not in X.columns
    assert list(X.columns) == [f"f{i}" for i in range(12)]
    assert len(W) == 200
    assert len(y) == 200


def test_train_val_test_split_uplift() -> None:
    df = generate_synthetic_uplift_data(n_samples=500, random_state=42)
    tr, val, te = train_val_test_split_uplift(df, val_size=0.2, test_size=0.2, random_state=42)

    assert len(tr) == 300
    assert len(val) == 100
    assert len(te) == 100

    # Ensure no overlap
    assert len(set(tr.index).intersection(set(te.index))) == 0


def test_extract_missing_columns_error() -> None:
    df_missing = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Missing required treatment"):
        extract_features_and_targets(df_missing)
