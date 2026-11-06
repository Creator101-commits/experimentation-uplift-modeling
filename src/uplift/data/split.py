"""Stratified partitioning by treatment and outcome."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def train_val_test_split_uplift(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data deterministically into train, validation, and test sets.

    Stratified by joint strata of (treatment, conversion).
    """
    total_eval = val_size + test_size
    if total_eval >= 1.0:
        raise ValueError("Sum of val_size and test_size must be strictly less than 1.0.")

    # Create joint strata column
    strata = (df["treatment"].astype(str) + "_" + df["conversion"].astype(str)).values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_eval, random_state=random_state)
    train_idx, eval_idx = next(sss1.split(df, strata))

    train_df = df.iloc[train_idx].copy()
    eval_df = df.iloc[eval_idx].copy()

    eval_strata = (
        eval_df["treatment"].astype(str) + "_" + eval_df["conversion"].astype(str)
    ).values
    rel_test_size = test_size / total_eval
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=random_state)
    val_rel_idx, test_rel_idx = next(sss2.split(eval_df, eval_strata))

    val_df = eval_df.iloc[val_rel_idx].copy()
    test_df = eval_df.iloc[test_rel_idx].copy()

    return train_df, val_df, test_df
