"""Base class and protocol for uplift estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from uplift.data.schema import validate_feature_columns


class BaseUpliftLearner(ABC):
    """Abstract base class for causal uplift learners."""

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"Estimator {self.__class__.__name__} is not fitted yet. "
                f"Call 'fit' before predicting."
            )

    def _prepare_inputs(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series | None = None,
        y: np.ndarray | pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Convert inputs to validated NumPy arrays."""
        if isinstance(X, pd.DataFrame):
            validate_feature_columns(list(X.columns))
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X_arr.shape}")
        if not X_arr.size or not np.all(np.isfinite(X_arr)):
            raise ValueError("X must be nonempty and finite.")

        W_arr = None
        if treatment is not None:
            if isinstance(treatment, pd.Series):
                W_arr = treatment.to_numpy()
            else:
                W_arr = np.asarray(treatment)
            if W_arr.ndim != 1 or not np.all(np.isin(W_arr, [0, 1])):
                raise ValueError("Treatment must be a binary one-dimensional array.")
            if len(np.unique(W_arr)) != 2:
                raise ValueError("Both treatment arms must be present.")
            if len(W_arr) != len(X_arr):
                raise ValueError("Length mismatch between X and treatment indicator.")

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim != 1 or not np.all(np.isin(y_arr, [0, 1])):
                raise ValueError("Outcome must be a binary one-dimensional array.")
            if len(y_arr) != len(X_arr):
                raise ValueError("Length mismatch between X and outcome target.")

        return X_arr, W_arr, y_arr

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> Any:
        """Fit the uplift learner."""

    @abstractmethod
    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate potential outcome probabilities (mu0, mu1)."""

    def predict_uplift(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Estimate conditional average treatment effect tau(x) = mu1(x) - mu0(x)."""
        mu0, mu1 = self.predict_outcomes(X)
        return cast(np.ndarray, mu1 - mu0)


def fit_binary_model(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Use an empirical constant when an arm/fold contains only one outcome class."""
    if len(np.unique(y)) == 1:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
    return model.fit(X, y)


def positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    """Read the positive-class column, including single-class fitted models."""
    indices = np.flatnonzero(model.classes_ == 1)
    if not len(indices):
        return np.zeros(len(X), dtype=np.float64)
    return cast(np.ndarray, model.predict_proba(X)[:, indices[0]])
