"""Base class and protocol for uplift estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd


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
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X_arr.shape}")

        W_arr = None
        if treatment is not None:
            if isinstance(treatment, pd.Series):
                W_arr = treatment.to_numpy()
            else:
                W_arr = np.asarray(treatment, dtype=int)
            if len(W_arr) != len(X_arr):
                raise ValueError("Length mismatch between X and treatment indicator.")

        y_arr = None
        if y is not None:
            y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y, dtype=int)
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
