"""Baseline policies and outcome-only comparison models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from uplift.models.base import BaseUpliftLearner, fit_binary_model, positive_probability


class TreatNonePolicy(BaseUpliftLearner):
    """Policy that assigns zero uplift to all individuals."""

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> TreatNonePolicy:
        self._prepare_inputs(X, treatment, y)
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        zeros = np.zeros(len(X_arr), dtype=np.float64)
        return zeros, zeros


class TreatAllPolicy(BaseUpliftLearner):
    """Policy that treats everyone, assigning constant sample average treatment effect."""

    def __init__(self) -> None:
        super().__init__()
        self.ate_: float = 0.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> TreatAllPolicy:
        _, W_arr, y_arr = self._prepare_inputs(X, treatment, y)
        assert W_arr is not None and y_arr is not None
        p1 = float(np.mean(y_arr[W_arr == 1]))
        p0 = float(np.mean(y_arr[W_arr == 0]))
        self.ate_ = p1 - p0
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        zeros = np.zeros(len(X_arr), dtype=np.float64)
        return zeros, zeros + self.ate_


class RandomPolicy(BaseUpliftLearner):
    """Policy that ranks users randomly with a fixed seed."""

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> RandomPolicy:
        self._prepare_inputs(X, treatment, y)
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        rng = np.random.default_rng(self.random_state)
        scores = rng.uniform(0.0, 1.0, size=len(X_arr))
        return np.zeros(len(X_arr)), scores


class OutcomeOnlyLearner(BaseUpliftLearner):
    """Baseline targeting model that predicts conversion probability without treatment."""

    def __init__(self, max_iter: int = 1000) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> OutcomeOnlyLearner:
        X_arr, _, y_arr = self._prepare_inputs(X, treatment, y)
        assert y_arr is not None
        self.model = fit_binary_model(self.model, X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        prob = positive_probability(self.model, X_arr)
        return np.zeros(len(X_arr)), prob
