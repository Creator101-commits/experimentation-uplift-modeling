"""T-Learner (Two Models) meta-learner."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from uplift.models.base import BaseUpliftLearner


class TLearner(BaseUpliftLearner):
    """T-Learner meta-estimator fitting separate models for control and treatment.

    Parameters
    ----------
    estimator_type : {"logistic", "gradient_boosting"}, default="logistic"
        Type of underlying classification model to fit.
    random_state : int, default=42
        Random seed for model training.
    """

    def __init__(
        self,
        estimator_type: Literal["logistic", "gradient_boosting"] = "logistic",
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.estimator_type = estimator_type
        self.random_state = random_state

        if estimator_type == "logistic":
            self.model_control = LogisticRegression(max_iter=1000, random_state=random_state)
            self.model_treated = LogisticRegression(max_iter=1000, random_state=random_state + 1)
        elif estimator_type == "gradient_boosting":
            self.model_control = HistGradientBoostingClassifier(random_state=random_state)
            self.model_treated = HistGradientBoostingClassifier(random_state=random_state + 1)
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> TLearner:
        """Fit model_control on W=0 and model_treated on W=1."""
        X_arr, W_arr, y_arr = self._prepare_inputs(X, treatment, y)
        assert W_arr is not None and y_arr is not None

        mask_c = W_arr == 0
        mask_t = W_arr == 1

        if np.sum(mask_c) == 0 or np.sum(mask_t) == 0:
            raise ValueError("Both treatment arms must have samples for T-Learner.")

        self.model_control.fit(X_arr[mask_c], y_arr[mask_c])
        self.model_treated.fit(X_arr[mask_t], y_arr[mask_t])
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate mu0(x) with control model and mu1(x) with treated model."""
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)

        mu0 = self.model_control.predict_proba(X_arr)[:, 1]
        mu1 = self.model_treated.predict_proba(X_arr)[:, 1]

        return mu0, mu1
