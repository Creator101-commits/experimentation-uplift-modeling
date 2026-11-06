"""S-Learner (Single Model) meta-learner."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from uplift.models.base import BaseUpliftLearner


class SLearner(BaseUpliftLearner):
    """S-Learner meta-estimator using treatment as a feature.

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
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif estimator_type == "gradient_boosting":
            self.model = HistGradientBoostingClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> SLearner:
        """Fit single model on concatenated features and treatment indicator."""
        X_arr, W_arr, y_arr = self._prepare_inputs(X, treatment, y)
        assert W_arr is not None and y_arr is not None

        # Augment features with treatment indicator column
        X_augmented = np.column_stack([X_arr, W_arr])
        self.model.fit(X_augmented, y_arr)
        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate mu0(x) with treatment=0 and mu1(x) with treatment=1."""
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        n = len(X_arr)

        X_control = np.column_stack([X_arr, np.zeros(n, dtype=np.float64)])
        X_treated = np.column_stack([X_arr, np.ones(n, dtype=np.float64)])

        mu0 = self.model.predict_proba(X_control)[:, 1]
        mu1 = self.model.predict_proba(X_treated)[:, 1]

        return mu0, mu1
