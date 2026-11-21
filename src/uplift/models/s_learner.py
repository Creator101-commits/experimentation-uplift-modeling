"""S-Learner (Single Model) meta-learner."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from uplift.models.base import BaseUpliftLearner, fit_binary_model, positive_probability


class SLearner(BaseUpliftLearner):
    """S-Learner meta-estimator using treatment as a feature.

    Parameters
    ----------
    estimator_type : {"logistic", "gradient_boosting", "xgboost"}, default="logistic"
        Type of underlying classification model to fit.
    random_state : int, default=42
        Random seed for model training.
    xgb_params : dict or None, default=None
        Optional XGBoost hyperparameter overrides when estimator_type="xgboost".
    xgb_early_stopping : bool, default=False
        When True and estimator_type="xgboost", early-stop on a held-out
        10% slice of each fit call (outcome log loss, 100 rounds).
    """

    def __init__(
        self,
        estimator_type: Literal["logistic", "gradient_boosting", "xgboost"] = "logistic",
        random_state: int = 42,
        xgb_params: dict[str, Any] | None = None,
        xgb_early_stopping: bool = False,
    ) -> None:
        super().__init__()
        self.estimator_type = estimator_type
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.xgb_early_stopping = xgb_early_stopping

        if estimator_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif estimator_type == "gradient_boosting":
            self.model = HistGradientBoostingClassifier(random_state=random_state)
        elif estimator_type == "xgboost":
            from uplift.models.xgboost import make_early_stop_xgb, make_xgb_classifier

            factory = make_early_stop_xgb if xgb_early_stopping else make_xgb_classifier
            self.model = factory(random_state, **(xgb_params or {}))
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
        self.model = fit_binary_model(self.model, X_augmented, y_arr)
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

        mu0 = positive_probability(self.model, X_control)
        mu1 = positive_probability(self.model, X_treated)

        return mu0, mu1
