"""Doubly Robust (AIPW) meta-learner."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from uplift.models.base import BaseUpliftLearner


class DoublyRobustLearner(BaseUpliftLearner):
    """Doubly Robust (Augmented Inverse Propensity Weighted) uplift learner.

    Combines outcome models and propensity weighting to create unbiased
    pseudo-outcomes for CATE regression.

    Parameters
    ----------
    n_splits : int, default=3
        Number of cross-fitting folds to prevent outcome-model overfitting.
    known_propensity : float or None, default=None
        Known randomized treatment assignment probability. If None, propensity
        is estimated with logistic regression.
    random_state : int, default=42
        Random seed for cross-fitting and estimators.
    """

    def __init__(
        self,
        n_splits: int = 3,
        known_propensity: float | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_splits = n_splits
        self.known_propensity = known_propensity
        self.random_state = random_state

        self.outcome_model_c = LogisticRegression(max_iter=1000, random_state=random_state)
        self.outcome_model_t = LogisticRegression(max_iter=1000, random_state=random_state + 1)
        self.effect_regressor = Ridge(alpha=1.0, random_state=random_state)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> DoublyRobustLearner:
        """Fit doubly robust learner using cross-fitted pseudo-outcomes."""
        X_arr, W_arr, y_arr = self._prepare_inputs(X, treatment, y)
        assert W_arr is not None and y_arr is not None

        n = len(X_arr)
        mu0_oof = np.zeros(n, dtype=np.float64)
        mu1_oof = np.zeros(n, dtype=np.float64)
        e_oof = np.zeros(n, dtype=np.float64)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            W_tr = W_arr[train_idx]
            y_tr = y_arr[train_idx]

            # Fit outcome models
            m_c = LogisticRegression(max_iter=1000, random_state=self.random_state)
            m_t = LogisticRegression(max_iter=1000, random_state=self.random_state + 1)

            c_mask = W_tr == 0
            t_mask = W_tr == 1

            if np.sum(c_mask) > 0:
                m_c.fit(X_tr[c_mask], y_tr[c_mask])
                mu0_oof[val_idx] = m_c.predict_proba(X_val)[:, 1]
            if np.sum(t_mask) > 0:
                m_t.fit(X_tr[t_mask], y_tr[t_mask])
                mu1_oof[val_idx] = m_t.predict_proba(X_val)[:, 1]

            # Propensity
            if self.known_propensity is not None:
                e_oof[val_idx] = self.known_propensity
            else:
                prop_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                prop_model.fit(X_tr, W_tr)
                e_oof[val_idx] = prop_model.predict_proba(X_val)[:, 1]

        # Clip propensity to avoid division by zero
        e_clipped = np.clip(e_oof, 0.01, 0.99)

        # Doubly robust pseudo-outcome (AIPW)
        # Y_dr = [mu1 + W*(Y - mu1)/e] - [mu0 + (1-W)*(Y - mu0)/(1-e)]
        dr_treated = mu1_oof + (W_arr * (y_arr - mu1_oof)) / e_clipped
        dr_control = mu0_oof + ((1 - W_arr) * (y_arr - mu0_oof)) / (1.0 - e_clipped)
        Y_dr = dr_treated - dr_control

        # Train final effect regressor on entire dataset predicting Y_dr from X
        self.effect_regressor.fit(X_arr, Y_dr)

        # Also fit final outcome models for predict_outcomes
        c_all = W_arr == 0
        t_all = W_arr == 1
        if np.sum(c_all) > 0:
            self.outcome_model_c.fit(X_arr[c_all], y_arr[c_all])
        if np.sum(t_all) > 0:
            self.outcome_model_t.fit(X_arr[t_all], y_arr[t_all])

        self.is_fitted_ = True
        return self

    def predict_outcomes(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate baseline mu0(x) and mu1(x) = mu0(x) + tau(x)."""
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        mu0 = self.outcome_model_c.predict_proba(X_arr)[:, 1]
        tau = self.effect_regressor.predict(X_arr)
        mu1 = np.clip(mu0 + tau, 0.0, 1.0)
        return mu0, mu1

    def predict_uplift(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Directly predict conditional treatment effect tau(x)."""
        self._check_is_fitted()
        X_arr, _, _ = self._prepare_inputs(X)
        return cast(np.ndarray, self.effect_regressor.predict(X_arr))
