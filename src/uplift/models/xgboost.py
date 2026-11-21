"""XGBoost classifier factory for tree-boosted uplift learners."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

# Conservative defaults sized for ~8.4M-row training on CPU hist; the
# benchmark selects among a small declared grid on validation only.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1,
}


def make_xgb_classifier(random_state: int, **overrides: Any) -> xgb.XGBClassifier:
    """Build a seeded binary XGBClassifier with the default uplift config."""
    params = {**DEFAULT_XGB_PARAMS, **overrides, "random_state": random_state}
    return xgb.XGBClassifier(**params)


class EarlyStoppingXGBClassifier:
    """XGBClassifier wrapper that early-stops on a held-out train slice.

    The uplift learners call ``fit(X, y)`` without an eval set, so this
    wrapper carves a stratified 10% slice of each fit call for early
    stopping on outcome log loss. ``n_estimators`` should be set large
    (the benchmark uses up to 2000) so early stopping selects the
    iteration count; ``early_stopping_rounds`` defaults to 100.
    """

    def __init__(
        self,
        random_state: int,
        early_stopping_rounds: int = 100,
        **overrides: Any,
    ) -> None:
        from sklearn.model_selection import train_test_split

        self._train_test_split = train_test_split
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self._base_overrides = dict(overrides)
        es_overrides = {**overrides, "early_stopping_rounds": early_stopping_rounds}
        self.inner = make_xgb_classifier(random_state, **es_overrides)
        self.classes_: np.ndarray | None = None
        self.best_iteration_: int | None = None
        self.device_ = "cuda" if _cuda_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray) -> EarlyStoppingXGBClassifier:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        counts = np.bincount(y_arr.astype(int))
        can_holdout = (
            len(np.unique(y_arr)) == 2
            and len(y_arr) >= 20
            and int(counts.min()) >= 2
        )
        if not can_holdout:
            # Too little or too imbalanced data to hold out; fit a plain
            # classifier (no early stopping, so no eval set is required).
            self.inner = make_xgb_classifier(self.random_state, **self._base_overrides)
            self.inner.fit(X_arr, y_arr)
        else:
            try:
                X_tr, X_es, y_tr, y_es = self._train_test_split(
                    X_arr, y_arr, test_size=0.1, stratify=y_arr,
                    random_state=self.random_state,
                )
            except ValueError:
                X_tr, X_es, y_tr, y_es = self._train_test_split(
                    X_arr, y_arr, test_size=0.1, random_state=self.random_state,
                )
            self.inner.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
            self.best_iteration_ = int(getattr(self.inner, "best_iteration", -1))
        self.classes_ = np.asarray(self.inner.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.predict_proba(np.asarray(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.predict(np.asarray(X)))


def _cuda_available() -> bool:
    try:
        import xgboost as _xgb

        return bool(_xgb.build_info().get("USE_CUDA", False))
    except Exception:
        return False


def make_early_stop_xgb(
    random_state: int,
    early_stopping_rounds: int = 100,
    **overrides: Any,
) -> EarlyStoppingXGBClassifier:
    """Build a seeded early-stopping XGBoost classifier for the learners."""
    return EarlyStoppingXGBClassifier(
        random_state, early_stopping_rounds=early_stopping_rounds, **overrides
    )
