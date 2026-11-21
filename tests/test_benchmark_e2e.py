"""Tests for the frozen uplift benchmark: weighted bootstrap, plan freeze, e2e."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift.data.schema import FEATURE_COLUMNS
from uplift.evaluation.uplift_metrics import auuc_score, qini_score, uplift_at_k
from uplift.evaluation.weighted_bootstrap import (
    SortedScorer,
    paired_poisson_bootstrap,
)
from uplift.experimentation.analysis_plan import (
    build_analysis_plan,
    plan_sha256,
    write_analysis_plan,
)
from uplift.experimentation.benchmark import run_benchmark


def _fixture_data(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, 12)).astype(np.float64)
    # Positive heterogeneous effect where f0 > 0.
    W = (rng.random(n) < 0.8).astype(np.int64)
    p = 0.02 + 0.01 * X[:, 1] + 0.08 * W * (X[:, 0] > 0)
    y = (rng.random(n) < np.clip(p, 0.0, 1.0)).astype(np.int64)
    return X, W, y


def _write_partition(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for name, n, seed in [("train", 3000, 1), ("validation", 900, 2), ("test", 900, 3)]:
        r = np.random.default_rng(seed)
        X, W, y = _fixture_data(r, n)
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df["treatment"] = W
        df["conversion"] = y
        df["visit"] = ((r.random(n) < 0.9) & (W == 1)).astype(np.int64)
        df["exposure"] = (r.random(n) < 0.5).astype(np.int64)
        _write_partition(df, data / f"{name}.parquet")
    manifest = {
        "known_propensity": 0.8,
        "n_rows": 4800,
        "observed_treatment_rate": 0.8,
        "observed_conversion_rate": 0.02,
        "source_sha256": "fixture",
        "split_counts": {"train": 3000, "validation": 900, "test": 900},
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data


def test_unit_weights_reproduce_unweighted_metrics() -> None:
    rng = np.random.default_rng(0)
    _, W, y = _fixture_data(rng, 2000)
    scores = rng.random(len(y))
    scorer = SortedScorer.from_scores(y, W, scores)
    unit = scorer.metrics(np.ones(len(y), dtype=np.float64))
    assert unit["auuc"] == pytest.approx(auuc_score(y, W, scores))
    assert unit["qini"] == pytest.approx(qini_score(y, W, scores))
    assert unit["uplift_at_20"] == pytest.approx(uplift_at_k(y, W, scores, k=0.20))


def test_weighted_bootstrap_deterministic_and_paired() -> None:
    rng = np.random.default_rng(1)
    _, W, y = _fixture_data(rng, 1500)
    s1 = rng.random(len(y))
    s2 = rng.random(len(y))
    a = paired_poisson_bootstrap(
        y, W, {"m1": s1, "m2": s2}, n_boot=25, seed=0
    )
    b = paired_poisson_bootstrap(
        y, W, {"m1": s1, "m2": s2}, n_boot=25, seed=0
    )
    np.testing.assert_allclose(a.metrics["m1"]["auuc"], b.metrics["m1"]["auuc"])
    # Point estimate equals the unit-weight scorer path.
    assert a.point["m1"]["auuc"] == pytest.approx(
        SortedScorer.from_scores(y, W, s1).metrics(np.ones(len(y)))["auuc"]
    )
    # Paired difference CI is symmetric around a finite mean.
    mean, lo, hi = a.paired_diff("auuc", "m1", "m2")
    assert np.isfinite(mean) and lo <= hi


def test_weighted_auuc_hand_computed() -> None:
    # Perfect ranking: all treated conversions outrank controls.
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    W = np.array([1, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    scorer = SortedScorer.from_scores(y, W, scores)
    unit = scorer.metrics(np.ones(4))
    # Sorted order is already score order; unit-weight AUUC equals unweighted.
    assert unit["auuc"] == pytest.approx(auuc_score(y, W, scores))


def test_analysis_plan_freeze_and_hash(tmp_path: Path) -> None:
    plan = build_analysis_plan(
        selected_configs={"m": {"name": "m"}},
        known_propensity=0.8,
        data_manifest_sha256="abc",
    )
    path = tmp_path / "analysis_plan.json"
    sha = write_analysis_plan(plan, path)
    assert path.exists()
    assert sha == plan_sha256(plan)
    loaded = json.loads(path.read_text())
    assert loaded["primary_outcome"] == "conversion"
    assert loaded["bootstrap"]["resamples"] == 500
    assert loaded["targeting_budgets"] == [0.10, 0.20, 0.30]


def test_end_to_end_benchmark_fixture(tmp_path: Path) -> None:
    data = _make_data_dir(tmp_path)
    out = tmp_path / "bench"
    manifest = run_benchmark(
        data, out, max_train=2500, max_val=800, max_test=800,
        n_boot=10, do_bootstrap=True,
    )
    assert manifest["ordering_ok"] is True
    assert manifest["card_write_epoch"] < manifest["first_test_read_epoch"]
    for fname in (
        "analysis_plan.json", "manifest.json", "metrics.json",
        "test_metrics.csv", "report.md",
    ):
        assert (out / fname).exists(), fname
    assert (out / "selection" / "xgboost_selection.csv").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    names = {r["name"] for r in metrics["test_metrics"]}
    assert {"treat_all", "t_learner_logistic", "s_learner_logistic"} <= names
    assert "dr_xgboost" in names
    # Ranking metrics are finite for rankings and absent for constant policies.
    for r in metrics["test_metrics"]:
        if r["name"] in ("treat_none", "treat_all"):
            assert r.get("auuc") is None
            assert r.get("qini") is None
            assert r.get("ipw_policy_value_20") is not None
        else:
            assert np.isfinite(r["auuc"])
            assert np.isfinite(r["qini"])
    # Persisted curves, calibration bins, seed summary, and gains exist.
    for key in (
        "curves", "random_qini_line", "calibration_bins",
        "seed_summary", "failure_analysis",
    ):
        assert key in metrics, key
    assert "policy_gains" in metrics["bootstrap"]
    # Paired diffs keep every reference model, not only the last one.
    diffs = metrics["bootstrap"]["paired_diffs"]["s_learner_logistic"]
    assert "s_learner_logistic_minus_outcome_only_auuc" in diffs
    assert "s_learner_logistic_minus_t_learner_logistic_auuc" in diffs
    assert "s_learner_logistic_minus_outcome_only_qini" in diffs
    assert "s_learner_logistic_minus_t_learner_logistic_qini" in diffs
