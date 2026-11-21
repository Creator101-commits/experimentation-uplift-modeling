"""Tests for tie-aware uplift metric evaluation.

Tied scores must not inherit source row order: the cumulative curve is
evaluated only at distinct-score block boundaries, areas connect boundary
points linearly, and cutoff metrics interpolate inside straddling blocks
(the expected value under random tie-breaking).
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.evaluation.policy_value import budget_policy_values
from uplift.evaluation.uplift_metrics import (
    auuc_score,
    compute_uplift_curve,
    qini_score,
    uplift_at_k,
)
from uplift.evaluation.weighted_bootstrap import (
    SortedScorer,
    paired_poisson_bootstrap,
)


def test_constant_scores_match_random_line() -> None:
    # Conversions clustered in file order with constant scores: the old
    # stable-sort path would score this as a strong ranking.
    n = 200
    W = np.array([1] * 160 + [0] * 40, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    y[:20] = 1
    scores = np.zeros(n)

    assert qini_score(y, W, scores) == pytest.approx(0.0, abs=1e-9)
    p_t = float(np.mean(y[W == 1]))
    p_c = float(np.mean(y[W == 0]))
    # Constant scores reproduce the random-targeting triangle exactly.
    assert auuc_score(y, W, scores) == pytest.approx(0.5 * (p_t - p_c), rel=1e-9)

    fracs, ug, qq = compute_uplift_curve(y, W, scores, n_bins=100)
    # Both curves are straight lines to their endpoints.
    np.testing.assert_allclose(qq, fracs * qq[-1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ug, fracs * ug[-1], rtol=1e-9, atol=1e-12)


def test_metrics_invariant_within_tie_blocks() -> None:
    rng = np.random.default_rng(0)
    n = 300
    W = rng.binomial(1, 0.8, n)
    scores = rng.choice([0.9, 0.5, 0.2], size=n)
    y = rng.binomial(1, 0.05, n)

    # Permute rows only inside each tie block.
    groups = [np.flatnonzero(scores == s) for s in np.unique(scores)]
    perm = np.concatenate([g[rng.permutation(len(g))] for g in groups])
    y2, W2, s2 = y[perm], W[perm], scores[perm]

    for k in (0.1, 0.2, 0.3):
        assert uplift_at_k(y, W, scores, k=k) == pytest.approx(
            uplift_at_k(y2, W2, s2, k=k), rel=1e-12, abs=1e-15
        )
    assert auuc_score(y, W, scores) == pytest.approx(auuc_score(y2, W2, s2), rel=1e-12)
    assert qini_score(y, W, scores) == pytest.approx(qini_score(y2, W2, s2), rel=1e-12)
    pv1 = budget_policy_values(y, W, scores, [0.2], propensity=0.8)
    pv2 = budget_policy_values(y2, W2, s2, [0.2], propensity=0.8)
    assert pv1 == pytest.approx(pv2, rel=1e-12)


def test_hand_computed_tied_case() -> None:
    # Two tie blocks of three rows each.
    y = np.array([1, 0, 1, 0, 0, 0], dtype=np.int64)
    W = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.7, 0.7, 0.7, 0.3, 0.3, 0.3])

    # Boundary points: (0, 0), (0.5, 0.5), (1, 2/3) for the uplift curve.
    assert auuc_score(y, W, scores) == pytest.approx(5.0 / 12.0, rel=1e-12)
    # Qini points (0, 0), (0.5, 2), (1, 2): area 1.5 minus half of 2.
    assert qini_score(y, W, scores) == pytest.approx(0.5, rel=1e-12)
    # k=0.5 lands exactly on the block boundary: block A only, rates 1 vs 0.
    assert uplift_at_k(y, W, scores, k=0.5) == pytest.approx(1.0, rel=1e-12)
    # k=0.2 interpolates inside block A (phi=0.4): same composition, rates 1 vs 0.
    assert uplift_at_k(y, W, scores, k=0.2) == pytest.approx(1.0, rel=1e-12)
    # k=0.6 straddles into block B (phi=0.2): cum_t=2.2, cum_y_t=2.
    assert uplift_at_k(y, W, scores, k=0.6) == pytest.approx(2.0 / 2.2, rel=1e-12)

    # IPW policy value at k=0.5 (no straddle) equals the explicit top-3
    # assignment evaluated by the Horvitz-Thompson formula.
    pv = budget_policy_values(y, W, scores, [0.5], propensity=0.5)
    assert pv["ipw_policy_value_50"] == pytest.approx(4.0 / 6.0, rel=1e-12)
    # k=0.2 straddles block A with phi=0.4: expected IPW value is
    # (0.4 * sum_block_a) / n since every b term is zero here.
    pv2 = budget_policy_values(y, W, scores, [0.2], propensity=0.5)
    assert pv2["ipw_policy_value_20"] == pytest.approx(1.6 / 6.0, rel=1e-12)


def test_bootstrap_unit_weights_match_plain_path_with_ties() -> None:
    rng = np.random.default_rng(0)
    n = 400
    W = rng.binomial(1, 0.8, n)
    scores = rng.choice([0.9, 0.5, 0.2], size=n)
    y = rng.binomial(1, 0.05, n)
    mu0 = np.full(n, 0.02)
    mu1 = np.full(n, 0.04)

    scorer = SortedScorer.from_scores(
        y,
        W,
        scores,
        policies={
            "ipw": (W * y / 0.8, (1 - W) * y / 0.2),
            "dr": (mu1 + W * (y - mu1) / 0.8, mu0 + (1 - W) * (y - mu0) / 0.2),
        },
    )
    unit = scorer.metrics(np.ones(n, dtype=np.float64))
    assert unit["auuc"] == pytest.approx(auuc_score(y, W, scores), rel=1e-12)
    assert unit["qini"] == pytest.approx(qini_score(y, W, scores), rel=1e-12)
    assert unit["uplift_at_20"] == pytest.approx(
        uplift_at_k(y, W, scores, k=0.20), rel=1e-12
    )
    plain = budget_policy_values(
        y, W, scores, [0.1, 0.2, 0.3], propensity=0.8, mu0=mu0, mu1=mu1
    )
    for k in ("10", "20", "30"):
        assert unit[f"ipw_policy_value_{k}"] == pytest.approx(
            plain[f"ipw_policy_value_{k}"], rel=1e-12
        )
        assert unit[f"dr_policy_value_{k}"] == pytest.approx(
            plain[f"dr_policy_value_{k}"], rel=1e-12
        )


def test_bootstrap_point_includes_policy_and_const_series() -> None:
    rng = np.random.default_rng(1)
    n = 300
    W = rng.binomial(1, 0.8, n)
    scores = rng.random(n)
    y = rng.binomial(1, 0.05, n)
    e = 0.8
    a_vec = W * y / e
    b_vec = (1 - W) * y / (1 - e)
    res = paired_poisson_bootstrap(
        y,
        W,
        {"m": scores},
        n_boot=10,
        seed=0,
        policy_vectors={
            "m": {"ipw": (a_vec, b_vec), "dr": (a_vec, b_vec)}
        },
        const_series={
            "policy_treat_all": {"ipw_policy_value_20": a_vec},
            "policy_treat_none": {"ipw_policy_value_20": b_vec},
        },
    )
    # Const pseudo-entries share the same resamples: the paired diff of a
    # model's policy value against treat-all is a gain estimate.
    assert "policy_treat_all" in res.metrics
    mean, lo, hi = res.paired_diff("ipw_policy_value_20", "m", "policy_treat_all")
    assert np.isfinite(mean) and lo <= hi
    assert len(res.metrics["m"]["ipw_policy_value_20"]) == 10
    assert len(res.metrics["policy_treat_all"]["ipw_policy_value_20"]) == 10
