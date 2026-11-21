"""End-to-end uplift benchmark on the Criteo partitions.

Pipeline: load typed Parquet partitions, screen candidate learners on the
validation partition only, freeze the analysis plan, then evaluate the
selected models and baselines exactly once on the frozen test partition.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from uplift.data.schema import FEATURE_COLUMNS
from uplift.evaluation.calibration import compute_uplift_calibration
from uplift.evaluation.policy_value import (
    budget_policy_values,
    constant_policy_values,
)
from uplift.evaluation.uplift_metrics import (
    auuc_score,
    boundary_areas,
    build_sorted_prefix,
    resampled_curve,
    uplift_at_cutoff,
)
from uplift.evaluation.weighted_bootstrap import paired_poisson_bootstrap
from uplift.experimentation.ab_test import ABTestAnalyzer
from uplift.experimentation.analysis_plan import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_BINS,
    TARGETING_BUDGETS,
    XGB_GRID,
    build_analysis_plan,
    now_epoch_ns,
    write_analysis_plan,
)
from uplift.experimentation.power import PowerCalculator
from uplift.models.baselines import (
    OutcomeOnlyLearner,
    RandomPolicy,
    TreatAllPolicy,
    TreatNonePolicy,
)
from uplift.models.doubly_robust import DoublyRobustLearner
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner

BUDGETS = TARGETING_BUDGETS
N_BINS = CALIBRATION_BINS
STOCHASTIC_SEEDS = (0, 1, 2)
# Learners with a genuine uplift estimate: ranking metrics, uplift
# calibration, and outcome-model scores all apply to these families.
UPLIFT_FAMILIES = {
    "s_logistic", "t_logistic", "s_xgb", "t_xgb", "dr_logistic", "dr_xgb",
}


def _load_partition(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X float64, W int, y int) for one Parquet partition."""
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS + ["treatment", "conversion"])
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    W = df["treatment"].to_numpy(dtype=np.int64)
    y = df["conversion"].to_numpy(dtype=np.int64)
    return X, W, y


def _outcome_model_metrics(
    y: np.ndarray, W: np.ndarray, mu0: np.ndarray, mu1: np.ndarray
) -> dict[str, float | None]:
    """Log loss and Brier for mu0 on controls and mu1 on treated rows."""
    eps = 1e-15
    out: dict[str, float | None] = {}
    for arm, mu in ((0, mu0), (1, mu1)):
        mask = arm == W
        yy = y[mask]
        pp = np.clip(np.asarray(mu, dtype=np.float64)[mask], eps, 1.0 - eps)
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            out[f"mu{arm}_logloss"] = None
            out[f"mu{arm}_brier"] = None
        else:
            out[f"mu{arm}_logloss"] = float(log_loss(yy, pp))
            out[f"mu{arm}_brier"] = float(brier_score_loss(yy, pp))
    return out


@dataclass
class Candidate:
    """A model candidate: a name plus a factory producing a fitted learner.

    ``kind`` is ``"ranking"`` for score-based models or ``"const_policy"``
    for the degenerate treat-none/treat-all policies, which have no
    ranking and therefore no AUUC/Qini/uplift@k or calibration.
    ``seeds`` lists the stochastic seeds to evaluate; ``None`` uses the
    learner default.
    """

    name: str
    factory: Callable[[int | None], Any]
    family: str
    kind: str = "ranking"
    seeds: tuple[int | None, ...] = (None,)


def _seeded(seed: int | None, default: int = 42) -> int:
    return default if seed is None else seed


def _xgb_params(depth: int, lr: float) -> dict[str, Any]:
    return {
        "max_depth": depth,
        "learning_rate": lr,
        "n_estimators": int(XGB_GRID["n_estimators_max"]),
    }


def _s_xgb_factory(params: dict[str, Any]) -> Callable[[int | None], SLearner]:
    def make(seed: int | None) -> SLearner:
        return SLearner(
            "xgboost",
            random_state=_seeded(seed),
            xgb_params=params,
            xgb_early_stopping=True,
        )

    return make


def _t_xgb_factory(params: dict[str, Any]) -> Callable[[int | None], TLearner]:
    def make(seed: int | None) -> TLearner:
        return TLearner(
            "xgboost",
            random_state=_seeded(seed),
            xgb_params=params,
            xgb_early_stopping=True,
        )

    return make


def _dr_xgb_factory(
    params: dict[str, Any], propensity: float
) -> Callable[[int | None], DoublyRobustLearner]:
    def make(seed: int | None) -> DoublyRobustLearner:
        return DoublyRobustLearner(
            known_propensity=propensity,
            random_state=_seeded(seed),
            outcome_estimator="xgboost",
            xgb_params=params,
            xgb_early_stopping=True,
        )

    return make


def build_candidates(propensity: float) -> tuple[list[Candidate], list[Candidate]]:
    """Return (fixed_candidates, xgb_grid_specs) for validation screening."""
    fixed = [
        Candidate(
            "treat_none",
            lambda seed: TreatNonePolicy(),
            "baseline",
            kind="const_policy",
        ),
        Candidate(
            "treat_all",
            lambda seed: TreatAllPolicy(),
            "baseline",
            kind="const_policy",
        ),
        Candidate(
            "random",
            lambda seed: RandomPolicy(random_state=seed if seed is not None else 0),
            "baseline",
            seeds=STOCHASTIC_SEEDS,
        ),
        Candidate(
            "outcome_only",
            lambda seed: OutcomeOnlyLearner(),
            "baseline",
        ),
        Candidate(
            "s_learner_logistic",
            lambda seed: SLearner("logistic", random_state=_seeded(seed)),
            "s_logistic",
        ),
        Candidate(
            "t_learner_logistic",
            lambda seed: TLearner("logistic", random_state=_seeded(seed)),
            "t_logistic",
        ),
        Candidate(
            "dr_logistic",
            lambda seed: DoublyRobustLearner(
                known_propensity=propensity,
                random_state=_seeded(seed),
                outcome_estimator="logistic",
            ),
            "dr_logistic",
            seeds=STOCHASTIC_SEEDS,
        ),
    ]
    grid = []
    for depth in XGB_GRID["max_depth"]:
        for lr in XGB_GRID["learning_rate"]:
            params = _xgb_params(int(depth), float(lr))
            grid.append(
                Candidate(
                    f"s_learner_xgb_d{depth}_lr{lr}",
                    _s_xgb_factory(params),
                    "s_xgb",
                    seeds=STOCHASTIC_SEEDS,
                )
            )
            grid.append(
                Candidate(
                    f"t_learner_xgb_d{depth}_lr{lr}",
                    _t_xgb_factory(params),
                    "t_xgb",
                    seeds=STOCHASTIC_SEEDS,
                )
            )
    return fixed, grid


def _select_xgb(
    grid: list[Candidate],
    X_tr: np.ndarray,
    W_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    W_va: np.ndarray,
    y_va: np.ndarray,
    propensity: float,
) -> tuple[dict[str, Candidate], pd.DataFrame]:
    """Fit each grid candidate on train, score on validation, keep best/family."""
    rows: list[dict[str, Any]] = []
    best: dict[str, tuple[float, Candidate]] = {}
    for cand in grid:
        t0 = time.time()
        # Screening uses the first declared stochastic seed.
        model = cand.factory(cand.seeds[0])
        model.fit(X_tr, W_tr, y_tr)
        tau = model.predict_uplift(X_va)
        val_auuc = float(auuc_score(y_va, W_va, tau))
        elapsed = time.time() - t0
        rows.append(
            {
                "name": cand.name,
                "family": cand.family,
                "seed": cand.seeds[0],
                "val_auuc": val_auuc,
                "fit_seconds": elapsed,
            }
        )
        fam = cand.family
        if fam not in best or val_auuc > best[fam][0]:
            best[fam] = (val_auuc, cand)
    sel = pd.DataFrame(rows).sort_values(["family", "val_auuc"], ascending=[True, False])
    return {fam: c for fam, (_, c) in best.items()}, sel


def run_ab_report(
    W_all: np.ndarray,
    y_all: np.ndarray,
    *,
    train_baseline_rate: float,
    train_n_control: int,
    train_n_treated: int,
) -> dict[str, Any]:
    """A/B analysis on the full dataset plus a train-based power/MDE table."""
    ab = ABTestAnalyzer.analyze(W_all, y_all)
    control_rate = float(np.mean(y_all[W_all == 0]))
    treated_rate = float(np.mean(y_all[W_all == 1]))
    n_c = int(np.sum(W_all == 0))
    n_t = int(np.sum(W_all == 1))
    power = PowerCalculator.calculate_power(
        n_control=train_n_control,
        n_treated=train_n_treated,
        baseline_rate=train_baseline_rate,
        mde_absolute=ab.absolute_lift,
    )
    return {
        "n_control": n_c,
        "n_treated": n_t,
        "control_rate": control_rate,
        "treated_rate": treated_rate,
        "absolute_lift": ab.absolute_lift,
        "absolute_lift_ci": [ab.ci_lower, ab.ci_upper],
        "relative_lift": ab.relative_lift,
        "p_value": ab.p_value,
        "number_needed_to_treat": ab.number_needed_to_treat,
        "power_at_observed_effect": {
            "power": power,
            "baseline_rate": train_baseline_rate,
            "n_control": train_n_control,
            "n_treated": train_n_treated,
            "mde_absolute": ab.absolute_lift,
        },
    }


def run_benchmark(
    data_dir: Path,
    out_dir: Path,
    *,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    boot_seed: int = BOOTSTRAP_SEED,
    do_bootstrap: bool = True,
) -> dict[str, Any]:
    """Run the full benchmark; write artifacts under ``out_dir``.

    ``max_*`` cap partition sizes for the smoke run; ``None`` uses all rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = out_dir / "selection"
    sel_dir.mkdir(exist_ok=True)
    t_start = time.time()
    timings: dict[str, float] = {}

    # ---- Load partitions -------------------------------------------------
    t0 = time.time()
    X_tr, W_tr, y_tr = _load_partition(data_dir / "train.parquet")
    X_va, W_va, y_va = _load_partition(data_dir / "validation.parquet")
    X_te, W_te, y_te = _load_partition(data_dir / "test.parquet")
    if max_train is not None:
        X_tr, W_tr, y_tr = X_tr[:max_train], W_tr[:max_train], y_tr[:max_train]
    if max_val is not None:
        X_va, W_va, y_va = X_va[:max_val], W_va[:max_val], y_va[:max_val]
    if max_test is not None:
        X_te, W_te, y_te = X_te[:max_test], W_te[:max_test], y_te[:max_test]
    timings["load_seconds"] = time.time() - t0

    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    propensity = float(data_manifest["known_propensity"])
    data_manifest_sha = hashlib.sha256(
        json.dumps(data_manifest, sort_keys=True).encode()
    ).hexdigest()

    # ---- Validation screening -------------------------------------------
    fixed, grid = build_candidates(propensity)
    t0 = time.time()
    selected_xgb, sel_df = _select_xgb(
        grid, X_tr, W_tr, y_tr, X_va, W_va, y_va, propensity
    )
    timings["validation_seconds"] = time.time() - t0
    sel_df.to_csv(sel_dir / "xgboost_selection.csv", index=False)

    # Fixed candidates also get a validation AUUC for the selection table.
    fixed_rows: list[dict[str, Any]] = []
    for cand in fixed:
        model = cand.factory(cand.seeds[0])
        model.fit(X_tr, W_tr, y_tr)
        tau = model.predict_uplift(X_va)
        fixed_rows.append(
            {"name": cand.name, "family": cand.family,
             "val_auuc": float(auuc_score(y_va, W_va, tau))}
        )
    pd.DataFrame(fixed_rows).to_csv(sel_dir / "fixed_selection.csv", index=False)

    # ---- Freeze analysis plan BEFORE any test read ----------------------
    selected_configs: dict[str, Any] = {
        fam: {"name": c.name} for fam, c in selected_xgb.items()
    }
    for r in fixed_rows:
        selected_configs[str(r["name"])] = {"name": str(r["name"])}
    plan = build_analysis_plan(
        selected_configs=selected_configs,
        known_propensity=propensity,
        data_manifest_sha256=data_manifest_sha,
    )
    plan_path = out_dir / "analysis_plan.json"
    p_sha = write_analysis_plan(plan, plan_path)
    card_write_epoch = now_epoch_ns()

    # ---- Full dataset for the A/B report (after plan freeze) -------------
    t0 = time.time()
    W_all = np.concatenate([W_tr, W_va, W_te])
    y_all = np.concatenate([y_tr, y_va, y_te])
    train_baseline = float(np.mean(y_tr[W_tr == 0])) if np.any(W_tr == 0) else 0.0
    ab_report = run_ab_report(
        W_all, y_all,
        train_baseline_rate=train_baseline,
        train_n_control=int(np.sum(W_tr == 0)),
        train_n_treated=int(np.sum(W_tr == 1)),
    )
    timings["ab_report_seconds"] = time.time() - t0

    # ---- Frozen test evaluation (exactly once) ---------------------------
    first_test_read_epoch = now_epoch_ns()
    candidates: list[Candidate] = list(fixed)
    # Add the selected XGBoost learners and DR-xgboost variant.
    for cand in selected_xgb.values():
        candidates.append(cand)
    # DR with the selected xgboost outcome config (best t_xgb config params).
    t_xgb = selected_xgb.get("t_xgb")
    if t_xgb is not None:
        # Reuse the same hyperparameters for DR nuisance models.
        m = re.search(r"d(\d+)_lr([\d.]+)", t_xgb.name)
        if m:
            params = _xgb_params(int(m.group(1)), float(m.group(2)))
            candidates.append(
                Candidate(
                    "dr_xgboost",
                    _dr_xgb_factory(params, propensity),
                    "dr_xgb",
                    seeds=STOCHASTIC_SEEDS,
                )
            )

    y_te_f = y_te.astype(np.float64)
    W_te_f = W_te.astype(np.float64)
    has_both_arms = bool(np.any(W_tr == 0) and np.any(W_tr == 1))
    train_ate = (
        float(np.mean(y_tr[W_tr == 1]) - np.mean(y_tr[W_tr == 0]))
        if has_both_arms
        else 0.0
    )
    const_vals = constant_policy_values(y_te, W_te, propensity, train_ate)

    test_rows: list[dict[str, Any]] = []
    curves_out: dict[str, dict[str, Any]] = {}
    cal_out: dict[str, dict[str, Any]] = {}
    scores_seed0: dict[str, np.ndarray] = {}
    policy_vectors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    tau_seed0: dict[str, np.ndarray] = {}
    t_xgb_model_seed0: Any = None
    t0 = time.time()
    for cand in candidates:
        for seed in cand.seeds:
            model = cand.factory(seed)
            t_fit = time.time()
            model.fit(X_tr, W_tr, y_tr)
            tau = np.asarray(model.predict_uplift(X_te), dtype=np.float64)
            mu = model.predict_outcomes(X_te)
            elapsed = time.time() - t_fit
            row: dict[str, Any] = {
                "name": cand.name,
                "family": cand.family,
                "seed": seed,
                "runtime_seconds": float(elapsed),
                "model_size_bytes": int(len(pickle.dumps(model))),
            }
            if cand.kind == "const_policy":
                vals = const_vals[cand.name]
                for est in ("ipw", "dr"):
                    for k in BUDGETS:
                        tag = f"{int(round(k * 100))}"
                        row[f"{est}_policy_value_{tag}"] = vals[est]
            else:
                sp = build_sorted_prefix(y_te, W_te, tau)
                auuc, qini = boundary_areas(sp)
                row["auuc"] = float(auuc)
                row["qini"] = float(qini)
                for k in BUDGETS:
                    tag = f"{int(round(k * 100))}"
                    row[f"uplift_at_{tag}"] = float(uplift_at_cutoff(sp, k))
                row.update(
                    budget_policy_values(
                        y_te, W_te, tau, BUDGETS, propensity,
                        mu0=mu[0], mu1=mu[1], prefix=sp,
                    )
                )
                fracs, ug, qq = resampled_curve(sp, n_bins=100)
                curves_out.setdefault(cand.name, {})[str(seed)] = {
                    "fracs": fracs.tolist(),
                    "uplift_gain": ug.tolist(),
                    "qini": qq.tolist(),
                }
                if cand.family in UPLIFT_FAMILIES:
                    cal = compute_uplift_calibration(y_te, W_te, tau, n_bins=N_BINS)
                    cal_out.setdefault(cand.name, {})[str(seed)] = [
                        asdict(b) for b in cal
                    ]
                    row["calibration_mean_gap"] = float(
                        np.mean([b.calibration_gap for b in cal]) if cal else 0.0
                    )
                    row.update(_outcome_model_metrics(y_te, W_te, mu[0], mu[1]))
            test_rows.append(row)
            if seed == cand.seeds[0] and cand.kind == "ranking":
                scores_seed0[cand.name] = tau
                tau_seed0[cand.name] = tau
                mu0, mu1 = mu
                e = float(propensity)
                policy_vectors[cand.name] = {
                    "ipw": (W_te_f * y_te_f / e,
                            (1.0 - W_te_f) * y_te_f / (1.0 - e)),
                    "dr": (
                        mu1 + W_te_f * (y_te_f - mu1) / e,
                        mu0 + (1.0 - W_te_f) * (y_te_f - mu0) / (1.0 - e),
                    ),
                }
                if cand.family == "t_xgb" and t_xgb_model_seed0 is None:
                    t_xgb_model_seed0 = model
    timings["test_eval_seconds"] = time.time() - t0
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(out_dir / "test_metrics.csv", index=False)

    metric_keys = sorted(
        {k for r in test_rows for k, v in r.items()
         if k not in ("name", "family", "seed")
         and isinstance(v, (int, float)) and not isinstance(v, bool)}
    )
    seed_summary: dict[str, Any] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in test_rows:
        rows_by_name[r["name"]].append(r)
    for name, rows in rows_by_name.items():
        summary: dict[str, Any] = {}
        for key in metric_keys:
            num_vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if not num_vals:
                continue
            arr = np.asarray(num_vals, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_seeds": len(rows),
            }
        seed_summary[name] = summary

    # Random-targeting reference line for the Qini panel: the expected
    # curve under random ordering is linear to the full-population Qini.
    ref_sp = build_sorted_prefix(y_te, W_te, np.zeros(len(y_te)))
    _, _, ref_qini_curve = resampled_curve(ref_sp, n_bins=100)
    random_line = {
        "fracs": np.linspace(0.0, 1.0, 101).tolist(),
        "qini": ref_qini_curve.tolist(),
    }

    # Failure analysis: T-XGB gain importances pick the top-3 features.
    top_features = _top_gain_features(t_xgb_model_seed0)
    feature_slices = _feature_slices(
        X_te, y_te, W_te, tau_seed0, top_features
    ) if top_features else {}

    # ---- Paired bootstrap CIs -------------------------------------------
    boot: dict[str, Any] = {"enabled": do_bootstrap, "resamples": n_boot, "seed": boot_seed}
    if do_bootstrap and scores_seed0:
        t0 = time.time()
        e = float(propensity)
        a_vec = W_te_f * y_te_f / e
        b_vec = (1.0 - W_te_f) * y_te_f / (1.0 - e)
        ta_dr_vec = train_ate + W_te_f * (y_te_f - train_ate) / e
        const_series = {
            "policy_treat_none": {
                **{f"ipw_policy_value_{t}": b_vec for t in ("10", "20", "30")},
                **{f"dr_policy_value_{t}": b_vec for t in ("10", "20", "30")},
            },
            "policy_treat_all": {
                **{f"ipw_policy_value_{t}": a_vec for t in ("10", "20", "30")},
                **{f"dr_policy_value_{t}": ta_dr_vec for t in ("10", "20", "30")},
            },
        }
        result = paired_poisson_bootstrap(
            y_te, W_te, scores_seed0, n_boot=n_boot, seed=boot_seed,
            policy_vectors=policy_vectors, const_series=const_series,
        )
        boot["models"] = {}
        for name in list(scores_seed0) + ["policy_treat_none", "policy_treat_all"]:
            entry: dict[str, Any] = {"point": result.point[name], "ci": {}}
            for metric in result.point[name]:
                if metric in result.metrics[name]:
                    lo, hi = result.ci(metric, name)
                    entry["ci"][metric] = [lo, hi]
            boot["models"][name] = entry
        # Paired differences vs the best validation baseline (outcome_only)
        # and vs the logistic T-learner; bootstrap uses seed-0 scores.
        refs = [n for n in ("outcome_only", "t_learner_logistic") if n in scores_seed0]
        boot["paired_diffs"] = {}
        for name in scores_seed0:
            entry_diffs: dict[str, Any] = {}
            for ref in refs:
                if name == ref:
                    continue
                for metric in ("auuc", "qini"):
                    mean, lo, hi = result.paired_diff(metric, name, ref)
                    entry_diffs[f"{name}_minus_{ref}_{metric}"] = {
                        "mean": mean, "ci": [lo, hi]
                    }
            boot["paired_diffs"][name] = entry_diffs
        # Policy gains vs the constant treat-all value and vs random.
        boot["policy_gains"] = {}
        for name in scores_seed0:
            gains: dict[str, Any] = {}
            for est in ("ipw", "dr"):
                for k in BUDGETS:
                    tag = f"{int(round(k * 100))}"
                    metric = f"{est}_policy_value_{tag}"
                    for gain_ref, label in (
                        ("policy_treat_all", "treat_all"),
                        ("random", "random"),
                    ):
                        if name == gain_ref:
                            continue
                        mean, lo, hi = result.paired_diff(metric, name, gain_ref)
                        gains[f"{est}_gain_vs_{label}_{tag}"] = {
                            "mean": mean, "ci": [lo, hi]
                        }
            boot["policy_gains"][name] = gains
        timings["bootstrap_seconds"] = time.time() - t0

    # ---- Manifest ---------------------------------------------------------
    ordering_ok = card_write_epoch < first_test_read_epoch
    manifest = {
        "git_sha": _git_sha(),
        "protocol": plan["protocol_version"],
        "analysis_plan_sha256": p_sha,
        "card_write_epoch": card_write_epoch,
        "first_test_read_epoch": first_test_read_epoch,
        "ordering_ok": bool(ordering_ok),
        "known_propensity": propensity,
        "timings": timings,
        "total_seconds": time.time() - t_start,
        "partition_sizes": {
            "train": int(len(y_tr)), "validation": int(len(y_va)), "test": int(len(y_te)),
        },
        "xgboost_device": _xgb_device(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metrics = {
        "analysis_plan": plan,
        "ab_report": ab_report,
        "test_metrics": test_rows,
        "seed_summary": seed_summary,
        "curves": curves_out,
        "random_qini_line": random_line,
        "calibration_bins": cal_out,
        "failure_analysis": {
            "uplift_deciles": cal_out,
            "top_t_xgb_features": top_features,
            "feature_slices": feature_slices,
        },
        "bootstrap": boot,
        "data_manifest": data_manifest,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    _write_report(out_dir, manifest, test_df, ab_report, sel_df)
    return manifest


def _xgb_device() -> str:
    try:
        from uplift.models.xgboost import _cuda_available

        return "cuda" if _cuda_available() else "cpu"
    except Exception:
        return "unknown"


def _git_sha() -> str:
    import os
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return os.environ.get("GIT_SHA", "unknown")


def _xgb_gain_scores(model: Any) -> dict[str, float]:
    """Summed split-gain importances over a learner's arm models."""
    total: dict[str, float] = {}
    for attr in ("model_treated", "model_control", "model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        inner = getattr(m, "inner", m)
        get_booster = getattr(inner, "get_booster", None)
        if get_booster is None:
            continue
        try:
            scores = get_booster().get_score(importance_type="gain")
        except Exception:
            continue
        for key, val in scores.items():
            total[key] = total.get(key, 0.0) + float(val)
    return total


def _top_gain_features(model: Any, n: int = 3) -> list[str]:
    """Top-``n`` feature names by summed T-learner XGBoost gain."""
    if model is None:
        return []
    scores = _xgb_gain_scores(model)
    if not scores:
        return []
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def _feature_slices(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    tau_by_model: dict[str, np.ndarray],
    features: list[str],
    n_slices: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Equal-count quantile slices per feature with arm rates and uplift."""
    col_index = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    out: dict[str, list[dict[str, Any]]] = {}
    for fname in features:
        j = col_index.get(fname)
        if j is None:
            continue
        order = np.argsort(X[:, j], kind="stable")
        rows: list[dict[str, Any]] = []
        for i, idx in enumerate(np.array_split(order, n_slices), start=1):
            if len(idx) == 0:
                continue
            yt = y[idx]
            wt = W[idx]
            n_t = int(np.sum(wt == 1))
            n_c = int(np.sum(wt == 0))
            p_t = float(np.mean(yt[wt == 1])) if n_t else None
            p_c = float(np.mean(yt[wt == 0])) if n_c else None
            rows.append(
                {
                    "slice": i,
                    "n": int(len(idx)),
                    "treated_rate": p_t,
                    "control_rate": p_c,
                    "observed_uplift": (
                        p_t - p_c if p_t is not None and p_c is not None else None
                    ),
                    "mean_predicted_uplift": {
                        name: float(tau[idx].mean())
                        for name, tau in tau_by_model.items()
                    },
                }
            )
        out[fname] = rows
    return out


def _write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    test_df: pd.DataFrame,
    ab: dict[str, Any],
    sel_df: pd.DataFrame,
) -> None:
    lines = [
        "# Uplift Benchmark Report",
        "",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- protocol: `{manifest['protocol']}`",
        f"- ordering_ok: `{manifest['ordering_ok']}`",
        f"- known_propensity: `{manifest['known_propensity']:.4f}`",
        "",
        "## A/B report (full dataset)",
        "",
        f"- control rate: {ab['control_rate']:.5f} (n={ab['n_control']})",
        f"- treated rate: {ab['treated_rate']:.5f} (n={ab['n_treated']})",
        f"- absolute lift: {ab['absolute_lift']:.5f} "
        f"CI [{ab['absolute_lift_ci'][0]:.5f}, {ab['absolute_lift_ci'][1]:.5f}]",
        f"- relative lift: {ab['relative_lift']:.2%}",
        f"- p-value: {ab['p_value']:.3e}",
        "",
        "## Test metrics (frozen, single read)",
        "",
        "| model | seed | AUUC | Qini | uplift@20 | ipw@20 | dr@20 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _fmt(v: Any, spec: str = ".4f") -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return format(float(v), spec)
        except (TypeError, ValueError):
            return "n/a"

    for _, r in test_df.iterrows():
        seed_label = r.get("seed")
        if seed_label is None or (
            isinstance(seed_label, float) and np.isnan(seed_label)
        ):
            seed_label = "-"
        lines.append(
            f"| {r['name']} | {seed_label} | {_fmt(r.get('auuc'))} | "
            f"{_fmt(r.get('qini'), '.2f')} | {_fmt(r.get('uplift_at_20'))} | "
            f"{_fmt(r.get('ipw_policy_value_20'), '.5f')} | "
            f"{_fmt(r.get('dr_policy_value_20'), '.5f')} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
