"""Reproduce image.png from the curated Kaggle benchmark outputs.

Reads results/metrics.json, results/manifest.json, and
results/data-manifest.json, writes the plotted values to
scripts/figure-data.json, and renders image.png. Requires Matplotlib;
run with `uv run --with matplotlib python scripts/render_figure.py`.
Set UPLIFT_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("UPLIFT_RESULTS_DIR", ROOT / "results"))

FAMILY_COLOR = {
    "baseline": "#64748b",
    "s_logistic": "#4169e1",
    "t_logistic": "#10a68b",
    "dr_logistic": "#d95f45",
    "s_xgb": "#8659cf",
    "t_xgb": "#e8a13a",
    "dr_xgb": "#5aa9e6",
}
BUDGETS = ("uplift_at_10", "uplift_at_20", "uplift_at_30")
BUDGET_LABELS = ("top 10%", "top 20%", "top 30%")
GAIN_KEYS_RANDOM = tuple(
    f"dr_gain_vs_random_{b.removeprefix('uplift_at_')}" for b in BUDGETS
)
LEARNER_PREFIXES = ("s_learner", "t_learner", "dr_")
LEGEND_KW = {
    "frameon": True,
    "facecolor": "white",
    "framealpha": 0.85,
    "edgecolor": "#cbd5e1",
}


def _display(name: str) -> str:
    if name == "outcome_only":
        return "outcome-only"
    if name == "random":
        return "random"
    if name.startswith("s_learner_xgb"):
        return "S-XGB"
    if name.startswith("t_learner_xgb"):
        return "T-XGB"
    if name.startswith("dr_xgb"):
        return "DR-XGB"
    if name.startswith("s_learner"):
        return "S-logistic"
    if name.startswith("t_learner"):
        return "T-logistic"
    if name.startswith("dr_"):
        return "DR-logistic"
    return name


def _is_learner(name: str) -> bool:
    return name.startswith(LEARNER_PREFIXES)


def _load() -> tuple[dict, dict, dict]:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    data_manifest = json.loads((RESULTS / "data-manifest.json").read_text())
    return metrics, manifest, data_manifest


def _family_by_name(metrics: dict) -> dict[str, str]:
    fam: dict[str, str] = {}
    for row in metrics["test_metrics"]:
        fam.setdefault(row["name"], row["family"])
    return fam


def _ranking_models(metrics: dict) -> list[str]:
    """Models with ranking metrics, ordered by AUUC point estimate."""
    models = metrics["bootstrap"]["models"]
    names = [n for n in models if not n.startswith("policy_")]
    return sorted(
        names,
        key=lambda n: models[n]["point"].get("auuc") or float("-inf"),
        reverse=True,
    )


def _seed_entries(section: dict, model: str) -> dict[str, object]:
    return section.get(model, {})


def _mean_bins(seed_bins: dict[str, list[dict]]) -> list[dict]:
    """Average bin statistics across stochastic seeds, keyed by bin_index."""
    by_index: dict[int, list[dict]] = {}
    for bins in seed_bins.values():
        for b in bins:
            by_index.setdefault(int(b["bin_index"]), []).append(b)
    out = []
    for idx in sorted(by_index):
        group = by_index[idx]
        row = {"bin_index": idx, "n_seeds": len(group)}
        for key in (
            "n_samples",
            "mean_predicted_uplift",
            "observed_uplift",
            "treated_rate",
            "control_rate",
            "calibration_gap",
        ):
            vals = [b[key] for b in group if b.get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def main() -> None:
    metrics, manifest, data_manifest = _load()
    n_test = manifest["partition_sizes"]["test"]
    ab = metrics["ab_report"]
    boot_models = metrics["bootstrap"]["models"]
    fam = _family_by_name(metrics)
    ranking = _ranking_models(metrics)
    learners = [n for n in ranking if _is_learner(n)]
    color_of = {
        n: FAMILY_COLOR.get(fam.get(n, ""), "#334155") for n in ranking
    }

    # Panel helpers shared across the grid.
    def model_color(n: str) -> str:
        return color_of[n]

    figure_data = {
        "source": f"kaggle {manifest['protocol']} sha {manifest['git_sha'][:7]}",
        "git_sha": manifest["git_sha"],
        "protocol": manifest["protocol"],
        "ordering_ok": manifest["ordering_ok"],
        "test_rows": n_test,
        "ab_report": ab,
        "models": {
            n: {
                "display": _display(n),
                "family": fam.get(n),
                "color": color_of[n],
                "auuc_point": boot_models[n]["point"].get("auuc"),
                "auuc_ci": boot_models[n]["ci"].get("auuc"),
                "qini_point": boot_models[n]["point"].get("qini"),
                "qini_ci": boot_models[n]["ci"].get("qini"),
            }
            for n in ranking
        },
        "constant_policies": {
            n: {
                "ipw_policy_value_10": boot_models[n]["point"].get(
                    "ipw_policy_value_10"
                ),
                "ipw_policy_value_10_ci": boot_models[n]["ci"].get(
                    "ipw_policy_value_10"
                ),
                "dr_policy_value_10": boot_models[n]["point"].get(
                    "dr_policy_value_10"
                ),
                "dr_policy_value_10_ci": boot_models[n]["ci"].get(
                    "dr_policy_value_10"
                ),
            }
            for n in ("policy_treat_none", "policy_treat_all")
            if n in boot_models
        },
        "policy_gains_dr_vs_random": {
            n: {
                k: metrics["bootstrap"]["policy_gains"][n][k]
                for k in GAIN_KEYS_RANDOM
                if k in metrics["bootstrap"]["policy_gains"].get(n, {})
            }
            for n in ranking
            if n != "random"
        },
        "paired_diffs": metrics["bootstrap"].get("paired_diffs", {}),
        "data_manifest": {
            "n_rows": data_manifest["n_rows"],
            "duplicate_rows": data_manifest["duplicate_rows"],
            "source_sha256": data_manifest["source_sha256"],
            "license": data_manifest["license"],
        },
        "notes": [
            "Qini panel plots every persisted per-seed curve (fracs vs qini); "
            "multi-seed learners show all seeds faintly plus the across-seed mean.",
            "Uplift@k and policy-gain panels use DR/IPW bootstrap point "
            "estimates with paired Poisson 95% CIs.",
            "Calibration and decile panels average bin statistics across "
            "stochastic seeds (seeds 0, 1, 2) per bin_index.",
            "A/B panel bars are constant-policy IPW values with known "
            "propensity; annotations use the raw-rate ab_report statistics.",
        ],
    }

    fig, axes = plt.subplots(
        2, 3, figsize=(17, 9.5), layout="constrained"
    )
    ax_qini, ax_k, ax_cal = axes[0]
    ax_gain, ax_ab, ax_dec = axes[1]

    # Panel 1: Qini curves with the random line.
    curves = metrics["curves"]
    random_line = metrics["random_qini_line"]
    for name in ranking:
        seeds = _seed_entries(curves, name)
        col = model_color(name)
        all_qini = []
        for payload in seeds.values():
            fr = np.asarray(payload["fracs"], dtype=float) * 100.0
            qi = np.asarray(payload["qini"], dtype=float)
            all_qini.append(qi)
            ax_qini.plot(
                fr,
                qi,
                color=col,
                lw=1.0 if len(seeds) > 1 else 1.7,
                alpha=0.45 if len(seeds) > 1 else 0.95,
            )
        if len(seeds) > 1:
            ax_qini.plot(
                fr,
                np.mean(all_qini, axis=0),
                color=col,
                lw=2.0,
                alpha=0.95,
            )
    ax_qini.plot(
        np.asarray(random_line["fracs"], dtype=float) * 100.0,
        np.asarray(random_line["qini"], dtype=float),
        color="#0f172a",
        lw=1.6,
        ls="--",
        label="random line",
    )
    ax_qini.set_title("Qini curve, frozen test", fontsize=10.5)
    ax_qini.set_xlabel("fraction of users targeted (%)", fontsize=8)
    ax_qini.set_ylabel("Qini", fontsize=8)
    ax_qini.legend(
        handles=[
            Line2D([0], [0], color=model_color(n), lw=2.0, label=_display(n))
            for n in ranking
        ]
        + [
            Line2D(
                [0],
                [0],
                color="#0f172a",
                lw=1.6,
                ls="--",
                label="random line",
            )
        ],
        fontsize=7,
        loc="upper left",
        ncol=2,
        **LEGEND_KW,
    )
    ax_qini.xaxis.set_major_locator(MaxNLocator(6))

    # Panel 2: uplift@k with CIs, learners and random only.
    panel2 = [n for n in ranking if n == "random" or _is_learner(n)]
    xs = np.arange(len(BUDGETS))
    width = 0.8 / max(len(panel2), 1)
    for i, name in enumerate(panel2):
        point = boot_models[name]["point"]
        ci = boot_models[name]["ci"]
        vals = np.array([point[k] for k in BUDGETS]) * 100.0
        lo = np.array([ci[k][0] for k in BUDGETS]) * 100.0
        hi = np.array([ci[k][1] for k in BUDGETS]) * 100.0
        ax_k.bar(
            xs + (i - (len(panel2) - 1) / 2) * width,
            vals,
            width,
            color=model_color(name),
            alpha=0.9,
            yerr=[vals - lo, hi - vals],
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#334155"},
        )
    ax_k.set_xticks(xs)
    ax_k.set_xticklabels(BUDGET_LABELS, fontsize=8.5)
    ax_k.set_title("Uplift@k, learners and random", fontsize=10.5)
    ax_k.set_ylabel("treated minus control rate (pp)", fontsize=8)
    ax_k.legend(
        handles=[
            Patch(facecolor=model_color(n), alpha=0.9, label=_display(n))
            for n in panel2
        ],
        fontsize=7,
        loc="upper right",
        ncol=2,
        **LEGEND_KW,
    )
    ax_k.axhline(0.0, color="#94a3b8", lw=0.8)

    # Panel 3: calibration for the top-3 learners by AUUC point.
    cal_bins = metrics["calibration_bins"]
    top3 = learners[:3]
    all_pts: list[float] = []
    cal_series = {}
    for name in top3:
        avg = _mean_bins(_seed_entries(cal_bins, name))
        xs_c = [
            b["mean_predicted_uplift"] * 100.0
            for b in avg
            if b["mean_predicted_uplift"] is not None
        ]
        ys_c = [
            b["observed_uplift"] * 100.0
            for b in avg
            if b["observed_uplift"] is not None
        ]
        cal_series[name] = (xs_c, ys_c)
        all_pts += xs_c + ys_c
    lo_d = min(all_pts)
    hi_d = max(all_pts)
    pad = 0.1 * (hi_d - lo_d) + 1e-9
    x0, x1 = lo_d - pad, hi_d + pad
    ax_cal.plot([x0, x1], [x0, x1], color="#94a3b8", lw=1.0, ls="--")
    for name in top3:
        xs_c, ys_c = cal_series[name]
        ax_cal.plot(
            xs_c,
            ys_c,
            "o-",
            color=model_color(name),
            lw=1.5,
            markersize=4.5,
            label=_display(name),
        )
    ax_cal.set_title("Calibration, top-3 learners", fontsize=10.5)
    ax_cal.set_xlabel("mean predicted uplift (pp)", fontsize=8)
    ax_cal.set_ylabel("observed uplift (pp)", fontsize=8)
    ax_cal.legend(fontsize=7.5, loc="upper left", **LEGEND_KW)
    ax_cal.set_xlim(x0, x1)
    ax_cal.set_ylim(x0, x1)

    # Panel 4: DR policy gain over the random policy at each budget.
    # Random is the reference (gain vs itself is identically zero) and is
    # excluded. All gains are plotted in percentage points.
    gains = metrics["bootstrap"]["policy_gains"]
    gain_models = [n for n in ranking if n != "random"]
    xs = np.arange(len(BUDGETS))
    width = 0.8 / max(len(gain_models), 1)
    gain_lo = 0.0
    gain_hi = 0.0
    for i, name in enumerate(gain_models):
        vals = []
        los = []
        his = []
        for key in GAIN_KEYS_RANDOM:
            g = gains[name][key]
            vals.append(g["mean"] * 100.0)
            ci_lo = g["ci"][0] * 100.0
            ci_hi = g["ci"][1] * 100.0
            los.append(vals[-1] - ci_lo)
            his.append(ci_hi - vals[-1])
            gain_lo = min(gain_lo, ci_lo)
            gain_hi = max(gain_hi, ci_hi)
        ax_gain.bar(
            xs + (i - (len(gain_models) - 1) / 2) * width,
            vals,
            width,
            color=model_color(name),
            alpha=0.9,
            yerr=[los, his],
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#334155"},
        )
    ax_gain.set_xticks(xs)
    ax_gain.set_xticklabels(BUDGET_LABELS, fontsize=8.5)
    ax_gain.set_title("DR policy gain vs random", fontsize=10.5)
    ax_gain.set_ylabel("policy value gain (pp)", fontsize=8)
    ax_gain.legend(
        handles=[
            Patch(facecolor=model_color(n), alpha=0.9, label=_display(n))
            for n in gain_models
        ],
        fontsize=6.5,
        loc="best",
        ncol=2,
        **LEGEND_KW,
    )
    ax_gain.axhline(0.0, color="#94a3b8", lw=0.8)
    ax_gain.text(
        0.02,
        0.02,
        "every budgeted policy is below treat-all: the average\n"
        "treatment effect is positive, so treating everyone\n"
        "maximizes total conversions",
        transform=ax_gain.transAxes,
        fontsize=7,
        va="bottom",
        color="#334155",
    )
    span = gain_hi - min(gain_lo, 0.0)
    pad_g = 0.15 * span + 1e-9
    ax_gain.set_ylim(min(gain_lo, 0.0) - pad_g, gain_hi + pad_g)

    # Panel 5: A/B effect, control vs treated rate with bootstrap CIs.
    pol_none = boot_models["policy_treat_none"]
    pol_all = boot_models["policy_treat_all"]
    bars_x = np.array([0, 1])
    rates = np.array([ab["control_rate"], ab["treated_rate"]]) * 100.0
    # Bar heights use the raw observed rates; error bars use the bootstrap
    # CI of the matching constant-policy IPW value (same estimand up to
    # the IPW weighting, differences are far below the CI width).
    ci_none = pol_none["ci"]["ipw_policy_value_10"]
    ci_all = pol_all["ci"]["ipw_policy_value_10"]
    lows = np.maximum(
        np.array(
            [ab["control_rate"] - ci_none[0], ab["treated_rate"] - ci_all[0]]
        )
        * 100.0,
        0.0,
    )
    highs = np.maximum(
        np.array(
            [ci_none[1] - ab["control_rate"], ci_all[1] - ab["treated_rate"]]
        )
        * 100.0,
        0.0,
    )
    tops = rates + highs
    ax_ab.bar(
        bars_x,
        rates,
        width=0.55,
        color=["#4169e1", "#e8a13a"],
        alpha=0.9,
        yerr=[lows, highs],
        error_kw={"elinewidth": 1.6, "capsize": 5, "ecolor": "#0f172a"},
    )
    for xi, top in zip(bars_x, tops, strict=True):
        ax_ab.annotate(
            f"{rates[xi]:.3f}%",
            (xi, top + 0.008),
            ha="center",
            fontsize=8.5,
            color="#334155",
        )
    ax_ab.set_xticks(bars_x)
    ax_ab.set_xticklabels(["control", "treated"], fontsize=9)
    ax_ab.set_title("A/B effect, full dataset", fontsize=10.5)
    ax_ab.set_ylabel("conversion rate (%)", fontsize=8)
    ax_ab.text(
        0.02,
        0.97,
        (
            f"absolute lift {ab['absolute_lift'] * 100:.3f} pp "
            f"[{ab['absolute_lift_ci'][0] * 100:.3f}, "
            f"{ab['absolute_lift_ci'][1] * 100:.3f}]\n"
            f"relative lift {ab['relative_lift']:.1%}\n"
            f"p = {ab['p_value']:.3f}, "
            f"NNT ~ {ab['number_needed_to_treat']:.0f}"
        ),
        transform=ax_ab.transAxes,
        fontsize=7.5,
        va="top",
        color="#334155",
    )
    ax_ab.margins(y=0.35)

    # Panel 6: per-decile observed uplift for the best learner.
    best = learners[0]
    dec = _mean_bins(
        _seed_entries(metrics["failure_analysis"]["uplift_deciles"], best)
    )
    dec_x = [b["bin_index"] for b in dec]
    dec_obs = [
        (b["observed_uplift"] or 0.0) * 100.0 for b in dec
    ]
    dec_pred = [
        (b["mean_predicted_uplift"] or 0.0) * 100.0 for b in dec
    ]
    ax_dec.bar(
        dec_x,
        dec_obs,
        color=model_color(best),
        alpha=0.85,
        width=0.65,
    )
    ax_dec.plot(
        dec_x,
        dec_pred,
        "o-",
        color="#0f172a",
        lw=1.4,
        markersize=4,
        label="mean predicted",
    )
    ax_dec.axhline(0.0, color="#94a3b8", lw=0.8)
    ax_dec.set_xticks(dec_x)
    ax_dec.set_xticklabels([f"d{d}" for d in dec_x], fontsize=8)
    ax_dec.set_xlabel("predicted-uplift decile (d1 lowest)", fontsize=8)
    ax_dec.set_title(
        f"Observed uplift by decile, {_display(best)}", fontsize=10.5
    )
    ax_dec.set_ylabel("observed uplift (pp)", fontsize=8)
    ax_dec.legend(fontsize=7.5, loc="best", **LEGEND_KW)

    # Persist the exact plotted values for provenance.
    figure_data["qini_curves"] = {
        n: {
            seed: {
                "fracs": _seed_entries(curves, n)[seed]["fracs"],
                "qini": _seed_entries(curves, n)[seed]["qini"],
            }
            for seed in _seed_entries(curves, n)
        }
        for n in ranking
        if n in curves
    }
    figure_data["random_qini_line"] = random_line
    figure_data["uplift_at_k"] = {
        n: {
            "point": {
                k: boot_models[n]["point"][k] for k in BUDGETS
            },
            "ci": {k: boot_models[n]["ci"][k] for k in BUDGETS},
        }
        for n in panel2
    }
    figure_data["calibration_top3"] = {
        n: _mean_bins(_seed_entries(cal_bins, n)) for n in top3
    }
    figure_data["deciles_best_learner"] = {"model": best, "bins": dec}

    (ROOT / "scripts" / "figure-data.json").write_text(
        json.dumps(figure_data, indent=2) + "\n"
    )

    fig.suptitle(
        f"Criteo uplift benchmark v2, frozen test (n={n_test:,}); "
        "95% paired Poisson bootstrap CIs",
        fontsize=11,
    )
    fig.savefig(ROOT / "image.png", dpi=160)
    print(f"wrote {ROOT / 'image.png'}")


if __name__ == "__main__":
    main()
