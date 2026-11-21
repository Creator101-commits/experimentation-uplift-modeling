"""Reproduce image.png from the curated Kaggle benchmark outputs.

Reads results/metrics.json, results/manifest.json, and
results/data-manifest.json, writes the plotted values to
scripts/figure-data.json, and renders image.png. Requires Matplotlib;
run with `uv run --with matplotlib python scripts/render_figure.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
RESULTS = ROOT / "results"

FAMILY_COLOR = {
    "baseline": "#64748b",
    "s_logistic": "#4169e1",
    "t_logistic": "#10a68b",
    "dr_logistic": "#d95f45",
    "s_xgb": "#8659cf",
    "t_xgb": "#e8a13a",
    "dr_xgb": "#5aa9e6",
}
DISPLAY = {
    "treat_none": "none",
    "treat_all": "all",
    "random": "random",
    "outcome_only": "outcome",
    "s_learner_logistic": "S-log",
    "t_learner_logistic": "T-log",
    "dr_logistic": "DR-log",
    "s_learner_xgb_d4_lr0.1": "S-XGB",
    "t_learner_xgb_d4_lr0.05": "T-XGB",
    "dr_xgboost": "DR-XGB",
}
K_COLORS = {"uplift_at_10": "#4169e1", "uplift_at_20": "#10a68b", "uplift_at_30": "#d95f45"}
K_LABELS = {"uplift_at_10": "top 10%", "uplift_at_20": "top 20%", "uplift_at_30": "top 30%"}


def _load() -> tuple[dict, dict, dict]:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    data_manifest = json.loads((RESULTS / "data-manifest.json").read_text())
    return metrics, manifest, data_manifest


def _model_rows(metrics: dict) -> list[dict]:
    boot = metrics["bootstrap"]["models"]
    rows = []
    for row in metrics["test_metrics"]:
        name = row["name"]
        entry = {
            "name": name,
            "display": DISPLAY.get(name, name),
            "family": row["family"],
            "color": FAMILY_COLOR.get(row["family"], "#334155"),
            "auuc": row["auuc"],
            "qini": row["qini"],
            "calibration_mean_gap": row["calibration_mean_gap"],
            "uplift_at_k": {
                k: {
                    "point": row[k],
                    "ci": boot[name]["ci"][k],
                }
                for k in ("uplift_at_10", "uplift_at_20", "uplift_at_30")
            },
            "ci": {
                "auuc": boot[name]["ci"]["auuc"],
                "qini": boot[name]["ci"]["qini"],
            },
        }
        rows.append(entry)
    return rows


def _dot_panel(
    ax, rows: list[dict], key: str, title: str, fmt: str, nbins: int = 6
) -> None:
    order = sorted(rows, key=lambda r: r[key], reverse=True)
    y = np.arange(len(order))[::-1]
    for yi, row in zip(y, order, strict=True):
        lo, hi = row["ci"][key]
        point = row[key]
        ax.errorbar(
            point,
            yi,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            elinewidth=1.4,
            capsize=3,
            markersize=6,
        )
        ax.annotate(
            fmt.format(point),
            (hi, yi),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="#334155",
            va="center",
        )
    ax.axvline(0.0, color="#94a3b8", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["display"] for r in order], fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    lo_all = min(min(r["ci"][key]) for r in rows)
    hi_all = max(max(r["ci"][key]) for r in rows)
    ax.set_xlim(min(lo_all, 0.0) - 0.08 * (hi_all - min(lo_all, 0.0)), hi_all * 1.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins))


def main() -> None:
    metrics, manifest, data_manifest = _load()
    rows = _model_rows(metrics)
    n_test = manifest["partition_sizes"]["test"]
    ab = metrics["ab_report"]

    figure_data = {
        "source": f"kaggle {manifest['protocol']} sha {manifest['git_sha'][:7]}",
        "git_sha": manifest["git_sha"],
        "protocol": manifest["protocol"],
        "ordering_ok": manifest["ordering_ok"],
        "note": (
            "Per-model Qini curves and calibration bins were not persisted by the "
            "benchmark, so panels show the scalar curve summaries (Qini coefficient, "
            "AUUC) with paired Poisson bootstrap confidence intervals instead."
        ),
        "test_rows": n_test,
        "ab_report": ab,
        "models": rows,
        "paired_diffs": metrics["bootstrap"].get("paired_diffs", {}),
        "data_manifest": {
            "n_rows": data_manifest["n_rows"],
            "duplicate_rows": data_manifest["duplicate_rows"],
            "source_sha256": data_manifest["source_sha256"],
            "license": data_manifest["license"],
        },
    }
    (ROOT / "scripts" / "figure-data.json").write_text(
        json.dumps(figure_data, indent=2) + "\n"
    )

    fig, axes = plt.subplots(
        1, 5, figsize=(21, 4.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.7, 1.0, 1.1]},
    )

    _dot_panel(axes[0], rows, "qini", "Qini coefficient, frozen test", "{:.0f}")
    _dot_panel(axes[1], rows, "auuc", "AUUC, frozen test", "{:.5f}", nbins=4)
    axes[0].set_xlabel("Qini coefficient", fontsize=8)
    axes[1].set_xlabel("AUUC", fontsize=8)

    # Panel 3: uplift@k grouped bars with bootstrap CIs.
    ax = axes[2]
    order = sorted(rows, key=lambda r: r["auuc"], reverse=True)
    labels = [r["display"] for r in order]
    x = np.arange(len(order))
    width = 0.26
    for i, key in enumerate(("uplift_at_10", "uplift_at_20", "uplift_at_30")):
        offset = (i - 1) * width
        points = [r["uplift_at_k"][key]["point"] for r in order]
        lows = [p - r["uplift_at_k"][key]["ci"][0] for p, r in zip(points, order, strict=True)]
        highs = [r["uplift_at_k"][key]["ci"][1] - p for p, r in zip(points, order, strict=True)]
        ax.bar(
            x + offset,
            points,
            width,
            color=K_COLORS[key],
            alpha=0.9,
            label=K_LABELS[key],
            yerr=[lows, highs],
            error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": "#334155"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_title("Uplift@k gain, frozen test", fontsize=10.5)
    ax.set_ylabel("treated minus control rate", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # Panel 4: mean calibration gap on a log scale.
    ax = axes[3]
    order_cal = sorted(rows, key=lambda r: r["calibration_mean_gap"])
    y = np.arange(len(order_cal))
    gaps = [r["calibration_mean_gap"] for r in order_cal]
    ax.barh(
        y,
        gaps,
        color=[r["color"] for r in order_cal],
        alpha=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["display"] for r in order_cal], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_title("Mean calibration gap, 10 bins (log)", fontsize=10.5)
    ax.set_xlabel("mean |predicted - observed| uplift", fontsize=8)
    for yi, g in zip(y, gaps, strict=True):
        ax.annotate(
            f"{g:.1e}",
            (g, yi),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="#334155",
            va="center",
        )

    # Panel 5: A/B effect with CI.
    ax = axes[4]
    lift = ab["absolute_lift"]
    lo, hi = ab["absolute_lift_ci"]
    ax.bar([0], [lift * 100], width=0.55, color="#4169e1")
    ax.errorbar(
        [0],
        [lift * 100],
        yerr=[[(lift - lo) * 100], [(hi - lift) * 100]],
        fmt="none",
        ecolor="#0f172a",
        elinewidth=1.6,
        capsize=5,
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["absolute lift"], fontsize=9)
    ax.set_title("A/B effect, full dataset", fontsize=10.5)
    ax.set_ylabel("percentage points", fontsize=8)
    ax.text(
        0.02,
        0.97,
        (
            f"absolute lift {lift * 100:.3f} pp\n"
            f"95% CI [{lo * 100:.3f}, {hi * 100:.3f}]\n"
            f"relative lift {ab['relative_lift']:.1%}\n"
            f"NNT ~ {ab['number_needed_to_treat']:.0f}\n"
            f"control {ab['control_rate']:.3%} (n={ab['n_control']:,})\n"
            f"treated {ab['treated_rate']:.3%} (n={ab['n_treated']:,})"
        ),
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        color="#334155",
    )

    fig.suptitle(
        f"Criteo uplift benchmark, frozen test (n={n_test:,}); "
        "95% paired Poisson bootstrap CIs (500 resamples)",
        fontsize=11,
    )
    fig.savefig(ROOT / "image.png", dpi=160)
    print(f"wrote {ROOT / 'image.png'}")


if __name__ == "__main__":
    main()
