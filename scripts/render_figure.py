"""Render image.png from the committed results/ benchmark outputs.

Panels: (a) Qini curves for every learner plus the random line, (b)
cumulative uplift gain for every learner plus the random mean, (c)
uplift@k with bootstrap CIs, (d) calibration for the top-3 learners,
(e) DR policy gain vs random, (f) A/B effect on the full dataset,
(g) observed uplift by predicted-uplift decile for the best learner.

Requires Matplotlib; run with
`uv run --with matplotlib python scripts/render_figure.py`.
Set UPLIFT_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("UPLIFT_RESULTS_DIR", ROOT / "results"))
OUT = ROOT / "image.png"

BUDGETS = ("uplift_at_10", "uplift_at_20", "uplift_at_30")
BUDGET_LABELS = ("top 10%", "top 20%", "top 30%")
GAIN_KEYS_RANDOM = tuple(
    f"dr_gain_vs_random_{b.removeprefix('uplift_at_')}" for b in BUDGETS
)
LEARNER_PREFIXES = ("s_learner", "t_learner", "dr_")


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


def _model_style(name: str) -> tuple[str, str, str]:
    """(color, linestyle, bar hatch) per learner.

    Hue encodes the learner family (S/T/DR), dash and hatch encode the
    backend, so all eight series stay distinguishable inside the
    six-color palette.
    """
    if name == "random":
        return GRAY, "--", ""
    if name == "outcome_only":
        return AMBER, "-", ""
    if name.startswith("s_learner"):
        xgb = "xgb" in name
        return BLUE, ("--" if xgb else "-"), ("//" if xgb else "")
    if name.startswith("t_learner"):
        xgb = "xgb" in name
        return TEAL, ("--" if xgb else "-"), ("//" if xgb else "")
    if name.startswith("dr_"):
        xgb = "xgb" in name
        return CORAL, ("--" if xgb else "-"), ("//" if xgb else "")
    return INK, "-", ""


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


def _line_legend_handles(names: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0], [0],
            color=_model_style(n)[0],
            ls=_model_style(n)[1],
            lw=1.8,
            label=_display(n),
        )
        for n in names
    ]


def _bar_legend_handles(names: list[str]) -> list[Patch]:
    return [
        Patch(
            facecolor=_model_style(n)[0],
            hatch=_model_style(n)[2] or None,
            edgecolor="white",
            label=_display(n),
        )
        for n in names
    ]


INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150

def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color")
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)




def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    ab = metrics["ab_report"]
    boot_models = metrics["bootstrap"]["models"]
    fam = {}
    for row in metrics["test_metrics"]:
        fam.setdefault(row["name"], row["family"])
    ranking = sorted(
        [n for n in boot_models if not n.startswith("policy_")],
        key=lambda n: boot_models[n]["point"].get("auuc") or float("-inf"),
        reverse=True,
    )
    learners = [n for n in ranking if _is_learner(n)]

    fig, axes = plt.subplots(
        2, 4, figsize=(18.2, 7.6), dpi=DPI, layout="constrained"
    )
    (a1, a2, a3, a4), (a5, a6, a7, _a8) = axes

    # (a) Qini curves for every learner plus the random line.
    curves = metrics["curves"]
    rql = metrics["random_qini_line"]
    for name in ranking:
        color, ls, _h = _model_style(name)
        seeds = _seed_entries(curves, name)
        all_qini = []
        fr = None
        for payload in seeds.values():
            fr = np.asarray(payload["fracs"], dtype=float) * 100.0
            qi = np.asarray(payload["qini"], dtype=float)
            all_qini.append(qi)
            a1.plot(fr, qi, color=color, ls=ls, lw=0.8, alpha=0.30)
        if len(seeds) > 1:
            a1.plot(
                fr, np.mean(all_qini, axis=0),
                color=color, ls=ls, lw=1.9, alpha=0.95,
            )
        elif seeds:
            a1.plot(
                fr, all_qini[0], color=color, ls=ls, lw=1.9, alpha=0.95
            )
    a1.plot(
        np.asarray(rql["fracs"], dtype=float) * 100.0,
        np.asarray(rql["qini"], dtype=float),
        color=INK, lw=1.5, ls="--", label="random line",
    )
    a1.set_xlim(0, 100)
    a1.xaxis.set_major_locator(MaxNLocator(6))
    style(a1, "a", "Qini curves (frozen test)",
          xlabel="fraction of users targeted (%)", ylabel="Qini")


    # (b) Cumulative uplift gain for every learner plus the random mean.
    for name in ranking:
        color, ls, _h = _model_style(name)
        seeds = _seed_entries(curves, name)
        all_gain = []
        fr = None
        for payload in seeds.values():
            if "uplift_gain" not in payload:
                continue
            fr = np.asarray(payload["fracs"], dtype=float) * 100.0
            gn = np.asarray(payload["uplift_gain"], dtype=float)
            all_gain.append(gn)
            a2.plot(fr, gn, color=color, ls=ls, lw=0.8, alpha=0.30)
        if len(all_gain) > 1:
            a2.plot(
                fr, np.mean(all_gain, axis=0),
                color=color, ls=ls, lw=1.9, alpha=0.95,
            )
        elif all_gain:
            a2.plot(fr, all_gain[0], color=color, ls=ls, lw=1.9,
                    alpha=0.95)
    a2.set_xlim(0, 100)
    a2.xaxis.set_major_locator(MaxNLocator(6))
    style(a2, "b", "Cumulative uplift gain (frozen test)",
          xlabel="fraction of users targeted (%)", ylabel="uplift gain")


    # (c) Uplift@k with CIs for learners and random.
    panel2 = [n for n in ranking if n == "random" or _is_learner(n)]
    xs = np.arange(len(BUDGETS))
    width = 0.8 / max(len(panel2), 1)
    for i, name in enumerate(panel2):
        color, _ls, hatch = _model_style(name)
        point = boot_models[name]["point"]
        ci = boot_models[name]["ci"]
        vals = np.array([point[k] for k in BUDGETS]) * 100.0
        lo = np.array([ci[k][0] for k in BUDGETS]) * 100.0
        hi = np.array([ci[k][1] for k in BUDGETS]) * 100.0
        a3.bar(
            xs + (i - (len(panel2) - 1) / 2) * width,
            vals, width,
            color=color, alpha=0.9, hatch=hatch or None,
            edgecolor="white", linewidth=0.4,
            yerr=[vals - lo, hi - vals],
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK},
        )
    a3.set_xticks(xs)
    ci_top3 = max(
        boot_models[n]["ci"][k][1] for n in panel2 for k in BUDGETS
    ) * 100.0
    a3.set_ylim(min(0.0, -0.08), ci_top3 * 1.42)
    style(a3, "c", "Uplift@k, learners and random",
          ylabel="treated minus control rate (pp)")
    cat(a3, "x", xs, list(BUDGET_LABELS))
    a3.axhline(0.0, color=SPINE, lw=0.8)
    a3.legend(
        handles=_bar_legend_handles(panel2),
        fontsize=6.5, loc="upper right", ncol=3,
    )

    # (d) Calibration for the top-3 learners by AUUC.
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
    a4.plot([x0, x1], [x0, x1], color=SPINE, lw=1.0, ls="--")
    for name in top3:
        color, ls, _h = _model_style(name)
        xs_c, ys_c = cal_series[name]
        a4.plot(
            xs_c, ys_c, color=color, ls=ls, lw=1.5, marker="o",
            markersize=4, label=_display(name),
        )
    a4.set_xlim(x0, x1)
    a4.set_ylim(x0, x1)
    style(a4, "d", "Calibration, top-3 learners",
          xlabel="mean predicted uplift (pp)",
          ylabel="observed uplift (pp)")
    a4.legend(fontsize=7, loc="upper left", ncol=1)

    # (e) DR policy gain over the random policy at each budget.
    gains = metrics["bootstrap"]["policy_gains"]
    gain_models = [n for n in ranking if n != "random"]
    xs = np.arange(len(BUDGETS))
    width = 0.8 / max(len(gain_models), 1)
    gain_lo = 0.0
    gain_hi = 0.0
    for i, name in enumerate(gain_models):
        color, _ls, hatch = _model_style(name)
        vals, los, his = [], [], []
        for key in GAIN_KEYS_RANDOM:
            g = gains[name][key]
            vals.append(g["mean"] * 100.0)
            ci_lo = g["ci"][0] * 100.0
            ci_hi = g["ci"][1] * 100.0
            los.append(vals[-1] - ci_lo)
            his.append(ci_hi - vals[-1])
            gain_lo = min(gain_lo, ci_lo)
            gain_hi = max(gain_hi, ci_hi)
        a5.bar(
            xs + (i - (len(gain_models) - 1) / 2) * width,
            vals, width,
            color=color, alpha=0.9, hatch=hatch or None,
            edgecolor="white", linewidth=0.4,
            yerr=[los, his],
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK},
        )
    a5.set_xticks(xs)
    style(
        a5, "e",
        "DR policy gain vs random; every budgeted policy is "
        "below treat-all",
        ylabel="policy value gain (pp)",
    )
    cat(a5, "x", xs, list(BUDGET_LABELS))
    a5.axhline(0.0, color=SPINE, lw=0.8)
    span = gain_hi - min(gain_lo, 0.0)
    a5.set_ylim(
        min(gain_lo, 0.0) - 0.12 * span,
        gain_hi + 0.34 * span,
    )
    a5.legend(
        handles=_bar_legend_handles(gain_models),
        fontsize=6.5, loc="upper right", ncol=3,
    )
    # (f) A/B effect, control vs treated rate with bootstrap CIs.
    pol_none = boot_models["policy_treat_none"]
    pol_all = boot_models["policy_treat_all"]
    bars_x = np.array([0, 1])
    rates = np.array([ab["control_rate"], ab["treated_rate"]]) * 100.0
    ci_none = pol_none["ci"]["ipw_policy_value_10"]
    ci_all = pol_all["ci"]["ipw_policy_value_10"]
    lows = np.maximum(
        np.array(
            [ab["control_rate"] - ci_none[0], ab["treated_rate"] - ci_all[0]]
        ) * 100.0,
        0.0,
    )
    highs = np.maximum(
        np.array(
            [ci_none[1] - ab["control_rate"], ci_all[1] - ab["treated_rate"]]
        ) * 100.0,
        0.0,
    )
    tops = rates + highs
    a6.bar(
        bars_x, rates, width=0.55,
        color=[BLUE, AMBER], alpha=0.9,
        yerr=[lows, highs],
        error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": INK},
    )
    for xi, top in zip(bars_x, tops, strict=True):
        a6.annotate(
            f"{rates[xi]:.3f}%", (xi, top + 0.008),
            ha="center", fontsize=8, color=INK,
        )
    a6.set_xticks(bars_x)
    style(a6, "f", "A/B effect, full dataset",
          ylabel="conversion rate (%)")
    cat(a6, "x", bars_x, ["control", "treated"])
    a6.text(
        0.02, 0.97,
        (
            f"absolute lift {ab['absolute_lift'] * 100:.3f} pp "
            f"[{ab['absolute_lift_ci'][0] * 100:.3f}, "
            f"{ab['absolute_lift_ci'][1] * 100:.3f}]\n"
            f"relative lift {ab['relative_lift']:.1%}\n"
            f"p = {ab['p_value']:.3f}, "
            f"NNT ~ {ab['number_needed_to_treat']:.0f}"
        ),
        transform=a6.transAxes, fontsize=7.5, va="top", color=TICK_LABEL,
    )
    a6.margins(y=0.35)

    # (g) Per-decile observed uplift for the best learner.
    best = learners[0]
    dec = _mean_bins(
        _seed_entries(metrics["failure_analysis"]["uplift_deciles"], best)
    )
    dec_x = [b["bin_index"] for b in dec]
    dec_obs = [(b["observed_uplift"] or 0.0) * 100.0 for b in dec]
    dec_pred = [(b["mean_predicted_uplift"] or 0.0) * 100.0 for b in dec]
    bcolor = _model_style(best)[0]
    a7.bar(dec_x, dec_obs, color=bcolor, alpha=0.85, width=0.65)
    a7.plot(
        dec_x, dec_pred, "o-", color=INK, lw=1.3, markersize=4,
        label="mean predicted",
    )
    a7.axhline(0.0, color=SPINE, lw=0.8)
    a7.set_xticks(dec_x)
    style(a7, "g", f"Observed uplift by decile, {_display(best)}",
          xlabel="predicted-uplift decile (d1 lowest)",
          ylabel="observed uplift (pp)")
    cat(a7, "x", dec_x, [f"d{d}" for d in dec_x])
    a7.legend(fontsize=7.5, loc="best")

    # Shared legends for the two curve panels live in the empty cell.
    _a8.set_xticks([])
    _a8.set_yticks([])
    for _sp in _a8.spines.values():
        _sp.set_visible(False)
    _a8.set_xlim(0, 1)
    _a8.set_ylim(0, 1)
    _a8.legend(
        handles=_line_legend_handles(ranking)
        + [Line2D([0], [0], color=INK, lw=1.5, ls="--",
                  label="random line")],
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=1,
    )
    from matplotlib.legend import Legend

    handles_b = _line_legend_handles(ranking)
    leg_b = Legend(
        _a8,
        handles_b,
        [_display(n) for n in ranking],
        fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=1,
    )
    _a8.add_artist(leg_b)

    finish(fig, OUT)


if __name__ == "__main__":
    main()
