"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uplift.data.schema import extract_features_and_targets
from uplift.data.split import train_val_test_split_uplift
from uplift.data.synthetic import generate_synthetic_uplift_data
from uplift.evaluation.uplift_metrics import compute_uplift_curve
from uplift.experimentation.power import PowerCalculator
from uplift.models.doubly_robust import DoublyRobustLearner
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


train, _, test = train_val_test_split_uplift(generate_synthetic_uplift_data(12000))
x_train, w_train, y_train = extract_features_and_targets(train)
x_test, w_test, y_test = extract_features_and_targets(test)
models = {
    "S-learner": SLearner(),
    "T-learner": TLearner(),
    "Doubly robust": DoublyRobustLearner(known_propensity=0.85),
}
curves = {}
for i, (name, model) in enumerate(models.items()):
    model.fit(x_train, w_train, y_train)
    tau = model.predict_uplift(x_test)
    fraction, _, qini = compute_uplift_curve(y_test, w_test, tau)
    axes[0].plot(fraction, qini, label=name, color=COLORS[i], lw=2)
    curves[name] = {"fraction": fraction.tolist(), "qini": qini.tolist()}
    if name == "Doubly robust":
        axes[1].hist(tau, bins=24, color=COLORS[i], alpha=0.85)
axes[0].plot([0, 1], [0, qini[-1]], "--", color=COLORS[4], label="Random baseline")
axes[0].set(
    xlabel="Targeted population fraction",
    ylabel="Qini incremental conversions",
    title="Held-out synthetic targeting",
)
axes[0].legend(fontsize=8)
axes[1].axvline(0, color=COLORS[4], ls="--")
axes[1].set(
    xlabel="Estimated conditional treatment effect",
    ylabel="Held-out observations",
    title="Estimated CATE, not observed ITE",
)
mde = np.linspace(0.003, 0.02, 50)
sizes = [PowerCalculator.calculate_sample_size(0.04, float(d)).total_sample_size for d in mde]
axes[2].plot(mde * 100, sizes, color=COLORS[0], lw=2)
axes[2].set(
    xlabel="Minimum detectable effect (percentage points)",
    ylabel="Required total sample size",
    title="Prospective normal approximation",
)
fig.suptitle("Uplift modeling | synthetic randomized experiment and power calculation", fontsize=16)
fig.supxlabel(
    "12,000 synthetic rows; 60/20/20 split; treatment probability 0.85; seed 42. "
    "Power: alpha 0.05, target 0.80.",
    fontsize=10,
)
payload = {
    "source": "seeded synthetic data, not Criteo",
    "sample_count": 12000,
    "curves": curves,
    "mde": mde.tolist(),
    "sample_sizes": sizes,
    "dr_cate": tau.tolist(),
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
