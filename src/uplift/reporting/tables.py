"""Markdown table formatters and report generators."""

from __future__ import annotations

from uplift.evaluation.calibration import UpliftCalibrationBin
from uplift.experimentation.ab_test import ABTestResult
from uplift.experimentation.power import PowerResult


def format_ab_test_table(result: ABTestResult) -> str:
    """Format A/B test results as a markdown table."""
    ci_str = (
        f"+{result.absolute_lift * 100:.3f}% "
        f"[{result.ci_lower * 100:.3f}%, {result.ci_upper * 100:.3f}%]"
    )
    lines = [
        "| Metric | Control | Treatment | Absolute Lift (95% CI) | Relative Lift | p-value |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        (
            f"| Conversion Rate | {result.control_rate * 100:.2f}% (N={result.n_control}) | "
            f"{result.treated_rate * 100:.2f}% (N={result.n_treated}) | "
            f"{ci_str} | +{result.relative_lift * 100:.2f}% | {result.p_value:.4e} |"
        ),
    ]
    return "\n".join(lines)


def format_power_table(result: PowerResult) -> str:
    """Format power analysis results as a markdown table."""
    lines = [
        "| Parameter | Value |",
        "| :--- | :--- |",
        f"| Baseline Control Conversion Rate | {result.baseline_rate * 100:.2f}% |",
        f"| Minimum Detectable Effect (Absolute) | {result.mde_absolute * 100:.3f}% |",
        f"| Minimum Detectable Effect (Relative) | {result.mde_relative * 100:.2f}% |",
        f"| Significance Level (Alpha) | {result.alpha:.2f} |",
        f"| Target Power (1 - Beta) | {result.power * 100:.0f}% |",
        f"| Required Control Sample Size | {result.n_control:,} |",
        f"| Required Treatment Sample Size | {result.n_treated:,} |",
        f"| Total Required Sample Size | {result.total_sample_size:,} |",
    ]
    return "\n".join(lines)


def format_model_comparison_table(
    models_metrics: dict[str, dict[str, float]],
) -> str:
    """Format uplift benchmark comparison table."""
    lines = [
        "| Model | AUUC | Qini Score | Uplift@10% | Uplift@20% | Policy Value (Top 20%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for model_name, m in models_metrics.items():
        lines.append(
            f"| `{model_name}` | {m.get('auuc', 0.0):.4f} | "
            f"{m.get('qini', 0.0):.2f} | "
            f"{m.get('uplift@10%', 0.0) * 100:.2f}% | "
            f"{m.get('uplift@20%', 0.0) * 100:.2f}% | "
            f"{m.get('policy_value_20%', 0.0):.4f} |"
        )
    return "\n".join(lines)


def format_calibration_table(bins: list[UpliftCalibrationBin]) -> str:
    """Format uplift calibration deciles table."""
    lines = [
        "| Bin | N Samples | N Treated | N Control | Mean Pred CATE | Observed Lift | Gap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for b in bins:
        lines.append(
            f"| {b.bin_index} | {b.n_samples} | {b.n_treated} | {b.n_control} | "
            f"{b.mean_predicted_uplift * 100:.2f}% | {b.observed_uplift * 100:.2f}% | "
            f"{b.calibration_gap * 100:.2f}% |"
        )
    return "\n".join(lines)
