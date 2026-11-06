"""Command-line interface for experimentation and uplift modeling."""

from __future__ import annotations

from pathlib import Path

import typer

from uplift import __version__
from uplift.data.schema import extract_features_and_targets
from uplift.data.split import train_val_test_split_uplift
from uplift.data.synthetic import generate_synthetic_uplift_data
from uplift.evaluation.policy_value import compute_policy_assignment, doubly_robust_policy_value
from uplift.evaluation.uplift_metrics import auuc_score, qini_score, uplift_at_k
from uplift.experimentation.ab_test import ABTestAnalyzer
from uplift.experimentation.power import PowerCalculator
from uplift.models.baselines import OutcomeOnlyLearner, RandomPolicy, TreatAllPolicy
from uplift.models.doubly_robust import DoublyRobustLearner
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner
from uplift.reporting.tables import (
    format_ab_test_table,
    format_model_comparison_table,
    format_power_table,
)

app = typer.Typer(help="Experimentation and Uplift Modeling CLI.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def power(
    baseline_rate: float = typer.Option(0.04, help="Control conversion rate (e.g. 0.04)."),  # noqa: B008
    mde: float = typer.Option(0.005, help="Absolute minimum detectable effect (e.g. 0.005)."),  # noqa: B008
    alpha: float = typer.Option(0.05, help="Two-sided alpha."),  # noqa: B008
    target_power: float = typer.Option(0.80, help="Target power (1 - beta)."),  # noqa: B008
    treatment_ratio: float = typer.Option(0.85, help="Treatment allocation ratio."),  # noqa: B008
) -> None:
    """Run prospective power analysis and sample size determination."""
    res = PowerCalculator.calculate_sample_size(
        baseline_rate=baseline_rate,
        mde_absolute=mde,
        alpha=alpha,
        power=target_power,
        treatment_ratio=treatment_ratio,
    )
    typer.echo(format_power_table(res))


@app.command()
def ab_test(
    samples: int = typer.Option(5000, help="Number of samples to generate for A/B test."),  # noqa: B008
    alpha: float = typer.Option(0.05, help="Significance level."),  # noqa: B008
) -> None:
    """Run randomized A/B experiment evaluation and covariate balance audit."""
    df = generate_synthetic_uplift_data(n_samples=samples, random_state=42)
    res = ABTestAnalyzer.analyze(df["treatment"], df["conversion"], alpha=alpha)
    typer.echo("\nRandomized A/B Test Results:")
    typer.echo(format_ab_test_table(res))

    X, W, _ = extract_features_and_targets(df)
    smds = ABTestAnalyzer.check_covariate_balance(X, W)
    max_smd = max(smds.values())
    typer.echo(f"\nCovariate Balance: Max SMD = {max_smd:.4f} (< 0.10 indicates balanced arms)")


@app.command()
def compare(
    samples: int = typer.Option(6000, help="Sample size for benchmark comparison."),  # noqa: B008
) -> None:
    """Benchmark baselines, S-Learner, T-Learner, and Doubly Robust estimators."""
    df = generate_synthetic_uplift_data(n_samples=samples, random_state=42)
    train_df, val_df, test_df = train_val_test_split_uplift(df, random_state=42)

    X_tr, W_tr, y_tr = extract_features_and_targets(train_df)
    X_te, W_te, y_te = extract_features_and_targets(test_df)

    models = {
        "Random": RandomPolicy(),
        "TreatAll": TreatAllPolicy(),
        "OutcomeOnly": OutcomeOnlyLearner(),
        "S-Learner (Logistic)": SLearner(estimator_type="logistic"),
        "T-Learner (Logistic)": TLearner(estimator_type="logistic"),
        "Doubly Robust (AIPW)": DoublyRobustLearner(known_propensity=0.85),
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_tr, W_tr, y_tr)
        tau_pred = model.predict_uplift(X_te)
        mu0_pred, mu1_pred = model.predict_outcomes(X_te)

        actions_20 = compute_policy_assignment(tau_pred, budget_fraction=0.20)
        dr_val = doubly_robust_policy_value(
            y_te, W_te, actions_20, mu0_pred, mu1_pred, propensity=0.85
        )

        results[name] = {
            "auuc": auuc_score(y_te, W_te, tau_pred),
            "qini": qini_score(y_te, W_te, tau_pred),
            "uplift@10%": uplift_at_k(y_te, W_te, tau_pred, k=0.10),
            "uplift@20%": uplift_at_k(y_te, W_te, tau_pred, k=0.20),
            "policy_value_20%": dr_val,
        }

    typer.echo("\nUplift Modeling Benchmark Results:")
    typer.echo(format_model_comparison_table(results))


@app.command()
def report(
    output_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Optional path to write markdown report to.",
    ),
    samples: int = typer.Option(6000, help="Sample size for report."),  # noqa: B008
) -> None:
    """Generate and display complete research report."""
    df = generate_synthetic_uplift_data(n_samples=samples, random_state=42)
    train_df, _, test_df = train_val_test_split_uplift(df, random_state=42)

    ab_res = ABTestAnalyzer.analyze(df["treatment"], df["conversion"])
    power_res = PowerCalculator.calculate_sample_size(0.04, 0.005)

    X_tr, W_tr, y_tr = extract_features_and_targets(train_df)
    X_te, W_te, y_te = extract_features_and_targets(test_df)

    models = {
        "Random": RandomPolicy(),
        "TreatAll": TreatAllPolicy(),
        "OutcomeOnly": OutcomeOnlyLearner(),
        "S-Learner": SLearner(estimator_type="logistic"),
        "T-Learner": TLearner(estimator_type="logistic"),
        "Doubly Robust": DoublyRobustLearner(known_propensity=0.85),
    }

    comp_results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_tr, W_tr, y_tr)
        tau_pred = model.predict_uplift(X_te)
        mu0_pred, mu1_pred = model.predict_outcomes(X_te)
        actions_20 = compute_policy_assignment(tau_pred, budget_fraction=0.20)
        dr_val = doubly_robust_policy_value(
            y_te, W_te, actions_20, mu0_pred, mu1_pred, propensity=0.85
        )
        comp_results[name] = {
            "auuc": auuc_score(y_te, W_te, tau_pred),
            "qini": qini_score(y_te, W_te, tau_pred),
            "uplift@10%": uplift_at_k(y_te, W_te, tau_pred, k=0.10),
            "uplift@20%": uplift_at_k(y_te, W_te, tau_pred, k=0.20),
            "policy_value_20%": dr_val,
        }

    report_text = f"""# Experimentation & Uplift Modeling Research Report

## Prospective Power Analysis

{format_power_table(power_res)}

## Randomized A/B Experiment Analysis

{format_ab_test_table(ab_res)}

## Uplift Benchmark Evaluation

{format_model_comparison_table(comp_results)}
"""
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_text, encoding="utf-8")
        typer.echo(f"Saved report to {output_file}")
    else:
        typer.echo(report_text)


@app.callback()
def _main() -> None:
    """Run Uplift Modeling commands."""


if __name__ == "__main__":
    app()
