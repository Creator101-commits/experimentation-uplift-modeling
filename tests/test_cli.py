"""Tests for Typer CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from uplift.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.output


def test_cli_power() -> None:
    res = runner.invoke(app, ["power", "--baseline-rate", "0.04", "--mde", "0.005"])
    assert res.exit_code == 0
    assert "Total Required Sample Size" in res.output


def test_cli_ab_test() -> None:
    res = runner.invoke(app, ["ab-test", "--samples", "1000"])
    assert res.exit_code == 0
    assert "Randomized A/B Test Results" in res.output
    assert "Covariate Balance" in res.output


def test_cli_compare() -> None:
    res = runner.invoke(app, ["compare", "--samples", "1000"])
    assert res.exit_code == 0
    assert "Uplift Modeling Benchmark Results" in res.output
    assert "S-Learner" in res.output
    assert "Doubly Robust" in res.output


def test_cli_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    res = runner.invoke(app, ["report", "-o", str(out), "--samples", "1000"])
    assert res.exit_code == 0
    assert out.exists()
    assert "Prospective Power Analysis" in out.read_text(encoding="utf-8")
