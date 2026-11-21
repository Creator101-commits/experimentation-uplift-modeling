"""Tests for Criteo acquisition, validation, split, and manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift.data import criteo
from uplift.data.criteo import (
    EXPECTED_COLUMNS,
    stratified_partition_indices,
    validate_criteo_frame,
)


def _write_csv(path: Path, n: int = 400, seed: int = 0, header: bool = True) -> None:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.85, size=n)
    conversion = np.zeros(n, dtype=int)
    # Rare conversions concentrated in treated rows.
    conv_idx = rng.choice(n, size=max(4, n // 50), replace=False)
    conversion[conv_idx] = 1
    visit = rng.binomial(1, 0.1, size=n)
    exposure = np.where(treatment == 1, rng.binomial(1, 0.9, size=n), 0)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    df["treatment"] = treatment
    df["conversion"] = conversion
    df["visit"] = visit
    df["exposure"] = exposure
    df = df[EXPECTED_COLUMNS]
    df.to_csv(path, index=False, header=header)


def test_schema_mismatch_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    df = pd.DataFrame({"f0": [1.0], "treatment": [1]})
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_criteo_frame(df, first=True)


def test_binary_domain_rejected(tmp_path: Path) -> None:
    df = pd.DataFrame({c: [0] for c in EXPECTED_COLUMNS})
    df.loc[0, "treatment"] = 2
    with pytest.raises(ValueError, match="not binary"):
        validate_criteo_frame(df, first=True)


def test_stratified_split_deterministic_disjoint() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    treatment = rng.binomial(1, 0.85, size=n)
    conversion = rng.binomial(1, 0.01, size=n)
    tr1, va1, te1 = stratified_partition_indices(treatment, conversion)
    tr2, va2, te2 = stratified_partition_indices(treatment, conversion)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2)
    assert np.array_equal(te1, te2)
    # Disjoint and cover all rows.
    all_idx = np.concatenate([tr1, va1, te1])
    assert len(np.unique(all_idx)) == n
    assert set(tr1).isdisjoint(va1) and set(tr1).isdisjoint(te1)
    # Roughly 60/20/20.
    assert abs(len(tr1) / n - 0.6) < 0.05
    assert abs(len(va1) / n - 0.2) < 0.05
    assert abs(len(te1) / n - 0.2) < 0.05


def test_prepare_criteo_writes_parquet_and_manifest(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteo.csv"
    _write_csv(csv_path, n=400, header=True)
    out = tmp_path / "processed"
    manifest = criteo.prepare_criteo(csv_path, out, chunksize=100, propensity_tolerance=0.05)

    assert manifest["n_rows"] == 400
    assert manifest["split_counts"]["train"] > 0
    assert manifest["known_propensity"] == pytest.approx(
        manifest["split_treatment_rates"]["train"]
    )
    for name in ("train", "validation", "test"):
        assert (out / f"{name}.parquet").exists()
    assert (out / "observations.parquet").exists()
    assert (out / "manifest.json").exists()
    # Treatment rates agree across splits within the fixture tolerance.
    rates = manifest["split_treatment_rates"]
    assert abs(rates["validation"] - rates["train"]) <= 0.05
    assert abs(rates["test"] - rates["train"]) <= 0.05
    # Feature columns stored as float32, flags as int8.
    import pyarrow.parquet as pq

    schema = pq.read_schema(out / "train.parquet")
    assert schema.field("f0").type == "float"
    assert schema.field("treatment").type == "int8"
    assert schema.field("row_id").type == "int64"


def test_prepare_criteo_detects_duplicates(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteo.csv"
    _write_csv(csv_path, n=200, header=False)
    # Duplicate the first 10 data rows at the end.
    lines = csv_path.read_text().strip().split("\n")
    dup = lines[:10]
    csv_path.write_text("\n".join(lines + dup) + "\n")
    out = tmp_path / "processed"
    manifest = criteo.prepare_criteo(csv_path, out, chunksize=64, propensity_tolerance=0.05)
    assert manifest["n_rows"] == 210
    assert manifest["duplicate_rows"] == 10


def test_row_hashes_distinguish_beyond_float32(tmp_path: Path) -> None:
    """Rows differing only past float32 precision are not duplicates."""
    csv_path = tmp_path / "criteo.csv"
    _write_csv(csv_path, n=120, header=True)
    base = pd.read_csv(csv_path)
    # Two rows identical except f0 = 2**24 vs 2**24 + 1. float32 rounds both
    # to 16777216.0, but the raw float64 values are distinct, so they must not
    # be counted as duplicates of each other.
    edge = pd.DataFrame([dict.fromkeys(EXPECTED_COLUMNS, 0.0)])
    edge["f0"] = [16777216.0]
    edge["treatment"] = [1]
    edge["conversion"] = [0]
    edge["visit"] = [0]
    edge["exposure"] = [1]
    edge2 = edge.copy()
    edge2["f0"] = [16777217.0]
    # Three identical copies of a distinct row -> exactly 2 duplicate rows.
    dup = edge.copy()
    dup["f0"] = [3.5]
    dup["treatment"] = [0]
    out = pd.concat([base, edge, edge2, dup, dup, dup], ignore_index=True)
    out.to_csv(csv_path, index=False)
    manifest = criteo.prepare_criteo(csv_path, tmp_path / "p", chunksize=64,
                                     propensity_tolerance=0.05)
    assert manifest["n_rows"] == 125
    # Only the two extra dup copies count; the float32-edge pair does not.
    assert manifest["duplicate_rows"] == 2


def test_prepare_criteo_headerless(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteo.csv"
    _write_csv(csv_path, n=120, header=False)
    out = tmp_path / "processed"
    manifest = criteo.prepare_criteo(csv_path, out, chunksize=50, propensity_tolerance=0.05)
    assert manifest["n_rows"] == 120
    assert manifest["null_counts"]["f0"] == 0
    # Production default keeps the brief's 0.001 agreement requirement.
    assert criteo.PROPENSITY_TOLERANCE == 0.001
