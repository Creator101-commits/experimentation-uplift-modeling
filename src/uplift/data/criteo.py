"""Criteo uplift dataset acquisition, validation, and partitioning.

The full dataset is ~14M rows and is processed on Kaggle, not a laptop.
Rows are validated chunk-by-chunk, stored as typed Parquet, and split
deterministically into train/validation/test stratified by the joint
treatment x conversion strata.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from uplift.data.schema import FEATURE_COLUMNS

CRITEO_REVISION = "2424920019e49d52d72c13ac1143ec5d53af276b"
CRITEO_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/"
    f"{CRITEO_REVISION}/{CRITEO_FILENAME}"
)
CRITEO_LICENSE = "CC BY-NC-SA 4.0"
TRANSFORMATION_VERSION = "criteo-uplift-parquet-v1"

EXPECTED_COLUMNS = FEATURE_COLUMNS + ["treatment", "conversion", "visit", "exposure"]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
SPLIT_SEED = 42
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
PROPENSITY_TOLERANCE = 0.001


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_criteo(dest_dir: Path) -> Path:
    """Download the pinned Criteo CSV.gz if absent; return the local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / CRITEO_FILENAME
    if not target.exists():
        urllib.request.urlretrieve(CRITEO_URL, target)  # noqa: S310
    return target


def _detect_header(csv_path: Path) -> int | None:
    """Return 0 if the file has a header row, else None for headerless data."""
    with csv_path.open("rb") as fh:
        first = fh.readline().decode("utf-8", "replace").strip().split(",")
    return 0 if first and first[0].strip().lower() == "f0" else None


def _typed_schema() -> pa.Schema:
    fields = [pa.field("row_id", pa.int64())]
    fields += [pa.field(c, pa.float32()) for c in FEATURE_COLUMNS]
    fields += [pa.field(c, pa.int8()) for c in BINARY_COLUMNS]
    return pa.schema(fields)


@dataclass
class CriteoStats:
    """Accumulated validation and provenance statistics."""

    n_rows: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    treatment_sum: int = 0
    conversion_sum: int = 0
    visit_sum: int = 0
    exposure_sum: int = 0
    stratum_counts: dict[int, int] = field(default_factory=dict)

    def add_chunk(self, df: pd.DataFrame) -> None:
        self.n_rows += len(df)
        for col in EXPECTED_COLUMNS:
            self.null_counts[col] = self.null_counts.get(col, 0) + int(df[col].isna().sum())
        t = df["treatment"].to_numpy(dtype=np.int64)
        c = df["conversion"].to_numpy(dtype=np.int64)
        self.treatment_sum += int(t.sum())
        self.conversion_sum += int(c.sum())
        self.visit_sum += int(df["visit"].to_numpy(dtype=np.int64).sum())
        self.exposure_sum += int(df["exposure"].to_numpy(dtype=np.int64).sum())
        codes = t * 2 + c
        for code in np.unique(codes):
            self.stratum_counts[int(code)] = self.stratum_counts.get(int(code), 0) + int(
                (codes == code).sum()
            )


def validate_criteo_frame(df: pd.DataFrame, *, first: bool) -> None:
    """Validate one chunk's schema and binary domains; raise on violation."""
    if first and set(df.columns) != set(EXPECTED_COLUMNS):
        raise ValueError(
            f"Criteo schema mismatch: expected columns {EXPECTED_COLUMNS}, "
            f"got {list(df.columns)}"
        )
    for col in BINARY_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Column {col} contains nulls; binary flags must be complete")
        vals = df[col].unique()
        if not set(vals).issubset({0, 1}):
            raise ValueError(f"Column {col} is not binary: {sorted(set(vals))}")


def _typed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a validated chunk to the storage dtypes (float32 features, int8 flags)."""
    out = df.copy()
    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype("float32")
    for col in BINARY_COLUMNS:
        out[col] = out[col].astype("int8")
    return out


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    """Deterministic 64-bit hash per row over all data columns (dup counting)."""
    hashed = pd.util.hash_pandas_object(
        df[EXPECTED_COLUMNS], index=False, categorize=False
    )
    return np.asarray(hashed.to_numpy(), dtype=np.uint64)


def _frame_to_table(df: pd.DataFrame, row_ids: np.ndarray) -> pa.Table:
    arrays = [pa.array(row_ids, type=pa.int64())]
    for col in FEATURE_COLUMNS:
        arrays.append(pa.array(df[col].to_numpy(), type=pa.float32()))
    for col in BINARY_COLUMNS:
        arrays.append(pa.array(df[col].to_numpy(), type=pa.int8()))
    return pa.Table.from_arrays(arrays, schema=_typed_schema())


def read_and_validate_criteo(
    csv_path: Path,
    observations_path: Path,
    *,
    chunksize: int = 500_000,
) -> tuple[CriteoStats, np.ndarray]:
    """Stream the CSV, validate, and write typed observations.parquet.

    Returns accumulated stats and the per-row 64-bit hashes used for the
    duplicate-row count.
    """
    header = _detect_header(csv_path)
    names = None if header == 0 else EXPECTED_COLUMNS
    stats = CriteoStats()
    hash_chunks: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None
    row_offset = 0
    try:
        for chunk_idx, raw in enumerate(
            pd.read_csv(
                csv_path,
                header=header,
                names=names,
                chunksize=chunksize,
                dtype={c: "float64" for c in FEATURE_COLUMNS}
                | {c: "Int64" for c in BINARY_COLUMNS},
            )
        ):
            validate_criteo_frame(raw, first=chunk_idx == 0)
            # Hash the raw float64 chunk so rows differing only beyond
            # float32 precision are not miscounted as duplicates.
            hash_chunks.append(_row_hashes(raw))
            typed = _typed_frame(raw)
            n = len(typed)
            row_ids = np.arange(row_offset, row_offset + n, dtype=np.int64)
            row_offset += n
            stats.add_chunk(typed)
            table = _frame_to_table(typed, row_ids)
            if writer is None:
                writer = pq.ParquetWriter(observations_path, _typed_schema())
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return stats, np.concatenate(hash_chunks)


def stratified_partition_indices(
    treatment: np.ndarray,
    conversion: np.ndarray,
    *,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 60/20/20 split stratified by treatment x conversion."""
    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1):
        raise ValueError("fractions must satisfy 0 < train < 1 and 0 <= val < 1")
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction < 0:
        raise ValueError("train_fraction + val_fraction must not exceed 1")
    strata = treatment.astype(np.int64) * 2 + conversion.astype(np.int64)
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []
    for code in np.unique(strata):
        idx = np.flatnonzero(strata == code)
        perm = rng.permutation(idx)
        n = len(idx)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        train_idx.append(np.sort(perm[:n_train]))
        val_idx.append(np.sort(perm[n_train : n_train + n_val]))
        test_idx.append(np.sort(perm[n_train + n_val :]))
    return (
        np.sort(np.concatenate(train_idx)),
        np.sort(np.concatenate(val_idx)),
        np.sort(np.concatenate(test_idx)),
    )


def prepare_criteo(
    csv_path: Path,
    out_dir: Path,
    *,
    chunksize: int = 500_000,
    propensity_tolerance: float = PROPENSITY_TOLERANCE,
) -> dict[str, Any]:
    """Validate, partition, and write the Criteo dataset plus a manifest.

    ``csv_path`` may be a local file or produced by :func:`download_criteo`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    observations_path = out_dir / "observations.parquet"

    stats, hashes = read_and_validate_criteo(csv_path, observations_path, chunksize=chunksize)
    stats.duplicate_rows = int(stats.n_rows - np.unique(hashes).size)

    # Load the typed table once; it is needed for both the split and the
    # partition writes. ~14M rows x 60 bytes is well within Kaggle memory.
    table = pq.read_table(observations_path)
    treatment = table.column("treatment").to_numpy().astype(np.int64)
    conversion = table.column("conversion").to_numpy().astype(np.int64)

    train_idx, val_idx, test_idx = stratified_partition_indices(treatment, conversion)

    # Propensity = train-partition treatment rate; verify val/test agreement.
    propensity = float(treatment[train_idx].mean())
    rates = {
        "train": propensity,
        "validation": float(treatment[val_idx].mean()),
        "test": float(treatment[test_idx].mean()),
    }
    for name in ("validation", "test"):
        if abs(rates[name] - propensity) > propensity_tolerance:
            raise AssertionError(
                f"{name} treatment rate {rates[name]} diverges from train "
                f"{propensity} beyond {propensity_tolerance}"
            )

    split_counts = {
        "train": int(len(train_idx)),
        "validation": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    for name, idx in (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        part = table.take(pa.array(idx, type=pa.int64()))
        pq.write_table(part, out_dir / f"{name}.parquet")

    manifest = {
        "source_url": CRITEO_URL,
        "source_revision": CRITEO_REVISION,
        "source_file": CRITEO_FILENAME,
        "license": CRITEO_LICENSE,
        "source_sha256": _sha256_file(csv_path),
        "source_size_bytes": int(csv_path.stat().st_size),
        "transformation_version": TRANSFORMATION_VERSION,
        "split_seed": SPLIT_SEED,
        "n_rows": stats.n_rows,
        "duplicate_rows": stats.duplicate_rows,
        "null_counts": stats.null_counts,
        "observed_treatment_rate": stats.treatment_sum / stats.n_rows,
        "observed_conversion_rate": stats.conversion_sum / stats.n_rows,
        "observed_visit_rate": stats.visit_sum / stats.n_rows,
        "observed_exposure_rate": stats.exposure_sum / stats.n_rows,
        "stratum_counts": {str(k): v for k, v in sorted(stats.stratum_counts.items())},
        "split_counts": split_counts,
        "split_treatment_rates": rates,
        "known_propensity": propensity,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
