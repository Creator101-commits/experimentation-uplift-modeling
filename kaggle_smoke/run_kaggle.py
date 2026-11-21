"""Kaggle smoke entrypoint: ~200k-row sample through the full pipeline.

Downloads the pinned Criteo CSV, draws a bounded random sample, runs the
same validation/partition/benchmark path as the full kernel at reduced
scale, and writes ``benchmark_smoke/`` artifacts. Used to validate the
end-to-end pipeline before the multi-hour full run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CRITEO_CACHE = WORK / "criteo"
SAMPLE_CSV = WORK / "criteo-sample.csv.gz"
DATA = WORK / "data_smoke"
BENCH = WORK / "benchmark_smoke"

TARGET_ROWS = 200_000
PER_CHUNK = 8_000
CHUNKSIZE = 500_000


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "uplift").is_dir():
            return candidate.parent
    return None


def _install_package() -> str:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError("no installable project under /kaggle/input")
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    info = proj / "run_info.json"
    sha = "unknown"
    if info.is_file():
        sha = str(json.loads(info.read_text()).get("git_sha", "unknown"))
    print(f"git_sha={sha}", flush=True)
    os.environ["GIT_SHA"] = sha
    return sha


def _sample_csv(csv_path: Path) -> Path:
    """Draw ~TARGET_ROWS random rows, preserving the joint distribution."""
    sys.path.insert(0, str(WORK / "project" / "src"))
    from uplift.data.criteo import EXPECTED_COLUMNS, _detect_header

    header = _detect_header(csv_path)
    names = None if header == 0 else EXPECTED_COLUMNS
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(
        csv_path, header=header, names=names, chunksize=CHUNKSIZE
    ):
        if total >= TARGET_ROWS:
            break
        take = min(PER_CHUNK, TARGET_ROWS - total, len(chunk))
        kept.append(chunk.sample(n=take, random_state=0))
        total += take
    sample = pd.concat(kept, ignore_index=True)
    sample = sample[list(EXPECTED_COLUMNS)]
    sample.to_csv(SAMPLE_CSV, index=False, compression="gzip")
    print(f"sampled {len(sample)} rows -> {SAMPLE_CSV}", flush=True)
    return SAMPLE_CSV


def main() -> None:
    t0 = time.time()
    sha = _install_package()
    DATA.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(WORK / "project" / "src"))
    from uplift.data.criteo import download_criteo, prepare_criteo
    from uplift.experimentation.benchmark import run_benchmark

    t1 = time.time()
    csv_path = download_criteo(CRITEO_CACHE)
    print(f"download seconds={time.time() - t1:.1f}", flush=True)

    t2 = time.time()
    sample_csv = _sample_csv(csv_path)
    manifest = prepare_criteo(sample_csv, DATA)
    print(
        f"sample-prepare seconds={time.time() - t2:.1f} "
        f"rows={manifest['n_rows']} prop={manifest['known_propensity']:.4f}",
        flush=True,
    )

    t3 = time.time()
    run_benchmark(DATA, BENCH, n_boot=50)
    print(f"benchmark seconds={time.time() - t3:.1f}", flush=True)
    print(f"total seconds={time.time() - t0:.1f} git_sha={sha}", flush=True)


if __name__ == "__main__":
    main()
