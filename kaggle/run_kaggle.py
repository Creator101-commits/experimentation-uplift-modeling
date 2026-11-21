"""Kaggle entrypoint for the full Criteo uplift benchmark.

Installs ``uplift`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, downloads and validates the full Criteo uplift
dataset, partitions it, runs the frozen benchmark (validation selection,
analysis-plan freeze, single test read, paired bootstrap, A/B report), and
writes ``benchmark/`` artifacts including ``manifest.json`` with the git SHA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA = WORK / "data"
CRITEO_CACHE = WORK / "criteo"
BENCH = WORK / "benchmark"


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
        raise FileNotFoundError(
            "no installable project under /kaggle/input (expected pyproject.toml + src/uplift)"
        )
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


def main() -> None:
    t0 = time.time()
    sha = _install_package()
    DATA.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    # Stage 1: download, validate, and partition the full Criteo dataset.
    t1 = time.time()
    _run(
        [
            sys.executable, "-m", "uplift.cli", "prepare-data",
            "--out", str(DATA),
            "--data-dir", str(CRITEO_CACHE),
        ]
    )
    print(f"prepare-data seconds={time.time() - t1:.1f}", flush=True)

    # Stage 2: frozen benchmark on the full partitions.
    t2 = time.time()
    _run(
        [
            sys.executable, "-m", "uplift.cli", "benchmark",
            "--data", str(DATA),
            "--out", str(BENCH),
        ]
    )
    print(f"benchmark seconds={time.time() - t2:.1f}", flush=True)
    print(f"total seconds={time.time() - t0:.1f} git_sha={sha}", flush=True)


if __name__ == "__main__":
    main()
