"""Seeding and the experiment log: one row per run in experiments/experiments.csv.

Usage:
    set_seed(cfg["seed"])
    ...
    log_run("lightgbm_baseline", config=cfg, seed=cfg["seed"], metrics={"pr_auc": 0.31})
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from earlymule.config import load_paths, repo_root

LOG_COLUMNS = ["timestamp", "run_name", "dataset", "commit", "dirty", "seed", "config", "metrics", "notes"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and, if installed, torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# The log itself changes on every run, so it never marks a run dirty
LOG_DIR_PREFIX = "experiments/"


def dirty_paths(porcelain: str, ignore_prefix: str = LOG_DIR_PREFIX) -> list[str]:
    """Changed or untracked paths from `git status --porcelain`, without the experiment log folder."""
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in porcelain.splitlines() if line.strip()]
    return [p for p in paths if not p.startswith(ignore_prefix)]


def git_state() -> tuple[str, bool]:
    """Return (short commit hash, has uncommitted changes); ("unknown", True) outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        # Untracked files count: new, uncommitted code would otherwise be logged as the clean commit.
        # Scratch outputs stay out because they are git-ignored.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root(),
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(dirty_paths(status))


def run_dir(run_name: str, seed: int) -> Path:
    """Output folder of one run: <experiments>/runs/<run_name>/<commit>/seed<N> (git-ignored).

    The commit is in the path so a rerun after a code change never overwrites the files of a logged run.
    """
    commit, dirty = git_state()
    d = load_paths().experiments / "runs" / run_name / (commit + ("-dirty" if dirty else "")) / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_run(
    run_name: str,
    config: dict,
    seed: int,
    metrics: dict,
    notes: str = "",
    log_file: str | Path | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> dict:
    """Append one run to the experiment log and return the row.

    Args:
        config: everything needed to rerun (model params, split days, windows); stored as JSON.
        metrics: metric name to value; stored as JSON.
        log_file: defaults to <experiments>/experiments.csv from configs/paths.yaml.
        commit, dirty: code state of a run made on another machine; default the local git state.
    """
    paths = load_paths()
    log_file = Path(log_file) if log_file else paths.experiments / "experiments.csv"
    if commit is None:
        commit, dirty = git_state()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "dataset": paths.dataset,
        "commit": commit,
        "dirty": dirty,
        "seed": seed,
        "config": json.dumps(config, sort_keys=True, default=str),
        "metrics": json.dumps(metrics, sort_keys=True),
        "notes": notes,
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.is_file()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_log(log_file: str | Path | None = None):
    """Load the experiment log as a DataFrame with config and metrics parsed back to dicts."""
    import pandas as pd

    log_file = Path(log_file) if log_file else load_paths().experiments / "experiments.csv"
    df = pd.read_csv(log_file, dtype={"commit": str})
    df["config"] = df["config"].map(json.loads)
    df["metrics"] = df["metrics"].map(json.loads)
    return df


def select_runs(log, run_name: str, commit: str | None = None):
    """Clean logged rows of run_name at `commit` (default: the latest clean commit), one per seed (latest row).

    Args:
        log: read_log output.
    Returns:
        DataFrame of the selected rows; ValueError if there is none.
    """
    runs = log[log.run_name.eq(run_name) & ~log.dirty.astype(str).str.lower().eq("true")]
    if runs.empty:
        raise ValueError(f"no clean run of {run_name} in the log")
    commit = commit or runs.commit.iloc[-1]
    runs = runs[runs.commit.eq(commit)].drop_duplicates("seed", keep="last")
    if runs.empty:
        raise ValueError(f"no clean run of {run_name} at commit {commit}")
    return runs


def run_folder(run_name: str, commit: str, seed: int) -> Path:
    """Existing output folder of a logged run (see run_dir)."""
    d = load_paths().experiments / "runs" / run_name / commit / f"seed{seed}"
    if not d.is_dir():
        raise FileNotFoundError(f"no saved outputs for {run_name} {commit} seed {seed}: {d}")
    return d
