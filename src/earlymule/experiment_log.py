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


def git_state() -> tuple[str, bool]:
    """Return (short commit hash, has uncommitted changes); ("unknown", True) outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        # Ignore untracked files so scratch outputs do not mark every run dirty
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root(),
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(status.strip())


def log_run(
    run_name: str,
    config: dict,
    seed: int,
    metrics: dict,
    notes: str = "",
    log_file: str | Path | None = None,
) -> dict:
    """Append one run to the experiment log and return the row.

    Args:
        config: everything needed to rerun (model params, split days, windows); stored as JSON.
        metrics: metric name to value; stored as JSON.
        log_file: defaults to <experiments>/experiments.csv from configs/paths.yaml.
    """
    paths = load_paths()
    log_file = Path(log_file) if log_file else paths.experiments / "experiments.csv"
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
