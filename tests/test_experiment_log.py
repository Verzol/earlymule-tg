"""Tests for seeding and the experiment log. Run: pytest tests/test_experiment_log.py"""

import random

import numpy as np

from earlymule.config import load_paths
from earlymule.experiment_log import LOG_COLUMNS, dirty_paths, git_state, log_run, read_log, set_seed


def test_same_seed_same_numbers():
    """Two runs with the same seed draw the same numbers."""
    set_seed(7)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(7)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_log_appends_rows_with_one_header(tmp_path):
    """Each call adds one row; the header is written once."""
    log = tmp_path / "experiments.csv"
    log_run("a", config={"lr": 0.1}, seed=0, metrics={"pr_auc": 0.5}, log_file=log)
    log_run("b", config={"lr": 0.2}, seed=1, metrics={"pr_auc": 0.6}, log_file=log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(LOG_COLUMNS)
    assert len(lines) == 3


def test_log_roundtrip(tmp_path):
    """Config and metrics come back as the dicts that were logged."""
    log = tmp_path / "experiments.csv"
    log_run("a", config={"windows": [1, 7, 30]}, seed=0, metrics={"pr_auc": 0.5}, log_file=log)
    row = read_log(log).iloc[0]
    assert row["config"] == {"windows": [1, 7, 30]}
    assert row["metrics"] == {"pr_auc": 0.5}
    assert row["dataset"] == load_paths().dataset


def test_git_state_reads_commit():
    """Inside the repo the commit hash is found."""
    commit, _ = git_state()
    assert commit != "unknown" and len(commit) >= 7


def test_dirty_paths_count_untracked_code_but_not_the_log():
    """A new source file makes a run dirty; appending to the experiment log does not."""
    porcelain = " M experiments/experiments.csv\n?? src/earlymule/models/new.py\nR  a.py -> b.py\n"
    assert dirty_paths(porcelain) == ["src/earlymule/models/new.py", "b.py"]
    assert dirty_paths("?? experiments/experiments.csv\n") == []
