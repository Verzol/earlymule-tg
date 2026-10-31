"""Resolve data locations from configs/paths.yaml, overridden by configs/local.yaml if present."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_KEYS = ("amlsim_outputs", "dataset", "processed", "experiments")


def repo_root() -> Path:
    """Return the repo root: EARLYMULE_ROOT if set, else derived from this file's location."""
    env_root = os.environ.get("EARLYMULE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Absolute data locations; `dataset` is the AMLSim run folder name."""

    amlsim_outputs: Path
    dataset: str
    processed: Path
    experiments: Path

    @property
    def dataset_dir(self) -> Path:
        """Folder with the CSV files of the selected AMLSim run."""
        return self.amlsim_outputs / self.dataset


def _read_yaml(path: Path) -> dict:
    """Read a YAML mapping; an empty file gives {}."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_paths(
    base: str | Path = "configs/paths.yaml",
    local: str | Path = "configs/local.yaml",
) -> Paths:
    """Load data locations as absolute paths.

    Args:
        base: shared config. Relative paths are resolved from the repo root.
        local: optional per-machine config; its keys replace those in `base`.
    """
    root = repo_root()

    # 1. Shared config, then per-machine override
    cfg = _read_yaml(root / base)
    if (root / local).is_file():
        cfg.update(_read_yaml(root / local))

    # 2. Check required keys
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"paths config is missing keys: {missing}")

    # 3. Resolve from the repo root, not the cwd, so notebooks/ and tests/ agree
    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (root / p).resolve()

    return Paths(
        amlsim_outputs=resolve(cfg["amlsim_outputs"]),
        dataset=str(cfg["dataset"]),
        processed=resolve(cfg["processed"]),
        experiments=resolve(cfg["experiments"]),
    )
