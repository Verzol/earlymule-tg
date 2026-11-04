"""Chronological train / validation / test split by day, weekly snapshot days and snapshot labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """A block of consecutive days; first_day and last_day are inclusive day indices."""

    name: str
    first_day: int
    last_day: int


def chronological_splits(total_days: int, fractions: tuple[float, float, float]) -> list[Split]:
    """Cut days 0..total_days-1 into train, validation and test blocks by the given fractions."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    ends = np.round(np.cumsum(fractions) * total_days).astype(int)
    starts = np.r_[0, ends[:-1]]
    return [Split(n, int(s), int(e) - 1) for n, s, e in zip(["train", "val", "test"], starts, ends)]


def snapshot_days(split: Split, horizon: int, min_history: int) -> list[int]:
    """Days t at which accounts are scored, every `horizon` days.

    The label window (t - horizon, t] stays inside the split, so labels never cross a split
    boundary, and t >= min_history so the longest feature window is full.
    """
    first = max(split.first_day + horizon - 1, min_history)
    return list(range(first, split.last_day + 1, horizon))


def snapshot_labels(onset: pd.Series, acct_ids: pd.Index, t: int, horizon: int) -> pd.Series:
    """Label of every account at day t (end-of-day cut-off).

    Args:
        onset: onset_day of mule candidates, indexed by acct_id.
        acct_ids: all accounts.
    Returns:
        1.0 if the mule onset falls in (t - horizon, t], 0.0 otherwise, NaN for mules that started
        earlier: they were already due at an earlier snapshot, so they are left out here.
        Mules that start after t are negatives at t.
    """
    o = onset.reindex(acct_ids)
    label = pd.Series(0.0, index=acct_ids)
    label[o.gt(t - horizon) & o.le(t)] = 1.0
    label[o.le(t - horizon)] = np.nan
    return label


def active_labels(onset: pd.Series, completion: pd.Series, acct_ids: pd.Index, t: int) -> pd.Series:
    """Daily label at day t: 1.0 while a mule is active (onset <= t <= completion), 0.0 otherwise.

    Mules after their completion day are NaN (left out): the case is over, so neither label fits.
    """
    o, c = onset.reindex(acct_ids), completion.reindex(acct_ids)
    label = pd.Series(0.0, index=acct_ids)
    label[o.le(t) & c.ge(t)] = 1.0
    label[c.lt(t)] = np.nan
    return label
