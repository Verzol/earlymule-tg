"""Read AMLSim output files and convert timestamps to day indices (day 0 = conf base_date)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_base_date(conf_file: Path) -> pd.Timestamp:
    """Return general.base_date of an AMLSim conf file as a UTC timestamp."""
    with Path(conf_file).open(encoding="utf-8") as f:
        conf = json.load(f)
    return pd.Timestamp(conf["general"]["base_date"], tz="UTC")


def to_day(timestamps: pd.Series, base_date: pd.Timestamp) -> pd.Series:
    """Convert tran_timestamp strings to integer days since base_date.

    AMLSim writes one step per day at 00:00, so the day index loses nothing.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    if (ts != ts.dt.normalize()).any():
        raise ValueError("timestamps with a time of day; day index would drop information")
    return (ts - base_date).dt.days.astype("int64")


def load_alert_accounts(run_dir: Path) -> pd.DataFrame:
    """Alert members: alert_id, alert_type, acct_id."""
    df = pd.read_csv(Path(run_dir) / "alert_accounts.csv", usecols=["alert_id", "alert_type", "acct_id"])
    return df.astype({"alert_id": "int64", "acct_id": "int64"})


def load_alert_transactions(run_dir: Path, base_date: pd.Timestamp) -> pd.DataFrame:
    """Illicit transactions: alert_id, alert_type, tran_id, orig_acct, bene_acct, base_amt, day."""
    cols = ["alert_id", "alert_type", "tran_id", "orig_acct", "bene_acct", "base_amt", "tran_timestamp"]
    df = pd.read_csv(Path(run_dir) / "alert_transactions.csv", usecols=cols)
    df["day"] = to_day(df.pop("tran_timestamp"), base_date)
    return df


def load_alert_members(tmp_run_dir: Path) -> pd.DataFrame:
    """Generator view of alert members: alert_id, acct_id, is_main.

    Only the simulator knows the main account, so is_main is evaluation-only.
    """
    df = pd.read_csv(Path(tmp_run_dir) / "alert_members.csv", usecols=["alertID", "accountID", "isMain"])
    df = df.rename(columns={"alertID": "alert_id", "accountID": "acct_id", "isMain": "is_main"})
    # pandas may read true/false as bool or as strings depending on the file
    df["is_main"] = df["is_main"].astype(str).str.lower().eq("true")
    return df
