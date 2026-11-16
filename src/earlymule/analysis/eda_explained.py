"""Simple, step-by-step EDA helpers for notebooks/03_eda_explained.ipynb (EMTG-69, explained version).

Every function does one small thing with pandas, so each step can be checked by hand. Day indices use the
end-of-day cut-off of load_transactions; `seq` is the execution order within a day. Labels (is_sar, mule
ground truth) are read for analysis only and never enter a feature. Chart helpers draw slide-ready PNGs in one
fixed palette: regular accounts blue, mules orange, non-mule case members aqua.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from earlymule.data import amlsim  # noqa: E402
from earlymule.features.point_in_time import snapshot_features  # noqa: E402

# Slide palette (dataviz reference palette, validated for colour-blind readers as a set of three)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GROUP_COLOURS = {"regular account": BLUE, "mule": ORANGE, "case member, not mule": AQUA}

# One sentence per feature base name; the window suffix (_1d, _7d, _30d) says over how many days
DESCRIPTIONS = {
    "n_out": "number of transfers sent", "n_in": "number of transfers received",
    "amt_out": "total amount sent", "amt_in": "total amount received",
    "peers_out": "distinct accounts paid", "peers_in": "distinct accounts that paid in",
    "out_in_ratio": "amount sent divided by amount received",
    "imbalance": "(sent - received) / (sent + received), from -1 (only receives) to +1 (only sends)",
    "new_peers_out": "accounts paid for the first time ever inside the window",
    "new_peers_in": "senders seen for the first time ever inside the window",
    "new_share_out": "share of the accounts paid that are new", "new_share_in": "share of the senders that are new",
    "fwd_0d": "share of the money received that is sent on the same day",
    "fwd_1d": "share of the money received that is sent on within 1 day",
    "fwd_3d": "share of the money received that is sent on within 3 days",
    "dwell_med": "median days from a day with money in to the next day with money out",
    "dwell_min": "shortest such wait in days",
    "up_amt_in": "money that the account's senders themselves received in the window",
    "down_amt_out": "money that the accounts it paid themselves sent on in the window",
    "ent_in": "how evenly the money received is spread over senders (entropy)",
    "ent_out": "how evenly the money sent is spread over receivers (entropy)",
    "burst": "(sd - mean) / (sd + mean) of the gaps between transfers; -1 = perfectly regular",
    "days_since_last_tx": "days since the last transfer of any kind",
    "days_since_last_out": "days since the last transfer sent", "days_since_last_in": "days since the last transfer received",
    "initial_deposit": "opening balance of the account",
}


def slide_style() -> None:
    """Matplotlib defaults for slides: light surface, recessive grid, readable fonts."""
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "font.size": 10,
                         "axes.axisbelow": True, "axes.titlelocation": "left"})   # grid behind the marks


def save(fig, path: Path) -> None:
    """Tight layout, save at slide resolution, close."""
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- 1. Data ----------

def labelled_transfers(run_dir: Path, base_date: pd.Timestamp) -> pd.DataFrame:
    """All transfers with day, execution order (seq), illicit flag and case id (analysis only)."""
    tx = pd.read_csv(run_dir / "transactions.csv",
                     usecols=["tran_id", "orig_acct", "bene_acct", "base_amt", "tran_timestamp", "is_sar", "alert_id"])
    tx["day"] = amlsim.to_day(tx.pop("tran_timestamp"), base_date)
    return tx.rename(columns={"tran_id": "seq"}).sort_values(["day", "seq"]).reset_index(drop=True)


def account_sides(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, transfer): the account, its partner, day, amount, direction ('in' or 'out')."""
    out = tx.assign(acct=tx.orig_acct, partner=tx.bene_acct, direction="out")
    inn = tx.assign(acct=tx.bene_acct, partner=tx.orig_acct, direction="in")
    cols = ["acct", "partner", "day", "seq", "base_amt", "direction", "is_sar"]
    return pd.concat([out[cols], inn[cols]]).sort_values(["acct", "day", "seq"]).reset_index(drop=True)


def regular_start_day(sides: pd.DataFrame, min_tx: int = 20) -> pd.Series:
    """Day on which each account reaches min_tx transfers; it is scored ('regular') from that day on."""
    nth = sides.groupby("acct").cumcount() + 1
    return sides[nth.eq(min_tx)].set_index("acct").day.rename("regular_from")


# ---------- 2. Feature dictionary ----------

def feature_dictionary(columns: list[str]) -> pd.DataFrame:
    """Name, base name, window and plain description of every model feature."""
    rows = []
    for col in columns:
        window = next((w for w in ["_7d_over_30d", "_1d", "_7d", "_30d"] if col.endswith(w)), "")
        base = col[: len(col) - len(window)] if window else col
        text = (f"7-day {DESCRIPTIONS.get(base, base)} as a share of the 30-day value" if window == "_7d_over_30d"
                else DESCRIPTIONS.get(base, base))
        rows.append({"feature": col, "base": base, "window": window.strip("_") or "whole history", "meaning": text})
    return pd.DataFrame(rows)


def features_at(inputs: dict, rows: pd.DataFrame, windows=(1, 7, 30), exclude=("n_tx_hist", "days_since_first_tx")) -> pd.DataFrame:
    """Features at the end of each row's day for rows with columns acct_id and day (other columns kept)."""
    parts = []
    for day, group in rows.groupby("day"):
        ids = pd.Index(group.acct_id.unique(), name="acct_id")
        f = snapshot_features(inputs["tx"], ids, int(day), windows, inputs["first_day"])
        f["initial_deposit"] = inputs["accounts"].initial_deposit.reindex(ids).to_numpy()
        parts.append(f.drop(columns=list(exclude)).reset_index().assign(day=day))
    return rows.merge(pd.concat(parts, ignore_index=True), on=["acct_id", "day"], how="left")


# ---------- 3. How regular accounts behave ----------

def partner_profile(sides: pd.DataFrame, accounts: pd.Index) -> pd.DataFrame:
    """Per account over the whole period: transfers, distinct partners, share of transfers with the top partner."""
    s = sides[sides.acct.isin(accounts)]
    per_pair = s.groupby(["acct", "partner"]).size()
    top_share = per_pair.groupby("acct").max() / per_pair.groupby("acct").sum()
    return pd.DataFrame({"transfers": s.groupby("acct").size(), "partners": per_pair.groupby("acct").size(),
                         "top_partner_share": top_share})


def gap_regularity(sides: pd.DataFrame, accounts: pd.Index) -> pd.Series:
    """Per account: share of gaps between consecutive transfer days equal to its most common gap (1 = clockwork)."""
    days = sides[sides.acct.isin(accounts)].drop_duplicates(["acct", "day"])
    gaps = days.groupby("acct").day.diff().dropna().to_frame("gap").join(days.acct)
    counts = gaps.groupby(["acct", "gap"]).size()
    return (counts.groupby("acct").max() / counts.groupby("acct").sum()).rename("same_gap_share")


# ---------- 4–5. Mule timelines and one example per typology ----------

def mule_timeline(mules: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Per mule: typology, onset, first outflow and completion day, days in between, same-day order."""
    illicit = tx[tx.is_sar]
    first_in = illicit.groupby("bene_acct").seq.min()
    first_out = illicit.groupby("orig_acct").seq.min()
    t = mules[["typology", "onset_day", "first_illicit_outflow_day", "completion_day"]].copy()
    t["days_to_first_out"] = t.first_illicit_outflow_day - t.onset_day
    t["receives_first"] = first_in.reindex(t.index) < first_out.reindex(t.index)
    return t


def account_story(sides: pd.DataFrame, acct: int, first_day: int, last_day: int) -> pd.DataFrame:
    """Transfers of one account between two days, with a flag for partners never seen before."""
    s = sides[sides.acct.eq(acct)].copy()
    s["new_partner"] = ~s.duplicated("partner")
    return s[s.day.between(first_day, last_day)][["day", "seq", "direction", "partner", "base_amt", "is_sar", "new_partner"]]


# ---------- 6–10. Comparing groups ----------

def random_regular(regular_from: pd.Series, exclude: pd.Index, days, per_day: int, seed: int = 0) -> pd.DataFrame:
    """per_day random scored accounts outside every case on each given day: rows (acct_id, day).

    Early days have few or no scored accounts; then all of them are taken, or the day is skipped.
    """
    rng = np.random.default_rng(seed)
    pool = regular_from.drop(exclude, errors="ignore")
    rows = []
    for d in sorted(set(days)):
        candidates = pool.index[pool.to_numpy() <= d]
        picked = rng.choice(candidates, min(per_day, len(candidates)), replace=False)
        rows.append(pd.DataFrame({"acct_id": picked, "day": d}))
    return pd.concat(rows, ignore_index=True)


def separation_table(frame: pd.DataFrame, features: list[str], group: str = "group", positive: str = "mule",
                     reference: str = "regular account") -> pd.DataFrame:
    """Median per group and single-feature ROC-AUC of `positive` against `reference` (0.5 = no separation)."""
    two = frame[frame[group].isin([positive, reference])]
    y = two[group].eq(positive)
    rows = [{"feature": c, f"median {positive}": two.loc[y, c].median(), f"median {reference}": two.loc[~y, c].median(),
             "roc_auc": roc_auc_score(y, two[c].fillna(two[c].median()))} for c in features]
    out = pd.DataFrame(rows)
    out["strength"] = (out.roc_auc - 0.5).abs() + 0.5   # same scale whether the feature is higher or lower in mules
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def correlated_pairs(frame: pd.DataFrame, features: list[str], threshold: float = 0.9) -> pd.DataFrame:
    """Pairs of features whose rank correlation is above the threshold (they carry the same information)."""
    rho = frame[features].rank().corr()
    pairs = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
    pairs = pairs[pairs.abs() > threshold].rename("spearman").reset_index()
    return pairs.rename(columns={"level_0": "feature_a", "level_1": "feature_b"}).sort_values("spearman", ascending=False)


# ---------- 13. Leakage check ----------

def future_change_check(inputs: dict, acct: int, day: int) -> pd.DataFrame:
    """Features of one account at `day`, before and after adding a large fake transfer on day + 1."""
    before = features_at(inputs, pd.DataFrame({"acct_id": [acct], "day": [day]}))
    fake = pd.DataFrame({"orig_acct": [acct], "bene_acct": [acct + 1], "base_amt": [1e6], "day": [day + 1], "seq": [10**9]})
    changed = {**inputs, "tx": pd.concat([inputs["tx"], fake]).sort_values(["day", "seq"]).reset_index(drop=True)}
    after = features_at(changed, pd.DataFrame({"acct_id": [acct], "day": [day]}))
    return pd.concat([before, after], keys=["original", "with a transfer on day + 1"]).droplevel(1).T


# ---------- Charts ----------

def bar_chart(values: pd.Series, title: str, xlabel: str, path: Path, colour: str = BLUE, fmt: str = "{:.0f}",
              reference: float | None = None) -> None:
    """Horizontal bars with the value written at the end of each bar (largest at the top); optional reference line."""
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(v) + 1.2))
    ax.barh(v.index.astype(str), v.to_numpy(), color=colour, height=0.6)
    for y, x in enumerate(v.to_numpy()):
        ax.text(x, y, " " + fmt.format(x), va="center", color=INK, fontsize=9)
    if reference is not None:
        ax.axvline(reference, color=MUTED, ls="--", lw=1)
    ax.set(xlabel=xlabel)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=13)   # tight_layout keeps a suptitle on the canvas
    ax.grid(axis="y", visible=False)
    save(fig, path)


def histogram_by_group(frame: pd.DataFrame, feature: str, group: str, path: Path, title: str, log: bool = False) -> None:
    """Overlaid density histograms of one feature, one per group, in the group colours."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = frame[feature].dropna()
    edges = np.geomspace(max(x[x > 0].min(), 1e-3), x.max() + 1, 40) if log else np.linspace(x.min(), x.max(), 40)
    for name, g in frame.groupby(group):
        ax.hist(g[feature].dropna(), bins=edges, density=True, alpha=0.55, color=GROUP_COLOURS.get(name, MUTED), label=name)
    if log:
        ax.set_xscale("log")
    ax.set(title=title, xlabel=feature, ylabel="density")
    ax.legend()
    save(fig, path)


def lines_by_group(medians: pd.DataFrame, features: list[str], path: Path, title: str, xlabel: str) -> None:
    """Small multiples: median of each feature by day offset, one line per group (columns of `medians`)."""
    fig, axes = plt.subplots(1, len(features), figsize=(3.4 * len(features), 3.2), sharex=True)
    for ax, f in zip(np.atleast_1d(axes), features):
        for name, series in medians[f].items():
            ax.plot(series.index, series.to_numpy(), lw=2, color=GROUP_COLOURS.get(name, MUTED), label=name)
        ax.axvline(0, color=MUTED, lw=0.8, ls="--")
        ax.set(title=f, xlabel=xlabel)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    save(fig, path)
