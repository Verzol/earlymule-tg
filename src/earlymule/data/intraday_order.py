"""Check that the execution order of transfers within a day can be recovered from AMLSim output.

AMLSim stamps every transfer at 00:00 of its day, but tx_log.csv lists transfers in the order the simulator
executed them, with each account's balance before and after. If every transfer's old balance equals the
new balance of the account's previous transfer in file order, that order is the execution order, also
within a day. transactions.csv carries tran_id = tx_log row number, so the order reaches the features as
load_transactions' seq column. Reads tx_log.csv and transactions.csv of the configured dataset.
Run: python -m earlymule.data.intraday_order [--dataset NAME]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from earlymule.config import load_paths

# Balances are written with 2 decimals
TOLERANCE = 0.011


def balance_chain(log: pd.DataFrame) -> pd.DataFrame:
    """One row per (transfer, side) after the account's first transfer: does the balance chain hold?

    Args:
        log: tx_log.csv columns step, nameOrig, nameDest and the four balance columns, in file order.
    Returns:
        DataFrame with acct, row, same_day (previous transfer of the account on the same step) and ok.
    """
    # 1. Stack both sides; each side has the account's balance before and after the transfer
    row = np.arange(len(log))
    sides = pd.concat([
        pd.DataFrame({"acct": log.nameOrig, "row": row, "step": log.step,
                      "old": log.oldbalanceOrig, "new": log.newbalanceOrig}),
        pd.DataFrame({"acct": log.nameDest, "row": row, "step": log.step,
                      "old": log.oldbalanceDest, "new": log.newbalanceDest})])
    # 2. File order per account; the old balance must equal the previous new balance
    sides = sides.sort_values(["acct", "row"], kind="stable")
    prev = sides.groupby("acct")[["new", "step"]].shift()
    has_prev = prev.new.notna()
    out = sides[has_prev].assign(same_day=prev.step[has_prev].eq(sides.step[has_prev]),
                                 ok=(sides.old[has_prev] - prev.new[has_prev]).abs().lt(TOLERANCE))
    return out[["acct", "row", "same_day", "ok"]].reset_index(drop=True)


def tran_id_matches_log(tx: pd.DataFrame, log: pd.DataFrame) -> bool:
    """True if transaction tran_id points at its tx_log row (1-based): same sender, receiver and amount."""
    x = log.iloc[tx.tran_id.to_numpy() - 1]
    return bool((x.nameOrig.to_numpy() == tx.orig_acct.to_numpy()).all()
                and (x.nameDest.to_numpy() == tx.bene_acct.to_numpy()).all()
                and np.allclose(x.amount.to_numpy(), tx.base_amt.to_numpy()))


def main() -> None:
    """Print the balance-chain and tran_id checks for one dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    paths = load_paths()
    paths = paths.with_dataset(args.dataset) if args.dataset else paths
    log = pd.read_csv(paths.dataset_dir / "tx_log.csv")
    tx = pd.read_csv(paths.dataset_dir / "transactions.csv", usecols=["tran_id", "orig_acct", "bene_acct", "base_amt"])
    chain = balance_chain(log)
    print(json.dumps({"dataset": paths.dataset, "tx_log_rows": len(log), "transactions": len(tx),
                      "steps_non_decreasing": bool(log.step.is_monotonic_increasing),
                      "chain_pairs": len(chain), "chain_ok": round(float(chain.ok.mean()), 6),
                      "same_day_pairs": int(chain.same_day.sum()),
                      "same_day_ok": round(float(chain.ok[chain.same_day].mean()), 6),
                      "tran_id_matches_log_row": tran_id_matches_log(tx, log),
                      "log_rows_without_transaction": sorted(set(range(1, len(log) + 1)) - set(tx.tran_id))[:10]},
                     indent=1))


if __name__ == "__main__":
    main()
