"""Hold rule of the transfer-time task: cooldown, threshold for a daily budget, mule detection flags."""

import numpy as np
import pandas as pd

from earlymule.evaluation.transfer_holds import first_outflows, hold_table, holds, threshold_for_budget


def rows(data):
    """Scored transfers from (seq, day, orig_acct, score) tuples."""
    return pd.DataFrame(data, columns=["seq", "day", "orig_acct", "score"])


def test_cooldown_skips_holds_of_a_reviewed_account():
    """Account 1 is held on day 0; its next high scores are skipped until day 30; account 2 is independent."""
    r = rows([(1, 0, 1, 5.0), (2, 3, 1, 6.0), (3, 29, 1, 6.0), (4, 30, 1, 6.0), (5, 3, 2, 4.0), (6, 4, 2, 0.1)])
    held = holds(r, theta=1.0, cooldown=30)
    assert held.seq.tolist() == [1, 4, 5]
    assert holds(r, theta=1.0, cooldown=0).seq.tolist() == [1, 2, 3, 4, 5]


def test_threshold_meets_the_daily_budget():
    """On random scores, the chosen threshold holds at most H new accounts a day, and one step lower holds more."""
    rng = np.random.default_rng(0)
    n = 5000
    r = rows(list(zip(range(n), np.sort(rng.integers(0, 10, n)), rng.integers(0, 2000, n), rng.normal(size=n))))
    theta = threshold_for_budget(r, per_day=5, cooldown=30)
    assert len(holds(r, theta, 30)) / 10 <= 5
    lower = np.sort(r.score[r.score < theta].to_numpy())[-1]
    assert len(holds(r, lower, 30)) / 10 > 5


def test_detection_flags():
    """Mule 10 is held on its first outflow, mule 11 after it, mule 12 before its onset only, mule 13 not scored."""
    illicit = pd.DataFrame({"tran_id": [5, 8, 9, 20, 30], "orig_acct": [10, 10, 11, 12, 13],
                            "base_amt": [100.0, 50.0, 70.0, 30.0, 10.0], "day": [2, 3, 2, 6, 6]})
    mules = pd.DataFrame({"acct_id": [10, 11, 12, 13], "alert_id": [1, 1, 2, 3], "typology": ["a", "a", "b", "b"],
                          "onset_day": [2, 1, 5, 6]})
    held = pd.DataFrame({"seq": [5, 12, 2], "day": [2, 4, 0], "orig_acct": [10, 11, 12]})
    t = hold_table(held, first_outflows(illicit, mules), pd.Index([5, 8, 9, 20])).set_index("acct_id")
    assert t.before_forwarding.tolist() == [True, False, False] and t.first_out_held.tolist() == [True, False, False]
    assert t.loc[10, "same_day"] and not t.loc[11, "same_day"] and 13 not in t.index
