"""Tests for the walking skeleton parts: chronological split, snapshot labels, point-in-time features, metrics."""

import numpy as np
import pandas as pd
import pytest

from earlymule.data.splits import Split, chronological_splits, snapshot_days, snapshot_labels
from earlymule.evaluation.metrics import evaluate, recall_at_k
from earlymule.features.point_in_time import snapshot_features

TX = pd.DataFrame({
    "orig_acct": [1, 1, 2, 3, 1],
    "bene_acct": [2, 3, 1, 1, 2],
    "base_amt": [100.0, 50.0, 30.0, 20.0, 999.0],
    "day": [3, 9, 9, 10, 11],
})
ACCTS = pd.Index([1, 2, 3, 4], name="acct_id")


def test_chronological_splits_cover_all_days():
    """60/15/25 of 720 days gives contiguous blocks 0-431, 432-539, 540-719."""
    s = chronological_splits(720, (0.6, 0.15, 0.25))
    assert [(x.first_day, x.last_day) for x in s] == [(0, 431), (432, 539), (540, 719)]


def test_snapshot_days_keep_label_window_inside_split():
    """The first snapshot leaves a full label window and a full feature history."""
    assert snapshot_days(Split("val", 432, 539), horizon=7, min_history=30)[:2] == [438, 445]
    assert snapshot_days(Split("train", 0, 431), horizon=7, min_history=30)[0] == 30


def test_snapshot_labels():
    """Onset in (t-7, t] is positive, earlier onset is left out, later onset is negative."""
    onset = pd.Series([10.0, 3.0, 11.0], index=[1, 2, 3])
    label = snapshot_labels(onset, ACCTS, t=10, horizon=7)
    assert label[1] == 1.0 and np.isnan(label[2]) and label[3] == 0.0 and label[4] == 0.0


def test_window_boundaries():
    """Window (t-w, t]: a transfer on day t counts, one on day t-w does not."""
    f = snapshot_features(TX, ACCTS, t=10, windows=(1, 7))
    assert f.loc[1, "n_in_1d"] == 1          # day 10 from account 3
    assert f.loc[1, "n_out_7d"] == 1         # day 9 only; day 3 is outside (3, 10]
    assert f.loc[1, "out_in_ratio_7d"] == pytest.approx(50 / 50)
    assert f.loc[4, "n_tx_hist"] == 0 and np.isnan(f.loc[4, "days_since_last_tx"])


def test_features_ignore_transfers_after_t():
    """Adding transfers after day t leaves every feature at t unchanged."""
    before = snapshot_features(TX[TX.day.le(10)], ACCTS, t=10, windows=(1, 7, 30))
    after = snapshot_features(TX, ACCTS, t=10, windows=(1, 7, 30))
    pd.testing.assert_frame_equal(before, after)


def test_recall_at_k_ranks_within_each_snapshot():
    """Top-k is taken per snapshot, not over the pooled rows."""
    frame = pd.DataFrame({"snapshot": [1, 1, 1, 2, 2], "label": [1, 0, 0, 1, 0],
                          "score": [0.2, 0.9, 0.1, 0.05, 0.01], "typology": ["cycle", None, None, "m", None]})
    assert recall_at_k(frame, k=1) == 0.5
    m = evaluate(frame, k=1)
    assert m["positives"] == 2 and m["recall_at_1_by_typology"] == {"cycle": 0.0, "m": 1.0}


def test_new_peers_burst_and_recency():
    """A peer is new if the pair first transacts inside the window; burst = 7-day share of 30 days."""
    tx = pd.DataFrame({"orig_acct": [1, 1, 1, 2], "bene_acct": [2, 2, 3, 1],
                       "base_amt": [10.0, 10.0, 10.0, 5.0], "day": [1, 20, 20, 15]})
    f = snapshot_features(tx, ACCTS, t=20, windows=(7, 30))
    assert f.loc[1, "peers_out_7d"] == 2 and f.loc[1, "new_peers_out_7d"] == 1   # 2 was paid on day 1
    assert f.loc[1, "new_share_out_7d"] == 0.5 and f.loc[1, "new_peers_out_30d"] == 2
    assert f.loc[1, "n_out_7d_over_30d"] == pytest.approx(2 / 3)
    assert f.loc[1, "days_since_last_in"] == 5 and f.loc[1, "days_since_last_out"] == 0


def test_features_with_precomputed_first_days_ignore_future():
    """The all-days pair table gives the same features as one built from the past only."""
    from earlymule.features.point_in_time import pair_first_day
    full = snapshot_features(TX, ACCTS, t=10, windows=(1, 7, 30), first_day=pair_first_day(TX))
    pd.testing.assert_frame_equal(full, snapshot_features(TX[TX.day.le(10)], ACCTS, t=10, windows=(1, 7, 30)))


def test_active_labels():
    """Positive from onset to completion, negative before, left out after completion."""
    from earlymule.data.splits import active_labels
    onset, comp = pd.Series([5.0], index=[1]), pd.Series([7.0], index=[1])
    assert [active_labels(onset, comp, ACCTS, t)[1] for t in (4, 5, 7)] == [0.0, 1.0, 1.0]
    assert np.isnan(active_labels(onset, comp, ACCTS, 8)[1]) and active_labels(onset, comp, ACCTS, 8)[2] == 0.0


def test_early_detection_timing():
    """First alert on or after onset; same-day detection counts as by-completion, not early."""
    from earlymule.evaluation.early_detection import early_detection
    scores = pd.DataFrame({"acct_id": [1, 2, 1, 2, 1, 2, 3, 3, 3], "snapshot": [5, 5, 6, 6, 7, 7, 5, 6, 7],
                           "score": [0.1, 0.9, 0.8, 0.2, 0.9, 0.1, 0.0, 0.0, 0.5]})
    mules = pd.DataFrame({"acct_id": [1, 3], "onset_day": [5, 7], "completion_day": [7, 7],
                          "first_illicit_outflow_day": [7, 7], "typology": ["a", "b"]})
    r = early_detection(scores, mules, k=1)
    # Mule 1 alerted on day 6 (< completion 7, < first outflow 7): early, TTD 1, lead 1. Mule 3 never in the top 1.
    assert (r["mules"], r["detected_by_completion"], r["detected_before_completion"]) == (2, 0.5, 0.5)
    # Mule 3 outflows on its onset day, so only mule 1 could be caught before forwarding
    assert (r["mules_can_precede_forwarding"], r["detected_before_forwarding"]) == (1, 1.0)
    assert (r["median_ttd_days"], r["median_lead_days"], r["multi_day_mules"]) == (1.0, 1.0, 1)
    assert r["alert_precision_active_mule"] == pytest.approx(2 / 3, abs=1e-4)


def test_early_detection_counts_misses_with_nullable_days():
    """Int64 day columns (as in the ground truth) must not drop undetected mules or non-mule alerts."""
    from earlymule.evaluation.early_detection import early_detection
    scores = pd.DataFrame({"acct_id": [1, 2, 3, 1, 2, 3], "snapshot": [5, 5, 5, 6, 6, 6],
                           "score": [0.1, 0.9, 0.0, 0.1, 0.9, 0.0]})
    mules = pd.DataFrame({"acct_id": [1, 3], "onset_day": pd.array([5, 5], dtype="Int64"),
                          "completion_day": pd.array([6, 6], dtype="Int64"),
                          "first_illicit_outflow_day": pd.array([6, pd.NA], dtype="Int64"), "typology": ["a", "b"]})
    r = early_detection(scores, mules, k=1)
    # Only non-mule 2 is ever alerted: nothing detected, precision 0
    assert (r["detected_by_completion"], r["detected_before_completion"], r["alert_precision_active_mule"]) == (0.0, 0.0, 0.0)
    assert (r["mules_can_precede_forwarding"], r["detected_before_forwarding"]) == (1, 0.0)


def test_fast_average_precision_matches_sklearn():
    """The rank formula equals sklearn's average precision, with and without ties."""
    from sklearn.metrics import average_precision_score
    from earlymule.evaluation.uncertainty import average_precision_fast
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 30), rng.normal(0, 1, 500)
    y = np.r_[np.ones(30), np.zeros(500)]
    assert average_precision_fast(pos, np.sort(neg)) == pytest.approx(average_precision_score(y, np.r_[pos, neg]))
    # Ties within and across classes, as LightGBM's piecewise-constant scores produce
    pos_t, neg_t = np.round(pos, 1), np.round(neg, 1)
    assert average_precision_fast(pos_t, np.sort(neg_t)) == pytest.approx(average_precision_score(y, np.r_[pos_t, neg_t]))


def test_case_bootstrap_intervals():
    """Intervals contain the point estimate here and repeat with the same seed."""
    from earlymule.evaluation.uncertainty import bootstrap_detection, bootstrap_ranking
    frame = pd.DataFrame({"acct_id": [1, 2, 3, 4, 5, 6], "snapshot": [1, 1, 1, 2, 2, 2],
                          "label": [1, 0, 1, 1, 0, 0], "score": [0.9, 0.5, 0.1, 0.8, 0.7, 0.2]})
    case_of = pd.Series({1: 10, 3: 11, 4: 12})
    a = bootstrap_ranking(frame, case_of, k=1, n=200, seed=0)
    assert a == bootstrap_ranking(frame, case_of, k=1, n=200, seed=0)
    assert a["recall_at_1_ci95"][0] <= 2 / 3 <= a["recall_at_1_ci95"][1]
    table = pd.DataFrame({"alert_id": [10, 11, 12], "by_completion": [True, False, True],
                          "before_completion": [True, False, False], "before_forwarding": [True, False, False],
                          "can_precede_forwarding": [True, True, False]})
    b = bootstrap_detection(table, n=200, seed=0)
    assert 0 <= b["detected_before_completion_ci95"][0] <= b["detected_before_completion_ci95"][1] <= 1
    # Only cases 10 and 11 count for forwarding; the share is 0, 0.5 or 1 per draw
    assert set(b["detected_before_forwarding_ci95"]) <= {0.0, 0.5, 1.0}


def test_alert_share_on_hard_negatives():
    """Alerts on non-mule alert members are reported as a share of all alerts."""
    from earlymule.evaluation.early_detection import early_detection
    scores = pd.DataFrame({"acct_id": [1, 2, 1, 2], "snapshot": [5, 5, 6, 6], "score": [0.1, 0.9, 0.9, 0.1]})
    mules = pd.DataFrame({"acct_id": [1], "onset_day": [5], "completion_day": [6], "first_illicit_outflow_day": [6],
                          "typology": ["a"]})
    r = early_detection(scores, mules, k=1, other_members=pd.Index([2]))
    assert r["alert_share_other_alert_members"] == 0.5


def test_cooldown_gives_slots_to_new_accounts():
    """An alerted account is skipped until the cooldown ends; its slot goes to the next account."""
    from earlymule.evaluation.early_detection import daily_alerts
    scores = pd.DataFrame({"acct_id": [1, 2, 1, 2, 1, 2], "snapshot": [0, 0, 1, 1, 2, 2],
                           "score": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]})
    assert daily_alerts(scores, k=1).acct_id.tolist() == [1, 1, 1]
    assert daily_alerts(scores, k=1, cooldown=2).acct_id.tolist() == [1, 2, 1]


def test_finished_mule_takes_a_slot_but_no_label():
    """Unlabelled rows (finished mules) stay in the top-k ranking but not in PR-AUC or counts."""
    frame = pd.DataFrame({"snapshot": [1, 1, 1], "label": [np.nan, 1.0, 0.0],
                          "score": [0.9, 0.5, 0.1], "typology": [None, "m", None]})
    m = evaluate(frame, k=1)
    assert (m["rows"], m["unlabelled_rows"], m["recall_at_1"], m["pr_auc"]) == (2, 1, 0.0, 1.0)


def test_paired_bootstrap_difference():
    """Same cases on both sides; a model that catches every eligible mule beats one that catches none."""
    from earlymule.evaluation.uncertainty import paired_bootstrap_difference
    base = pd.DataFrame({"acct_id": [1, 2, 3, 4], "alert_id": [10, 10, 11, 12],
                         "can_precede_forwarding": [True, True, True, False]})
    a = base.assign(before_forwarding=[True, True, True, False])
    b = base.assign(before_forwarding=[False, False, False, True])
    r = paired_bootstrap_difference(a, b, n=200, seed=0)
    assert r["difference"] == 1.0 and r["difference_ci95"] == [1.0, 1.0]
    with pytest.raises(ValueError):
        paired_bootstrap_difference(a, b.iloc[:3], n=10, seed=0)


def test_ties_break_by_account_id():
    """Equal scores rank the lower acct_id first, whatever the row order."""
    from earlymule.evaluation.early_detection import daily_alerts
    scores = pd.DataFrame({"acct_id": [7, 3, 5], "snapshot": [0, 0, 0], "score": [1.0, 1.0, 0.5]})
    assert daily_alerts(scores, k=1).acct_id.tolist() == [3]
    assert daily_alerts(scores.iloc[::-1], k=1, cooldown=30).acct_id.tolist() == [3]


def test_cooldown_ignores_duplicate_index():
    """Concatenated frames with repeated index labels still give one alert per slot."""
    from earlymule.evaluation.early_detection import daily_alerts
    day0 = pd.DataFrame({"acct_id": [1, 2], "snapshot": [0, 0], "score": [0.9, 0.1]})
    day1 = pd.DataFrame({"acct_id": [1, 2], "snapshot": [1, 1], "score": [0.9, 0.1]})
    a = daily_alerts(pd.concat([day0, day1]), k=1, cooldown=30)
    assert list(zip(a.snapshot, a.acct_id)) == [(0, 1), (1, 2)]


def test_rankable_before_forwarding():
    """A mule counts as rankable only if it is scored on a day between onset and first outflow."""
    from earlymule.evaluation.early_detection import detection_table
    # Mule 1 is scored on day 5 (onset) before its outflow on day 6; mule 2 only from its outflow day on
    scores = pd.DataFrame({"acct_id": [1, 1, 2, 9], "snapshot": [5, 6, 6, 5], "score": [0.1, 0.2, 0.3, 0.9]})
    mules = pd.DataFrame({"acct_id": [1, 2], "alert_id": [10, 11], "onset_day": [5, 5], "completion_day": [6, 6],
                          "first_illicit_outflow_day": [6, 6], "typology": ["a", "b"]})
    t = detection_table(scores, mules, k=1).set_index("acct_id")
    assert t.can_precede_forwarding.tolist() == [True, True]
    assert t.rankable_before_forwarding.tolist() == [True, False]


def test_fit_stops_on_pr_auc_only():
    """Early stopping tracks validation PR-AUC alone and keeps its best round."""
    from earlymule.models.tabular import fit
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 4))
    y = (x[:, 0] + rng.normal(scale=2, size=3000) > 2.5).astype(float)
    frame = pd.DataFrame(x, columns=list("abcd")).assign(label=y)
    cfg = {"seed": 0, "early_stopping_rounds": 20,
           "lightgbm": {"n_estimators": 400, "learning_rate": 0.1, "num_leaves": 7, "verbose": -1}}
    model = fit(frame.iloc[:2000], frame.iloc[2000:], list("abcd"), cfg)
    curve = model.evals_result_["valid_0"]
    assert list(curve) == ["average_precision"]
    assert model.best_iteration_ == int(np.argmax(curve["average_precision"])) + 1


def test_threshold_rule_detects_first_crossing():
    """The plan's rule alerts every account-day at or above the validation threshold; no budget, no cooldown."""
    from earlymule.evaluation.early_detection import early_detection, threshold_alerts, threshold_on_validation
    val = pd.DataFrame({"label": [1, 1, 0, 0, 0, np.nan], "score": [0.9, 0.6, 0.7, 0.2, 0.1, 5.0]})
    # F1 is best at 0.6 (both positives, one false alert); the unlabelled row is ignored
    assert threshold_on_validation(val) == 0.6
    scores = pd.DataFrame({"acct_id": [1, 2, 1, 2, 1, 2], "snapshot": [5, 5, 6, 6, 7, 7],
                           "score": [0.7, 0.8, 0.7, 0.1, 0.2, 0.9]})
    mules = pd.DataFrame({"acct_id": [1], "onset_day": [5], "completion_day": [7], "first_illicit_outflow_day": [6],
                          "typology": ["a"]})
    alerts = threshold_alerts(scores, 0.6)
    r = early_detection(scores, mules, None, alerts=alerts)
    # Mule 1 crosses on day 5, before its outflow on day 6; 4 alerts over 3 days, 2 of them on the active mule
    assert (r["detected_before_forwarding"], r["alerts"], r["alerts_per_day"]) == (1.0, 4, 1.33)
    assert r["alert_precision_active_mule"] == 0.5


def test_summary_skips_dirty_runs_and_repeated_seeds():
    """Only clean runs of the latest commit count, each seed once with its latest row."""
    from earlymule.evaluation.summarize_runs import summarize
    m = lambda v: {"test": {"all": {"pr_auc": v}}}  # noqa: E731
    log = pd.DataFrame({"run_name": ["w"] * 5, "commit": ["a", "b", "b", "b", "b"],
                        "dirty": [False, False, False, True, False], "seed": [0, 0, 0, 1, 1],
                        "metrics": [m(0.1), m(0.2), m(0.4), m(0.9), m(0.6)]})
    table = summarize(log, "w")
    assert table.loc["test PR-AUC (all)", "mean"] == 0.5 and table.loc["test PR-AUC (all)", "seeds"] == 2


def test_typology_table_and_threshold_classification():
    """Per-typology timing (plan section 24) and the plan's secondary metrics at a threshold."""
    from earlymule.evaluation.early_detection import early_detection
    from earlymule.evaluation.metrics import threshold_classification
    scores = pd.DataFrame({"acct_id": [1, 2, 3, 1, 2, 3], "snapshot": [5, 5, 5, 6, 6, 6],
                           "score": [0.9, 0.1, 0.5, 0.2, 0.8, 0.1], "label": [1, 1, 0, 1, 1, 0]})
    mules = pd.DataFrame({"acct_id": [1, 2], "onset_day": [5, 5], "completion_day": [6, 6],
                          "first_illicit_outflow_day": [6, 6], "typology": ["a", "b"]})
    r = early_detection(scores, mules, k=1)
    # Mule 1 alerted on day 5 (TTD 0, lead 1), mule 2 on day 6 (TTD 1, lead 0); each typology gets one of 2 alerts
    assert r["by_typology"]["a"] == {"mules": 1, "detected_before_forwarding": 1.0, "detected_by_completion": 1.0,
                                     "median_ttd_days": 0.0, "median_lead_days": 1.0, "alert_hit_share": 0.5}
    assert r["by_typology"]["b"]["median_ttd_days"] == 1.0 and r["by_typology"]["b"]["detected_before_forwarding"] == 0.0
    c = threshold_classification(scores, 0.5)
    # Predictions >= 0.5: rows 1, 3, 5 -> TP 2, FP 1, FN 2
    assert (c["precision"], c["recall"]) == (0.6667, 0.5)


def test_select_runs_takes_latest_clean_commit():
    """Run selection skips dirty rows, takes the latest clean commit and one row per seed."""
    from earlymule.experiment_log import select_runs
    log = pd.DataFrame({"run_name": ["d"] * 4 + ["x"], "commit": ["a", "b", "b", "c", "c"],
                        "dirty": [False, False, False, True, False], "seed": [0, 0, 0, 1, 0], "metrics": [1, 2, 3, 4, 5]})
    runs = select_runs(log, "d")
    assert runs.commit.tolist() == ["b"] and runs.metrics.tolist() == [3]
    assert select_runs(log, "d", "a").metrics.tolist() == [1]
    with pytest.raises(ValueError):
        select_runs(log, "d", "c")


def test_compare_tables_needs_every_seed_above_zero():
    """A is better only when the paired interval is above 0 for each shared seed."""
    from earlymule.evaluation.compare import compare_tables
    base = pd.DataFrame({"acct_id": range(6), "alert_id": range(6), "can_precede_forwarding": True, "k": 10})
    good, bad = base.assign(before_forwarding=True), base.assign(before_forwarding=False)
    r = compare_tables({0: good, 1: good}, {0: bad, 1: bad, 2: bad}, k=10, n=200)
    assert r.seed.tolist() == [0, 1] and r.a_better.all() and r.difference.tolist() == [1.0, 1.0]
    assert not compare_tables({0: bad}, {0: bad}, k=10, n=200).a_better.any()


def test_view_without_hard_negatives():
    """The comparison view drops the non-mule alert members and keeps every other row."""
    from earlymule.evaluation.views import without_hard_negatives
    scores = pd.DataFrame({"acct_id": [1, 2, 3, 2], "snapshot": [5, 5, 5, 6], "score": [0.1, 0.9, 0.5, 0.8]})
    assert without_hard_negatives(scores, pd.Index([2])).acct_id.tolist() == [1, 3]
