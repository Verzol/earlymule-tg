"""Window and leakage tests for the daily GraphSAGE; skipped where torch and PyG are not installed (CPU env)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from earlymule.models.graphsage import SAGEAccount, StatScaler, WindowGraph, score_days, train_epoch  # noqa: E402
from earlymule.models.tgn import Stream, rows_by_day  # noqa: E402

NODES = 8  # node 7 never transfers


def events(seed: int = 0) -> pd.DataFrame:
    """Random transfers among nodes 0-6 over days 0-11, several per day."""
    rng = np.random.default_rng(seed)
    n = 120
    return pd.DataFrame({"src": rng.integers(0, 7, n), "dst": rng.integers(0, 7, n), "day": rng.integers(0, 12, n),
                         "log_amount": rng.uniform(1, 9, n).astype("float32"), "dow": rng.integers(0, 7, n)})


def model(n_features: int = 0) -> SAGEAccount:
    """Small untrained model with fixed weights."""
    torch.manual_seed(0)
    return SAGEAccount(torch.zeros(NODES), dim=8, layers=2, device="cpu", n_features=n_features)


def all_rows(days) -> dict:
    """Every node scored on every given day."""
    rows = pd.DataFrame([(a, d, a) for d in days for a in range(NODES)], columns=["acct_id", "snapshot", "node"])
    return rows_by_day(rows, "cpu")


def test_window_graph_holds_only_the_window():
    """The graph of day t has one edge per pair of days (t - window, t]; the statistics count those transfers."""
    e = events()
    g = WindowGraph(Stream(e, 11, "cpu"), 7, 3, NODES)
    win = e[e.day.between(5, 7)]
    pairs = set(zip(win.src, win.dst))
    assert set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist())) == pairs
    np.testing.assert_allclose(g.stats[:, 1].numpy(), np.log1p(np.bincount(win.src, minlength=NODES)), rtol=1e-5)
    np.testing.assert_allclose(g.stats[:, 2].numpy(),
                               np.log1p(np.bincount(win.dst, np.expm1(win.log_amount), NODES)), rtol=1e-4)
    assert g.stats[7].sum() == 0 and WindowGraph(Stream(e, 11, "cpu"), 11, 30, NODES).edge_index.shape[1] > 0


def test_scores_ignore_transfers_outside_the_window():
    """Scores of day t do not change when transfers after t, or before t - window + 1, are changed."""
    e, m = events(), model()
    sc = StatScaler(Stream(e, 11, "cpu"), range(0, 12), 3, NODES)
    base = score_days(m, Stream(e, 11, "cpu"), sc, all_rows([7]), 3)
    later = e.assign(log_amount=np.where(e.day.gt(7), 99.0, e.log_amount), dst=np.where(e.day.gt(7), 0, e.dst))
    earlier = e.assign(log_amount=np.where(e.day.lt(5), 99.0, e.log_amount), src=np.where(e.day.lt(5), 0, e.src))
    for changed in (later, earlier, e[e.day.between(5, 7)]):
        s = score_days(m, Stream(changed, 11, "cpu"), sc, all_rows([7]), 3)
        np.testing.assert_allclose(base.score, s.score, atol=1e-6)
    # Transfers inside the window do matter
    inside = e.assign(dst=np.where(e.day.eq(6), 0, e.dst))
    assert not np.allclose(base.score, score_days(m, Stream(inside, 11, "cpu"), sc, all_rows([7]), 3).score)


def test_same_day_order_does_not_matter():
    """Shuffling the stored rows gives the same scores."""
    e, m = events(), model()
    sc = StatScaler(Stream(e, 11, "cpu"), range(0, 12), 3, NODES)
    a = score_days(m, Stream(e, 8, "cpu"), sc, all_rows(range(12)), 3)
    b = score_days(m, Stream(e.sample(frac=1, random_state=1), 8, "cpu"), sc, all_rows(range(12)), 3)
    np.testing.assert_allclose(a.score, b.score, atol=1e-6)


def test_train_epoch_with_features_gives_finite_loss():
    """One training pass with row features in the account head runs and returns a finite loss."""
    m = model(n_features=2)
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 2)).astype(np.float32)
    stream = Stream(events(), 8, "cpu")
    sc = StatScaler(stream, range(0, 9), 3, NODES)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, stream, sc, rows_by_day(rows, "cpu", x), {"window": 3, "train_negative_rate": 1.0}, opt,
                       np.random.default_rng(0))
    assert np.isfinite(loss) and loss > 0
