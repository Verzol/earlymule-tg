"""Relation, token and leakage tests for the STA-GT-inspired model; skipped without torch and PyG (CPU env)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from earlymule.models.graphsage import StatScaler, WindowGraph, score_days, train_epoch  # noqa: E402
from earlymule.models.stagt import IN, OUT, STAGTAccount, recent_neighbours, shared_partner_edges  # noqa: E402
from earlymule.models.tgn import Stream, rows_by_day  # noqa: E402

NODES = 8


def events(seed: int = 0) -> pd.DataFrame:
    """Random transfers among nodes 0-6 over days 0-11, several per day."""
    rng = np.random.default_rng(seed)
    n = 120
    return pd.DataFrame({"src": rng.integers(0, 7, n), "dst": rng.integers(0, 7, n), "day": rng.integers(0, 12, n),
                         "log_amount": rng.uniform(1, 9, n).astype("float32"), "dow": rng.integers(0, 7, n)})


def model(n_features: int = 0) -> STAGTAccount:
    """Small untrained model with fixed weights, in eval mode (no dropout)."""
    torch.manual_seed(0)
    return STAGTAccount(torch.zeros(NODES), dim=8, layers=2, tokens=3, device="cpu", n_features=n_features,
                        shared_cap=3).eval()


def all_rows(days) -> dict:
    """Every node scored on every given day."""
    rows = pd.DataFrame([(a, d, a) for d in days for a in range(NODES)], columns=["acct_id", "snapshot", "node"])
    return rows_by_day(rows, "cpu")


def test_shared_partner_edges_pair_co_payers_under_the_cap():
    """1, 2, 3 all pay 0 on different days: pairs among them with the earlier day; 4 payers of 5 exceed cap 3."""
    ei = torch.tensor([[1, 2, 3, 0, 0, 6, 7, 1, 2], [0, 0, 0, 1, 2, 5, 5, 5, 5]])
    day = torch.tensor([3, 5, 9, 1, 4, 1, 1, 1, 1])
    sh, sh_day = shared_partner_edges(ei, day, NODES, cap=3)
    pairs = dict(zip(zip(sh[0].tolist(), sh[1].tolist()), sh_day.tolist()))
    assert pairs[(1, 2)] == 3 and pairs[(2, 1)] == 3 and pairs[(1, 3)] == 3 and pairs[(2, 3)] == 5
    # 1 and 2 are also payees of 0: that link (days 1 and 4 -> 1) is older, so the pair keeps day 3
    assert pairs[(1, 2)] == 3
    assert not any(a in (6, 7) or b in (6, 7) for a, b in pairs)


def test_recent_neighbours_keeps_newest_partners():
    """Node 0 has four partners; the three most recent are kept, newest first, ties by partner id."""
    ei = torch.tensor([[1, 0, 3, 4], [0, 2, 0, 0]])
    day = torch.tensor([2, 9, 9, 1])
    nb, rel, d, pad = recent_neighbours(torch.tensor([0, 5]), ei, day, NODES, k=3)
    assert nb[0].tolist() == [2, 3, 1] and rel[0].tolist() == [OUT, IN, IN] and d[0].tolist() == [9, 9, 2]
    assert not pad[0].any() and pad[1].all()


def test_scores_ignore_transfers_outside_the_window():
    """Scores of day t do not change when transfers after t, or before the window, are changed."""
    e, m = events(), model()
    sc = StatScaler(Stream(e, 11, "cpu"), range(0, 12), 3, NODES)
    base = score_days(m, Stream(e, 11, "cpu"), sc, all_rows([7]), 3)
    later = e.assign(log_amount=np.where(e.day.gt(7), 99.0, e.log_amount), dst=np.where(e.day.gt(7), 0, e.dst))
    earlier = e.assign(log_amount=np.where(e.day.lt(5), 99.0, e.log_amount), src=np.where(e.day.lt(5), 0, e.src))
    for changed in (later, earlier, e[e.day.between(5, 7)]):
        s = score_days(m, Stream(changed, 11, "cpu"), sc, all_rows([7]), 3)
        np.testing.assert_allclose(base.score, s.score, atol=1e-5)
    inside = e.assign(dst=np.where(e.day.eq(6), 0, e.dst))
    assert not np.allclose(base.score, score_days(m, Stream(inside, 11, "cpu"), sc, all_rows([7]), 3).score)


def test_train_epoch_with_features_gives_finite_loss():
    """One training pass with row features in the account head runs and returns a finite loss."""
    m = model(n_features=2).train()
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 2)).astype(np.float32)
    stream = Stream(events(), 8, "cpu")
    sc = StatScaler(stream, range(0, 9), 3, NODES)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, stream, sc, rows_by_day(rows, "cpu", x), {"window": 3, "train_negative_rate": 1.0}, opt,
                       np.random.default_rng(0))
    assert np.isfinite(loss) and loss > 0
