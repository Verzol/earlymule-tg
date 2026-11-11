"""Leakage and determinism tests for the minimal TGN; skipped where torch and PyG are not installed (CPU env)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from earlymule.models.tgn import (NeighborBuffer, Stream, TGNAccount, rows_by_day,  # noqa: E402
                                  sample_negatives, score_pass, train_epoch)

NODES = 8  # node 7 never transfers


def events(seed: int = 0) -> pd.DataFrame:
    """Random transfers among nodes 0-6 over days 0-11, several per day."""
    rng = np.random.default_rng(seed)
    n = 120
    return pd.DataFrame({"src": rng.integers(0, 7, n), "dst": rng.integers(0, 7, n), "day": rng.integers(0, 12, n),
                         "log_amount": rng.uniform(1, 9, n).astype("float32"), "dow": rng.integers(0, 7, n)})


def model() -> TGNAccount:
    """Small untrained model with fixed weights."""
    torch.manual_seed(0)
    return TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu")


def all_rows(days) -> dict:
    """Every node scored on every given day."""
    rows = pd.DataFrame([(a, d, a) for d in days for a in range(NODES)], columns=["acct_id", "snapshot", "node"])
    return rows_by_day(rows, "cpu")


def test_scores_ignore_transfers_after_the_day():
    """Scores up to day 5 are the same when later transfers are removed or changed."""
    e, m = events(), model()
    full = score_pass(m, Stream(e, 3, "cpu"), {"s": all_rows(range(6))}, 5)["s"]
    cut = score_pass(m, Stream(e[e.day.le(5)], 3, "cpu"), {"s": all_rows(range(6))}, 5)["s"]
    changed = e.assign(log_amount=np.where(e.day.gt(5), 99.0, e.log_amount), dst=np.where(e.day.gt(5), 0, e.dst))
    moved = score_pass(m, Stream(changed, 3, "cpu"), {"s": all_rows(range(6))}, 5)["s"]
    np.testing.assert_allclose(full.score, cut.score, atol=1e-6)
    np.testing.assert_allclose(full.score, moved.score, atol=1e-6)


def test_same_day_order_does_not_matter():
    """Shuffling the stored rows gives the same scores: same-day transfers have no known order."""
    e, m = events(), model()
    a = score_pass(m, Stream(e, 8, "cpu"), {"s": all_rows(range(12))}, 11)["s"]
    b = score_pass(m, Stream(e.sample(frac=1, random_state=1), 8, "cpu"), {"s": all_rows(range(12))}, 11)["s"]
    np.testing.assert_allclose(a.score, b.score, atol=1e-6)


def test_memory_changes_only_with_own_transfers():
    """An account without transfers keeps an empty memory; accounts with transfers do not."""
    m = model()
    score_pass(m, Stream(events(), 8, "cpu"), {}, 11)
    assert torch.all(m.memory.memory[7] == 0) and m.memory.last_update[7] == 0
    assert torch.any(m.memory.memory[:7] != 0, dim=1).all()


def test_negatives_only_from_seen_accounts():
    """Negative destinations are drawn from accounts seen so far."""
    seen = torch.zeros(10, dtype=torch.bool)
    seen[[2, 5]] = True
    neg = sample_negatives(seen, 500, torch.Generator().manual_seed(0))
    assert set(neg.tolist()) == {2, 5}


def test_neighbor_buffer_keeps_newest_events():
    """With more same-batch events than slots, the newest ones stay, also across batches."""
    buf = NeighborBuffer(10, size=3, device="cpu")
    buf.insert(torch.zeros(5, dtype=torch.long), torch.tensor([1, 2, 3, 4, 5]))
    n_id, edge_index, e_id = buf(torch.tensor([0]))
    assert sorted(e_id.tolist()) == [2, 3, 4]
    buf.insert(torch.tensor([0]), torch.tensor([6]))
    _, _, e_id = buf(torch.tensor([0]))
    assert sorted(e_id.tolist()) == [3, 4, 5]


def test_train_epoch_gives_finite_loss():
    """One training pass with link and account losses runs and returns a finite loss."""
    m = model()
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    cfg = {"link_weight": 1.0, "train_negative_rate": 1.0}
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, Stream(events(), 8, "cpu"), rows_by_day(rows, "cpu"), 8, cfg, opt,
                       np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert np.isfinite(loss) and loss > 0
