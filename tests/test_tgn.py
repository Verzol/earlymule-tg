"""Leakage and determinism tests for the minimal TGN; skipped where torch and PyG are not installed (CPU env)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from earlymule.models.tgn import (FeatureScaler, NeighborBuffer, Stream, TGNAccount, load_rows,  # noqa: E402
                                  rows_by_day, sample_negatives, score_pass, train_epoch)

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


def test_feature_scaler_uses_train_statistics():
    """Train rows set the scale; a missing value becomes 0 with a flag; a signed column skips log1p; a negative
    value in an unsigned column is rejected."""
    train = pd.DataFrame({"a": [0.0, 1.0, 3.0], "b": [1.0, np.nan, 1.0], "s": [-1.0, 0.0, 1.0]})
    sc = FeatureScaler(train)
    mean = sc.mean.copy()
    assert sc.width == 4  # a, b, s and the missing flag of b
    assert list(sc.signed) == [False, False, True]
    z = sc.transform(pd.DataFrame({"a": [100.0, 0.0], "b": [np.nan, 1.0], "s": [-1.0, 1.0]}))
    assert z[0, 1] == 0 and z[0, 3] == 1 and z[1, 3] == 0
    np.testing.assert_allclose(sc.transform(train)[:, 0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(z[:, 2], [-np.sqrt(1.5), np.sqrt(1.5)], atol=1e-6)  # (−1 − 0) / std([−1, 0, 1])
    # Transforming other rows never refits the scale
    np.testing.assert_array_equal(sc.mean, mean)
    with pytest.raises(ValueError):
        sc.transform(pd.DataFrame({"a": [-1.0], "b": [1.0], "s": [0.0]}))


def test_row_features_reach_the_score(tmp_path):
    """Features follow their rows by day; they change the score; misordered feature files are refused."""
    rows = pd.DataFrame({"acct_id": [1, 2, 1, 2], "snapshot": [3, 3, 4, 4], "node": [1, 2, 1, 2]})
    feats = rows[["acct_id", "snapshot"]].assign(f=[0.0, 5.0, 1.0, 9.0])
    rows.to_parquet(tmp_path / "rows_test.parquet")
    feats.to_parquet(tmp_path / "features_test.parquet")
    sc = FeatureScaler(feats[["f"]])
    by_day = load_rows(tmp_path, "test", sc, "cpu")
    np.testing.assert_allclose(by_day[4][3].numpy()[:, 0], sc.transform(feats[["f"]])[2:, 0])
    torch.manual_seed(0)
    m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=sc.width)
    s = score_pass(m, Stream(events(), 8, "cpu"), {"s": by_day}, 4)["s"]
    zero = {d: (*v[:3], torch.zeros_like(v[3])) for d, v in by_day.items()}
    s0 = score_pass(m, Stream(events(), 8, "cpu"), {"s": zero}, 4)["s"]
    assert not np.allclose(s.score, s0.score)
    feats.iloc[[0, 1]] = feats.iloc[[1, 0]].to_numpy()
    feats.to_parquet(tmp_path / "features_test.parquet")
    with pytest.raises(ValueError):
        load_rows(tmp_path, "test", sc, "cpu")


def test_train_epoch_with_features_gives_finite_loss():
    """One training pass with row features in the account head runs and returns a finite loss."""
    torch.manual_seed(0)
    m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=2)
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 2)).astype(np.float32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, Stream(events(), 8, "cpu"), rows_by_day(rows, "cpu", x), 8,
                       {"link_weight": 1.0, "train_negative_rate": 1.0}, opt, np.random.default_rng(0),
                       torch.Generator().manual_seed(0))
    assert np.isfinite(loss) and loss > 0


def test_without_graph_scores_ignore_transfers():
    """use_graph=False: scores depend on deposit and features only, and a training pass still runs."""
    torch.manual_seed(0)
    m = TGNAccount(NODES, torch.arange(NODES, dtype=torch.float), dim=8, neighbors=3, device="cpu", n_features=2,
                   use_graph=False)
    assert not hasattr(m, "memory")
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 2)).astype(np.float32)
    a = score_pass(m, Stream(events(0), 8, "cpu"), {"s": rows_by_day(rows, "cpu", x)}, 5)["s"]
    b = score_pass(m, Stream(events(1), 8, "cpu"), {"s": rows_by_day(rows, "cpu", x)}, 5)["s"]
    np.testing.assert_allclose(a.score, b.score, atol=1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, Stream(events(), 8, "cpu"), rows_by_day(rows, "cpu", x), 8,
                       {"link_weight": 1.0, "train_negative_rate": 1.0}, opt, np.random.default_rng(0),
                       torch.Generator().manual_seed(0))
    assert np.isfinite(loss) and loss > 0


def test_role_groups_split_inputs_by_window():
    """Every input lands in exactly one window group or the rest; flags follow their column."""
    from earlymule.models.tgn import role_groups
    names = ["n_in_1d", "fwd_0d_7d", "burst_30d", "n_out_7d_over_30d", "days_since_last_tx", "fwd_0d_7d__missing"]
    groups, rest = role_groups(names)
    assert groups == {"1d": [0], "7d": [1, 5], "30d": [2]} and rest == [3, 4]
    assert sorted(sum(groups.values(), rest)) == list(range(len(names)))


def test_role_vector_depends_only_on_its_window():
    """Changing a 7-day input changes r through the 7-day MLP only; the rest passes through untouched."""
    from earlymule.models.tgn import RoleEncoder
    torch.manual_seed(0)
    enc = RoleEncoder({"1d": [0], "7d": [1], "30d": [2]}, [3], dim=4).eval()
    x = torch.tensor([[0.5, 1.0, -0.5, 9.0]])
    r, alpha, rest = enc(x)
    assert r.shape == (1, 4) and alpha.shape == (1, 3) and rest.tolist() == [[9.0]]
    assert torch.allclose(alpha.sum(1), torch.ones(1))
    per_window = [enc.mlps[i](x[:, [i]]) for i in range(3)]
    x2 = x.clone()
    x2[0, 1] = 5.0
    per_window2 = [enc.mlps[i](x2[:, [i]]) for i in range(3)]
    assert torch.allclose(per_window[0], per_window2[0]) and torch.allclose(per_window[2], per_window2[2])
    assert not torch.allclose(per_window[1], per_window2[1])


def test_train_epoch_with_role_encoder_gives_finite_loss():
    """One training pass with the role encoder in the account head runs, returns a finite loss and logs r."""
    from earlymule.models.tgn import RoleEncoder
    torch.manual_seed(0)
    role = RoleEncoder({"1d": [0], "7d": [1], "30d": [2]}, [3], dim=4)
    m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=4, role=role)
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = train_epoch(m, Stream(events(), 8, "cpu"), rows_by_day(rows, "cpu", x), 8,
                       {"link_weight": 1.0, "train_negative_rate": 1.0}, opt, np.random.default_rng(0),
                       torch.Generator().manual_seed(0))
    assert np.isfinite(loss) and loss > 0
    score_pass(m, Stream(events(), 8, "cpu"), {"val": rows_by_day(rows, "cpu", x)}, 5)
    assert m.role_log["r_std_mean"] >= 0 and len(m.role_log["alpha_mean"]) == 3


def test_transition_reads_scale_differences():
    """q changes when the 1-day role moves away from the 7-day role, even if r is held fixed."""
    from earlymule.models.tgn import TransitionEncoder
    torch.manual_seed(0)
    enc = TransitionEncoder(dim=4).eval()
    r = torch.zeros(1, 4)
    same = torch.ones(1, 3, 4)
    switched = same.clone()
    switched[0, 0] = 3.0
    assert not torch.allclose(enc(r, same), enc(r, switched))


def test_full_model_steps_run_and_log_the_gate():
    """Role, transition and gated fusion stack; a missing step is refused; the gate is logged in [0, 1]."""
    from earlymule.models.tgn import GatedFusion, RoleEncoder, TransitionEncoder
    rows = pd.DataFrame({"acct_id": [1, 2, 3] * 4, "snapshot": np.repeat([2, 3, 4, 5], 3), "node": [1, 2, 3] * 4,
                         "label": [1.0, 0.0, 0.0] * 4})
    x = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    groups = {"1d": [0], "7d": [1], "30d": [2]}
    with pytest.raises(ValueError):
        TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=4,
                   transition=TransitionEncoder(4))
    for with_fusion in (False, True):
        torch.manual_seed(0)
        m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=4,
                       role=RoleEncoder(groups, [3], dim=4), transition=TransitionEncoder(4),
                       fusion=GatedFusion(8, 4, 8) if with_fusion else None)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loss = train_epoch(m, Stream(events(), 8, "cpu"), rows_by_day(rows, "cpu", x), 8,
                           {"link_weight": 1.0, "train_negative_rate": 1.0}, opt, np.random.default_rng(0),
                           torch.Generator().manual_seed(0))
        assert np.isfinite(loss) and loss > 0
        score_pass(m, Stream(events(), 8, "cpu"), {"val": rows_by_day(rows, "cpu", x)}, 5)
        assert ("gate_mean" in m.role_log) == with_fusion
        if with_fusion:
            assert 0 <= m.role_log["gate_mean"] <= 1


def test_full_model_scores_ignore_transfers_after_the_day():
    """The full model, like the TGN, gives the same day-5 scores when later transfers change."""
    from earlymule.models.tgn import GatedFusion, RoleEncoder, TransitionEncoder
    torch.manual_seed(0)
    m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=4,
                   role=RoleEncoder({"1d": [0], "7d": [1], "30d": [2]}, [3], dim=4), transition=TransitionEncoder(4),
                   fusion=GatedFusion(8, 4, 8))
    rows = pd.DataFrame([(a, d, a) for d in range(6) for a in range(NODES)], columns=["acct_id", "snapshot", "node"])
    x = np.random.default_rng(0).normal(size=(len(rows), 4)).astype(np.float32)
    e = events()
    full = score_pass(m, Stream(e, 3, "cpu"), {"s": rows_by_day(rows, "cpu", x)}, 5)["s"]
    changed = e.assign(log_amount=np.where(e.day.gt(5), 99.0, e.log_amount), dst=np.where(e.day.gt(5), 0, e.dst))
    moved = score_pass(m, Stream(changed, 3, "cpu"), {"s": rows_by_day(rows, "cpu", x)}, 5)["s"]
    np.testing.assert_allclose(full.score, moved.score, atol=1e-6)


def test_temporal_transition_reads_the_lagged_role(tmp_path):
    """With lag rows the input is [x(t) ; x(t − 7)]; changing only the lagged half changes q, and a misordered lag
    file is refused."""
    from earlymule.models.tgn import RoleEncoder, TransitionEncoder
    rows = pd.DataFrame({"acct_id": [1, 2, 1, 2], "snapshot": [3, 3, 4, 4], "node": [1, 2, 1, 2]})
    cur = rows[["acct_id", "snapshot"]].assign(a_1d=[0.0, 5.0, 1.0, 9.0], b_7d=[1.0, 2.0, 3.0, 4.0], c_30d=[2.0] * 4,
                                               d=[0.0] * 4)
    lag = cur.assign(a_1d=[0.0] * 4)
    rows.to_parquet(tmp_path / "rows_test.parquet")
    cur.to_parquet(tmp_path / "features_test.parquet")
    lag.to_parquet(tmp_path / "features_lag7_test.parquet")
    sc = FeatureScaler(cur.drop(columns=["acct_id", "snapshot"]))
    by_day = load_rows(tmp_path, "test", sc, "cpu", lag=7)
    assert by_day[4][3].shape[1] == 2 * sc.width
    torch.manual_seed(0)
    m = TGNAccount(NODES, torch.zeros(NODES), dim=8, neighbors=3, device="cpu", n_features=sc.width,
                   role=RoleEncoder({"1d": [0], "7d": [1], "30d": [2]}, [3], dim=4),
                   transition=TransitionEncoder(4, temporal=True)).eval()
    s = score_pass(m, Stream(events(), 8, "cpu"), {"s": by_day}, 4)["s"]
    same = {d: (*v[:3], torch.cat([v[3][:, :sc.width], v[3][:, :sc.width]], dim=1)) for d, v in by_day.items()}
    s_same = score_pass(m, Stream(events(), 8, "cpu"), {"s": same}, 4)["s"]
    assert not np.allclose(s.score, s_same.score)
    lag.iloc[[0, 1]] = lag.iloc[[1, 0]].to_numpy()
    lag.to_parquet(tmp_path / "features_lag7_test.parquet")
    with pytest.raises(ValueError):
        load_rows(tmp_path, "test", sc, "cpu", lag=7)
