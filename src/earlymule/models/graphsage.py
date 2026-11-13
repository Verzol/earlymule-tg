"""Daily GraphSAGE for mule scoring on an exported dataset (python -m earlymule.data.export_events).

The graph of day t holds the transfers of days (t - window, t]: one directed edge per (sender, receiver) pair, built
on the GPU when the day is needed, never stored. Node inputs for message passing are the scaled deposit and the
window's in/out transfer counts and amounts of every account, computed from the same edges. Two SAGEConv layers,
each applied over in-edges and out-edges separately, give the embedding; the account head reads it next to the
deposit and the row's LightGBM features, exactly as the TGN with features does, so the two differ only in the encoder.
Days are scored from their own graph with no state carried over: the window is all the history the graph part sees.
Run: python -m earlymule.models.graphsage --data <export folder> [--config graphsage.yaml] [--seed N] [--out DIR]
     [--max-epochs N] [--lr X]
Writes scores_val.parquet, scores_test.parquet (acct_id, snapshot, score) and run_meta.json to --out
(default experiments/runs/<run_name>[_<dataset>]/<commit>/seed<N>/).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import coalesce

from earlymule.config import load_paths
from earlymule.experiment_log import git_state, run_dir, set_seed
from earlymule.models.tabular import load_config
from earlymule.models.tgn import FeatureScaler, Stream, load_rows, pr_auc


class WindowGraph:
    """Edges and per-account statistics of the transfers in (t - window, t].

    Attributes:
        t: the day.
        edge_index: [2, E] sender -> receiver, one edge per distinct pair.
        edge_day: [E] most recent day the pair transferred (for time encodings: t - edge_day).
        stats: [N, 4] log1p of in count, out count, in amount, out amount over the window (unscaled).
    """

    def __init__(self, stream: Stream, t: int, window: int, num_nodes: int):
        # 1. Rows of the window: day t - window + 1 .. t (empty before the first day)
        self.t = t
        first = max(0, t - window + 1)
        rows = slice(int(stream.start[first]), int(stream.start[min(t, stream.last_day) + 1])) if t >= 0 else slice(0, 0)
        src, dst = stream.src[rows], stream.dst[rows]
        amount = torch.expm1(stream.log_amount[rows])
        # 2. Per-account counts and amounts, from the raw transfers (parallel pairs count each time)
        ones = torch.ones_like(amount)
        cols = [torch.bincount(dst, ones, num_nodes), torch.bincount(src, ones, num_nodes),
                torch.bincount(dst, amount, num_nodes), torch.bincount(src, amount, num_nodes)]
        self.stats = torch.log1p(torch.stack(cols, dim=1))
        # 3. One edge per pair with its latest day; amounts enter through the node statistics, so the edges carry
        #    no weight
        if src.numel():
            self.edge_index, self.edge_day = coalesce(torch.stack([src, dst]), stream.t[rows], num_nodes, reduce="max")
        else:
            self.edge_index = torch.empty(2, 0, dtype=torch.long, device=src.device)
            self.edge_day = torch.empty(0, dtype=torch.long, device=src.device)


class StatScaler:
    """Standardise the window statistics with their mean and std over all accounts and train days."""

    def __init__(self, stream: Stream, days: range, window: int, num_nodes: int):
        s = torch.stack([WindowGraph(stream, t, window, num_nodes).stats for t in days])
        self.mean, self.std = s.mean(dim=(0, 1)), s.std(dim=(0, 1)).clamp_min(1e-6)

    def __call__(self, stats: torch.Tensor) -> torch.Tensor:
        """Scaled statistics."""
        return (stats - self.mean) / self.std


class SAGEAccount(torch.nn.Module):
    """GraphSAGE over the window graph with an account head (mule logit).

    Each layer sums a SAGEConv over in-edges (what an account received) and one over out-edges (what it sent),
    so direction survives the mean aggregation. n_features row features join the head input.
    """

    def __init__(self, deposit: torch.Tensor, dim: int, layers: int, device, n_features: int = 0):
        super().__init__()
        self.convs_in = torch.nn.ModuleList([SAGEConv(5 if i == 0 else dim, dim) for i in range(layers)])
        self.convs_out = torch.nn.ModuleList([SAGEConv(5 if i == 0 else dim, dim) for i in range(layers)])
        self.head = torch.nn.Sequential(Linear(dim + 1 + n_features, dim), torch.nn.ReLU(), Linear(dim, 1))
        self.register_buffer("deposit", deposit)
        self.to(device)

    def embed(self, stats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embedding of every account from the scaled window statistics and the window edges."""
        x = torch.cat([self.deposit.unsqueeze(-1), stats], dim=-1)
        rev = edge_index.flip(0)
        for i, (c_in, c_out) in enumerate(zip(self.convs_in, self.convs_out)):
            x = c_in(x, edge_index) + c_out(x, rev)
            if i + 1 < len(self.convs_in):
                x = x.relu()
        return x

    def score(self, nodes: torch.Tensor, z: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Mule logit of the given accounts; z = embed(...) of the day, x = the rows' features when the head takes them."""
        parts = [z[nodes], self.deposit[nodes].unsqueeze(-1)]
        return self.head(torch.cat(parts if x is None else [*parts, x], dim=-1)).squeeze(-1)

    def score_day(self, nodes: torch.Tensor, g: WindowGraph, stats: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
        """Mule logit of the given accounts at the end of day g.t; stats = the scaled window statistics.

        The interface every daily graph model shares (see earlymule.models.stagt): the training and scoring loops
        below only call this.
        """
        return self.score(nodes, self.embed(stats, g.edge_index), x)


def train_epoch(model, stream: Stream, scaler: StatScaler, train_rows: dict, cfg: dict, opt, rng) -> float:
    """One pass over the train days, one optimiser step per day; returns the mean loss per day."""
    model.train()
    loss_fn, total, steps = BCEWithLogitsLoss(), 0.0, 0
    for d in sorted(train_rows):
        nodes, label, _, x = train_rows[d]
        # All mules of the day, a share of the others, as in the TGN and the daily LightGBM run
        keep = (label == 1) | (rng.random(len(label)) < cfg["train_negative_rate"])
        if not keep.any():
            continue
        g = WindowGraph(stream, d, cfg["window"], model.deposit.numel())
        y = torch.as_tensor(label[keep], dtype=torch.float, device=nodes.device)
        k = torch.as_tensor(keep, device=nodes.device)
        loss = loss_fn(model.score_day(nodes[k], g, scaler(g.stats), None if x is None else x[k]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, steps = total + loss.item(), steps + 1
    return total / max(steps, 1)


@torch.no_grad()
def score_days(model, stream: Stream, scaler: StatScaler, rows: dict, window: int) -> pd.DataFrame:
    """Score each day's rows from that day's graph."""
    model.eval()
    out = []
    for d in sorted(rows):
        nodes, label, acct, x = rows[d]
        g = WindowGraph(stream, d, window, model.deposit.numel())
        frame = pd.DataFrame({"acct_id": acct, "snapshot": d,
                              "score": model.score_day(nodes, g, scaler(g.stats), x).cpu().numpy()})
        out.append(frame if label is None else frame.assign(label=label))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_sage(cfg: dict, deposit: torch.Tensor, n_features: int, device) -> SAGEAccount:
    """The GraphSAGE model of a config."""
    return SAGEAccount(deposit, cfg["dim"], cfg["layers"], device, n_features)


def main(build_model=build_sage, default_config: str = "graphsage.yaml") -> None:
    """Train with early stopping on validation PR-AUC, then write validation and test scores.

    Args:
        build_model: (cfg, deposit, n_features, device) -> model with a score_day method; other daily graph
            models reuse this loop by passing their own builder.
        default_config: config file used when --config is not given.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="export folder (manifest.json, events, nodes, rows_*, features_*)")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--lr", type=float, help="learning rate for a sweep; the run is logged as <run_name>_lr<X>")
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["seed"] = cfg["seed"] if args.seed is None else args.seed
    cfg["max_epochs"] = args.max_epochs or cfg["max_epochs"]
    if args.lr is not None:
        cfg["lr"], cfg["run_name"] = args.lr, f"{cfg['run_name']}_lr{args.lr:g}"
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    # 1. Export tables; feature and statistic scaling from the train period only
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    splits = manifest["splits"]
    stream = Stream(pd.read_parquet(data / "events.parquet"), splits["train"][1], device)
    nodes = pd.read_parquet(data / "nodes.parquet")
    deposit = np.log1p(nodes.initial_deposit.to_numpy())
    deposit = torch.as_tensor((deposit - deposit.mean()) / deposit.std(), dtype=torch.float, device=device)
    scaler = None
    if cfg.get("account_features"):
        scaler = FeatureScaler(pd.read_parquet(data / "features_train.parquet").drop(columns=["acct_id", "snapshot"]))
    train_rows = load_rows(data, "train", scaler, device)
    val_rows = load_rows(data, "val", scaler, device)
    stats = StatScaler(stream, range(splits["train"][0], splits["train"][1] + 1), cfg["window"], len(nodes))

    # 2. Train; keep the parameters of the epoch with the best validation PR-AUC
    model = build_model(cfg, deposit, scaler.width if scaler else 0, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_epoch, best_state, epochs = -1.0, 0, None, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss = train_epoch(model, stream, stats, train_rows, cfg, opt, rng)
        t1 = time.time()
        val = pr_auc(score_days(model, stream, stats, val_rows, cfg["window"]))
        epochs.append({"epoch": epoch, "loss": round(loss, 4), "val_pr_auc": round(val, 4),
                       "train_seconds": round(t1 - t0, 1), "val_seconds": round(time.time() - t1, 1),
                       "peak_mib": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None})
        print(epochs[-1], flush=True)
        if val > best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break

    # 3. Scores of the best epoch for validation and test
    model.load_state_dict(best_state)
    test_rows = load_rows(data, "test", scaler, device)
    scores = {"val": score_days(model, stream, stats, val_rows, cfg["window"]),
              "test": score_days(model, stream, stats, test_rows, cfg["window"])}

    # 4. Write scores and the run record; evaluation runs locally on these files
    commit, dirty = git_state()
    dataset = manifest["dataset"]
    run_name = cfg["run_name"] + ("" if dataset == load_paths().dataset else f"_{dataset}")
    out = Path(args.out) if args.out else run_dir(run_name, cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    scores["val"].to_parquet(out / "scores_val.parquet", index=False)
    scores["test"].to_parquet(out / "scores_test.parquet", index=False)
    meta = {"run_name": run_name, "dataset": dataset, "export_commit": manifest["commit"], "commit": commit,
            "dirty": dirty, "seed": cfg["seed"], "config": cfg, "device": torch.cuda.get_device_name(0)
            if device.type == "cuda" else "cpu", "params": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch, "best_val_pr_auc": round(best, 4), "epochs": epochs, "seconds": round(time.time() - start, 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ["run_name", "best_val_pr_auc", "seconds"]} | {"out": str(out)}))


if __name__ == "__main__":
    main()
