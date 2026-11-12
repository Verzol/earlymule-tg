"""Minimal TGN for daily mule scoring on an exported dataset (python -m earlymule.data.export_events).

Each day is one batch of transfers: the model predicts the day's transfers from the state before the day
(self-supervised link loss), then adds them to the account memory, then scores accounts at the end of the day
(account loss on train rows). Same-day order is unknown, so a day's messages are averaged and events are kept in a
fixed canonical order. Validation and test scores come from a pass from day 0 without training.
With account_features in the config, the account head also reads the LightGBM features of each row
(features_<split>.parquet of the export), which separates what the graph memory adds from what the features carry.
With use_graph: false the memory, attention and link loss are left out and the same head is trained on the
features alone, as the control for that comparison.
Run: python -m earlymule.models.tgn --data <export folder> [--config tgn.yaml] [--seed N] [--out DIR] [--max-epochs N]
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
from sklearn.metrics import average_precision_score
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastNeighborLoader, MeanAggregator

from earlymule.config import load_paths
from earlymule.experiment_log import git_state, run_dir, set_seed
from earlymule.models.tabular import load_config


class Stream:
    """Transfers of an export as tensors, in day order and a fixed order within each day.

    The fixed order (src, dst, amount) makes every result independent of how same-day rows were stored.
    """

    def __init__(self, events: pd.DataFrame, train_last_day: int, device):
        e = events.sort_values(["day", "src", "dst", "log_amount", "dow"], kind="stable").reset_index(drop=True)
        # Amount scaled with train-period statistics only
        train_amt = e.log_amount[e.day.le(train_last_day)]
        amount = (e.log_amount - train_amt.mean()) / train_amt.std()
        self.src = torch.tensor(e.src.to_numpy(), dtype=torch.long, device=device)
        self.dst = torch.tensor(e.dst.to_numpy(), dtype=torch.long, device=device)
        self.t = torch.tensor(e.day.to_numpy(), dtype=torch.long, device=device)
        self.msg = torch.as_tensor(np.c_[amount, e.dow / 6.0], dtype=torch.float, device=device)
        self.last_day = int(e.day.max()) if len(e) else -1
        # Row range of each day: day d is rows start[d]:start[d + 1]
        self.start = np.searchsorted(e.day.to_numpy(), np.arange(self.last_day + 2))

    def day(self, d: int) -> slice:
        """Rows of day d (empty after the last day)."""
        if d > self.last_day:
            return slice(0, 0)
        return slice(int(self.start[d]), int(self.start[d + 1]))


class NeighborBuffer(LastNeighborLoader):
    """LastNeighborLoader whose insert keeps, per account, exactly the most recent `size` events.

    PyG's insert writes several same-batch events of one account into the same slot when it has more than `size`
    of them, and which write wins is undefined on GPU; here the latest events by event id win, deterministically.
    """

    def insert(self, src, dst):
        """Add transfers (src, dst) with the next event ids, as undirected neighbours."""
        # 1. Both directions, with their event ids
        neighbors, nodes = torch.cat([src, dst]), torch.cat([dst, src])
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.numel(), device=src.device).repeat(2)
        self.cur_e_id += src.numel()
        # 2. Per node, newest events first; keep the first `size`
        perm = torch.argsort(e_id, descending=True, stable=True)
        perm = perm[torch.argsort(nodes[perm], stable=True)]
        nodes, neighbors, e_id = nodes[perm], neighbors[perm], e_id[perm]
        n_id, counts = torch.unique_consecutive(nodes, return_counts=True)
        rank = torch.arange(nodes.numel(), device=nodes.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        keep = rank < self.size
        # 3. Dense [node, slot] layout, merged with the stored neighbours; the `size` newest stay
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)
        dense = self._assoc[nodes[keep]] * self.size + rank[keep]
        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense] = e_id[keep]
        dense_nb = neighbors.new_zeros(n_id.numel() * self.size)
        dense_nb[dense] = neighbors[keep]
        all_e_id = torch.cat([self.e_id[n_id], dense_e_id.view(-1, self.size)], dim=-1)
        all_nb = torch.cat([self.neighbors[n_id], dense_nb.view(-1, self.size)], dim=-1)
        top, perm = all_e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = top
        self.neighbors[n_id] = torch.gather(all_nb, 1, perm)


class FeatureScaler:
    """log1p, then standardise with train-row statistics; a missing value becomes 0 plus a missing flag.

    The features are counts, amounts, ratios and day gaps: non-negative with long tails, hence log1p.
    NaN means "nothing to divide by" or "never happened", which the flag keeps apart from an average value.
    """

    def __init__(self, train: pd.DataFrame):
        x = self._log(train)
        self.columns = list(train.columns)
        self.mean, self.std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        self.mean, self.std = np.nan_to_num(self.mean), np.where(self.std > 0, self.std, 1.0)
        # Flags only for columns ever missing on train; the others would be constant
        self.flagged = np.isnan(x).any(axis=0)

    @staticmethod
    def _log(frame: pd.DataFrame) -> np.ndarray:
        """Values on a log scale; negative values would break log1p, so they are rejected."""
        x = frame.to_numpy(dtype=np.float32)
        if (x < 0).any():
            raise ValueError("account features must be non-negative")
        return np.log1p(x)

    @property
    def width(self) -> int:
        """Number of model inputs: one per feature plus one per missing flag."""
        return len(self.columns) + int(self.flagged.sum())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Model inputs of each row, float32."""
        x = self._log(frame[self.columns])
        missing = np.isnan(x)
        z = np.where(missing, 0.0, (x - self.mean) / self.std)
        return np.c_[z, missing[:, self.flagged]].astype(np.float32)


class TGNAccount(torch.nn.Module):
    """TGN memory and attention embedding with a link head and an account head (mule logit).

    n_features > 0 adds that many row features to the account head input. use_graph=False drops the memory,
    attention and link head: the account head then reads only the deposit and the row features.
    """

    def __init__(self, num_nodes: int, deposit: torch.Tensor, dim: int, neighbors: int, device, n_features: int = 0,
                 use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            msg_dim = 2
            self.memory = TGNMemory(num_nodes, msg_dim, dim, dim, message_module=IdentityMessage(msg_dim, dim, dim),
                                    aggregator_module=MeanAggregator())
            self.conv = TransformerConv(dim, dim // 2, heads=2, dropout=0.1, edge_dim=msg_dim + dim)
            self.link_src, self.link_dst, self.link_out = Linear(dim, dim), Linear(dim, dim), Linear(dim, 1)
        head_in = (dim if use_graph else 0) + 1 + n_features
        self.head = torch.nn.Sequential(Linear(head_in, dim), torch.nn.ReLU(), Linear(dim, 1))
        self.register_buffer("deposit", deposit)
        self.to(device)
        if use_graph:
            self.buffer = NeighborBuffer(num_nodes, size=neighbors, device=device)

    def reset(self):
        """Empty memory and neighbours: every pass starts at day 0."""
        if self.use_graph:
            self.memory.reset_state()
            self.buffer.reset_state()

    def detach(self):
        """Cut the memory's gradient history after a training step."""
        if self.use_graph:
            self.memory.detach()

    def embed(self, nodes: torch.Tensor, stream: Stream) -> torch.Tensor:
        """Embedding of each node from its memory and its recent neighbours (state after the last observe)."""
        n_id, edge_index, e_id = self.buffer(nodes)
        z, last_update = self.memory(n_id)
        rel_t = (last_update[edge_index[0]] - stream.t[e_id]).float()
        z = self.conv(z, edge_index, torch.cat([self.memory.time_enc(rel_t), stream.msg[e_id]], dim=-1))
        # The buffer's assoc table maps each account to its row in n_id (set by the call above)
        return z[self.buffer._assoc[nodes]]

    def link_logits(self, src, dst, stream: Stream) -> torch.Tensor:
        """Logit that src sends to dst, from embeddings before the day's transfers."""
        z = self.embed(torch.cat([src, dst]), stream)
        zs, zd = z[: src.numel()], z[src.numel():]
        return self.link_out((self.link_src(zs) + self.link_dst(zd)).relu()).squeeze(-1)

    def observe(self, rows: slice, stream: Stream):
        """Add one day's transfers to memory and neighbours."""
        src, dst = stream.src[rows], stream.dst[rows]
        if src.numel() == 0 or not self.use_graph:
            return
        self.memory.update_state(src, dst, stream.t[rows], stream.msg[rows])
        self.buffer.insert(src, dst)

    def score(self, nodes: torch.Tensor, stream: Stream, x: torch.Tensor | None = None) -> torch.Tensor:
        """Mule logit of each node at the end of the current day; x = the rows' features when the head takes them."""
        parts = [self.deposit[nodes].unsqueeze(-1)]
        if self.use_graph:
            parts.insert(0, self.embed(nodes, stream))
        return self.head(torch.cat(parts if x is None else [*parts, x], dim=-1)).squeeze(-1)


def sample_negatives(seen: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """n random accounts among those already seen (bool mask); unseen accounts would make the link task trivial."""
    pool = seen.nonzero().squeeze(-1)
    return pool[torch.randint(pool.numel(), (n,), generator=generator, device="cpu").to(pool.device)]


def rows_by_day(rows: pd.DataFrame, device, x: np.ndarray | None = None) -> dict[int, tuple]:
    """Day -> (node tensor, label array or None, acct_id array, feature tensor or None) of that day's rows.

    Args:
        x: optional model inputs per row, in the positional order of rows (FeatureScaler.transform).
    """
    out, pos = {}, np.arange(len(rows))
    for d, idx in pd.Series(pos).groupby(rows.snapshot.to_numpy(), sort=True):
        g = rows.iloc[idx.to_numpy()]
        label = g.label.to_numpy() if "label" in g else None
        feats = None if x is None else torch.as_tensor(x[idx.to_numpy()], device=device)
        out[int(d)] = (torch.tensor(g.node.to_numpy(), dtype=torch.long, device=device), label, g.acct_id.to_numpy(),
                       feats)
    return out


def load_rows(data: Path, split: str, scaler: FeatureScaler | None, device) -> dict[int, tuple]:
    """rows_<split> of an export by day, with the scaled row features when a scaler is given."""
    rows = pd.read_parquet(data / f"rows_{split}.parquet")
    if scaler is None:
        return rows_by_day(rows, device)
    feats = pd.read_parquet(data / f"features_{split}.parquet")
    # The export writes both files in one row order; check it rather than trust it
    if not (np.array_equal(feats.acct_id.to_numpy(), rows.acct_id.to_numpy())
            and np.array_equal(feats.snapshot.to_numpy(), rows.snapshot.to_numpy())):
        raise ValueError(f"features_{split} is not in the row order of rows_{split}")
    return rows_by_day(rows, device, scaler.transform(feats))


def train_epoch(model, stream: Stream, train_rows: dict, last_day: int, cfg: dict, opt, rng, gen) -> float:
    """One training pass over days 0..last_day; returns the mean loss per day."""
    model.train()
    model.reset()
    seen = torch.zeros(model.deposit.numel(), dtype=torch.bool, device=model.deposit.device)
    loss_fn, total, steps = BCEWithLogitsLoss(), 0.0, 0
    for d in range(last_day + 1):
        rows = stream.day(d)
        src, dst = stream.src[rows], stream.dst[rows]
        loss = torch.zeros((), device=seen.device)
        # 1. Link loss: predict the day's transfers from the state before them
        if model.use_graph and src.numel() and seen.any():
            pos = model.link_logits(src, dst, stream)
            neg = model.link_logits(src, sample_negatives(seen, src.numel(), gen), stream)
            loss = loss + cfg["link_weight"] * (loss_fn(pos, torch.ones_like(pos)) + loss_fn(neg, torch.zeros_like(neg)))
        # 2. The day's transfers enter memory only after their own prediction
        model.observe(rows, stream)
        seen[src] = True
        seen[dst] = True
        # 3. Account loss at the end of the day: all mules, a share of the others
        if d in train_rows:
            nodes, label, _, x = train_rows[d]
            keep = (label == 1) | (rng.random(len(label)) < cfg["train_negative_rate"])
            if keep.any():
                y = torch.as_tensor(label[keep], dtype=torch.float, device=nodes.device)
                k = torch.as_tensor(keep, device=nodes.device)
                loss = loss + loss_fn(model.score(nodes[k], stream, None if x is None else x[k]), y)
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        model.detach()
    return total / max(steps, 1)


@torch.no_grad()
def score_pass(model, stream: Stream, score_rows: dict[str, dict], last_day: int) -> dict[str, pd.DataFrame]:
    """Pass from day 0 to last_day without training; at the end of each day, score that day's rows of each split."""
    model.eval()
    model.reset()
    out = {name: [] for name in score_rows}
    for d in range(last_day + 1):
        model.observe(stream.day(d), stream)
        for name, rows in score_rows.items():
            if d in rows:
                nodes, label, acct, x = rows[d]
                frame = pd.DataFrame({"acct_id": acct, "snapshot": d,
                                      "score": model.score(nodes, stream, x).cpu().numpy()})
                out[name].append(frame if label is None else frame.assign(label=label))
    return {name: pd.concat(f, ignore_index=True) if f else pd.DataFrame() for name, f in out.items()}


def pr_auc(scores: pd.DataFrame) -> float:
    """Average precision over labelled rows."""
    s = scores[scores.label.notna()]
    return float(average_precision_score(s.label, s.score))


def main() -> None:
    """Train with early stopping on validation PR-AUC, then write validation and test scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="export folder (manifest.json, events, nodes, rows_*)")
    parser.add_argument("--config", default="tgn.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["seed"] = cfg["seed"] if args.seed is None else args.seed
    cfg["max_epochs"] = args.max_epochs or cfg["max_epochs"]
    set_seed(cfg["seed"])
    rng, gen = np.random.default_rng(cfg["seed"]), torch.Generator().manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    # 1. Export tables; train labels end on the last train day, test rows carry no labels
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    splits = manifest["splits"]
    stream = Stream(pd.read_parquet(data / "events.parquet"), splits["train"][1], device)
    nodes = pd.read_parquet(data / "nodes.parquet")
    deposit = np.log1p(nodes.initial_deposit.to_numpy())
    deposit = torch.as_tensor((deposit - deposit.mean()) / deposit.std(), dtype=torch.float, device=device)
    # Feature scaling from train rows only, as LightGBM fits on train rows only
    scaler = None
    if cfg.get("account_features"):
        scaler = FeatureScaler(pd.read_parquet(data / "features_train.parquet").drop(columns=["acct_id", "snapshot"]))
    train_rows = load_rows(data, "train", scaler, device)
    val_rows = load_rows(data, "val", scaler, device)

    # 2. Train; keep the parameters of the epoch with the best validation PR-AUC
    model = TGNAccount(len(nodes), deposit, cfg["dim"], cfg["neighbors"], device, scaler.width if scaler else 0,
                       cfg.get("use_graph", True))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_epoch, best_state, epochs = -1.0, 0, None, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss = train_epoch(model, stream, train_rows, splits["train"][1], cfg, opt, rng, gen)
        t1 = time.time()
        val = pr_auc(score_pass(model, stream, {"val": val_rows}, splits["val"][1])["val"])
        epochs.append({"epoch": epoch, "loss": round(loss, 4), "val_pr_auc": round(val, 4),
                       "train_seconds": round(t1 - t0, 1), "val_seconds": round(time.time() - t1, 1),
                       "peak_mib": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None})
        print(epochs[-1], flush=True)
        if val > best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break

    # 3. Scores of the best epoch for validation and test, from one pass over all days
    model.load_state_dict(best_state)
    test_rows = load_rows(data, "test", scaler, device)
    scores = score_pass(model, stream, {"val": val_rows, "test": test_rows}, splits["test"][1])

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
