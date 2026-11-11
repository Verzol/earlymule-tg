"""Kaggle smoke test for the temporal graph setup: versions, PyG's TGN example, one TGN epoch on our transfers.

Part 1 runs PyG's TGN link-prediction example on JODIE Wikipedia (downloaded, needs Internet) for a few epochs,
to check the environment against a known result (validation AP above 0.9).
Part 2 runs one training epoch of the same model over the train-period transfers of an exported dataset, for
timing and GPU memory only: batches of 200 events ignore that same-day order is arbitrary, and nothing is scored.
Run (Kaggle, GPU): python -m earlymule.models.tgn_smoke [--input /kaggle/input] [--out /kaggle/working/smoke.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.metrics import average_precision_score
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.data import TemporalData
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

# Files whose change would make an export stale for the code that reads it
EXPORT_SOURCES = ["src/earlymule/data/export_events.py", "src/earlymule/data/splits.py",
                  "configs/daily.yaml", "configs/lightgbm.yaml", "configs/kaggle.yaml"]


def environment() -> dict:
    """Python, torch, CUDA and PyG versions, and the GPU with its memory in MiB."""
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "pyg": torch_geometric.__version__, "gpu": None}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_mib=p.total_memory // 2**20, gpus=torch.cuda.device_count())
    return env


def export_matches_code(folder: Path, repo: Path) -> dict:
    """Manifest of an exported dataset, plus whether the export code is unchanged since its commit."""
    manifest = json.loads((folder / "manifest.json").read_text())
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", manifest["commit"], "HEAD", "--", *EXPORT_SOURCES])
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return {"dataset": manifest["dataset"], "export_commit": manifest["commit"], "code_commit": head.stdout.strip(),
            "export_code_unchanged": diff.returncode == 0, "rows": manifest["rows"]}


class GraphAttentionEmbedding(torch.nn.Module):
    """One attention layer over recent neighbours, with time-encoded edge features (PyG TGN example)."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc):
        super().__init__()
        self.time_enc = time_enc
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1,
                                    edge_dim=msg_dim + time_enc.out_channels)

    def forward(self, x, last_update, edge_index, t, msg):
        """Embeddings of the nodes in x; t and msg belong to the edges in edge_index."""
        rel_t = (last_update[edge_index[0]] - t).to(x.dtype)
        return self.conv(x, edge_index, torch.cat([self.time_enc(rel_t), msg], dim=-1))


class LinkPredictor(torch.nn.Module):
    """Logit that src transfers to dst, from their embeddings."""

    def __init__(self, channels: int):
        super().__init__()
        self.lin_src, self.lin_dst, self.lin_final = Linear(channels, channels), Linear(channels, channels), Linear(channels, 1)

    def forward(self, z_src, z_dst):
        return self.lin_final((self.lin_src(z_src) + self.lin_dst(z_dst)).relu())


class TGN:
    """Memory, embedding, link predictor and neighbour buffer of PyG's TGN example (dimension 100)."""

    def __init__(self, num_nodes: int, msg_dim: int, device, dim: int = 100):
        self.memory = TGNMemory(num_nodes, msg_dim, dim, dim, message_module=IdentityMessage(msg_dim, dim, dim),
                                aggregator_module=LastAggregator()).to(device)
        self.gnn = GraphAttentionEmbedding(dim, dim, msg_dim, self.memory.time_enc).to(device)
        self.link = LinkPredictor(dim).to(device)
        self.neighbors = LastNeighborLoader(num_nodes, size=10, device=device)
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters()) | set(self.link.parameters()), lr=1e-4)
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.params = sum(p.numel() for m in [self.memory, self.gnn, self.link] for p in m.parameters())

    def step(self, data: TemporalData, batch):
        """Link logits of a batch (positive, negative), then the memory and neighbour updates with its events."""
        n_id, edge_index, e_id = self.neighbors(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        src = z[self.assoc[batch.src]]
        pos, neg = self.link(src, z[self.assoc[batch.dst]]), self.link(src, z[self.assoc[batch.neg_dst]])
        # The batch's own events enter memory only after its predictions, so they cannot leak into them
        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbors.insert(batch.src, batch.dst)
        return pos, neg

    def epoch(self, data: TemporalData, loader, train: bool) -> float:
        """Train for one pass (returns the mean loss) or evaluate (returns AP); memory restarts at the pass start."""
        for m in [self.memory, self.gnn, self.link]:
            m.train(train)
        if train:
            self.memory.reset_state()
            self.neighbors.reset_state()
        loss_fn, total, seen, scores, labels = BCEWithLogitsLoss(), 0.0, 0, [], []
        for batch in loader:
            batch = batch.to(data.src.device)
            with torch.set_grad_enabled(train):
                pos, neg = self.step(data, batch)
            if train:
                self.optimizer.zero_grad()
                loss = loss_fn(pos, torch.ones_like(pos)) + loss_fn(neg, torch.zeros_like(neg))
                loss.backward()
                self.optimizer.step()
                self.memory.detach()
                total += loss.item() * batch.num_events
                seen += batch.num_events
            else:
                scores.append(torch.cat([pos, neg]).sigmoid().cpu())
                labels.append(torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu())
        if train:
            return total / seen
        return float(average_precision_score(torch.cat(labels).numpy(), torch.cat(scores).numpy()))


def timed(fn):
    """Run fn; return its result, seconds and peak GPU memory in MiB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() // 2**20 if torch.cuda.is_available() else None
    return out, round(time.time() - start, 1), peak


def jodie_example(root: Path, device, epochs: int) -> dict:
    """PyG's TGN example on JODIE Wikipedia: per epoch the train loss, validation AP, seconds and peak MiB."""
    data = JODIEDataset(str(root), name="wikipedia")[0].to(device)
    train, val, _ = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    model = TGN(data.num_nodes, data.msg.size(-1), device)
    rows = []
    for e in range(epochs):
        loss, sec, peak = timed(lambda: model.epoch(data, TemporalDataLoader(train, batch_size=200, neg_sampling_ratio=1.0), True))
        ap = model.epoch(data, TemporalDataLoader(val, batch_size=200, neg_sampling_ratio=1.0), False)
        rows.append({"epoch": e + 1, "loss": round(loss, 4), "val_ap": round(ap, 4), "seconds": sec, "peak_mib": peak})
        print(rows[-1], flush=True)
    return {"events": data.num_events, "nodes": data.num_nodes, "params": model.params, "epochs": rows}


def our_epoch(folder: Path, device, last_day: int) -> dict:
    """One training epoch over transfers up to last_day (the train period); seconds, events per second, peak MiB."""
    events = pd.read_parquet(folder / "events.parquet")
    events = events[events.day.le(last_day)]
    nodes = len(pd.read_parquet(folder / "nodes.parquet"))
    msg = np.c_[events.log_amount.to_numpy(), events.dow.to_numpy() / 6.0].astype("float32")
    data = TemporalData(src=torch.from_numpy(events.src.to_numpy().astype("int64")),
                        dst=torch.from_numpy(events.dst.to_numpy().astype("int64")),
                        t=torch.from_numpy(events.day.to_numpy().astype("int64")),
                        msg=torch.from_numpy(msg)).to(device)
    model = TGN(nodes, msg.shape[1], device)
    loss, sec, peak = timed(lambda: model.epoch(data, TemporalDataLoader(data, batch_size=200, neg_sampling_ratio=1.0), True))
    return {"events": len(events), "nodes": nodes, "params": model.params, "loss": round(loss, 4), "seconds": sec,
            "events_per_second": round(len(events) / sec), "peak_mib": peak}


def main() -> None:
    """Run both parts and write one JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input", help="folder searched for exported datasets")
    parser.add_argument("--repo", default="/tmp/earlymule-tg", help="clone of this repository, for the export check")
    parser.add_argument("--dataset", default="80K-earlymule-mule")
    parser.add_argument("--jodie-epochs", type=int, default=3)
    parser.add_argument("--out", default="/kaggle/working/smoke.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Environment and the exported dataset (found by its manifest, wherever Kaggle mounts it)
    report = {"environment": environment()}
    print(report, flush=True)
    folder = next(m.parent for m in Path(args.input).rglob("manifest.json")
                  if json.loads(m.read_text())["dataset"] == args.dataset)
    report["export"] = export_matches_code(folder, Path(args.repo))
    print(report["export"], flush=True)

    # 2. PyG's example as a known reference, then one epoch on our train-period transfers
    if args.jodie_epochs > 0:  # 0 skips the reference run, e.g. on the laptop once Kaggle has checked it
        report["jodie_wikipedia"] = jodie_example(Path("/tmp/jodie"), device, args.jodie_epochs)
    train_last_day = json.loads((folder / "manifest.json").read_text())["splits"]["train"][1]
    report["our_train_epoch"] = our_epoch(folder, device, train_last_day)
    print(report["our_train_epoch"], flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
