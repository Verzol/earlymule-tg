"""STA-GT-inspired daily account scorer on an exported dataset (python -m earlymule.data.export_events).

Adapts the Spatial-Temporal-Aware Graph Transformer (Tian and Liu, IEEE TII 2024) from transaction nodes linked by
shared devices to accounts scored at the end of each day, under the evaluation protocol (only transfers dated on
or before day t). Kept from the paper: an attribute embedding, a time encoding, the fraud-aware aggregation of
FRAUDRE (mean of neighbours and mean of the differences to them), attention between relations, the concatenation
of the layers, and a Transformer over each node's neighbourhood. Changed, and recorded in the protocol: nodes are
accounts on day t; the relations come from the money flow of the 30-day window (received from, sent to, shared
partner) instead of shared devices; the time encoding is sin/cos of t minus the pair's latest transfer day, not an
absolute timestamp; the per-node Transformer reads the node plus its most recent neighbours as tokens; the
account head reads the embedding, the deposit and the row's LightGBM features, as the other daily graph models do.
Run: python -m earlymule.models.stagt --data <export folder> [--config stagt.yaml] [--seed N] [--out DIR]
     [--max-epochs N] [--lr X]
Same outputs as earlymule.models.graphsage, whose training and scoring loop this module reuses.
"""

from __future__ import annotations

import math

import torch
from torch.nn import Linear, TransformerEncoderLayer
from torch_geometric.utils import coalesce, scatter

from earlymule.models.graphsage import WindowGraph, main

IN, OUT, SHARED = 0, 1, 2  # relations: received from, sent to, shared a partner with


def shared_partner_edges(edge_index: torch.Tensor, edge_day: torch.Tensor, num_nodes: int, cap: int) -> tuple:
    """Undirected edges between accounts that paid the same account, or were paid by the same account, in the window.

    Partners with more than `cap` payers (or payees) are skipped: a payroll or merchant account would otherwise
    link thousands of unrelated customers. The edge day is the earlier of the two contacts, when the relation
    became complete.
    Returns:
        (edge_index [2, E] both directions, edge_day [E]); empty when nothing qualifies.
    """
    parts = []
    for side in (0, 1):
        # 1. Group the edges by the shared partner (dst for payers, src for payees), small groups only
        partner, member = edge_index[1 - side], edge_index[side]
        order = torch.argsort(partner, stable=True)
        partner, member, day = partner[order], member[order], edge_day[order]
        _, counts = torch.unique_consecutive(partner, return_counts=True)
        starts = counts.cumsum(0) - counts
        keep = (counts >= 2) & (counts <= cap)
        counts, starts = counts[keep], starts[keep]
        if counts.numel() == 0:
            continue
        # 2. Every ordered pair inside each group, without the diagonal
        cells = counts * counts
        gid = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), cells)
        k = torch.arange(int(cells.sum()), device=counts.device) - torch.repeat_interleave(cells.cumsum(0) - cells, cells)
        a, b = starts[gid] + k // counts[gid], starts[gid] + k % counts[gid]
        a, b = a[a != b], b[a != b]
        parts.append((torch.stack([member[a], member[b]]), torch.minimum(day[a], day[b])))
    if not parts:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty.view(2, 0), empty
    ei = torch.cat([p[0] for p in parts], dim=1)
    d = torch.cat([p[1] for p in parts])
    # 3. One edge per pair; a pair linked through several partners keeps its most recent complete link
    return coalesce(ei, d, num_nodes, reduce="max")


def recent_neighbours(nodes: torch.Tensor, edge_index: torch.Tensor, edge_day: torch.Tensor, num_nodes: int,
                      k: int) -> tuple:
    """For each query node, its k most recent window partners (both directions) as padded tensors.

    Returns:
        nb [M, k] partner ids (0 where padded), rel [M, k] IN or OUT, day [M, k], mask [M, k] True where padded.
        Ties on the day are broken by partner id, so the choice does not depend on edge order.
    """
    pos = torch.full((num_nodes,), -1, dtype=torch.long, device=nodes.device)
    pos[nodes] = torch.arange(nodes.numel(), device=nodes.device)
    # 1. Candidate (query, partner, relation, day) rows from both edge directions
    q = torch.cat([pos[edge_index[1]], pos[edge_index[0]]])
    nb = torch.cat([edge_index[0], edge_index[1]])
    rel = torch.cat([torch.full_like(edge_index[0], IN), torch.full_like(edge_index[0], OUT)])
    day = torch.cat([edge_day, edge_day])
    sel = q >= 0
    q, nb, rel, day = q[sel], nb[sel], rel[sel], day[sel]
    # 2. Newest first within each query, then by partner id; keep the first k
    big = int(day.max()) + 1 if day.numel() else 1
    key = (q * big + (big - 1 - day)) * num_nodes + nb
    order = torch.argsort(key, stable=True)
    q, nb, rel, day = q[order], nb[order], rel[order], day[order]
    _, counts = torch.unique_consecutive(q, return_counts=True) if q.numel() else (None, q)
    rank = torch.arange(q.numel(), device=q.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    keep = rank < k
    slot = q[keep] * k + rank[keep]
    out = [torch.zeros(nodes.numel() * k, dtype=torch.long, device=nodes.device) for _ in range(3)]
    for tensor, value in zip(out, (nb[keep], rel[keep], day[keep])):
        tensor[slot] = value
    mask = torch.ones(nodes.numel() * k, dtype=torch.bool, device=nodes.device)
    mask[slot] = False
    return (*[o.view(-1, k) for o in out], mask.view(-1, k))


class TimeEncoding(torch.nn.Module):
    """sin/cos of a day gap at geometric frequencies (periods 2 to 64 days), then a linear map to `dim`."""

    def __init__(self, dim: int, n_freq: int = 6):
        super().__init__()
        self.register_buffer("omega", 2 * math.pi / (2.0 ** torch.arange(1, n_freq + 1)))
        self.lin = Linear(2 * n_freq, dim)

    def forward(self, gap: torch.Tensor) -> torch.Tensor:
        """Encoding of each gap (days, float or long)."""
        a = gap.float().unsqueeze(-1) * self.omega
        return self.lin(torch.cat([a.sin(), a.cos()], dim=-1))


class RelationLayer(torch.nn.Module):
    """One STA-GT layer: per relation, mean of time-aware neighbour messages and mean difference to the
    neighbours (FRAUDRE); attention between relations; a self transform; ReLU."""

    def __init__(self, dim: int, n_rel: int = 3):
        super().__init__()
        self.msg = torch.nn.ModuleList([Linear(2 * dim, dim) for _ in range(n_rel)])
        self.comb = torch.nn.ModuleList([Linear(2 * dim, dim) for _ in range(n_rel)])
        self.att = Linear(dim, dim)
        self.q = torch.nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.self_lin = Linear(dim, dim)

    def forward(self, h: torch.Tensor, rels: list[tuple], tenc: torch.Tensor | None = None) -> torch.Tensor:
        """h [N, dim]; rels = [(edge_index, time encoding [E, dim])] per relation, messages flow edge[0] -> edge[1]."""
        n = h.size(0)
        per_rel, has = [], []
        for r, (ei, te) in enumerate(rels):
            src, dst = ei
            # Mean of time-aware messages and mean difference h_i - h_j over the relation's neighbours
            agg = scatter(self.msg[r](torch.cat([h[src], te], dim=-1)), dst, 0, dim_size=n, reduce="mean")
            mean_nb = scatter(h[src], dst, 0, dim_size=n, reduce="mean")
            deg = scatter(torch.ones_like(dst, dtype=h.dtype), dst, 0, dim_size=n, reduce="sum")
            per_rel.append(self.comb[r](torch.cat([agg, h - mean_nb], dim=-1)).relu())
            has.append(deg > 0)
        stack, present = torch.stack(per_rel), torch.stack(has)  # [R, N, dim], [R, N]
        # Attention between relations; a node without neighbours in a relation gives that relation no weight
        score = torch.tanh(self.att(stack)) @ self.q
        # A large negative, not -inf: a node with no neighbours in any relation would otherwise give NaN
        score = score.masked_fill(~present, -1e9)
        alpha = torch.softmax(score, dim=0).masked_fill(~present, 0.0)
        return (self.self_lin(h) + (alpha.unsqueeze(-1) * stack).sum(0)).relu()


class STAGTAccount(torch.nn.Module):
    """STA-GT-inspired encoder over the window graph with the shared account head.

    Args:
        tokens: most recent partners read by the per-node Transformer next to the node itself.
        shared_cap: partners with more payers or payees than this create no shared-partner edges.
    """

    def __init__(self, deposit: torch.Tensor, dim: int, layers: int, tokens: int, device, n_features: int = 0,
                 shared_cap: int = 20, heads: int = 4):
        super().__init__()
        self.tokens, self.shared_cap = tokens, shared_cap
        self.embed_in = Linear(5, dim)
        self.time = TimeEncoding(dim)
        self.layers = torch.nn.ModuleList([RelationLayer(dim) for _ in range(layers)])
        self.jump = Linear(layers * dim, dim)
        self.rel_emb = torch.nn.Embedding(3, dim)  # token kinds: self, received from, sent to
        self.transformer = TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True)
        self.head = torch.nn.Sequential(Linear(dim + 1 + n_features, dim), torch.nn.ReLU(), Linear(dim, 1))
        self.register_buffer("deposit", deposit)
        self.to(device)

    def embed(self, g: WindowGraph, stats: torch.Tensor) -> torch.Tensor:
        """Embedding of every account after the relation layers (before the Transformer)."""
        n = self.deposit.numel()
        h = self.embed_in(torch.cat([self.deposit.unsqueeze(-1), stats], dim=-1)).relu()
        gap = g.t - g.edge_day
        sh_ei, sh_day = shared_partner_edges(g.edge_index, g.edge_day, n, self.shared_cap)
        # Messages flow from the neighbour to the account: senders -> me, receivers -> me, shared partners -> me
        rels = [(g.edge_index, self.time(gap)), (g.edge_index.flip(0), self.time(gap)), (sh_ei, self.time(g.t - sh_day))]
        outs = []
        for layer in self.layers:
            h = layer(h, rels)
            outs.append(h)
        return self.jump(torch.cat(outs, dim=-1))

    def score_day(self, nodes: torch.Tensor, g: WindowGraph, stats: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
        """Mule logit of the given accounts at the end of day g.t (interface shared with graphsage.SAGEAccount)."""
        z = self.embed(g, stats)
        # 1. Token sequence per account: itself, then its most recent partners with kind and recency
        nb, rel, day, pad = recent_neighbours(nodes, g.edge_index, g.edge_day, self.deposit.numel(), self.tokens)
        tok = torch.cat([z[nodes].unsqueeze(1), z[nb]], dim=1)
        kind = torch.cat([torch.zeros_like(nodes).unsqueeze(1), rel + 1], dim=1)
        gap = torch.cat([torch.zeros_like(nodes).unsqueeze(1), g.t - day], dim=1)
        tok = tok + self.rel_emb(kind) + self.time(gap)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)
        # 2. Self-attention over the sequence; the account's own token is the output
        out = self.transformer(tok, src_key_padding_mask=mask)[:, 0]
        parts = [out, self.deposit[nodes].unsqueeze(-1)]
        return self.head(torch.cat(parts if x is None else [*parts, x], dim=-1)).squeeze(-1)


def build_stagt(cfg: dict, deposit: torch.Tensor, n_features: int, device) -> STAGTAccount:
    """The STA-GT-inspired model of a config."""
    return STAGTAccount(deposit, cfg["dim"], cfg["layers"], cfg["tokens"], device, n_features, cfg["shared_cap"],
                        cfg["heads"])


if __name__ == "__main__":
    main(build_stagt, "stagt.yaml")
