"""Topology-only SUM-GIN encoder.  It deliberately ignores data.x."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv, global_add_pool


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))


class TopologyEncoder(nn.Module):
    """Two-layer SUM-GIN producing one graph token per root node."""

    def __init__(self, hidden_dim: int = 128, pooling: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.conv1 = GINConv(_mlp(hidden_dim, hidden_dim), train_eps=True)
        self.conv2 = GINConv(_mlp(hidden_dim, hidden_dim), train_eps=True)
        self.pool_projection = nn.Linear(hidden_dim * 2, hidden_dim) if pooling else None

    def forward(self, data, normalize: bool = False) -> torch.Tensor:
        device = data.edge_index.device
        x = torch.ones((data.num_nodes, self.hidden_dim), dtype=torch.float32, device=device)
        h = self.conv2(self.conv1(x, data.edge_index), data.edge_index)
        roots = h[data.root_mask]
        if self.pooling:
            batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long, device=device))
            roots = self.pool_projection(torch.cat([roots, global_add_pool(h, batch)], dim=-1))
        return F.normalize(roots, dim=-1) if normalize else roots


class StatHead(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
