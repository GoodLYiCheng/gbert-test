"""Online deterministic anchor-pair batches and fixed evaluation sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch

from .graphs import RootedGraph, descriptors, perturb, permute, sample_rooted_graph, stats, to_data


@dataclass
class PairSample:
    anchor: RootedGraph
    iso: RootedGraph
    a: RootedGraph
    b: RootedGraph
    d_a: float
    d_b: float
    stat: np.ndarray
    desc: dict[str, str]


def make_sample(seed: int, split: str, sample_id: int, version: str, tolerance: float, attempts: int,
                families=None) -> PairSample:
    anchor = sample_rooted_graph(seed, split, sample_id, version, families or __import__("topology_pretrain.graphs", fromlist=["TRAIN_FAMILIES"]).TRAIN_FAMILIES)
    rng = np.random.default_rng(seed + sample_id * 1009 + sum(ord(c) for c in split))
    iso = permute(anchor, rng)
    a, d_a, _ = perturb(anchor, float(rng.random()), rng, tolerance, attempts)
    b, d_b, _ = perturb(anchor, float(rng.random()), rng, tolerance, attempts)
    for _ in range(8):
        if abs(d_a - d_b) > 1e-8:
            break
        b, d_b, _ = perturb(anchor, float(rng.random()), rng, tolerance, attempts)
    return PairSample(anchor, iso, a, b, d_a, d_b, stats(anchor), descriptors(anchor))


def collate(samples: list[PairSample], device: torch.device) -> dict:
    def batch(name: str) -> Batch:
        return Batch.from_data_list([to_data(getattr(s, name)) for s in samples]).to(device)
    return {
        "anchor": batch("anchor"), "iso": batch("iso"), "a": batch("a"), "b": batch("b"),
        "d_a": torch.tensor([s.d_a for s in samples], dtype=torch.float32, device=device),
        "d_b": torch.tensor([s.d_b for s in samples], dtype=torch.float32, device=device),
        "stats": torch.tensor(np.stack([s.stat for s in samples]), dtype=torch.float32, device=device),
        "descriptors": [s.desc for s in samples],
    }
