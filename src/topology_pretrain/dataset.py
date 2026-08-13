"""Online deterministic anchor-pair batches and fixed evaluation sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch, Data

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


def make_sample_from_args(args: tuple) -> PairSample:
    """Pickle-friendly entry point for ProcessPoolExecutor workers."""
    return make_sample(*args)


def pack_samples_numpy(samples: list[PairSample]) -> dict:
    """Return a pure-NumPy batch safe for cross-process transfer.

    Returning torch tensors from ProcessPoolExecutor uses /dev/shm and file
    descriptors. Container shared-memory limits can deadlock even when regular
    RAM is plentiful, so workers return only pickle-owned NumPy arrays.
    """
    views = ("anchor", "iso", "a", "b")
    edge_parts: list[np.ndarray] = []
    roots: list[int] = []
    batch_parts: list[np.ndarray] = []
    node_offset = 0
    graph_index = 0
    for view in views:
        for sample in samples:
            rooted = getattr(sample, view)
            edges = np.asarray(list(rooted.graph.edges()), dtype=np.int32)
            if edges.size:
                directed = np.concatenate((edges, edges[:, ::-1]), axis=0).T
            else:
                directed = np.empty((2, 0), dtype=np.int32)
            edge_parts.append(directed + node_offset)
            roots.append(node_offset + rooted.root)
            node_count = rooted.graph.number_of_nodes()
            batch_parts.append(np.full(node_count, graph_index, dtype=np.int32))
            node_offset += node_count
            graph_index += 1
    return {
        "edge_index": np.concatenate(edge_parts, axis=1) if edge_parts else np.empty((2, 0), dtype=np.int32),
        "roots": np.asarray(roots, dtype=np.int64),
        "batch_index": np.concatenate(batch_parts),
        "num_nodes": node_offset,
        "d_a": np.asarray([s.d_a for s in samples], dtype=np.float32),
        "d_b": np.asarray([s.d_b for s in samples], dtype=np.float32),
        "stats": np.stack([s.stat for s in samples]).astype(np.float32, copy=False),
        "descriptors": [s.desc for s in samples],
        "size": len(samples),
    }


def make_packed_batch_from_args(args: tuple) -> dict:
    """Worker entry point returning no NetworkX/PyG/torch objects."""
    return pack_samples_numpy([make_sample_from_args(item) for item in args])


def numpy_batch_to_cpu(batch: dict) -> dict:
    """Construct the PyG object only in the main process."""
    root_mask = torch.zeros(int(batch["num_nodes"]), dtype=torch.bool)
    root_mask[torch.from_numpy(batch["roots"])] = True
    graphs = Data(edge_index=torch.from_numpy(batch["edge_index"]).long(), root_mask=root_mask,
                  batch=torch.from_numpy(batch["batch_index"]).long(), num_nodes=int(batch["num_nodes"]))
    return {
        "graphs": graphs,
        "d_a": torch.from_numpy(batch["d_a"]),
        "d_b": torch.from_numpy(batch["d_b"]),
        "stats": torch.from_numpy(batch["stats"]),
        "descriptors": batch["descriptors"],
        "size": batch["size"],
    }


def pack_samples(samples: list[PairSample]) -> dict:
    """Main-process convenience wrapper retained for tests/evaluation."""
    return numpy_batch_to_cpu(pack_samples_numpy(samples))


def move_packed_batch(batch: dict, device: torch.device) -> dict:
    if "edge_index" in batch:
        batch = numpy_batch_to_cpu(batch)
    # PyG Batch.to() mutates in place. Clone cache-backed tensors/graphs so a
    # fixed validation cache remains CPU-resident across every evaluation.
    batch = {key: (value.clone() if hasattr(value, "clone") else value) for key, value in batch.items()}
    if device.type == "cuda":
        batch = {key: (value.pin_memory() if hasattr(value, "pin_memory") else value) for key, value in batch.items()}
    return {key: (value.to(device, non_blocking=True) if hasattr(value, "to") else value)
            for key, value in batch.items()}


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
