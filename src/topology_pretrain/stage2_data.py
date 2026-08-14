"""Deterministic Stage 2 QA data, Stage 1 validation, and embedding caches."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from .graphs import RootedGraph, TRAIN_FAMILIES, sample_rooted_graph, to_data
from .model import TopologyEncoder


QA_TASKS = ("num_nodes", "num_edges", "root_degree", "num_hop2")
GRAPH_SLOT = "<|topology_graph_slot_7f3a9c|>"
PROMPT_BANK_VERSION = "stage2-qa-en-v1"
PROMPT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "num_nodes": (
        "How many nodes are in the rooted graph?",
        "What is the total number of nodes in this graph?",
        "Count all nodes in the represented graph.",
        "Give the graph's node count.",
        "How many vertices does the rooted graph contain?",
        "Return the total vertex count for this graph.",
    ),
    "num_edges": (
        "How many edges are in the rooted graph?",
        "What is the total number of edges in this graph?",
        "Count all edges in the represented graph.",
        "Give the graph's edge count.",
        "How many links does the rooted graph contain?",
        "Return the total edge count for this graph.",
    ),
    "root_degree": (
        "What is the degree of the root node?",
        "How many direct neighbors does the root have?",
        "Count the nodes exactly one hop from the root.",
        "How many nodes are adjacent to the root?",
        "Return the root node's degree.",
        "How many one-hop neighbors surround the root?",
    ),
    "num_hop2": (
        "How many nodes are exactly two hops from the root?",
        "Count the nodes at distance two from the root.",
        "What is the number of two-hop neighbors of the root?",
        "Give the count of nodes precisely two steps from the root.",
        "How many graph nodes lie at hop two from the root?",
        "Return the number of nodes whose root distance is two.",
    ),
}
TEMPLATE_SPLITS = {"train": (0, 1, 2, 3), "validation": (4,), "test": (5,)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def topology_facts(rooted: RootedGraph) -> np.ndarray:
    """Return the four non-redundant integer QA targets in ``QA_TASKS`` order."""
    graph, root = rooted.graph, rooted.root
    distances = nx.single_source_shortest_path_length(graph, root)
    return np.asarray(
        [
            graph.number_of_nodes(),
            graph.number_of_edges(),
            graph.degree[root],
            sum(distance == 2 for distance in distances.values()),
        ],
        dtype=np.int64,
    )


def rooted_topology_hash(rooted: RootedGraph) -> str:
    """Root-aware WL hash used as a conservative split de-duplication key.

    Rooted-isomorphic graphs necessarily share this key. Rejecting every key
    collision may discard rare WL collisions, but cannot let an isomorphic
    duplicate cross a split boundary.
    """
    graph = rooted.graph.copy()
    marker = "__stage2_root__"
    nx.set_node_attributes(graph, "node", marker)
    graph.nodes[rooted.root][marker] = "root"
    wl_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr=marker, iterations=5)
    degree_sequence = ",".join(map(str, sorted(dict(graph.degree()).values())))
    payload = (
        f"{graph.number_of_nodes()}|{graph.number_of_edges()}|"
        f"{graph.degree[rooted.root]}|{degree_sequence}|{wl_hash}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for external artifacts.
        return torch.load(path, map_location="cpu")


def _load_stage1_payload(run_dir: Path) -> tuple[dict, Path, str]:
    export_path = run_dir / "topology_encoder.pt"
    if export_path.is_file():
        payload = _load_torch(export_path)
        return payload, export_path, "encoder_export"

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Stage 1 model not found; expected either "
            f"{export_path} or {checkpoint_path}"
        )
    # best.pt is produced by this repository and contains RNG/optimizer state
    # that is not accepted by PyTorch's restricted weights-only loader.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage 1 best.pt must contain a checkpoint dictionary")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Stage 1 best.pt is missing its config dictionary")
    hidden_dim = int(config.get("hidden_dim", -1))
    payload = {
        "encoder": checkpoint.get("encoder"),
        "input_contract": "internal fixed all-one vector; ignores data.x",
        "input_dim": hidden_dim,
        "output_dim": hidden_dim,
        "layers": 2,
        "aggregation": "sum-gin",
    }
    return payload, checkpoint_path, "training_checkpoint"


def validate_stage1_artifact(stage1_run_dir: Path, allow_smoke: bool = False) -> tuple[TopologyEncoder, dict]:
    run_dir = Path(stage1_run_dir)
    payload, artifact_path, artifact_type = _load_stage1_payload(run_dir)
    evidence_paths = {
        "run_manifest": run_dir / "run_manifest.json",
        "metrics": run_dir / "metrics.json",
        "report": run_dir / "report.md",
    }
    missing = [name for name, path in evidence_paths.items() if not path.is_file()]
    engineering_only = bool(missing)
    if missing and not allow_smoke:
        raise FileNotFoundError(
            "Formal Stage 1 evidence is incomplete: " + ", ".join(missing) +
            ". Use allow_smoke_artifact only for engineering checks."
        )

    expected = {
        "input_dim": 128,
        "output_dim": 128,
        "layers": 2,
        "aggregation": "sum-gin",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Stage 1 contract mismatch for {key}: {payload.get(key)!r} != {value!r}")
    if "fixed all-one" not in str(payload.get("input_contract", "")) or "ignores data.x" not in str(
        payload.get("input_contract", "")
    ):
        raise ValueError("Stage 1 input contract must use internal all-one features and ignore data.x")
    state = payload.get("encoder")
    if not isinstance(state, dict) or not state:
        raise ValueError("Stage 1 export does not contain a non-empty encoder state_dict")
    if any(key.startswith("pool_projection") for key in state):
        raise ValueError("Stage 2 requires the root-only, non-pooled Stage 1 encoder")

    encoder = TopologyEncoder(hidden_dim=128, pooling=False)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    provenance = {
        "run_dir": str(run_dir.resolve()),
        "artifact_type": artifact_type,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        # Backward-compatible provenance keys used by existing Stage 2
        # manifests; for a best.pt fallback these identify the checkpoint.
        "export_path": str(artifact_path.resolve()),
        "export_sha256": sha256_file(artifact_path),
        "contract": {key: payload[key] for key in (*expected, "input_contract")},
        "engineering_only": engineering_only,
        "missing_formal_evidence": missing,
        "evidence_sha256": {
            name: sha256_file(path) for name, path in evidence_paths.items() if path.is_file()
        },
    }
    return encoder, provenance


def _unique_graphs(
    *, seed: int, split: str, count: int, version: str, used_hashes: set[str]
) -> Iterator[tuple[RootedGraph, str]]:
    accepted = 0
    candidate_id = 0
    maximum_candidates = max(count * 100, count + 1000)
    while accepted < count and candidate_id < maximum_candidates:
        rooted = sample_rooted_graph(
            seed, f"stage2_{split}", candidate_id, version, families=TRAIN_FAMILIES
        )
        fingerprint = rooted_topology_hash(rooted)
        candidate_id += 1
        if fingerprint in used_hashes:
            continue
        used_hashes.add(fingerprint)
        accepted += 1
        yield rooted, fingerprint
    if accepted != count:
        raise RuntimeError(
            f"Could not generate {count} unique {split} graphs after {candidate_id} candidates"
        )


def _encode_split(
    encoder: TopologyEncoder,
    records: Iterable[tuple[RootedGraph, str]],
    record_count: int,
    device: torch.device,
    batch_size: int,
    no_progress: bool,
) -> dict[str, np.ndarray]:
    embeddings: list[np.ndarray] = []
    facts: list[np.ndarray] = []
    sample_ids: list[int] = []
    families: list[str] = []
    hashes: list[str] = []
    iterator = records if no_progress else tqdm(records, total=record_count, desc="Stage 2 embeddings", unit="graph")
    encoder = encoder.to(device)
    pending: list[tuple[RootedGraph, str]] = []

    def encode_pending(batch_records: list[tuple[RootedGraph, str]]) -> None:
        graph_batch = Batch.from_data_list([to_data(rooted) for rooted, _ in batch_records]).to(device)
        z = encoder(graph_batch, normalize=False).float().cpu().numpy()
        embeddings.append(z)
        for rooted, fingerprint in batch_records:
            facts.append(topology_facts(rooted))
            sample_ids.append(rooted.sample_id)
            families.append(rooted.family)
            hashes.append(fingerprint)

    with torch.no_grad():
        for record in iterator:
            pending.append(record)
            if len(pending) == batch_size:
                encode_pending(pending)
                pending = []
        if pending:
            encode_pending(pending)
    encoder.to("cpu")
    if len(facts) != record_count:
        raise RuntimeError(f"Encoded {len(facts)} graphs, expected {record_count}")
    return {
        "embeddings": np.concatenate(embeddings).astype(np.float32, copy=False),
        "facts": np.stack(facts).astype(np.int64, copy=False),
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "families": np.asarray(families, dtype="U32"),
        "topology_hashes": np.asarray(hashes, dtype="U64"),
    }


def prepare_stage2_cache(config: dict) -> dict:
    seed = int(config["seed"])
    version = str(config.get("generator_version", "topology-v1"))
    data_config = config["data"]
    cache_dir = Path(data_config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder, stage1 = validate_stage1_artifact(
        Path(config["stage1_run_dir"]), bool(config.get("allow_smoke_artifact", False))
    )
    requested_device = str(data_config.get("prepare_device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    used_hashes: set[str] = set()
    split_sizes = {
        "train": int(data_config["train_graphs"]),
        "validation": int(data_config["validation_graphs"]),
        "test": int(data_config["test_graphs"]),
    }
    split_summary: dict[str, dict] = {}
    for split, size in split_sizes.items():
        records = _unique_graphs(
            seed=seed, split=split, count=size, version=version, used_hashes=used_hashes
        )
        arrays = _encode_split(
            encoder,
            records,
            size,
            device,
            int(data_config.get("embedding_batch_size", 256)),
            bool(config.get("no_progress", False)),
        )
        output_path = cache_dir / f"{split}.npz"
        np.savez_compressed(output_path, **arrays)
        split_summary[split] = {
            "graphs": size,
            "cache_file": output_path.name,
            "cache_sha256": sha256_file(output_path),
            "family_counts": dict(sorted(Counter(arrays["families"].tolist()).items())),
            "fact_min": dict(zip(QA_TASKS, arrays["facts"].min(axis=0).tolist())),
            "fact_max": dict(zip(QA_TASKS, arrays["facts"].max(axis=0).tolist())),
        }

    manifest = {
        "schema_version": "stage2-cache-v1",
        "seed": seed,
        "generator_version": version,
        "qa_tasks": list(QA_TASKS),
        "prompt_bank_version": PROMPT_BANK_VERSION,
        "template_splits": {key: list(value) for key, value in TEMPLATE_SPLITS.items()},
        "train_families": list(TRAIN_FAMILIES),
        "split_isolation": "root-aware WL hash rejected globally across all splits",
        "stage1": stage1,
        "engineering_only": stage1["engineering_only"],
        "splits": split_summary,
    }
    manifest_path = cache_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_stage2_split(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    manifest_path = Path(cache_dir) / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 data manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(cache_dir) / manifest["splits"][split]["cache_file"]
    if sha256_file(path) != manifest["splits"][split]["cache_sha256"]:
        raise ValueError(f"Stage 2 cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    if arrays["embeddings"].shape[1] != 128 or arrays["facts"].shape[1] != len(QA_TASKS):
        raise ValueError(f"Invalid Stage 2 cache tensor shapes in {path}")
    return arrays


def task_permutation(seed: int, sample_id: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, "task-permutation", sample_id))
    return rng.permutation(len(QA_TASKS))


def question_for(
    task_index: int, split: str, *, seed: int, sample_id: int, epoch: int = 0
) -> tuple[str, int]:
    task = QA_TASKS[int(task_index)]
    allowed = TEMPLATE_SPLITS[split]
    if split == "train":
        choice = stable_seed(seed, "template", sample_id, epoch, task) % len(allowed)
        template_index = allowed[choice]
    else:
        template_index = allowed[0]
    return PROMPT_TEMPLATES[task][template_index], int(template_index)


class Stage2QADataset(torch.utils.data.Dataset):
    """One task per graph/epoch for train; all four tasks for fixed splits."""

    def __init__(self, arrays: dict[str, np.ndarray], split: str, seed: int) -> None:
        if split not in TEMPLATE_SPLITS:
            raise ValueError(f"unknown Stage 2 split: {split}")
        self.arrays = arrays
        self.split = split
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        graphs = len(self.arrays["embeddings"])
        return graphs if self.split == "train" else graphs * len(QA_TASKS)

    def __getitem__(self, index: int) -> dict:
        if self.split == "train":
            graph_index = int(index)
            permutation = task_permutation(self.seed, int(self.arrays["sample_ids"][graph_index]))
            task_index = int(permutation[self.epoch % len(QA_TASKS)])
        else:
            graph_index, task_index = divmod(int(index), len(QA_TASKS))
        sample_id = int(self.arrays["sample_ids"][graph_index])
        question, template_index = question_for(
            task_index,
            self.split,
            seed=self.seed,
            sample_id=sample_id,
            epoch=self.epoch,
        )
        answer = int(self.arrays["facts"][graph_index, task_index])
        return {
            "embedding": torch.from_numpy(self.arrays["embeddings"][graph_index]),
            "graph_index": graph_index,
            "sample_id": sample_id,
            "task_index": task_index,
            "task": QA_TASKS[task_index],
            "question": question,
            "template_index": template_index,
            "answer": answer,
        }
