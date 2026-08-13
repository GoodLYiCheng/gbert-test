"""Reproducible rooted two-hop graph generation and pair construction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

TRAIN_FAMILIES = (
    "er", "tree", "star", "ba", "community", "dense", "sparse", "regular",
    "bipartite", "core_periphery", "mixed",
)
OOD_FAMILIES = ("cycle", "barbell")


@dataclass(frozen=True)
class RootedGraph:
    graph: nx.Graph
    root: int
    family: str
    sample_id: int


def _rng(seed: int, split: str, sample_id: int, version: str) -> np.random.Generator:
    payload = f"{seed}|{split}|{sample_id}|{version}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def _n(rng: np.random.Generator) -> int:
    return int(rng.integers(4, 65))


def _raw_graph(family: str, n: int, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(2**31 - 1))
    if family == "er":
        return nx.gnp_random_graph(n, float(rng.uniform(.03, .75)), seed=seed)
    if family == "tree":
        return nx.random_labeled_tree(n, seed=seed)
    if family == "star":
        return nx.star_graph(n - 1)
    if family == "ba":
        return nx.barabasi_albert_graph(n, int(rng.integers(1, min(6, n))), seed=seed)
    if family == "community":
        a = int(rng.integers(2, n - 1)); sizes = [a, n - a]
        return nx.stochastic_block_model(sizes, [[.55, .04], [.04, .55]], seed=seed)
    if family == "dense":
        return nx.gnp_random_graph(n, float(rng.uniform(.55, .95)), seed=seed)
    if family == "sparse":
        return nx.gnp_random_graph(n, float(rng.uniform(.02, .12)), seed=seed)
    if family == "regular":
        degree = int(rng.integers(2, min(7, n)))
        if (n * degree) % 2:
            degree = max(1, degree - 1)
        return nx.random_regular_graph(degree, n, seed=seed)
    if family == "bipartite":
        a = int(rng.integers(2, n - 1)); b = n - a
        g = nx.complete_bipartite_graph(a, b)
        for edge in list(g.edges()):
            if rng.random() < .25:
                g.remove_edge(*edge)
        return g
    if family == "core_periphery":
        core = int(rng.integers(3, min(n, 12)))
        g = nx.complete_graph(core)
        for node in range(core, n):
            for hub in rng.choice(core, size=int(rng.integers(1, min(4, core) + 1)), replace=False):
                g.add_edge(node, int(hub))
        return g
    if family == "mixed":
        g = nx.random_labeled_tree(n, seed=seed)
        candidates = [(u, v) for u in g for v in g if u < v and not g.has_edge(u, v)]
        rng.shuffle(candidates)
        g.add_edges_from(candidates[:int(rng.integers(1, max(2, n // 2)))])
        return g
    if family == "cycle":
        return nx.cycle_graph(n)
    if family == "barbell":
        m = max(2, n // 3); path = max(0, n - 2 * m)
        return nx.barbell_graph(m, path)
    raise ValueError(f"unknown family: {family}")


def _rooted_ego(g: nx.Graph, root: int) -> nx.Graph:
    nodes = list(nx.single_source_shortest_path_length(g, root, cutoff=2))
    out = g.subgraph(nodes).copy()
    return nx.convert_node_labels_to_integers(out, label_attribute="original")


def _root_after_relabel(g: nx.Graph, old_root: int) -> int:
    for node, attrs in g.nodes(data=True):
        if attrs.get("original") == old_root:
            return int(node)
    raise RuntimeError("root lost during relabel")


def valid_rooted(g: nx.Graph, root: int) -> bool:
    return (3 <= g.number_of_nodes() <= 64 and nx.is_connected(g)
            and all(d <= 2 for d in nx.single_source_shortest_path_length(g, root).values()))


def sample_rooted_graph(seed: int, split: str, sample_id: int, version: str = "topology-v1",
                        families: Iterable[str] = TRAIN_FAMILIES) -> RootedGraph:
    rng = _rng(seed, split, sample_id, version)
    choices = tuple(families)
    for _ in range(200):
        family = choices[int(rng.integers(len(choices)))]
        raw = _raw_graph(family, _n(rng), rng)
        if not nx.is_connected(raw):
            raw = raw.subgraph(max(nx.connected_components(raw), key=len)).copy()
        if raw.number_of_nodes() < 3:
            continue
        root_old = int(rng.choice(list(raw.nodes())))
        g = _rooted_ego(raw, root_old)
        root = _root_after_relabel(g, root_old)
        if valid_rooted(g, root):
            return RootedGraph(g, root, family, sample_id)
    raise RuntimeError("could not generate a valid rooted graph")


def edge_set(g: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in g.edges()}


def edge_jaccard(a: nx.Graph, b: nx.Graph) -> float:
    ea, eb = edge_set(a), edge_set(b)
    union = ea | eb
    return 0.0 if not union else len(ea ^ eb) / len(union)


def permute(rooted: RootedGraph, rng: np.random.Generator) -> RootedGraph:
    nodes = list(rooted.graph.nodes()); shuffled = list(rng.permutation(nodes))
    mapping = dict(zip(nodes, shuffled))
    return RootedGraph(nx.relabel_nodes(rooted.graph, mapping, copy=True), mapping[rooted.root], rooted.family, rooted.sample_id)


def rooted_isomorphic(a: RootedGraph, b: RootedGraph) -> bool:
    ga, gb = a.graph.copy(), b.graph.copy()
    nx.set_node_attributes(ga, {n: n == a.root for n in ga}, "is_root")
    nx.set_node_attributes(gb, {n: n == b.root for n in gb}, "is_root")
    ha = nx.weisfeiler_lehman_graph_hash(ga, node_attr="is_root")
    hb = nx.weisfeiler_lehman_graph_hash(gb, node_attr="is_root")
    if ha != hb:
        return False
    return nx.is_isomorphic(ga, gb, node_match=lambda x, y: x["is_root"] == y["is_root"])


def _protected_bfs_edges(g: nx.Graph, root: int) -> set[tuple[int, int]]:
    """Edges whose retention proves rooted connectivity and the two-hop bound."""
    tree = nx.bfs_tree(g, root, depth_limit=2)
    return {tuple(sorted((int(u), int(v)))) for u, v in tree.edges()}


def _edit_counts(edge_count: int, additions: int, deletions: int, target: float) -> tuple[int, int, float]:
    """Choose legal add/delete counts minimizing Jaccard-distance error.

    With fixed correspondence and no repeated toggle, d=(add+delete)/(E+add).
    The search is over integer addition counts only; for each count the best
    deletion count is one of the two integers around the analytic solution.
    """
    best = (0, 0, 0.0)
    best_error = abs(target)
    for add in range(additions + 1):
        ideal = target * (edge_count + add) - add
        for delete in {int(np.floor(ideal)), int(np.ceil(ideal)), 0, deletions}:
            delete = min(deletions, max(0, delete))
            distance = (add + delete) / (edge_count + add)
            error = abs(distance - target)
            if error < best_error:
                best, best_error = (add, delete, distance), error
    return best


def perturb(rooted: RootedGraph, target: float, rng: np.random.Generator, tolerance: float = .02,
            attempts: int = 256) -> tuple[RootedGraph, float, set[tuple[int, int]]]:
    """Construct one valid perturbation instead of trial-and-error candidates.

    The BFS tree rooted at ``root`` is protected from deletion.  Therefore all
    remaining nodes stay connected and at distance at most two without calling
    an expensive graph check after every edit. ``attempts`` remains accepted for
    backwards-compatible configs but is intentionally unused.
    """
    del tolerance, attempts
    base_edges = edge_set(rooted.graph)
    nodes = list(rooted.graph.nodes())
    protected = _protected_bfs_edges(rooted.graph, rooted.root)
    removable = tuple(base_edges - protected)
    possible_additions = tuple((u, v) for index, u in enumerate(nodes) for v in nodes[index + 1:]
                               if (u, v) not in base_edges)
    add_count, delete_count, distance = _edit_counts(len(base_edges), len(possible_additions), len(removable), target)
    added = {possible_additions[int(i)] for i in rng.choice(len(possible_additions), size=add_count, replace=False)} if add_count else set()
    removed = {removable[int(i)] for i in rng.choice(len(removable), size=delete_count, replace=False)} if delete_count else set()
    g = rooted.graph.copy()
    g.remove_edges_from(removed); g.add_edges_from(added)
    out = RootedGraph(g, rooted.root, rooted.family, rooted.sample_id)
    # The direct construction is valid by the protected BFS tree invariant.
    assert valid_rooted(out.graph, out.root)
    # rooted_isomorphic first rejects unequal WL hashes, then performs the exact
    # rooted check required by the supervision contract.
    actual = 0.0 if rooted_isomorphic(rooted, out) else distance
    return out, actual, added | removed


def to_data(rooted: RootedGraph) -> Data:
    edges = list(edge_set(rooted.graph))
    directed = edges + [(v, u) for u, v in edges]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous() if directed else torch.empty((2, 0), dtype=torch.long)
    root_mask = torch.zeros(rooted.graph.number_of_nodes(), dtype=torch.bool); root_mask[rooted.root] = True
    return Data(edge_index=edge_index, num_nodes=rooted.graph.number_of_nodes(), root_mask=root_mask)


def stats(rooted: RootedGraph) -> np.ndarray:
    g, root = rooted.graph, rooted.root
    lengths = nx.single_source_shortest_path_length(g, root)
    n = g.number_of_nodes(); edges = g.number_of_edges()
    return np.asarray([n / 64.0, edges / (64 * 63 / 2), sum(d == 1 for d in lengths.values()) / 64.0,
                       sum(d == 2 for d in lengths.values()) / 64.0], dtype=np.float32)


def descriptors(rooted: RootedGraph) -> dict[str, str]:
    g, r = rooted.graph, rooted.root; n = g.number_of_nodes(); m = g.number_of_edges()
    dist = nx.single_source_shortest_path_length(g, r)
    density = nx.density(g); root_ratio = g.degree(r) / max(n - 1, 1)
    def bucket(x: float, cuts: list[float]) -> str:
        return str(sum(x >= c for c in cuts))
    return {
        "family": rooted.family,
        "nodes": bucket(n, [9, 17, 33]), "density": bucket(density, [.1, .25, .5]),
        "root_ratio": bucket(root_ratio, [.25, .5, .75]),
        "n1_ratio": bucket(sum(d == 1 for d in dist.values()) / n, [.25, .5, .75]),
        "n2_ratio": bucket(sum(d == 2 for d in dist.values()) / n, [.25, .5, .75]),
        "cycle_rank": "0" if m - n + 1 == 0 else "1" if m - n + 1 == 1 else "2-4" if m - n + 1 < 5 else "5+",
        "triangles": "0" if sum(nx.triangles(g).values()) == 0 else "1-2" if sum(nx.triangles(g).values()) // 3 <= 2 else "3-8" if sum(nx.triangles(g).values()) // 3 <= 8 else "9+",
    }
