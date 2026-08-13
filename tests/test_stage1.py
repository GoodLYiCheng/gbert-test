import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Batch

from topology_pretrain.dataset import collate, make_sample
from topology_pretrain.graphs import (OOD_FAMILIES, TRAIN_FAMILIES, edge_jaccard,
                                      edge_set, permute, rooted_isomorphic,
                                      sample_rooted_graph, valid_rooted, perturb)
from topology_pretrain.model import TopologyEncoder
from topology_pretrain.coverage import CoverageController
from topology_pretrain.training import _family_schedule
from topology_pretrain.dataset import pack_samples


def test_rooted_generator_contract():
    for family in (*TRAIN_FAMILIES, *OOD_FAMILIES):
        item = sample_rooted_graph(7, "test", 2, families=(family,))
        assert valid_rooted(item.graph, item.root)
        assert nx.number_of_selfloops(item.graph) == 0


def test_reproducible_sample_and_permutation():
    a = make_sample(11, "validation", 9, "topology-v1", .02, 32)
    b = make_sample(11, "validation", 9, "topology-v1", .02, 32)
    assert edge_set(a.anchor.graph) == edge_set(b.anchor.graph)
    assert rooted_isomorphic(a.anchor, a.iso)
    # The positive's edge labels are intentionally permuted, so its edge-set
    # Jaccard is not a topology distance.  The isomorphic pair has target d=0.


def test_encoder_ignores_external_x_and_is_permutation_invariant():
    rooted = sample_rooted_graph(1, "test", 1)
    permuted = permute(rooted, np.random.default_rng(5))
    d1 = collate([make_sample(1, "test", 1, "topology-v1", .02, 32)], torch.device("cpu"))["anchor"]
    d1.x = torch.randn(d1.num_nodes, 17)
    from topology_pretrain.graphs import to_data
    d2 = Batch.from_data_list([to_data(permuted)])
    model = TopologyEncoder().eval()
    with torch.no_grad():
        z1 = model(d1); z2 = model(d2)
    assert z1.shape == (1, 128)
    assert torch.allclose(z1, z2, atol=1e-6)


def test_pair_batch_contract():
    samples = [make_sample(13, "train", i, "topology-v1", .02, 32) for i in range(3)]
    batch = collate(samples, torch.device("cpu"))
    assert batch["anchor"].root_mask.sum().item() == 3
    assert batch["stats"].shape == (3, 4)


def test_coverage_schedule_keeps_all_families_reachable():
    coverage = CoverageController()
    coverage.add({"family": "er"})
    schedule = _family_schedule(coverage)
    assert set(TRAIN_FAMILIES).issubset(schedule)


def test_changed_edge_set_is_final_symmetric_difference():
    rooted = sample_rooted_graph(19, "test", 3)
    changed, _, delta = perturb(rooted, .5, np.random.default_rng(3), attempts=32)
    assert delta == edge_set(rooted.graph) ^ edge_set(changed.graph)


def test_direct_perturbation_preserves_rooted_contract():
    rooted = sample_rooted_graph(23, "test", 4)
    for target in (0.0, 0.2, 0.5, 0.9):
        changed, _, _ = perturb(rooted, target, np.random.default_rng(100 + int(target * 10)), attempts=256)
        assert valid_rooted(changed.graph, changed.root)


def test_packed_batch_contains_four_graph_views():
    samples = [make_sample(31, "train", i, "topology-v1", .02, 1) for i in range(2)]
    packed = pack_samples(samples)
    assert packed["size"] == 2
    assert packed["graphs"].root_mask.sum().item() == 8
