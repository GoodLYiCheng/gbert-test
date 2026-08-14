from __future__ import annotations

from types import SimpleNamespace
import json
import shutil

import networkx as nx
import numpy as np
import pytest
import torch
from torch import nn

from topology_pretrain.graphs import RootedGraph
from topology_pretrain.stage2_data import (
    GRAPH_SLOT,
    PROMPT_TEMPLATES,
    QA_TASKS,
    Stage2QADataset,
    question_for,
    prepare_stage2_cache,
    rooted_topology_hash,
    task_permutation,
    topology_facts,
    validate_stage1_artifact,
    sha256_file,
)
from topology_pretrain.stage2_model import (
    GraphProjector,
    build_injected_batch,
    freeze_module,
    prompt_token_ids,
)
from topology_pretrain.stage2_training import (
    _device_and_dtype,
    _majority_baselines,
    _qwen_load_source,
    _shuffle_map,
    _validate_dual_gpu_device_map,
    assert_frozen_training_contract,
    train_stage2,
)


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self) -> None:
        characters = "\n <>|_/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?.,'-:+"
        self.char_to_id = {character: index + 2 for index, character in enumerate(characters)}
        self.id_to_char = {value: key for key, value in self.char_to_id.items()}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [self.char_to_id[character] for character in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.id_to_char.get(int(value), "") for value in ids if int(value) > 1)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            initializer_range=0.02,
            use_cache=False,
        )
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        return_dict=True,
        **kwargs,
    ):
        del return_dict, kwargs
        values = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        hidden, state = self.rnn(values, past_key_values)
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits, past_key_values=state)


def test_topology_facts_merge_degree_and_hop_one():
    star = RootedGraph(nx.star_graph(4), 0, "star", 0)
    assert topology_facts(star).tolist() == [5, 4, 4, 0]
    path = RootedGraph(nx.path_graph(5), 2, "path", 1)
    assert topology_facts(path).tolist() == [5, 4, 2, 2]


def test_rooted_hash_is_permutation_invariant_and_root_sensitive():
    graph = nx.path_graph(5)
    original = RootedGraph(graph, 0, "path", 0)
    relabeled_graph = nx.relabel_nodes(graph, {0: 4, 1: 3, 2: 2, 3: 1, 4: 0})
    relabeled = RootedGraph(relabeled_graph, 4, "path", 1)
    different_root = RootedGraph(graph, 2, "path", 2)
    assert rooted_topology_hash(original) == rooted_topology_hash(relabeled)
    assert rooted_topology_hash(original) != rooted_topology_hash(different_root)


def test_task_schedule_and_templates_are_deterministic_and_isolated():
    first = task_permutation(7, 11)
    second = task_permutation(7, 11)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(QA_TASKS)))
    train_indices = {question_for(0, "train", seed=7, sample_id=11, epoch=e)[1] for e in range(12)}
    validation_index = question_for(0, "validation", seed=7, sample_id=11)[1]
    test_index = question_for(0, "test", seed=7, sample_id=11)[1]
    assert train_indices <= {0, 1, 2, 3}
    assert validation_index == 4 and test_index == 5
    assert len(PROMPT_TEMPLATES["root_degree"]) >= 6


def test_fixed_split_dataset_expands_every_graph_to_four_tasks():
    arrays = {
        "embeddings": np.zeros((2, 128), dtype=np.float32),
        "facts": np.asarray([[3, 2, 1, 1], [5, 4, 2, 2]], dtype=np.int64),
        "sample_ids": np.asarray([4, 8], dtype=np.int64),
    }
    dataset = Stage2QADataset(arrays, "test", 9)
    assert len(dataset) == 8
    assert [dataset[index]["task"] for index in range(4)] == list(QA_TASKS)


def test_shuffle_is_a_bijection_and_always_changes_the_task_answer():
    facts = np.asarray(
        [
            [4, 3, 1, 0],
            [5, 4, 2, 1],
            [6, 5, 3, 2],
            [7, 6, 4, 3],
        ],
        dtype=np.int64,
    )
    mapping = _shuffle_map({"facts": facts})
    for task_index in range(len(QA_TASKS)):
        assert sorted(mapping[:, task_index].tolist()) == list(range(len(facts)))
        assert np.all(facts[:, task_index] != facts[mapping[:, task_index], task_index])


def test_majority_label_is_selected_on_train_but_scored_on_test():
    train = {"facts": np.asarray([[4, 1, 1, 1], [4, 2, 2, 2], [5, 3, 3, 3]])}
    test = {"facts": np.asarray([[4, 8, 8, 8], [5, 8, 8, 8], [5, 8, 8, 8]])}
    baseline = _majority_baselines(train, test)["num_nodes"]
    assert baseline["label"] == 4
    assert baseline["accuracy"] == pytest.approx(1 / 3)


def test_dual_v100_runtime_requires_fp16_and_exactly_two_visible_gpus(monkeypatch):
    config = {
        "hardware_profile": "dual_v100",
        "device": "cuda:0",
        "llm": {"dtype": "float16"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device, dtype = _device_and_dtype(config)
    assert device == torch.device("cuda:0") and dtype == torch.float16
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="exactly two visible GPUs"):
        _device_and_dtype(config)
    config["llm"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="requires llm.dtype=float16"):
        _device_and_dtype(config)


def test_dual_gpu_device_map_rejects_offload_and_requires_both_gpus():
    valid = SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "model.layers.20": "cuda:1"})
    assert set(_validate_dual_gpu_device_map(valid).values()) == {"cuda:0", "cuda:1"}
    with pytest.raises(RuntimeError, match="forbidden"):
        _validate_dual_gpu_device_map(
            SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "model.layers.20": "cpu"})
        )
    with pytest.raises(RuntimeError, match="must use both"):
        _validate_dual_gpu_device_map(SimpleNamespace(hf_device_map={"": 0}))


def test_qwen_local_source_is_offline_and_keeps_pinned_identity(tmp_path):
    local_model = tmp_path / "Qwen3-8B"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    source, kwargs = _qwen_load_source(
        {
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "local_path": str(local_model),
        }
    )
    assert source == str(local_model.resolve())
    assert kwargs == {"local_files_only": True}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _qwen_load_source(
            {
                "model_id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "local_path": str(tmp_path / "missing"),
            }
        )


def test_stage1_smoke_requires_explicit_engineering_override():
    run_dir = __import__("pathlib").Path("artifacts/micro_smoke")
    with pytest.raises(FileNotFoundError):
        validate_stage1_artifact(run_dir, allow_smoke=False)
    encoder, provenance = validate_stage1_artifact(run_dir, allow_smoke=True)
    assert provenance["engineering_only"] is True
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_graph_slot_injection_masks_prompt_and_backpropagates_only_to_projector():
    tokenizer = CharacterTokenizer()
    llm = freeze_module(TinyCausalLM(max(tokenizer.id_to_char) + 2))
    projector = GraphProjector(128, 12, llm.config.hidden_size, 4, 0.02)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    assert_frozen_training_contract(projector, llm, optimizer)
    topology = torch.randn(2, 128)
    graph_tokens = projector(topology)
    injected = build_injected_batch(
        tokenizer=tokenizer,
        embedding_layer=llm.get_input_embeddings(),
        questions=["How many nodes are in the rooted graph?", "What is the root degree?"],
        graph_tokens=graph_tokens,
        answers=[5, 2],
        max_length=512,
    )
    assert injected.inputs_embeds.shape[0] == 2
    assert injected.position_ids.shape == injected.attention_mask.shape
    assert torch.all(injected.labels[:, : injected.prompt_lengths.min()] == -100)
    assert all(GRAPH_SLOT not in tokenizer.decode(row.tolist()) for row in injected.labels)
    output = llm(
        inputs_embeds=injected.inputs_embeds,
        attention_mask=injected.attention_mask,
        position_ids=injected.position_ids,
        labels=injected.labels,
        use_cache=False,
        return_dict=True,
    )
    output.loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in llm.parameters())
    assert all(not parameter.requires_grad for parameter in llm.parameters())


def test_graph_slot_must_appear_once():
    tokenizer = CharacterTokenizer()
    ids, start, end = prompt_token_ids(tokenizer, "How many nodes?", 512)
    slot = tokenizer(GRAPH_SLOT, add_special_tokens=False)["input_ids"]
    assert ids[start:end] == slot
    assert sum(ids[index : index + len(slot)] == slot for index in range(len(ids) - len(slot) + 1)) == 1


def test_prepare_cache_is_deterministic_and_split_isolated(tmp_path):
    stage1 = tmp_path / "formal_stage1"
    stage1.mkdir()
    source = __import__("pathlib").Path("artifacts/micro_smoke")
    for name in ("topology_encoder.pt", "metrics.json", "report.md"):
        shutil.copy2(source / name, stage1 / name)
    (stage1 / "run_manifest.json").write_text("{}", encoding="utf-8")

    def prepare(cache_dir):
        return prepare_stage2_cache(
            {
                "seed": 19,
                "generator_version": "topology-v1",
                "stage1_run_dir": str(stage1),
                "allow_smoke_artifact": False,
                "no_progress": True,
                "data": {
                    "cache_dir": str(cache_dir),
                    "train_graphs": 3,
                    "validation_graphs": 2,
                    "test_graphs": 2,
                    "prepare_device": "cpu",
                    "embedding_batch_size": 2,
                },
            }
        )

    first_dir, second_dir = tmp_path / "cache_a", tmp_path / "cache_b"
    first, second = prepare(first_dir), prepare(second_dir)
    all_hashes = []
    for split in ("train", "validation", "test"):
        with np.load(first_dir / f"{split}.npz", allow_pickle=False) as left, np.load(
            second_dir / f"{split}.npz", allow_pickle=False
        ) as right:
            assert np.array_equal(left["embeddings"], right["embeddings"])
            assert np.array_equal(left["facts"], right["facts"])
            assert np.array_equal(left["topology_hashes"], right["topology_hashes"])
            all_hashes.extend(left["topology_hashes"].tolist())
        assert first["splits"][split]["fact_min"] == second["splits"][split]["fact_min"]
    assert len(all_hashes) == len(set(all_hashes))


def test_tiny_stage2_training_writes_best_and_resumable_checkpoint(tmp_path):
    tokenizer = CharacterTokenizer()
    llm = TinyCausalLM(max(tokenizer.id_to_char) + 2)
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    cache_dir.mkdir(); run_dir.mkdir()
    split_sizes = {"train": 4, "validation": 2, "test": 2}
    splits = {}
    rng = np.random.default_rng(31)
    for split, size in split_sizes.items():
        path = cache_dir / f"{split}.npz"
        np.savez_compressed(
            path,
            embeddings=rng.normal(size=(size, 128)).astype(np.float32),
            facts=np.asarray([[4 + i, 3 + i, 1 + i % 2, 2] for i in range(size)], dtype=np.int64),
            sample_ids=np.arange(size, dtype=np.int64),
            families=np.asarray(["tree"] * size, dtype="U32"),
            topology_hashes=np.asarray([f"{split}-{i}" for i in range(size)], dtype="U64"),
        )
        splits[split] = {"cache_file": path.name, "cache_sha256": sha256_file(path)}
    (cache_dir / "data_manifest.json").write_text(
        json.dumps({"engineering_only": True, "splits": splits}), encoding="utf-8"
    )
    config = {
        "seed": 31,
        "device": "cpu",
        "allow_smoke_artifact": True,
        "no_progress": True,
        "output_dir": str(tmp_path),
        "data": {"cache_dir": str(cache_dir)},
        "llm": {"dtype": "bfloat16", "enable_thinking": False, "max_input_length": 512, "max_new_tokens": 2},
        "projector": {"graph_dim": 128, "hidden_dim": 12, "num_tokens": 4},
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "grad_clip": 1.0,
            "data_workers": 0,
            "resume_from": None,
        },
        "evaluation": {"bootstrap_samples": 10},
    }
    result = train_stage2(config, run_dir, tokenizer=tokenizer, llm=llm)
    assert result["engineering_only"] is True
    assert (run_dir / "best_projector.safetensors").stat().st_size > 0
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 0 and checkpoint["global_step"] == 2
    assert checkpoint["optimizer"] and checkpoint["scheduler"]
    assert "scaler" in checkpoint
