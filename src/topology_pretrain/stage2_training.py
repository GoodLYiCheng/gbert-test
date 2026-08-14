"""Training, dependency evaluation, reporting, and export for Stage 2 QA."""
from __future__ import annotations

import json
import math
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .stage2_data import QA_TASKS, Stage2QADataset, load_stage2_split, sha256_file
from .stage2_model import (
    GraphProjector,
    build_injected_batch,
    freeze_module,
    greedy_generate_from_embeds,
)


QWEN_MODEL_ID = "Qwen/Qwen3-8B"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEPENDENCY_CONDITIONS = ("normal", "zero", "random", "shuffle", "text-only")


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_data_manifest(config: dict) -> tuple[Path, dict]:
    cache_dir = Path(config["data"]["cache_dir"])
    path = cache_dir / "data_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run stage2-prepare first; missing {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("engineering_only") and not config.get("allow_smoke_artifact", False):
        raise ValueError("Engineering-only Stage 1 data cache cannot be used for a formal Stage 2 run")
    return path, manifest


def _device_and_dtype(config: dict) -> tuple[torch.device, torch.dtype]:
    profile = str(config.get("hardware_profile", "a100_single"))
    if profile not in {"a100_single", "dual_v100"}:
        raise ValueError(f"Unsupported Stage 2 hardware_profile: {profile}")
    requested = str(config.get("device", "cuda:0"))
    if torch.device(requested).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 2 Qwen3-8B training requires CUDA")
    device = torch.device(requested)
    dtype_name = str(config["llm"].get("dtype", "bfloat16"))
    if profile == "a100_single":
        if dtype_name != "bfloat16":
            raise ValueError("The single-A100 profile requires llm.dtype=bfloat16")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured CUDA device does not support native BF16")
        return device, torch.bfloat16
    if device.type != "cuda" or device.index not in {None, 0}:
        raise ValueError("The dual-V100 profile must start on visible CUDA device 0")
    if dtype_name != "float16":
        raise ValueError("The dual-V100 profile requires llm.dtype=float16")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            "The dual-V100 profile requires exactly two visible GPUs; set CUDA_VISIBLE_DEVICES to the chosen pair"
        )
    return torch.device("cuda:0"), torch.float16


def _normalise_device(value: object) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value)
    return f"cuda:{text}" if text.isdigit() else text


def _validate_dual_gpu_device_map(model) -> dict[str, str]:
    raw = getattr(model, "hf_device_map", None)
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("Dual-V100 loading did not produce an hf_device_map")
    normalised = {str(name): _normalise_device(value) for name, value in raw.items()}
    devices = set(normalised.values())
    forbidden = devices - {"cuda:0", "cuda:1"}
    if forbidden:
        raise RuntimeError(f"CPU/disk/off-profile Qwen placement is forbidden: {sorted(forbidden)}")
    if devices != {"cuda:0", "cuda:1"}:
        raise RuntimeError(f"Qwen must use both visible V100 GPUs; observed {sorted(devices)}")
    return normalised


def _projector_device(llm, fallback: torch.device) -> torch.device:
    device = llm.get_input_embeddings().weight.device
    return device if device.type != "meta" else fallback


def _qwen_load_source(llm_config: dict) -> tuple[str, dict]:
    """Resolve a pinned Hub model or an explicitly supplied offline copy."""
    model_id = str(llm_config.get("model_id", QWEN_MODEL_ID))
    revision = str(llm_config.get("revision", QWEN_REVISION))
    if model_id != QWEN_MODEL_ID or revision != QWEN_REVISION:
        raise ValueError("Formal Stage 2 protocol pins Qwen/Qwen3-8B and its approved revision")
    local_path = llm_config.get("local_path")
    if local_path in {None, ""}:
        return model_id, {"revision": revision}
    path = Path(str(local_path)).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Configured local Qwen directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local Qwen directory is missing config.json: {path}")
    return str(path.resolve()), {"local_files_only": True}


def load_frozen_qwen(config: dict, device: torch.device, dtype: torch.dtype):
    try:
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers>=4.51.0 and safetensors>=0.4 before Stage 2") from exc
    if Version(transformers.__version__) < Version("4.51.0"):
        raise RuntimeError(f"Qwen3 requires transformers>=4.51.0; found {transformers.__version__}")
    llm_config = config["llm"]
    model_source, source_kwargs = _qwen_load_source(llm_config)
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, **source_kwargs)
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": str(llm_config.get("attn_implementation", "sdpa")),
        **source_kwargs,
    }
    if str(config.get("hardware_profile", "a100_single")) == "dual_v100":
        max_memory = llm_config.get("max_memory", {0: "28GiB", 1: "30GiB"})
        load_kwargs.update(
            device_map=str(llm_config.get("device_map", "balanced")),
            max_memory={int(key): str(value) for key, value in max_memory.items()},
            low_cpu_mem_usage=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        _validate_dual_gpu_device_map(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs).to(device)
    if int(model.config.hidden_size) != 4096 or int(model.config.num_hidden_layers) != 36:
        raise ValueError("Loaded model is not the pinned Qwen3-8B architecture")
    freeze_module(model)
    model.config.use_cache = False
    return tokenizer, model


def make_projector(config: dict, llm) -> GraphProjector:
    projector_config = config["projector"]
    return GraphProjector(
        graph_dim=int(projector_config.get("graph_dim", 128)),
        hidden_dim=int(projector_config.get("hidden_dim", 512)),
        llm_dim=int(llm.config.hidden_size),
        num_tokens=int(projector_config.get("num_tokens", 4)),
        initializer_range=float(llm.config.initializer_range),
    )


def assert_frozen_training_contract(projector: torch.nn.Module, llm: torch.nn.Module, optimizer) -> None:
    if any(parameter.requires_grad for parameter in llm.parameters()):
        raise RuntimeError("Frozen LLM contains trainable parameters")
    projector_ids = {id(parameter) for parameter in projector.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != projector_ids:
        raise RuntimeError("Optimizer must contain every Projector parameter and no other parameters")


def _collate(batch: list[dict]) -> dict:
    return {
        "embedding": torch.stack([item["embedding"] for item in batch]),
        "graph_index": torch.tensor([item["graph_index"] for item in batch], dtype=torch.long),
        "sample_id": torch.tensor([item["sample_id"] for item in batch], dtype=torch.long),
        "task_index": torch.tensor([item["task_index"] for item in batch], dtype=torch.long),
        "task": [item["task"] for item in batch],
        "question": [item["question"] for item in batch],
        "template_index": torch.tensor([item["template_index"] for item in batch], dtype=torch.long),
        "answer": torch.tensor([item["answer"] for item in batch], dtype=torch.long),
    }


def _loader(dataset, config: dict, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config["training"].get("data_workers", 0)),
        collate_fn=_collate,
        generator=generator,
        pin_memory=True,
    )


def _scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _grad_scaler(enabled: bool):
    """Use the current AMP API while retaining compatibility with older test hosts."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _save_projector(path: Path, projector: GraphProjector) -> None:
    from safetensors.torch import save_file

    state = {key: value.detach().cpu().contiguous() for key, value in projector.state_dict().items()}
    save_file(state, str(path))


def _load_projector(path: Path, projector: GraphProjector) -> None:
    from safetensors.torch import load_file

    projector.load_state_dict(load_file(str(path), device="cpu"), strict=True)


def _checkpoint(
    path: Path,
    *,
    projector,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best: dict,
    config: dict,
    data_manifest_sha256: str,
) -> None:
    torch.save(
        {
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": best,
            "config": config,
            "data_manifest_sha256": data_manifest_sha256,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def _restore(path: Path, projector, optimizer, scheduler, scaler, expected_hash: str, device: torch.device) -> tuple[int, int, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state["data_manifest_sha256"] != expected_hash:
        raise ValueError("Resume checkpoint uses a different Stage 2 data manifest")
    projector.load_state_dict(state["projector"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return int(state["epoch"]) + 1, int(state["global_step"]), dict(state["best"])


def _parse_integer(text: str) -> int | None:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    return int(match.group(1)) if match else None


def _decode_predictions(tokenizer, token_ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode(row.tolist(), skip_special_tokens=True).strip() for row in token_ids.cpu()]


def _predict_loader(
    *,
    projector,
    llm,
    tokenizer,
    loader,
    config: dict,
    device: torch.device,
    condition: str = "normal",
    graph_token_table: torch.Tensor | None = None,
    shuffle_indices: np.ndarray | None = None,
    random_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[dict]:
    projector.eval()
    llm.eval()
    llm.config.use_cache = True
    predictions: list[dict] = []
    rng = torch.Generator(device=device).manual_seed(int(config["seed"]) + 99173)
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device, non_blocking=True)
            if condition == "text-only":
                tokens = None
            elif condition == "shuffle":
                if graph_token_table is None or shuffle_indices is None:
                    raise ValueError("shuffle requires graph_token_table and shuffle_indices")
                source = torch.from_numpy(shuffle_indices[batch["graph_index"].numpy(), batch["task_index"].numpy()]).long()
                tokens = graph_token_table[source.to(graph_token_table.device)]
            else:
                tokens = projector(embeddings)
                if condition == "zero":
                    tokens = torch.zeros_like(tokens)
                elif condition == "random":
                    if random_stats is None:
                        raise ValueError("random requires fitted GraphToken statistics")
                    mean, std = random_stats
                    noise = torch.randn(tokens.shape, generator=rng, device=device, dtype=torch.float32)
                    tokens = mean + noise * std
                elif condition != "normal":
                    raise ValueError(f"unknown dependency condition: {condition}")
            injected = build_injected_batch(
                tokenizer=tokenizer,
                embedding_layer=llm.get_input_embeddings(),
                questions=batch["question"],
                graph_tokens=tokens,
                answers=None,
                max_length=int(config["llm"].get("max_input_length", 128)),
            )
            generated = greedy_generate_from_embeds(
                llm,
                injected,
                eos_token_id=int(tokenizer.eos_token_id),
                max_new_tokens=int(config["llm"].get("max_new_tokens", 8)),
            )
            texts = _decode_predictions(tokenizer, generated)
            for index, text in enumerate(texts):
                predictions.append(
                    {
                        "condition": condition,
                        "graph_index": int(batch["graph_index"][index]),
                        "sample_id": int(batch["sample_id"][index]),
                        "task": batch["task"][index],
                        "template_index": int(batch["template_index"][index]),
                        "answer": int(batch["answer"][index]),
                        "prediction": text,
                        "parsed_integer": _parse_integer(text),
                    }
                )
    llm.config.use_cache = False
    return predictions


def _condition_metrics(records: list[dict]) -> dict:
    by_task: dict[str, dict] = {}
    for task in QA_TASKS:
        rows = [row for row in records if row["task"] == task]
        parsed = [row["parsed_integer"] for row in rows]
        targets = [row["answer"] for row in rows]
        correct = [prediction == target for prediction, target in zip(parsed, targets)]
        valid_errors = [abs(prediction - target) for prediction, target in zip(parsed, targets) if prediction is not None]
        by_task[task] = {
            "n": len(rows),
            "strict_exact_match": float(np.mean([row["prediction"] == str(row["answer"]) for row in rows])),
            "numeric_accuracy": float(np.mean(correct)),
            "parse_rate": float(np.mean([value is not None for value in parsed])),
            "mae_on_parsed": float(np.mean(valid_errors)) if valid_errors else None,
        }
    return {
        "n": len(records),
        "macro_numeric_accuracy": float(np.mean([value["numeric_accuracy"] for value in by_task.values()])),
        "macro_strict_exact_match": float(np.mean([value["strict_exact_match"] for value in by_task.values()])),
        "by_task": by_task,
    }


def _majority_baselines(train_arrays: dict[str, np.ndarray], test_arrays: dict[str, np.ndarray]) -> dict:
    output = {}
    for task_index, task in enumerate(QA_TASKS):
        counter = Counter(map(int, train_arrays["facts"][:, task_index]))
        label, _ = counter.most_common(1)[0]
        test_accuracy = float(np.mean(test_arrays["facts"][:, task_index] == label))
        output[task] = {
            "label": label,
            "accuracy": test_accuracy,
            "selection_basis": "most frequent label in train; accuracy measured on fixed test split",
        }
    return output


def _shuffle_map(test_arrays: dict[str, np.ndarray]) -> np.ndarray:
    facts = test_arrays["facts"]
    mapping = np.empty((len(facts), len(QA_TASKS)), dtype=np.int64)
    for task_index in range(len(QA_TASKS)):
        order = np.argsort(facts[:, task_index], kind="stable")
        values = facts[order, task_index]
        largest_group = max(Counter(map(int, values)).values())
        if largest_group * 2 > len(order):
            raise ValueError(f"No answer-different shuffle derangement exists for {QA_TASKS[task_index]}")
        rotated = np.roll(order, -largest_group)
        if np.any(facts[order, task_index] == facts[rotated, task_index]) or np.any(order == rotated):
            raise RuntimeError(f"Could not construct deterministic shuffle for {QA_TASKS[task_index]}")
        mapping[order, task_index] = rotated
    return mapping


@torch.no_grad()
def _project_table(projector, embeddings: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    rows = []
    projector.eval()
    for start in range(0, len(embeddings), batch_size):
        batch = torch.from_numpy(embeddings[start : start + batch_size]).to(device)
        rows.append(projector(batch).cpu())
    return torch.cat(rows, dim=0)


def _fit_random_stats(projector, train_embeddings: np.ndarray, device: torch.device, batch_size: int):
    table = _project_table(projector, train_embeddings, device, batch_size).float()
    mean = table.mean(dim=0, keepdim=True).to(device)
    std = table.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6).to(device)
    return mean, std


def _paired_bootstrap(normal: list[dict], control: list[dict], seed: int, samples: int) -> dict:
    if len(normal) != len(control):
        raise ValueError("Paired conditions have different numbers of predictions")
    normal_correct = np.asarray([row["parsed_integer"] == row["answer"] for row in normal], dtype=np.float64)
    control_correct = np.asarray([row["parsed_integer"] == row["answer"] for row in control], dtype=np.float64)
    delta = normal_correct - control_correct
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 100
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        estimates[start : start + count] = delta[indices].mean(axis=1)
    return {
        "accuracy_delta": float(delta.mean()),
        "bootstrap_samples": samples,
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
    }


def _acceptance(metrics: dict, majority: dict) -> dict:
    normal = metrics["conditions"]["normal"]
    checks = {
        "normal_macro_at_least_0_60": normal["macro_numeric_accuracy"] >= 0.60,
        "all_tasks_above_majority_plus_0_05": all(
            normal["by_task"][task]["numeric_accuracy"] >= majority[task]["accuracy"] + 0.05
            for task in QA_TASKS
        ),
    }
    thresholds = {"shuffle": 0.15, "zero": 0.10, "random": 0.10, "text-only": 0.10}
    for condition, threshold in thresholds.items():
        comparison = metrics["dependency_deltas"][condition]
        checks[f"normal_over_{condition}"] = (
            comparison["accuracy_delta"] >= threshold and comparison["ci95"][0] > 0.0
        )
    checks["finite_metrics"] = all(
        math.isfinite(value["macro_numeric_accuracy"]) for value in metrics["conditions"].values()
    )
    return {"passed": all(checks.values()), "checks": checks}


def _write_report(run_dir: Path, metrics: dict, manifest: dict) -> None:
    lines = [
        "# Stage 2 Topology-to-LLM QA MVP Report",
        "",
        "Evaluation basis: fixed synthetic rooted-graph test split; Qwen3-8B and Stage 1 Encoder frozen; Projector-only alignment.",
        "",
        f"Verification status: **{'VERIFIED' if metrics['acceptance']['passed'] else 'UNVERIFIED'}**",
        f"Engineering-only Stage 1 artifact: **{manifest['engineering_only']}**",
        "",
        "## Dependency conditions",
        "",
        "| Condition | Macro numeric accuracy | Macro exact match |",
        "|---|---:|---:|",
    ]
    for condition in DEPENDENCY_CONDITIONS:
        value = metrics["conditions"][condition]
        lines.append(
            f"| {condition} | {value['macro_numeric_accuracy']:.4f} | {value['macro_strict_exact_match']:.4f} |"
        )
    lines.extend(["", "## Normal performance by task", "", "| Task | Numeric accuracy | Parse rate | MAE on parsed |", "|---|---:|---:|---:|"])
    for task in QA_TASKS:
        value = metrics["conditions"]["normal"]["by_task"][task]
        mae = "null" if value["mae_on_parsed"] is None else f"{value['mae_on_parsed']:.4f}"
        lines.append(f"| {task} | {value['numeric_accuracy']:.4f} | {value['parse_rate']:.4f} | {mae} |")
    lines.extend(["", "## Acceptance", ""])
    for name, passed in metrics["acceptance"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_stage2(
    run_dir: Path,
    *,
    config: dict | None = None,
    tokenizer=None,
    llm=None,
    projector=None,
) -> dict:
    run_dir = Path(run_dir)
    if config is None:
        config = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["config"]
    manifest_path, data_manifest = _load_data_manifest(config)
    device, dtype = _device_and_dtype(config)
    if tokenizer is None or llm is None:
        tokenizer, llm = load_frozen_qwen(config, device, dtype)
    projector_device = _projector_device(llm, device)
    if projector is None:
        projector = make_projector(config, llm)
        _load_projector(run_dir / "best_projector.safetensors", projector)
    projector.to(projector_device)
    train_arrays = load_stage2_split(Path(config["data"]["cache_dir"]), "train")
    test_arrays = load_stage2_split(Path(config["data"]["cache_dir"]), "test")
    test_dataset = Stage2QADataset(test_arrays, "test", int(config["seed"]))
    loader = _loader(test_dataset, config, False, int(config["seed"]))
    batch_size = int(config["training"].get("batch_size", 8))
    graph_table = _project_table(projector, test_arrays["embeddings"], projector_device, batch_size)
    shuffle_indices = _shuffle_map(test_arrays)
    random_stats = _fit_random_stats(projector, train_arrays["embeddings"], projector_device, batch_size)
    records_by_condition = {}
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for condition in DEPENDENCY_CONDITIONS:
            records = _predict_loader(
                projector=projector,
                llm=llm,
                tokenizer=tokenizer,
                loader=loader,
                config=config,
                device=projector_device,
                condition=condition,
                graph_token_table=graph_table.to(projector_device) if condition == "shuffle" else None,
                shuffle_indices=shuffle_indices if condition == "shuffle" else None,
                random_stats=random_stats if condition == "random" else None,
            )
            records_by_condition[condition] = records
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    conditions = {name: _condition_metrics(records) for name, records in records_by_condition.items()}
    bootstrap_samples = int(config["evaluation"].get("bootstrap_samples", 10000))
    dependency = {
        condition: _paired_bootstrap(
            records_by_condition["normal"],
            records_by_condition[condition],
            int(config["seed"]) + index,
            bootstrap_samples,
        )
        for index, condition in enumerate(DEPENDENCY_CONDITIONS[1:], start=1)
    }
    majority = _majority_baselines(train_arrays, test_arrays)
    metrics = {
        "conditions": conditions,
        "majority_baselines": majority,
        "dependency_deltas": dependency,
        "evaluation_basis": "fixed synthetic test split; paired predictions across all dependency conditions",
        "data_manifest_sha256": sha256_file(manifest_path),
    }
    metrics["acceptance"] = _acceptance(metrics, majority)
    _json_dump(run_dir / "metrics.json", metrics)
    _write_report(run_dir, metrics, data_manifest)
    return metrics


def train_stage2(config: dict, run_dir: Path, *, tokenizer=None, llm=None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    manifest_path, data_manifest = _load_data_manifest(config)
    if config["llm"].get("enable_thinking", False) is not False:
        raise ValueError("Stage 2 QA MVP requires llm.enable_thinking=false")
    manifest_hash = sha256_file(manifest_path)
    shutil.copy2(manifest_path, run_dir / "data_manifest.json")
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest.update(
            {
                "engineering_only": data_manifest["engineering_only"],
                "data_manifest_sha256": manifest_hash,
                "stage1_export_sha256": data_manifest.get("stage1", {}).get("export_sha256"),
            }
        )
        _json_dump(run_manifest_path, run_manifest)
    device, dtype = _device_and_dtype(config)
    if tokenizer is None or llm is None:
        tokenizer, llm = load_frozen_qwen(config, device, dtype)
    else:
        freeze_module(llm)
        llm.config.use_cache = False
    projector_device = _projector_device(llm, device)
    projector = make_projector(config, llm).to(device=projector_device, dtype=torch.float32)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    train_arrays = load_stage2_split(Path(config["data"]["cache_dir"]), "train")
    validation_arrays = load_stage2_split(Path(config["data"]["cache_dir"]), "validation")
    train_dataset = Stage2QADataset(train_arrays, "train", seed)
    validation_dataset = Stage2QADataset(validation_arrays, "validation", seed)
    epochs = int(training.get("epochs", 4))
    accumulation = int(training.get("gradient_accumulation_steps", 4))
    updates_per_epoch = math.ceil(math.ceil(len(train_dataset) / int(training.get("batch_size", 8))) / accumulation)
    scheduler = _scheduler(optimizer, updates_per_epoch * epochs, float(training.get("warmup_ratio", 0.05)))
    scaler = _grad_scaler(enabled=dtype == torch.float16 and projector_device.type == "cuda")
    assert_frozen_training_contract(projector, llm, optimizer)
    start_epoch, global_step, best = 0, 0, {"macro_numeric_accuracy": -1.0, "epoch": -1}
    resume = training.get("resume_from")
    if resume:
        start_epoch, global_step, best = _restore(
            Path(resume), projector, optimizer, scheduler, scaler, manifest_hash, projector_device
        )
        _save_projector(run_dir / "best_projector.safetensors", projector)
    history_path = run_dir / "validation_history.jsonl"
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        loader = _loader(train_dataset, config, True, seed + epoch)
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        rms_rows = []
        iterator = loader if config.get("no_progress", False) else tqdm(loader, desc=f"Stage 2 epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(iterator):
            topology = batch["embedding"].to(projector_device, non_blocking=True)
            graph_tokens = projector(topology)
            injected = build_injected_batch(
                tokenizer=tokenizer,
                embedding_layer=llm.get_input_embeddings(),
                questions=batch["question"],
                graph_tokens=graph_tokens,
                answers=batch["answer"].tolist(),
                max_length=int(config["llm"].get("max_input_length", 128)),
            )
            outputs = llm(
                inputs_embeds=injected.inputs_embeds,
                attention_mask=injected.attention_mask,
                position_ids=injected.position_ids,
                labels=injected.labels,
                use_cache=False,
                return_dict=True,
            )
            loss = outputs.loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Stage 2 loss at epoch={epoch}, batch={batch_index}")
            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * accumulation
            rms_rows.append((injected.graph_token_rms, injected.text_token_rms))
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                gradients = [parameter.grad for parameter in projector.parameters() if parameter.grad is not None]
                if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                    raise FloatingPointError("Projector gradients are missing or non-finite")
                if not any(bool(torch.count_nonzero(gradient)) for gradient in gradients):
                    raise RuntimeError(
                        "Projector gradient is identically zero; the sharded LLM may have broken the GraphToken gradient path"
                    )
                if any(parameter.grad is not None for parameter in llm.parameters()):
                    raise RuntimeError("Frozen Qwen parameter unexpectedly received a gradient")
                torch.nn.utils.clip_grad_norm_(projector.parameters(), float(training.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{total_loss / (batch_index + 1):.4f}")

        validation_loader = _loader(validation_dataset, config, False, seed)
        validation_records = _predict_loader(
            projector=projector,
            llm=llm,
            tokenizer=tokenizer,
            loader=validation_loader,
            config=config,
            device=projector_device,
        )
        validation_metrics = _condition_metrics(validation_records)
        validation_metrics.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / max(1, len(loader)),
                "graph_token_rms": float(np.mean([row[0] for row in rms_rows])),
                "text_token_rms": float(np.mean([row[1] for row in rms_rows])),
            }
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation_metrics) + "\n")
        if validation_metrics["macro_numeric_accuracy"] > best["macro_numeric_accuracy"]:
            best = validation_metrics
            _save_projector(run_dir / "best_projector.safetensors", projector)
        _checkpoint(
            run_dir / "last.pt",
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best=best,
            config=config,
            data_manifest_sha256=manifest_hash,
        )
    return {
        "best_validation": best,
        "global_step": global_step,
        "engineering_only": data_manifest["engineering_only"],
        "hardware_profile": str(config.get("hardware_profile", "a100_single")),
        "projector_device": str(projector_device),
        "llm_device_map": (
            _validate_dual_gpu_device_map(llm)
            if str(config.get("hardware_profile", "a100_single")) == "dual_v100"
            else {"": str(device)}
        ),
        "run_dir": str(run_dir.resolve()),
    }


def export_stage2(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    data_manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    projector_path = run_dir / "best_projector.safetensors"
    if not projector_path.is_file():
        raise FileNotFoundError(f"Best Projector not found: {projector_path}")
    meta = {
        "schema_version": "stage2-projector-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "projector_sha256": sha256_file(projector_path),
        "graph_dim": 128,
        "hidden_dim": int(config["projector"].get("hidden_dim", 512)),
        "num_graph_tokens": int(config["projector"].get("num_tokens", 4)),
        "llm_dim": 4096,
        "llm_model_id": QWEN_MODEL_ID,
        "llm_revision": QWEN_REVISION,
        "llm_mode": "non-thinking",
        "training_hardware_profile": str(config.get("hardware_profile", "a100_single")),
        "prompt_protocol": data_manifest["prompt_bank_version"],
        "stage1_export_sha256": data_manifest["stage1"]["export_sha256"],
        "data_manifest_sha256": sha256_file(run_dir / "data_manifest.json"),
        "engineering_only": data_manifest["engineering_only"],
    }
    _json_dump(run_dir / "projector_meta.json", meta)
    return meta
