"""Training, validation, checkpoint and export routines."""
from __future__ import annotations

import json
import random
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from tqdm import trange

from .coverage import CoverageController
from .dataset import (collate, make_packed_batch_from_args, make_sample,
                      make_sample_from_args, move_packed_batch, pack_samples)
from .graphs import OOD_FAMILIES, TRAIN_FAMILIES
from .model import StatHead, TopologyEncoder


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def _loss(model, head, batch, margin: float, lambda_rank: float, lambda_stat: float):
    z0, zi, za, zb = model(batch["graphs"]).split(batch["size"], dim=0)
    c_iso = F.cosine_similarity(z0, zi); c_a = F.cosine_similarity(z0, za); c_b = F.cosine_similarity(z0, zb)
    sim = (F.smooth_l1_loss(c_iso, torch.ones_like(c_iso), beta=.1) +
           F.smooth_l1_loss(c_a, 1 - batch["d_a"], beta=.1) +
           F.smooth_l1_loss(c_b, 1 - batch["d_b"], beta=.1)) / 3
    sign = torch.sign(batch["d_b"] - batch["d_a"])
    mask = sign != 0
    rank = F.relu(margin - sign[mask] * (c_a[mask] - c_b[mask])).mean() if mask.any() else sim.new_tensor(0.)
    stat = F.smooth_l1_loss(head(z0), batch["stats"], beta=.1)
    return sim + lambda_rank * rank + lambda_stat * stat, {"sim": sim.item(), "rank": rank.item(), "stat": stat.item()}


@torch.no_grad()
def evaluate(model, head, config: dict, split: str, size: int, ood: bool = False) -> dict:
    device = next(model.parameters()).device; model.eval(); head.eval()
    predictions: list[float] = []; targets: list[float] = []; ranks: list[bool] = []; isos: list[float] = []; losses: list[float] = []
    batch_size = int(config["batch_size"])
    fixed_samples = _fixed_samples(config, split, size, OOD_FAMILIES if ood else None)
    for start in range(0, size, batch_size):
        samples = fixed_samples[start:start + batch_size]
        b = move_packed_batch(pack_samples(samples), device); loss, _ = _loss(model, head, b, config["ranking_margin"], config["lambda_rank"], config["lambda_stat"])
        z0, zi, za, zb = model(b["graphs"]).split(b["size"], dim=0)
        ca, cb = F.cosine_similarity(z0, za), F.cosine_similarity(z0, zb)
        predictions.extend(torch.cat([ca, cb]).cpu().tolist()); targets.extend(torch.cat([1-b["d_a"], 1-b["d_b"]]).cpu().tolist())
        ranks.extend(((ca > cb) == (b["d_a"] < b["d_b"])).cpu().tolist()); isos.extend(F.cosine_similarity(z0, zi).cpu().tolist()); losses.append(loss.item())
    return {"split": split, "ood": ood, "n": size, "pearson": float(pearsonr(predictions, targets).statistic),
            "spearman": float(spearmanr(predictions, targets).statistic), "mae": float(np.mean(np.abs(np.asarray(predictions)-np.asarray(targets)))),
            "huber": float(np.mean(losses)), "ranking_accuracy": float(np.mean(ranks)),
            "isomorphic_cosine": {"mean": float(np.mean(isos)), "std": float(np.std(isos)), "min": float(np.min(isos)),
                                  "p5": float(np.percentile(isos, 5)), "p50": float(np.percentile(isos, 50)), "p95": float(np.percentile(isos, 95))}}


def _save(path: Path, model, head, optimizer, config, step, coverage, best) -> None:
    torch.save({"encoder": model.state_dict(), "stat_head": head.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "step": step, "coverage": coverage.state_dict(), "best": best,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, path)


def _family_schedule(coverage: CoverageController) -> tuple[str, ...]:
    """Oversample currently underrepresented generator families for one immutable block."""
    counts = coverage.counts.get("family", {})
    minimum = min((counts.get(f, 0) for f in TRAIN_FAMILIES), default=0)
    # All families remain reachable; low-count families receive additional tickets.
    tickets: list[str] = []
    for family in TRAIN_FAMILIES:
        gap = max(0, minimum + 1 - counts.get(family, 0))
        tickets.extend([family] * (1 + min(4, gap)))
    return tuple(tickets)


class _ImmediateFuture:
    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


class SamplePrefetcher:
    """Overlap worker-side packed batch generation with GPU optimization."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.workers = int(config.get("data_workers", 0))
        # Keep at least two batch jobs per worker queued; otherwise a setting
        # such as workers=8/prefetch=4 leaves half the CPU generator idle.
        self.depth = max(1, self.workers * 2, int(config.get("prefetch_batches", 1)))
        self.pool = ProcessPoolExecutor(max_workers=self.workers) if self.workers > 0 else None
        self.pending: dict[int, Future] = {}

    def submit(self, start: int, count: int, split: str, families) -> None:
        if start in self.pending:
            return
        args = [(self.config["seed"], split, sample_id, self.config["generator_version"],
                 self.config["perturb_tolerance"], self.config["perturb_attempts"], families)
                for sample_id in range(start, start + count)]
        self.pending[start] = self.pool.submit(make_packed_batch_from_args, args) if self.pool else _ImmediateFuture(make_packed_batch_from_args(args))

    def take(self, start: int):
        return self.pending.pop(start).result()

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


def _fixed_samples(config: dict, split: str, size: int, families=None):
    """Parallel deterministic fixed-set generation for validation and test."""
    args = [(config["seed"], split, i, config["generator_version"], config["perturb_tolerance"],
             config["perturb_attempts"], families) for i in range(size)]
    workers = int(config.get("data_workers", 0))
    if workers <= 0:
        return [make_sample_from_args(item) for item in args]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(make_sample_from_args, args, chunksize=8))


def train(config: dict, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True); device = _device(config["device"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    model = TopologyEncoder(config["hidden_dim"]).to(device); head = StatHead(config["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    coverage = CoverageController(); best = {"spearman": -float("inf")}; history = deque(maxlen=int(config["stability_evals"]))
    batches = range(0, int(config["max_anchors"]), int(config["batch_size"])); iterator = batches if config.get("no_progress") else trange(0, int(config["max_anchors"]), int(config["batch_size"]), desc="anchors")
    schedule: tuple[str, ...] = TRAIN_FAMILIES
    schedule_block = -1
    prefetcher = SamplePrefetcher(config)
    timing = {"wait_seconds": 0.0, "gpu_seconds": 0.0, "batches": 0}
    try:
        for start in iterator:
            block = start // config["coverage_update_every"]
            if block != schedule_block:
                schedule_block = block
                schedule = _family_schedule(coverage)
                (run_dir / "coverage_schedule.jsonl").open("a", encoding="utf-8").write(json.dumps({"start": start, "families": schedule}) + "\n")
            # Batches are generated and packed in workers; no NetworkX object
            # crosses the process boundary. Keep several future batches ready.
            prefetch_end = min(config["max_anchors"], start + config["batch_size"] * prefetcher.depth)
            for future_start in range(start, prefetch_end, config["batch_size"]):
                future_count = min(config["batch_size"], config["max_anchors"] - future_start)
                prefetcher.submit(future_start, future_count, "train", schedule)
            current_end = min(config["max_anchors"], start + config["batch_size"])
            wait_start = time.perf_counter()
            batch = move_packed_batch(prefetcher.take(start), device)
            timing["wait_seconds"] += time.perf_counter() - wait_start
            model.train(); head.train(); optimizer.zero_grad()
            gpu_start = time.perf_counter()
            loss, parts = _loss(model, head, batch, config["ranking_margin"], config["lambda_rank"], config["lambda_stat"]); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), config["grad_clip"]); optimizer.step()
            if device.type == "cuda": torch.cuda.synchronize(device)
            timing["gpu_seconds"] += time.perf_counter() - gpu_start; timing["batches"] += 1
            for desc in batch["descriptors"]: coverage.add(desc)
            consumed = current_end
            if not config.get("no_progress"):
                iterator.set_postfix(loss=f"{loss.item():.4f}", sim=f"{parts['sim']:.4f}",
                                     cpu_wait=f"{timing['wait_seconds']/timing['batches']:.3f}s",
                                     gpu=f"{timing['gpu_seconds']/timing['batches']:.3f}s")
            if consumed % config["eval_every"] < batch["size"]:
                metrics = evaluate(model, head, config, "validation", config["val_size"]); metrics["consumed_anchors"] = consumed
                history.append(metrics)
                (run_dir / "validation_history.jsonl").open("a", encoding="utf-8").write(json.dumps(metrics) + "\n")
                if metrics["spearman"] > best["spearman"]:
                    best = metrics; _save(run_dir / "best.pt", model, head, optimizer, config, consumed, coverage, best)
                stable = len(history) == history.maxlen and max(x["spearman"] for x in history) - min(x["spearman"] for x in history) < config["stability_delta"] and max(x["ranking_accuracy"] for x in history) - min(x["ranking_accuracy"] for x in history) < config["stability_delta"]
                coverage_ok = all(min(values["counts"].values()) >= config["coverage_min_count"] and values["js_divergence"] < config["coverage_js_threshold"] for values in coverage.summary().values())
                if consumed >= config["min_anchors"] and stable and coverage_ok: break
            if consumed % config["checkpoint_every"] < batch["size"]: _save(run_dir / "last.pt", model, head, optimizer, config, consumed, coverage, best)
    finally:
        prefetcher.close()
    _save(run_dir / "last.pt", model, head, optimizer, config, consumed, coverage, best)
    (run_dir / "coverage.json").write_text(json.dumps(coverage.summary(), indent=2), encoding="utf-8")
    timing["mean_cpu_wait_seconds"] = timing.pop("wait_seconds") / max(timing["batches"], 1)
    timing["mean_gpu_step_seconds"] = timing.pop("gpu_seconds") / max(timing["batches"], 1)
    (run_dir / "throughput.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return {"best": best, "consumed_anchors": consumed, "device": str(device), "timing": timing}


def load_checkpoint(path: Path, device: torch.device | str = "cpu"):
    state = torch.load(path, map_location=device); config = state["config"]
    model = TopologyEncoder(config["hidden_dim"]).to(device); head = StatHead(config["hidden_dim"]).to(device)
    model.load_state_dict(state["encoder"]); head.load_state_dict(state["stat_head"])
    return model, head, config, state


def export_encoder(checkpoint: Path, output: Path) -> None:
    model, _, config, _ = load_checkpoint(checkpoint)
    torch.save({"encoder": model.state_dict(), "input_contract": "internal fixed all-one vector; ignores data.x", "input_dim": config["hidden_dim"], "output_dim": config["hidden_dim"], "layers": 2, "aggregation": "sum-gin"}, output)
