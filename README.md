# Topology Pretraining Stage 1

This package learns a 128-dimensional rooted topology representation from synthetic 2-hop graphs only.  The encoder internally creates fixed all-one node features and ignores `data.x`; its output is the future GraphToken source vector.

## Installation

The target environment is Python 3.10, PyTorch 2.6.0, PyG 2.6.x, and CUDA
12.6. The project itself does not compile CUDA extensions. For a GPU image,
install the matching PyTorch wheel first, then install this package.

```powershell
# CUDA 12.6 / PyTorch 2.6.0
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install "torch-geometric>=2.6,<2.7"
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional PyG CUDA extension wheels for this exact stack are available from
`https://data.pyg.org/whl/torch-2.6.0+cu126.html`; install them only if a later
experiment requires an extension-backed operation. PyTorch/PyG binaries must
match the target CUDA and Python version.

## Run

```powershell
topology-pretrain prepare --config configs\smoke.yaml
topology-pretrain train --config configs\smoke.yaml
topology-pretrain evaluate --run-dir artifacts\stage1_<timestamp>
topology-pretrain export --run-dir artifacts\stage1_<timestamp>
```

`configs/smoke.yaml` is the 10,000-anchor engineering smoke configuration. `configs/stage1.yaml` is the 2M–10M online-pretraining configuration. Each run records its configuration, checkpoints, fixed-split metrics, coverage summary, immutable coverage schedule blocks, and encoder-only export.

## Throughput tuning

The trainer constructs perturbations directly while protecting the root BFS tree,
so it does not retry hundreds of NetworkX candidates per graph. CPU graph
generation is prefetched by `data_workers` processes while the main process
runs the GNN on the GPU. The stable Stage 1 V100 baseline uses `batch_size: 256`,
`data_workers: 16`, and `prefetch_batches: 16` on a 24-core host. `configs/v100_32gb.yaml` raises
this to batch size 1024 and 20 workers for a 32 GB V100 with at least 20 CPU cores.

On a shared machine, set `data_workers` to at most the number of CPU cores you
can reserve. Use `nvidia-smi` during a smoke run: if GPU utilisation is low and
CPU is saturated, increase workers; if GPU memory is exhausted, reduce batch
size to 256. Each run writes `throughput.json`: increase workers/prefetch if
`mean_cpu_wait_seconds` exceeds `mean_gpu_step_seconds`; increase batch size if
GPU utilisation is still low after CPU wait is comparable to GPU step time.

Workers return pure NumPy payloads, so torch tensor sharing cannot exhaust the
container's `/dev/shm` or file-descriptor budget. Prefetch is bounded to one
queued batch per worker. `worker_timeout_seconds` turns a stalled worker into an
error containing worker PIDs and exit codes instead of waiting forever.

The training validation set is a fixed, cached 1,024-anchor model-selection
set. It is generated once before CUDA training starts, so an evaluation at 1%
does not create a second CPU worker pool or stall the GPU. The 50,000-anchor
ID/OOD sets remain reserved for the final `evaluate` command.

## Stage 2: frozen Qwen3-8B QA alignment

Stage 2 converts the frozen 128-dimensional Stage 1 root embedding into four
continuous Qwen3-8B GraphTokens. Qwen and the topology encoder remain frozen;
only the FP32 two-layer Projector is trained. The MVP covers node count, edge
count, root degree/one-hop count, and two-hop count with English integer-answer
prompts.

Install `transformers>=4.51` for Qwen3 support, then edit
`configs/stage2_qwen3_8b_qa.yaml` so `stage1_run_dir` points to the completed
formal Stage 1 run on Linux. The directory must contain an Encoder artifact plus
`run_manifest.json`, `metrics.json`, and `report.md`.

If Stage 1 has not been exported yet, Stage 2 also accepts this repository's
`best.pt` training checkpoint in place of `topology_encoder.pt`. It extracts
only the Encoder state, validates the same 128-dimensional two-layer SUM-GIN
contract, and records the checkpoint hash in provenance. Formal runs still
require `run_manifest.json`, `metrics.json`, and `report.md`.

```bash
topology-pretrain stage2-prepare --config configs/stage2_qwen3_8b_qa.yaml
topology-pretrain stage2-train --config configs/stage2_qwen3_8b_qa.yaml
topology-pretrain stage2-evaluate --run-dir artifacts/stage2_<timestamp>
topology-pretrain stage2-export --run-dir artifacts/stage2_<timestamp>
```

The default formal profile targets one A100 with native BF16 and SDPA. Preparation
writes fixed, split-isolated embedding caches. Training writes a resumable
`last.pt` and `best_projector.safetensors`. Evaluation records paired predictions
for normal, zero, random, shuffled, and text-only GraphToken conditions, followed
by 10,000-sample paired bootstrap confidence intervals. A run that misses any
acceptance threshold remains `UNVERIFIED`; it never unfreezes Stage 1 or Qwen.

For interface-only development, `allow_smoke_artifact: true` permits a local
smoke export without formal Stage 1 evidence. Every resulting cache and run is
permanently marked `engineering_only` and must not be reported as a scientific
Stage 2 result.

### Dual V100 execution

`configs/stage2_qwen3_8b_qa_dual_v100.yaml` runs Qwen3-8B in FP16 with its
layers balanced across exactly two visible V100s. It is one Python process with
naive model parallelism, not DDP, so do not use `torchrun`. The Projector stays
FP32 on the GPU that owns Qwen's input embedding. FP16 loss scaling is enabled,
and the first optimizer step verifies that a finite, nonzero gradient crossed
the frozen sharded LLM back into the GraphTokens. A device map containing CPU,
disk, or only one GPU is rejected.

For an offline model copy, keep `model_id: Qwen/Qwen3-8B` and set
`llm.local_path` to the directory containing its `config.json`, tokenizer
files, and weight shards. Both tokenizer and model loading then use
`local_files_only=True`; no Hub connection is attempted. The pinned model ID
and revision remain in the run provenance.

```bash
export CUDA_VISIBLE_DEVICES=0,1
topology-pretrain stage2-prepare --config configs/stage2_qwen3_8b_qa_dual_v100.yaml
topology-pretrain stage2-train --config configs/stage2_qwen3_8b_qa_dual_v100.yaml
topology-pretrain stage2-evaluate --run-dir artifacts/stage2_<timestamp>
topology-pretrain stage2-export --run-dir artifacts/stage2_<timestamp>
```

The V100 profile keeps the A100 optimization hyperparameters unchanged and
does not quantize, checkpoint activations, or offload weights. Its default
12/14 GiB device caps also support 16 GB V100s while leaving more headroom on
GPU 0 for injected embeddings and Projector activations.
