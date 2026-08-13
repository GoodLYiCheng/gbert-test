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
runs the GNN on the GPU. The Stage 1 V100 baseline uses `batch_size: 512`,
`data_workers: 16`, and `prefetch_batches: 32`. `configs/v100_32gb.yaml` raises
this to batch size 1024 and 20 workers for a 32 GB V100 with at least 20 CPU cores.

On a shared machine, set `data_workers` to at most the number of CPU cores you
can reserve. Use `nvidia-smi` during a smoke run: if GPU utilisation is low and
The trainer automatically queues at least two batches per worker. CPU is saturated, increase workers; if GPU memory is exhausted, reduce batch
size to 256. Each run writes `throughput.json`: increase workers/prefetch if
`mean_cpu_wait_seconds` exceeds `mean_gpu_step_seconds`; increase batch size if
GPU utilisation is still low after CPU wait is comparable to GPU step time.
