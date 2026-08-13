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
