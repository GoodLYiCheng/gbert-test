# Topology Pretraining Stage 1

This package learns a 128-dimensional rooted topology representation from synthetic 2-hop graphs only.  The encoder internally creates fixed all-one node features and ignores `data.x`; its output is the future GraphToken source vector.

## Installation

The implementation was verified with Python 3.10, PyTorch 1.13.0, PyG 2.5.2,
and CUDA 11.7. For a GPU environment, install the PyTorch wheel that matches
your CUDA runtime first, then install the matching PyG extension wheels from
the official wheel index before installing this package.

```powershell
# Example: tested CUDA 11.7 environment
E:\anaconda\envs\gnn\python.exe -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --index-url https://download.pytorch.org/whl/cu117
E:\anaconda\envs\gnn\python.exe -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
E:\anaconda\envs\gnn\python.exe -m pip install -r requirements.txt
E:\anaconda\envs\gnn\python.exe -m pip install -e .
```

For CPU-only use, install the appropriate PyTorch CPU wheel first, then run
`pip install -r requirements.txt`. PyTorch/PyG binaries must be selected for
the target CUDA and Python version; the project does not bundle them.

## Run

```powershell
topology-pretrain prepare --config configs\smoke.yaml
topology-pretrain train --config configs\smoke.yaml
topology-pretrain evaluate --run-dir artifacts\stage1_<timestamp>
topology-pretrain export --run-dir artifacts\stage1_<timestamp>
```

`configs/smoke.yaml` is the 10,000-anchor engineering smoke configuration. `configs/stage1.yaml` is the 2M–10M online-pretraining configuration. Each run records its configuration, checkpoints, fixed-split metrics, coverage summary, immutable coverage schedule blocks, and encoder-only export.
