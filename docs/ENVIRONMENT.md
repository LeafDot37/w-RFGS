# Environments used for development and verification

Two environments are used. Both run the same code; only the launcher differs.

## 1. Local verification box (WSL + conda)

Windows 11 host with WSL2 Ubuntu. All local pipeline checks and the small CPU/GPU renders
were run here.

```bash
# from Windows (PowerShell / cmd)
wsl bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh; conda activate mvsplat; \
              cd /mnt/d/GS3D/w-RFGS && python pipeline.py --stage eval --checkpoint <ckpt>'
```

| Item | Value |
|---|---|
| Conda | miniforge3 at `~/miniforge3`, env name `mvsplat` |
| Python | 3.10.21 |
| PyTorch | 2.1.2+cu118 |
| CUDA | available (`torch.cuda.is_available() == True`) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8.6 GB |
| Data root | `datasets/` (see `docs/DATASETS.md`); not tracked |

Notes:

- A non-login shell does not source `~/.bashrc`, so the conda hook must be sourced explicitly as
  above. Alternatively use `conda run -n mvsplat python ...`.
- The Windows-native Python (e.g. `C:\Python314`) has **no torch** and is only used for
  repository tooling (plotting from JSON, LaTeX compilation outside this repo).
- 8.6 GB is enough for evaluation and single-view qualitative renders; full 100k training was
  run on the server.

## 2. Server (AutoDL)

The training runs reported in the paper use an AutoDL GPU instance:

| Item | Value |
|---|---|
| Repository root | `/root/autodl-tmp/rf` |
| Conda env | `/root/miniconda3` (Python 3.10) |
| GPU | RTX 4090-class, 24 GB (CUDA available) |
| Sionna generation env | `/root/miniconda3/envs/rfgen` (separate, for `code/sim`) |

The instance also runs in **no-GPU mode** for dataset packaging and file transfers; in that mode
`nvidia-smi` is unavailable and only CPU work is possible.

## 3. Installing from scratch

```bash
conda create -n wrfgs python=3.10 -y && conda activate wrfgs
pip install -r requirements.txt
```

`torch` must be the CUDA build matching your driver. The Gaussian rasteriser
(`diff-gaussian-rasterization`, `simple-knn`) is only needed for the per-scene SH/CORF baselines;
the main pipeline uses the analytic renderer.
