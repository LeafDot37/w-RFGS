# w-RFGS — Feed-Forward Radio Radiance Fields for Unseen Transmitters and Beams

w-RFGS renders radio-frequency angular spectra (CBF and MVDR heat maps) from **frozen optical 3D
Gaussians** for transmitters whose position, yaw, pitch, carrier frequency and beamforming weights
**w** were never seen during training. One model is trained on 144 transmitter configurations and
then answers any held-out configuration in a single forward pass — no per-transmitter refit.

The field is physical, not a colour map: per Gaussian the head predicts a direction-dependent
complex scattering coefficient and a geometric delay; the channel is assembled analytically and
the transmit array response applies **w** in closed form. CBF and MVDR spectra, the complex CIR,
and beam readouts all follow from that one channel.

## What is in this repository

| Path | Contents |
|---|---|
| `code/rfgs/csi/` | main model: training entry + analytic `render_cir`/CBF/MVDR renderer |
| `code/rfgs/core/` | data loading (`angular_data`), metrics, point-cloud initialisation |
| `code/rfgs/xscene/` | six-room cross-scene study (material-conditioned head) |
| `code/rfgs/evaluation/` | CBF / MVDR evaluation entry |
| `code/sim/`, `code/splat/` | Sionna spectrum generator + 3DGS kernels |
| `pipeline.py` | **one-command train → evaluate → figures** |
| `datasets_meta/` | instance tables, splits and protocol metadata (small, tracked) |
| `scripts/package_datasets.sh` | package spectra into release-sized checksummed archives |
| `results/` | precomputed headline results (see below) |
| `docs/DATASETS.md`, `docs/REPRODUCE.md` | data contract and reproduction protocol |

## Installation

```bash
conda create -n wrfgs python=3.10 -y && conda activate wrfgs
pip install -r requirements.txt
```

`torch` with CUDA is required. The Gaussian *rasteriser* (diff-gaussian-rasterization /
simple-knn) is only needed for the per-scene SH/CORF baselines; the main pipeline uses the
analytic renderer.

## Data

The spectra are several GB and are distributed as checksummed archives (GitHub Releases /
cloud drive), not in git:

- `wrfgs_cbf_params` — single-transmitter benchmark, CBF
- `wrfgs_mvdr_w_params` — beamforming-weight benchmark, MVDR (the main table)
- `wrfgs_mvdr_params`, `wrfgs_cbf_w_pairs` — operator/weight ablations
- `wrfgs_diffuse_subset` — 60-instance diffuse subset (CBF images + complex CIR)
- `wrfgs_cir_train` — complex CIR labels for the training instances

Unpack under `datasets/` following [`docs/DATASETS.md`](docs/DATASETS.md). Instance tables are in
`datasets_meta/`.

## One-command pipeline

```bash
python pipeline.py --stage all --out results/main_cross_tx
```

Stages: `train` (100k steps, image + MVDR + CIR supervision) → `eval` (CBF and MVDR on six
held-out axes, 64 views per instance) → `figures` (per-axis bars and a qualitative grid).
Useful flags: `--per-axis 6`, `--views 64`, `--checkpoint <ckpt>` to skip training,
`--qual-ids 144 196 232 268 318` for the qualitative grid. Every output lands in `--out`:
`metrics.json`, `logs/`, `figures/`.

## Headline results

Per-axis CBF over all held-out instances (52/36/36/36/13/36), 64 views each; LPIPS is the primary metric.
Full numbers and figures are in [`results/main_cross_tx/`](results/main_cross_tx/).

| Axis | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Position | 14.53 | 0.743 | 0.338 |
| Yaw | 15.06 | 0.749 | 0.325 |
| Pitch | 15.56 | 0.751 | 0.313 |
| Carrier | 15.11 | 0.749 | 0.327 |
| Joint position+carrier | 14.47 | 0.743 | 0.337 |
| Beamforming weights | 15.22 | 0.754 | 0.321 |
| **Pooled (210 instances)** | **15.01** | **0.743** | **0.327** |

Pooled MVDR from the same predicted channel: 15.18 / 0.763 / 0.299. Three-seed standard deviation
of the 100k recipe: 0.11 dB PSNR / 0.0010 SSIM / 0.0011 LPIPS.

### Beyond the main protocol

- **Diffuse channels** ([`results/diffuse/`](results/diffuse/)): per-scene SH/CORF fits trained on
  48 views fall below the per-instance constant-image floor (10.97 / 10.05 vs 11.22 dB), while the
  zero-shot field reaches 11.87 dB and a 5k fine-tune 12.23 ± 0.14 dB. The field is trained only on
  the specular benchmark; the advantage is on the same location-disjoint protocol used by the
  per-scene baselines.
- **Cross-scene** ([`results/cross_scene/`](results/cross_scene/)): leave-one-room-out over six
  synthetic rooms gives 17.50 ± 2.02 dB, beating per-room scratch trained on 4/8/16 views
  (15.63 / 16.57 / 16.94 dB) in every fold.
- **System level** ([`results/system/`](results/system/)): 52.5 ms per view (640 views in 33.6 s),
  7.8 GB peak, zero-measurement beam selection at 67.6 % of the oracle spectral efficiency
  (uniform 46.1 %, random 37.2 %, exhaustive 100 % at 58 scans).

## Citation and license

Citation information will be added with the paper. Code is released for non-commercial research
use; see [LICENSE](LICENSE). The NIST lobby scene and the original RF-3DGS dataset follow their
own licenses.
