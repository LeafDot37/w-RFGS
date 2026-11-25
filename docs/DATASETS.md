# Datasets

The spectra used in this project are 8-bit grayscale AoA images produced by Sionna RT on the
NIST lobby digital twin. They are large (several GB), so they are **not** part of the git
repository. This document covers three things:

1. [what the code expects on disk](#1-directory-contract),
2. [how to download the pre-generated archives](#2-download),
3. [how to regenerate them from scratch](#3-regeneration).

Instance tables and all dataset metadata (the part that defines *what* was rendered) **are** in
the repository, under [`datasets_meta/`](../datasets_meta). Copy them next to the images when you
unpack, or use them directly with `--instance_file`.

---

## 1. Directory contract

The code looks for datasets under `datasets/` (git-ignored). After unpacking you should have:

```
datasets/
├── RF-3DGS_dataset/                 # base data from RF-3DGS (optics + author spectra)  ~4.4 GB
│   ├── blender_visual_dataset/      #   optical images, 809 unique viewpoints
│   ├── blender_visual_trained/      #   trained optical 3DGS, chkpnt30000.pth (seed for the RF head)
│   ├── training-rf-spectrum/
│   │   └── 3dgs_CBF_100/            #   ← camera poses (3,200 views) used by every generator
│   └── RF-3DGS_trained_RRF/         #   released per-scene radio radiance fields
├── NIST_lobby_spatial_full/         # scene geometry + transmitter positions
│   ├── cloud.ply                    #   optical point cloud (1,014,142 Gaussians)
│   ├── query_xyz.npy                #   query points for the analytic field
│   ├── meta.json                    #   49 transmitter world coordinates + splits
│   └── rx_candidates.json           #   receiver pose candidates
├── RF-3DGS_cbf_params/              # ★ single-transmitter dataset (main tables)
│   ├── meta.json                    #   protocol + 318 instances
│   ├── instances.json               #   instance table (position, yaw, pitch, freq, role)
│   └── spectra/tx000 … tx317/       #   640 or 3,200 PNGs per instance
├── RF-3DGS_mvdr_params/             # same scenario, regularized MVDR operator
├── RF-3DGS_mvdr_w_params/           # ★ beamforming-weight dataset (w table)
│   ├── meta.json / instances.json
│   └── spectra/inst000 … inst353/
├── RF-3DGS_diffuse_subset/          # diffuse study (60 instances x 64 views)
│   ├── inst000 … inst059/           #   8-bit CBF images
│   └── inst000/cir.npz …            #   complex CIR labels (v<view>_h)
└── _cir_train/                      # complex CIR labels for the 144 training instances
    └── inst000.npz … inst143.npz
```

`RF-3DGS_dataset` is the upstream release of [RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS)
(download it from that repository). Everything else is produced by this project.

### What each instance looks like

| Field | `RF-3DGS_cbf_params` | `RF-3DGS_mvdr_w_params` |
|---|---|---|
| Instance ids | `tx000`–`tx317` | `inst000`–`inst353` |
| Rendered views | 3,200 (train role) / 640 (everything else) | 640 (all) |
| Total images | 572,160 | 226,560 |
| Archive size | ≈ 7.7 GB | ≈ 2.7 GB |
| Operator | CBF (conventional beamforming) | regularized MVDR |
| Varied | position × yaw × pitch × carrier | beamforming weights w (+ inherited parameters) |
| Roles | `train` (144), `pos_holdout` (52), `yaw/pitch/freq_holdout` (36 each), `double_holdout` (13) | `train` (144), `w_holdout` (36, ids 318–353) |

Each PNG is a `300 × 200` 8-bit image of the normalized AoA intensity, already min–max scaled
per image, in grayscale. The jet colour map is applied **at render/evaluation time** by the
fixed LUT in [`code/splat/color_scalar.py`](../code/splat/color_scalar.py); it is not baked into
the files. Do not mix the two domains when comparing numbers.

---

## 2. Download

> **Status.** The archives are uploaded as release assets split into ≈1.9 GB parts (GitHub's
> per-asset limit). The link below is the only place the data is distributed from; if it does not
> resolve, regenerate the data instead (§3) — the generation path is fully scripted and does not
> depend on any link.

Primary source (Hugging Face):

```
https://huggingface.co/datasets/LeafDot37/w-RFGS
```

| Archive | Contents | Size |
|---|---|---|
| `wrfgs_dataset` | base RF-3DGS release: optical images, trained optical Gaussians, 3,200 poses, released RRF | 583 MB |
| `wrfgs_mvdr_w_params` | beamforming-weight benchmark, regularized MVDR, 354 instances × 640 views | 2.2 GB |
| `wrfgs_cbf_w_params` | the same 354 instances under conventional beamforming | 2.5 GB |
| `wrfgs_diffuse_subset` | 60-instance diffuse channel, 64 views + complex CIR per instance | 463 MB |
| `wrfgs__cir_train` | complex CIR labels for the 144 training instances | 607 MB |

Each archive is split into parts of ≤1.9 GB. `SHA256SUMS.txt` and `MANIFEST.txt` are in the
Hugging Face repository and also tracked in `datasets_meta/release/`.

Download every part of an archive into a single directory, then:

```bash
cd ~/downloads

# 1) integrity: every part must match
sha256sum -c SHA256SUMS.txt

# 2) reassemble and unpack (parts are plain byte splits, `cat` order matters)
cat wrfgs_cbf_params.tar.zst.part-* | zstd -d | tar -xf - -C /path/to/w-RFGS/datasets
cat wrfgs_mvdr_w_params.tar.zst.part-* | zstd -d | tar -xf - -C /path/to/w-RFGS/datasets

# 3) sanity check: instance count and image count must match the table above
ls /path/to/w-RFGS/datasets/RF-3DGS_cbf_params/spectra | wc -l          # 318
find /path/to/w-RFGS/datasets/RF-3DGS_cbf_params/spectra -name '*.png' | wc -l   # 572160
```

If you do not have `zstd`, use `gzip` (the packaging script supports `--codec gzip`) and replace
`zstd -d` with `gzip -d`.

### Building your own archives

```bash
# from the repository root, with the datasets already in datasets/
bash scripts/package_datasets.sh --root datasets --out /data/release --split-mb 1900
#   → /data/release/wrfgs_cbf_params.tar.zst.part-*  +  SHA256SUMS.txt
```

Upload the parts and `SHA256SUMS.txt` as assets of a release tagged `datasets-v1`, then replace
`LeafDot37` above if the repository ever moves.

---

## 3. Regeneration

Everything is reproducible from the NIST lobby digital twin plus the RF-3DGS base data. The
scripts are in [`code/sim/`](../code/sim) and [`code/tools/`](../code/tools). Requires
[Sionna](https://nvlabs.github.io/sionna/) ≥ 0.19 (`pip install sionna`); the propagation solver
runs on CPU or GPU. Budget ≈ 5 h single-process for 572 k images, or shard by `--ids`.

```bash
cd /path/to/w-RFGS

# 0) Unpack the NIST lobby digital twin (from the Sionna RT tutorial assets, or the
#    RF-3DGS release) into datasets/NIST_lobby_scene/, then re-assign materials:
python code/sim/prep_nist_scene.py \
    --src datasets/NIST_lobby_scene \
    --out datasets/NIST_lobby_spatial_full

# 1) Instance tables first (they are pure numpy, no Sionna, seconds to run):
#    a. 49 transmitter positions, 36 train / 13 held out, optical query points, receiver poses
python code/sim/sample_nist_rx.py --data_root datasets/NIST_lobby_spatial_full \
       --n 96 --out datasets/NIST_lobby_spatial_full/rx_candidates.json
#    b. 318 single-transmitter instances (position x yaw x pitch x carrier)
python code/tools/make_tx_instances.py \
       --out datasets/RF-3DGS_cbf_params --seed 0
#    c. 354 beamforming-weight instances (adds the w axis, keeps position/orientation/carrier)
python code/tools/make_w_instances.py \
       --base datasets/RF-3DGS_cbf_params/instances.json \
       --out  datasets/RF-3DGS_mvdr_w_params --seed 0

# 2) Render the spectra. Poses come from the RF-3DGS release; `--views auto` picks
#    3,200 views for train-role instances and 640 for the rest.
python code/sim/gen_cbf_params.py \
       --instances datasets/RF-3DGS_cbf_params/instances.json \
       --out       datasets/RF-3DGS_cbf_params \
       --poses_root datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
       --views auto --M 10 --tx_pattern tr38901 --tx_rows 1 --tx_cols 1 \
       --mat_freq 60e9 --max_depth 1 --width 300 --height 200 --skip_existing

python code/sim/gen_mvdr_params.py \
       --instances datasets/RF-3DGS_mvdr_w_params/instances.json \
       --out       datasets/RF-3DGS_mvdr_w_params \
       --poses_root datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
       --views auto --M 10 --tx_pattern tr38901 --tx_rows 4 --tx_cols 4 \
       --mat_freq 60e9 --max_depth 1 --width 300 --height 200 --skip_existing
```

Sharding: both generators accept `--ids a b c …` and `--skip_existing`, so N processes can walk
disjoint instance id sets and be resumed at any time. A single instance is the atomic unit.

Batched shards are convenient:

```bash
for IDS in "0 1 2 3 4" "5 6 7 8 9"; do
  python code/sim/gen_cbf_params.py --instances ... --out ... --ids $IDS --skip_existing &
done
```

### Verification after generation

```bash
python code/tools/check_param_data.py \
    --meta    datasets/RF-3DGS_cbf_params/meta.json \
    --spectra datasets/RF-3DGS_cbf_params/spectra
```

checks instance table ↔ image count (3,200 for `train`, 640 otherwise), image size, and the LUT
range of every file; it also reports the pairwise PSNR between two instances that share a
position but differ only in one parameter, which must be far below the re-render noise floor.

### Diffuse subset and CIR labels

The diffuse subset is generated from the same instance table with the diffuse/diffraction
channels enabled:

```bash
python code/sim/gen_cbf_params.py \
  --instances datasets_meta/mvdr_w_params/meta.json \
  --out datasets/RF-3DGS_diffuse_subset \
  --ids 0 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 126 132 138 \
        144 152 160 168 176 184 196 202 208 214 220 226 232 238 244 250 256 262 \
        268 274 280 286 292 298 304 306 308 310 312 314 318 319 320 321 322 323 \
  --views test --views_limit 64 --tx_rows 4 --tx_cols 4 --w_from_instance \
  --diffuse 1 --diffraction 1 --export_cir 1 --max_depth 1 --samples 1e6
```

`wrfgs_cir_train` contains the same `--export_cir` output for the 144 training instances and is
what `train_csi_cbf.py --cir_root` reads (one `cir.npz` per instance, key `v<view>_h`).

### Instance table and roles (w benchmark)

`datasets_meta/mvdr_w_params/instances.json` stores, per instance: `id`, `xyz` (transmitter
position, m), `yaw_abs_rad`, `pitch_abs_rad`, `freq_hz`, `w_label`, `role`, `views`, `label`.

| Role | Count | Meaning |
|---|---|---|
| `train` | 144 | training configurations |
| `pos_holdout` | 52 | unseen positions |
| `yaw_holdout` / `pitch_holdout` / `freq_holdout` | 36 each | unseen orientation / carrier |
| `double_holdout` | 13 | unseen position + carrier |
| `w_holdout` | 36 | unseen beamforming weights |
| `orig` | 1 | original RF-3DGS transmitter |

`wrfgs_cbf_w_params` and `wrfgs_mvdr_w_params` share the same 354 instances and 640 views, so
CBF and MVDR can be paired instance by instance.

### Complex CIR files

- `_cir_train/instXXX.npz`: `v<view>_a` (complex path amplitudes), `v<view>_tau` (ns),
  `v<view>_valid` (bool), and `views` (640 entries).
- `RF-3DGS_diffuse_subset/instXXX/cir.npz`: `v<view>_h`, a complex `[100, T]` channel on a
  0.5 ns grid, plus `views` (64 entries).

The diffuse subset contains 60 instances (the 24 training ids plus the 36 held-out ids used
by the diffuse study), each with 64 views — the first 64 author-test views.
