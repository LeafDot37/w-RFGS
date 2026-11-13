# w-RFGS: Feed-Forward Radio Radiance Fields with Transmitter and Beamforming-Weight Conditioning

w-RFGS renders radio-frequency angular spectra (AoA heat maps) from an **optical** 3D Gaussian
representation, for transmitters whose **position, orientation, carrier frequency and — for an
array transmitter — beamforming weights w** were never seen during training.

The pipeline is a single feed-forward pass:

```
optical images ──► frozen MVSplat encoder ──► optical Gaussians (1,014,142, frozen)
                                                   │
                        transmitter parameters ────┤  per-Gaussian embedding + positional encoding
                        (position, yaw, pitch, f)  │  + transmitter block / array-weight block
                                                   ▼
                                          scalar head MLP (128×2)  ──►  u per Gaussian
                                                   ▼
                                     frozen 3DGS alpha blending  ──►  u per pixel
                                                   ▼
                                        fixed jet LUT (256)  ──►  RGB spectrum
```

Unlike per-scene radio radiance fields, nothing is optimised at test time: the model is trained once
on 144 transmitter instances and renders new instances with one forward pass.

---

## Results

All numbers are PSNR / SSIM / LPIPS on the 640 test views shared by every method, in the jet
delivery domain. LPIPS is the perceptual score of record, but all three metrics are reported
everywhere.

### Unseen transmitter: position, orientation and carrier all outside the training set

| Method | Optimisation | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| Constant image (floor) | — | 14.293 | 0.466 | 0.433 |
| RF-3DGS (legacy protocol) | per-scene | 14.132 | 0.723 | 0.446 |
| CORF-GS (legacy protocol) | per-scene | 14.633 | 0.755 | 0.403 |
| Prior feed-forward (legacy protocol) | amortised | 11.847 | 0.738 | 0.440 |
| **w-RFGS, 60 k steps** | **amortised** | 15.314 | 0.766 | 0.403 |
| **w-RFGS, 180 k steps** | **amortised** | **15.382** | **0.764** | 0.404 |

Against the published per-scene RF-3DGS, w-RFGS gains 1.25 dB PSNR, 0.041 SSIM and 0.042 LPIPS
while performing **no per-transmitter optimisation**. Against CORF-GS, which couples optical and
radio appearance, it gains 0.75 dB PSNR and 0.008 SSIM; CORF-GS is marginally better on LPIPS
(0.4029 vs 0.4040).

The reference rows were measured under the previous-generation protocol (isotropic transmitter).
Replacing the isotropic radiator with a TR 38.901 element perturbs the target image by 43.0 dB PSNR
(1.8 grey levels out of 255) against method errors of 50–65 grey levels, which moves a reading by
0.005 dB in quadrature. We therefore treat these rows as comparable to within ±0.1 dB and do not
interpret differences below that.

The constant-image floor is the strongest predictor that uses no transmitter information at all;
against it the PSNR margin is 1.09 dB, while SSIM rises 0.466 → 0.764 and LPIPS falls 0.433 →
0.404.

### Held-out positions (52 instances = 13 positions × 4 parameter draws)

| Metric | 60 k | 180 k | Δ | SE | higher |
|---|---|---|---|---|---|
| PSNR | 15.752 | 15.879 | +0.127 | 0.021 | 44/52 |
| SSIM | 0.7957 | 0.7953 | −0.00047 | 0.00019 | 19/52 |
| LPIPS | 0.3682 | 0.3683 | +0.00015 | 0.00026 | 22/52 |

The longer schedule buys PSNR reliably (t = 6.0) but slightly costs SSIM and leaves LPIPS
unchanged; we report this rather than only the PSNR gain.

### Beamforming weights w (36 held-out weight vectors, 640 views each)

Conditioning on the precoder of a 4×4 transmit array under the regularized MVDR operator. Every arm
shares the frozen geometry, schedule and hyperparameters.

| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Complex w, λ_p = 0 | 11.557 | 0.7020 | 0.4627 |
| Complex w, + LPIPS loss | 11.712 | 0.6871 | 0.4621 |
| **w-blind (control)** | **12.209** | **0.6894** | **0.4455** |
| *Floors computed from ground truth only* | | | |
| Same-position training-w prior | 13.534 | 0.6498 | 0.2902 |
| Sibling instance (other w) | 12.460 | 0.6946 | 0.2877 |
| Same-position median | 12.723 | 0.6943 | 0.2959 |

The w channel is **alive but weakly exploited**: swapping w moves ground truth by 74.50 grey levels,
whereas the trained model moves by 2.02 (λ_p = 0) and 3.10 (+LPIPS) — 2.7 % and 4.2 % of the
ground-truth response; the blind control moves by exactly 0. On the aggregate metrics the blind
control still wins by 0.50 dB / 0.0166 LPIPS. A ceiling analysis puts a perfect w channel at
+3.06 dB PSNR over the blind arm, so the headroom is real but bounded.

**This axis has no direct literature baseline.** No published radio radiance field conditions on
beamforming weights, so the w table compares only against w-blind controls and ground-truth-derived
floors. It is *not* comparable to the single-transmitter table above (4×4 vs 1×1 transmit array,
MVDR vs CBF operator, different instance set).

---

## Repository layout

```
w-RFGS/
├── code/
│   ├── train_rrf_angular.py      # main training entry (transmitter-conditioned scalar head)
│   ├── render_rrf_angular.py     # rendering / ground-truth export / evaluation entry
│   ├── angular_data.py           # dataset, instance tables, camera loading
│   ├── splat/                    # 3DGS engine + scalar heads (color_scalar*.py)
│   ├── feedforward/              # MVSplat geometry encoder (frozen)
│   ├── sim/                      # Sionna RT dataset construction
│   └── tools/                    # instance tables, aggregation, TTA, figures
├── datasets_meta/                # instance tables and metadata (JSON only, no images)
├── docs/
│   ├── DATASETS.md               # how to obtain / regenerate the datasets
│   ├── REPRODUCE.md              # exact commands behind every table above
│   └── README_mvsplat.md         # upstream MVSplat README (archived)
├── paper/                        # LaTeX source and figures
├── scripts/package_datasets.sh   # build the downloadable dataset archives + checksums
├── config.yaml                   # site paths (edit `root:` before first run)
└── requirements.txt
```

`datasets/`, `results/`, `checkpoints/` and `final/` are git-ignored; see
[docs/DATASETS.md](docs/DATASETS.md).

---

## Installation

w-RFGS inherits its engine from [MVSplat](https://github.com/donydchen/mvsplat) and
[RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS). A CUDA 11.8 / 12.x toolchain and Python ≥ 3.9
are assumed.

```bash
git clone https://github.com/LeafDot37/w-RFGS.git && cd w-RFGS

# 1) Python dependencies
pip install -r requirements.txt

# 2) The Gaussian rasteriser (MVSplat fork)
pip install "git+https://github.com/donydchen/mvsplat.git#subdirectory=submodules/diff-gaussian-rasterization"

# 3) Point the pipeline at your checkout
sed -i 's|root: /path/to/w-RFGS|root: '"$PWD"'|' config.yaml
```

Data generation additionally needs
[Sionna 0.19+](https://nvlabs.github.io/sionna/); training and evaluation do not.

---

## Data

The image archives are **not** in this repository (several GB). Two routes:

1. **Download** the pre-generated archives — links, SHA-256 checksums and the archive layout are in
   [docs/DATASETS.md](docs/DATASETS.md).
2. **Regenerate** from the NIST lobby digital twin with Sionna RT using the scripts in `code/sim/`,
   following the protocol below (≈ 5 h single process on a desktop CPU).

| Quantity | Value |
|---|---|
| Scene | NIST lobby digital twin (≈ 14 × 15 m) |
| Optical | Blender renders, 3,200 posed views (2,560 train / 640 test) |
| Image | 300 × 200, 90° FoV, 8-bit, per-image min–max |
| Receiver | 10 × 10 planar array, TR 38.901 pattern |
| Transmitter | 1 × 1 TR 38.901 element (4 × 4 for the w dataset), vertical polarisation |
| Propagation | Sionna RT, max depth 1, specular only |
| Materials | tutorial values at 60 GHz, frequency-update callback frozen |
| Carrier | 2.10 / 2.60 / 3.10 / 3.60 GHz (seen); 2.40 GHz main reading |
| Transmitter positions | 36 train / 13 held out |
| Instances / images | 36×4 train + 13×4 held out + 122 probes → 572,160 images |

Two protocol details are load-bearing and easy to get wrong:

* **The transmitter must not be isotropic.** With an isotropic radiator, rotating the transmitter
  leaves the image unchanged and the orientation axis is vacuous. We use a TR 38.901 element
  (65° azimuth beamwidth).
* **Freeze the material frequency callback.** Sionna's material objects carry a
  frequency-update callback that silently restores the ITU conductivity model when the scene
  frequency is set. Left in place, changing the carrier would change *both* propagation and
  material, and the frequency axis would not be attributable.

---

## Training and evaluation

Training is one feed-forward model on the 144 training instances:

```bash
cd code
python train_rrf_angular.py \
    -s ../datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
    -m ../results/run_wrfgs \
    --spectra_root ../datasets/RF-3DGS_cbf_params \
    --meta        ../datasets/RF-3DGS_cbf_params/meta.json \
    --start_checkpoint ../datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
    --iterations 180000 --views_per_step 2 \
    --lr_schedule cosine --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 180000 \
    --init_u 0.05 --embed_lr 2.5e-3 \
    --use_tx_orient 1 --use_tx_local_dir 1 --use_tx_freq 1 \
    --train_roles train --train_views train --train_view_random 640 --train_view_seed 0 \
    --save_iterations 20000 40000 60000 120000 180000 --log_every 500
```

Evaluation of one instance (ground-truth export + render + three metrics):

```bash
python render_rrf_angular.py -s <optical-data> \
    --spectra_root ../datasets/RF-3DGS_cbf_params --meta ../datasets/RF-3DGS_cbf_params/meta.json \
    -m ../results/run_wrfgs --checkpoint ../results/run_wrfgs/angular_180000.pth \
    --instance 317 --start_checkpoint <seed> --limit 640 --export_gt --out_root ../results/gt317
python render_rrf_angular.py ... --out_root ../results/render317        # same call, no --export_gt
python tools/tta_eval.py --renders ../results/render317 --gt ../results/gt317 \
    --poses_root <optical-data> --skip_flat_gt
```

`--skip_flat_gt` is mandatory: 0.2 % of the test views have an all-zero ground-truth field and
would otherwise dominate the average with a degenerate jet(0) image.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the per-table commands, the w-axis arms, and the
test-time-adaptation procedure.

---

## Citation

```bibtex
@misc{wrfgs2026,
  title  = {w-RFGS: Feed-Forward Radio Radiance Fields with Transmitter and
            Beamforming-Weight Conditioning},
  author = {LeafDot37},
  year   = {2026},
  note   = {Preprint}
}
```

Please also cite the two upstream systems this work builds on:

```bibtex
@inproceedings{chen2024mvsplat,
  title     = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
  author    = {Chen, Donny and others},
  booktitle = {ECCV}, year = {2024}
}
@article{rf3dgs2024,
  title   = {RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and
             Gaussian Splatting},
  author  = {SunLab-UGA and others},
  journal = {arXiv:2411.19420}, year = {2024}
}
```

## License

MIT for the code in this repository; see [LICENSE](LICENSE). `code/feedforward/` and
`code/splat/` are derived from MVSplat (MIT) and RF-3DGS, and keep their respective upstream terms.
The dataset archives are released for research use; the NIST lobby digital twin itself is
distributed by its original authors.

## Acknowledgements

This project stands on [MVSplat](https://github.com/donydchen/mvsplat) (feed-forward geometry),
[RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS) (radio radiance field and the base dataset) and
[Sionna RT](https://nvlabs.github.io/sionna/) (propagation solver).
