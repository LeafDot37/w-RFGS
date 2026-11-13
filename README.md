# w-RFGS: Feed-Forward Radio Radiance Fields with Transmitter and Beamforming-Weight Conditioning

w-RFGS renders radio-frequency angular spectra (AoA heat maps) from **optical** 3D Gaussians, for
transmitters whose position, orientation, carrier frequency — and, for an array transmitter,
beamforming weights **w** — were never seen during training.

```
optical images ──► frozen MVSplat encoder ──► 1,014,142 optical Gaussians (frozen)
                                                    │
                     transmitter parameters ────────┤  per-Gaussian embedding + positional encoding
                     (p, yaw, pitch, f) and w       │  + transmitter / array-weight block
                                                    ▼
                                          scalar head (128×2 MLP) ──► u per Gaussian
                                                    ▼
                                    frozen 3DGS alpha blending ──► u per pixel ──► jet LUT ──► RGB
```

One training run, one forward pass at test time; no per-transmitter optimisation.

---

## Results

PSNR / SSIM / LPIPS on 300×200 jet images, always all three. `Δ` rows are paired per instance.
The four blocks use **different operators, array sizes and instance sets** — compare inside a
block only, never across blocks.

### 1. Single transmitter, CBF operator — unseen position + orientation + carrier

1×1 TR 38.901 transmitter, 10×10 receiver array, instance 317 (author TX48 slot), 640 test views.

| Method | Per-scene training | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| Constant-image floor | — | 14.293 | 0.466 | 0.433 |
| RF-3DGS (released) | yes | 14.132 | 0.723 | 0.446 |
| CORF-GS | yes | 14.633 | 0.755 | 0.403 |
| Prior feed-forward | no | 11.847 | 0.738 | 0.440 |
| **w-RFGS, 60 k steps** | **no** | 15.314 | 0.766 | 0.403 |
| **w-RFGS, 180 k steps** | **no** | **15.382** | **0.764** | 0.404 |

Positions held out (52 instances = 13 positions × 4 parameter draws): 15.752 / 0.7957 / 0.3682 at
60 k and 15.879 / 0.7953 / 0.3683 at 180 k; the longer schedule buys +0.127 dB PSNR (SE 0.021,
44/52 instances) and is neutral on SSIM/LPIPS.

### 2. Single transmitter, MVDR (Capon) operator

Same 318-instance parameter set, operator swapped to regularised MVDR (ε = 0.01, diffuse on).
The target itself is harder: the constant-image floor drops to 11.38 dB.

| Method | Data | PSNR↑ | SSIM↑ | LPIPS↓ | Net over own floor |
|---|---|---|---|---|---|
| Constant-image floor (ours) | our protocol | 11.381 | 0.411 | 0.506 | — |
| Constant-image floor (authors) | original data | 12.164 | 0.602 | 0.439 | — |
| RF-3DGS (released, MVDR) | original data | 16.019 | 0.731 | 0.391 | +3.85 dB |
| CORF (full coupling) | original data | 19.444 | 0.803 | 0.287 | +7.28 dB |
| **w-RFGS, 180 k steps** | **our protocol** | 14.209 | 0.718 | 0.436 | **+2.83 dB** |

Held-out positions (52 instances): 13.795 / 0.713 / 0.432.

The reference rows were measured on the authors' original MVDR data (isotropic array, their delay
grid), so their absolute values are not subtractable from ours; each row is therefore also given
as a margin over its **own** constant-image floor. By that measure CORF and RF-3DGS remain ahead
on this operator, and w-RFGS is the only one of the three that is amortised.

### 3. Test-time fine-tuning (MVDR delivery checkpoint)

Zero-shot means no view of the target transmitter is used at any point.

| Setting | Protocol | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| Zero-shot, delivery | 53 inst. × 640 views | 13.803 | 0.7133 | 0.4325 |
| + perceptual tail (still feed-forward) | 53 inst. × 640 views | 13.711 | 0.7194 | **0.3957** |
| Zero-shot, adaptation protocol | 16 inst., held-out 320 views | 13.913 | 0.7174 | 0.4269 |
| + 320 target views, 11 k steps | 16 inst., held-out 320 views | **16.438** | **0.7704** | **0.3440** |
| In-sample oracle (upper bound) | 1 inst., all 640 views | 19.209 | 0.8463 | 0.2247 |

The perceptual tail improves LPIPS on 53/53 instances without seeing the target. Adaptation gains
+2.48 dB / +0.049 SSIM / −0.083 LPIPS over its own zero-shot baseline with all three metrics
improving on 16/16 instances; adaptation sets must be sampled uniformly over position *and*
orientation (prefix or strided selection flips the sign of the gain). The oracle uses the same
head, geometry and rasteriser, so the remaining 0.3440 → 0.2247 gap is the cost of amortisation,
not missing capacity.

Adaptation also repairs the failure mode reported in §5 below. On instance 150's held-out 320 views the
zero-shot checkpoint covers 3.26 % of the warm end of the palette against 5.07 % in ground truth;
after 320 target views and 20 k steps it covers 3.87 %. View 1503 is the clearest single case:
zero warm pixels zero-shot against 7.13 % in ground truth, 7.72 % after adaptation
(`paper/figs/fig14_qual_tta.png`). The bright core is therefore a property of amortisation, not of
the head or the colour map — but it is not recoverable without target views.

### 4. Array transmitter, beamforming weights **w** — 4×4 array, 36 held-out **w**

Every arm shares frozen geometry, schedule and hyperparameters. There is **no published baseline**
for this axis; the controls are a w-blind arm and three ground-truth-only floors.

**CBF operator**

| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| w-conditioned (5-d pointing block) | 14.545 | 0.7905 | 0.3857 |
| w-blind control | **14.578** | **0.7924** | **0.3832** |
| Paired Δ (conditioned − blind) | −0.032 | −0.0019 | +0.0024 |
| Same-position training-w prior | 13.534 | 0.6498 | 0.2902 |
| Sibling instance (different w) | 12.460 | 0.6946 | 0.2877 |
| Same-position median | 12.723 | 0.6943 | 0.2959 |

**MVDR operator**

| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Complex w, λ_p = 0 | 11.557 | 0.7020 | 0.4627 |
| Complex w, + LPIPS loss | 11.712 | 0.6871 | 0.4621 |
| w-blind control (complex run) | **12.209** | 0.6894 | **0.4455** |
| 5-d pointing w, 180 k steps | 12.190 | 0.6903 | 0.4597 |
| 5-d w-blind, 180 k steps | 12.188 | **0.6971** | 0.4486 |
| Ground-truth-only floors | 12.460 – 13.534 | 0.650 – 0.695 | 0.288 – 0.296 |

The **w** channel is alive but weakly exploited. Swapping **w** moves ground truth by 74.50 grey
levels; the trained models move by 2.02 (λ_p = 0) and 3.10 (+LPIPS) — 2.7 % and 4.2 % of the
ground-truth response — and the blind control moves by exactly 0. On the aggregate metrics no
**w** arm beats its blind control beyond the 0.365 dB repeat noise floor. A probe on **training**
weights shows the CBF model expresses 0.1 % of the ground-truth variation it has already seen
(pairwise model PSNR 52.19 dB against 22.95 dB for ground truth), so the limitation is
representational — one real scalar per Gaussian cannot carry the coherent complex sum over the
16 array elements — not a generalisation failure. A ceiling analysis puts a perfect **w** channel
at +3.06 dB over the blind arm: real headroom, but bounded by the base fidelity.

### 5. Where the error lives: bright regions

Averaged over all 640 views the model beats the per-scene baseline by 1.09 dB. Split by how much
of the ground truth is bright (mean luminance > 0.6), that margin is not spread evenly:

| Ground-truth bright fraction | Views | w-RFGS | RF-3DGS | Δ |
|---|---|---|---|---|
| Q1 (0.000–0.025) | 160 | 19.22 | 16.31 | **+2.91** |
| Q2 (0.025–0.039) | 160 | 17.78 | 15.13 | +2.65 |
| Q3 (0.039–0.084) | 160 | 13.78 | 13.34 | +0.43 |
| Q4 (0.084–0.186) | 160 | 10.75 | 11.05 | **−0.30** |

Both columns are recomputed on our views (the released renders score 13.96 dB here; the 14.13 dB of
Table 1 is the authors' own protocol-level number). The advantage is bought on the dim half of the
views; on the brightest quartile it inverts.

The mechanism is visible in the palette. Counting warm pixels (R > 150, G > 60): the
single-transmitter heads reach 0.85 % of the image against 2.41 % in ground truth (CBF) and
2.07 % against 5.93 % (MVDR) — about a third of the true coverage, but present in 414 / 640 and
96 / 160 views. The w-conditioned heads never reach it at all: zero warm pixels in all 1280 rendered
views of the two w·CBF instances, and in all conditioned w·MVDR views, against 1.99 % and 0.375 % in
ground truth. The blind w·MVDR control errs the other way, putting 3.55 % of the image in the warm
end — its small PSNR edge comes with a *worse* magnitude distribution, not a better one.

This is not a global-gain problem: sweeping u′ = a·u + b over eight gains and five offsets on 50
views never beats the identity (15.33 dB at (1, 0); best non-identity 14.41 dB at (1.30, −0.05)).
The error is in where the energy is placed, not in a missing multiplier. See
`paper/figs/fig13_where.pdf`.

### 6. Qualitative figures

`paper/figs/fig4_qual.png` (CBF: GT / RF-3DGS / ours), `fig10_qual_mvdr.png` (MVDR, two protocols
side by side — rows 1–2 the authors', rows 3–4 ours, each scored against its own ground truth),
`fig11_qual_w_cbf.png` (w·CBF: GT / conditioned / blind), `fig12_qual_w_mvdr.png` (w·MVDR: GT /
complex λ_p=0 / complex+LPIPS / blind), `fig14_qual_tta.png` (TTA: GT / zero-shot / adapted).
Views are drawn from the top quartile of per-view PSNR (or, for TTA, from the held-out half), i.e.
best-case illustrations; the tables above are the unbiased readings.

---

## Repository layout

```
w-RFGS/
├── code/
│   ├── train_rrf_angular.py      # training entry (transmitter-conditioned scalar head)
│   ├── render_rrf_angular.py     # render / ground-truth export / evaluation
│   ├── angular_data.py           # dataset, instance tables, camera loading
│   ├── splat/                    # 3DGS engine + scalar heads (color_scalar*.py)
│   ├── feedforward/              # MVSplat geometry encoder (frozen)
│   ├── sim/                      # Sionna RT dataset construction
│   └── tools/                    # instance tables, aggregation, TTA, figures
├── datasets_meta/                # instance tables and metadata (JSON only, no images)
├── docs/                         # DATASETS.md, REPRODUCE.md
├── paper/                        # LaTeX source and figures
├── scripts/package_datasets.sh   # dataset archives + checksums
├── config.yaml                   # site paths (edit `root:` before first run)
└── requirements.txt
```

`datasets/`, `results/`, `checkpoints/` and `final/` are git-ignored; see
[docs/DATASETS.md](docs/DATASETS.md).

---

## Installation

w-RFGS inherits its engine from [MVSplat](https://github.com/donydchen/mvsplat) and
[RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS). CUDA 11.8 / 12.x and Python ≥ 3.9.

```bash
git clone https://github.com/LeafDot37/w-RFGS.git && cd w-RFGS
pip install -r requirements.txt
pip install "git+https://github.com/donydchen/mvsplat.git#subdirectory=submodules/diff-gaussian-rasterization"
sed -i 's|root: /path/to/w-RFGS|root: '"$PWD"'|' config.yaml
```

Data generation additionally needs [Sionna 0.19+](https://nvlabs.github.io/sionna/); training and
evaluation do not.

---

## Data

The image archives are **not** in this repository. Download the pre-generated archives (links and
SHA-256 checksums in [docs/DATASETS.md](docs/DATASETS.md)), or regenerate them from the NIST lobby
digital twin with `code/sim/` (≈ 5 h single process).

| Quantity | Value |
|---|---|
| Scene | NIST lobby digital twin (≈ 14 × 15 m) |
| Optical | Blender renders, 3,200 posed views (2,560 train / 640 test) |
| Image | 300 × 200, 90° FoV, 8-bit, per-image min–max |
| Receiver | 10 × 10 planar array, TR 38.901 pattern |
| Transmitter | 1 × 1 TR 38.901 element (4 × 4 for the w datasets), vertical polarisation |
| Propagation | Sionna RT, max depth 1, specular only |
| Carrier | 2.10 / 2.60 / 3.10 / 3.60 GHz (seen); 2.40 GHz main reading |
| Instances / images | 36×4 train + 13×4 held out + 122 probes → 572,160 images |

Two protocol details are load-bearing:

* **The transmitter must not be isotropic.** An isotropic radiator makes the orientation axis
  vacuous. We use a TR 38.901 element (65° azimuth beamwidth).
* **Freeze the material frequency callback.** Sionna restores the ITU conductivity model when the
  scene frequency is set; left in place, changing the carrier would change *both* propagation and
  material and the frequency axis would not be attributable.

---

## Training and evaluation

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

python render_rrf_angular.py -s <optical-data> \
    --spectra_root ../datasets/RF-3DGS_cbf_params --meta ../datasets/RF-3DGS_cbf_params/meta.json \
    -m ../results/run_wrfgs --checkpoint ../results/run_wrfgs/angular_180000.pth \
    --instance 317 --start_checkpoint <seed> --limit 640 --export_gt --out_root ../results/gt317
python render_rrf_angular.py ... --out_root ../results/render317        # same call, no --export_gt
python tools/tta_eval.py --renders ../results/render317 --gt ../results/gt317 \
    --poses_root <optical-data> --skip_flat_gt
```

`--skip_flat_gt` is mandatory: 0.2 % of the test views have an all-zero ground-truth field and
would otherwise dominate with a degenerate jet(0) image.

[docs/REPRODUCE.md](docs/REPRODUCE.md) has the per-table commands, the w-axis arms, and the
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

Please also cite the upstream systems:

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

MIT for the code in this repository; see [LICENSE](LICENSE). `code/feedforward/` and `code/splat/`
are derived from MVSplat (MIT) and RF-3DGS and keep their upstream terms. Dataset archives are
released for research use; the NIST lobby digital twin is distributed by its original authors.

## Acknowledgements

This project stands on [MVSplat](https://github.com/donydchen/mvsplat) (feed-forward geometry),
[RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS) (radio radiance field and base dataset) and
[Sionna RT](https://nvlabs.github.io/sionna/) (propagation solver).
