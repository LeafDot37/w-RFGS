---
license: other
language:
  - en
tags:
  - radio-radiance-field
  - 3d-gaussian-splatting
  - beamforming
  - channel-state-information
  - wireless
---

# w-RFGS — radio spatial-spectrum datasets

These spectra were generated with **Sionna RT** on the **NIST lobby** digital twin for the
w-RFGS benchmark: a **4×4 transmitter array** (TR 38.901 elements, V polarization) and a
**10×10 receiver array**, carrier 2.1–3.6 GHz (unless stated), one reflection plus LOS.

Each image is a **300×200, 8-bit grayscale** angular spectrum of the normalized
beamforming power (per-image min–max). The colour map is **not** baked in: the renderer
applies the fixed 256-entry `jet` LUT at evaluation time. Do not compare a grayscale file
against a jet RGB render — they differ by 2–3 dB.

## Archives

| Archive | Contents | Instances × views | Images |
|---|---|---|---|
| `wrfgs_dataset.tar.gz` | base RF-3DGS release: optical images, trained optical 3D Gaussians (`chkpnt30000.pth`), 3,200 camera poses, released per-scene RRF | — | — |
| `wrfgs_mvdr_w_params.tar.gz` | **beamforming-weight benchmark, regularized MVDR** | 354 × 640 | 226,560 |
| `wrfgs_cbf_w_params.tar.gz` | the **same instances under conventional beamforming (CBF)** | 354 × 640 | 226,560 |
| `wrfgs_diffuse_subset.tar.gz` | **diffuse channel** (diffuse + diffraction), 64 views per instance, plus complex CIR per instance | 60 × 64 | 3,840 |
| `wrfgs__cir_train.tar.gz` | complex CIR labels for the **144 training instances** | 144 × 640 views | 144 npz |
| `wrfgs__cir_axis.tar.gz` | complex CIR labels for the **174 axis-holdout instances** | 174 × 640 views | 174 npz |
| `wrfgs__cir_w.tar.gz` | complex CIR labels for the **36 weight-holdout instances** | 36 × 640 views | 36 npz |

All archives are split into parts of ≤1.9 GB. Checksums: `SHA256SUMS.txt`; per-archive
counts and byte sizes: `MANIFEST.txt`.

Restore:

```bash
cat wrfgs_mvdr_w_params.tar.gz.part-* | tar xzf - -C datasets/
```

`wrfgs__cir_train` restores as `datasets/_cir_train`.

## Layout after restoring

```
datasets/
├── RF-3DGS_dataset/                 # base release (optics + poses + optical Gaussians)
├── RF-3DGS_mvdr_w_params/
│   ├── meta.json / instances.json   # instance table and protocol
│   └── spectra/inst000 … inst353/   # 640 PNG per instance
├── RF-3DGS_cbf_w_params/
│   └── spectra/inst000 … inst353/
├── RF-3DGS_diffuse_subset/
│   └── inst000 … inst059/           # 64 PNG + cir.npz each
├── _cir_train/                      # 144 training instances
│   └── inst000.npz … inst143.npz    # keys v<view>_a, v<view>_tau, v<view>_valid
├── _cir_axis/                       # 174 axis-holdout instances
└── _cir_w/                          # 36 weight-holdout instances
```

## Instance table

`instances.json` carries, per instance: `id`, `xyz` (transmitter position, m), `yaw_abs_rad`,
`pitch_abs_rad`, `freq_hz`, `w_label` (beamforming weights), `role`, `views`, `label`.

| Role | Count | Meaning |
|---|---|---|
| `train` | 144 | training configurations |
| `pos_holdout` | 52 | unseen positions |
| `yaw_holdout` / `pitch_holdout` / `freq_holdout` | 36 each | unseen orientation / carrier |
| `double_holdout` | 13 | unseen position + carrier |
| `w_holdout` | 36 | unseen beamforming weights |
| `orig` | 1 | original RF-3DGS transmitter |

The CBF and MVDR archives share the same 354 instances and the same 640 views, so the two
operators can be compared instance by instance.

## Complex CIR

- CIR labels for all 354 instances: `_cir_train` (144), `_cir_axis` (174), `_cir_w` (36); the three files use the same keys.
- Training labels (`_cir_train/instXXX.npz`): `v<view>_a` (complex amplitudes),
  `v<view>_tau` (ns), `v<view>_valid` (bool), plus `views`.
- Diffuse subset (`instXXX/cir.npz`): `v<view>_h`, a complex `[100, T]` channel on a 0.5 ns
  grid (`T` stored in the file), plus `views`.

## License and citation

Code is released for non-commercial research use. The NIST lobby scene and the original
RF-3DGS dataset follow their own licenses. If you use these spectra, cite RF-3DGS and the
w-RFGS paper (see the GitHub repository `LeafDot37/w-RFGS`).
