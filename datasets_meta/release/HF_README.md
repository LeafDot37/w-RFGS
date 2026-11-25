---
license: other
tags:
  - radio-radiance-field
  - 3d-gaussian-splatting
  - beamforming
  - channel-state-information
---

# w-RFGS — radio spatial-spectrum datasets

Generated with Sionna RT on the NIST lobby digital twin for the w-RFGS benchmark
(4×4 transmitter, 10×10 receiver, 2.1–3.6 GHz unless noted). Images are 300×200
8-bit grayscale scalar fields; the jet colour map is applied at render time.

| Archive | Contents | Views |
|---|---|---|
| `wrfgs_dataset.tar.gz` | base RF-3DGS release: optical dataset, trained optical Gaussians, 3,200 camera poses, released per-scene RRF | — |
| `wrfgs_mvdr_w_params.tar.gz` | beamforming-weight benchmark, regularized MVDR, 354 instances × 640 views | 226,560 |
| `wrfgs_cbf_w_params.tar.gz` | the same instances under conventional beamforming (CBF) | 226,560 |
| `wrfgs_diffuse_subset.tar.gz` | 60-instance diffuse channel (diffuse + diffraction), 64 views each, plus complex CIR (`cir.npz`, key `v<view>_h`) | 3,840 |
| `wrfgs__cir_train.tar.gz` | complex CIR labels for the 144 training instances | 144 files |

Each archive is split into ≤1.9 GB parts (`cat …part-* | tar xzf -`). Checksums are in
`SHA256SUMS.txt`, per-archive counts in `MANIFEST.txt`.

## Restore

```bash
cat wrfgs_mvdr_w_params.tar.gz.part-* | tar xzf - -C datasets/
```

`wrfgs__cir_train` restores as `datasets/_cir_train`.

## Licence

Code is released for non-commercial research use. The NIST lobby scene and the
original RF-3DGS dataset follow their own licences; cite RF-3DGS and the w-RFGS
paper when using these spectra.
