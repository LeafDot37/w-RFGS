# Dataset release package (2026-11-25)

Archives were produced with `scripts/package_datasets.sh --codec gzip --split-mb 1900` from
the generated spectra. The manifest and per-part SHA-256 sums are in this directory.

| Archive | Contents | Parts | Size |
|---|---|---|---|
| `wrfgs_dataset` | base RF-3DGS release: optical dataset, trained optical Gaussians, poses, released RRF | 1 | 583 MB |
| `wrfgs_mvdr_w_params` | beamforming-weight benchmark (MVDR, 4×4 TX, 354 instances × 640 views) | 2 | 2.2 GB |
| `wrfgs_cbf_w_params` | same instances, CBF operator | 2 | 2.5 GB |
| `wrfgs_diffuse_subset` | 60-instance diffuse subset, 64 views + complex CIR per instance | 1 | 463 MB |
| `wrfgs__cir_train` | complex CIR labels for the 144 training instances | 1 | 607 MB |

Restore into the repository:

```bash
cat wrfgs_mvdr_w_params.tar.gz.part-* | tar xzf - -C datasets
```

Note: the double underscore in `wrfgs__cir_train` comes from the leading underscore of the source
directory; it restores as `datasets/_cir_train`.

The archives are prepared and checksummed; uploading them as release assets requires a GitHub
token or a cloud-drive target and is pending.
