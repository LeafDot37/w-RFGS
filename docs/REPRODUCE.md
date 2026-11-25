# Reproducing the results

Every number comes from per-instance metric tables produced by the pipeline; nothing is
hand-transcribed. Datasets are unpacked under `datasets/` as described in
[DATASETS.md](DATASETS.md); instance tables live in `datasets_meta/`.

## 0. Metric protocol (read before comparing)

| Rule | Why |
|---|---|
| **Same domain on both sides.** Ground truth is the stored 8-bit scalar `u`; the renderer's `u` is min–max normalized per image, and both are mapped through the same 256-entry jet LUT before PSNR/SSIM/LPIPS. | Two encodings of the same field exist; mixing them costs 2–3 dB. |
| **Per-instance tables, then aggregate.** Report per-instance means and their standard deviation; LPIPS is the primary metric. | Per-image PSNR can saturate on near-identical images. |
| **All three metrics, always.** | PSNR, SSIM and LPIPS disagree at the 0.1 dB scale. |
| **Do not compare across protocols.** | The main table uses the 4×4 / 10×10 / 2–4 GHz benchmark; the original RF-3DGS 60 GHz protocol is a separate compatibility check. |

## 1. Main result (six held-out axes)

```bash
python pipeline.py --stage all --out results/main_cross_tx
```

`train` runs the paper recipe (100k steps, image + MVDR + CIR supervision, 144 training
configurations). To reuse a released checkpoint, skip training:

```bash
python pipeline.py --stage eval --checkpoint <csi_100000.pth> --out results/main_cross_tx
python pipeline.py --stage figures --checkpoint <csi_100000.pth> --out results/main_cross_tx
```

Outputs: `metrics.json` (per axis and pooled, CBF and MVDR), `logs/`, `figures/`
(`bars_psnr.pdf`, `bars_lpips.pdf`, and `qual_grid.pdf` when `--qual-ids` is given).

The precomputed headline numbers and the paper's qualitative grid are in
`results/main_cross_tx/`.

## 2. Beyond the main protocol

- **Diffuse channels** — see `results/diffuse/metrics.json` and the protocol note in
  `results/diffuse/README.md`. The study reuses the main field without retraining; per-scene
  SH/CORF baselines are trained on 48 views of the target instance.
- **Cross-scene** — six synthetic rooms, leave-one-room-out. Entry points:
  `code/rfgs/xscene/train_xscene.py` and `code/rfgs/xscene/eval_xscene.py`
  (see `results/cross_scene/metrics.json`).
- **System level** — `code/tools/eval/bench_system.py`, `eval_beam_mgmt.py`,
  `eval_tx_sensitivity.py` (see `results/system/metrics.json`).

## 3. Release checks

- Three-seed standard deviation of the 100k recipe: 0.11 dB PSNR, 0.0010 SSIM, 0.0011 LPIPS
  (`results/main_cross_tx/metrics.json` stores the pooled means).
- Data archives: `bash scripts/package_datasets.sh --root datasets --out dist --split-mb 1900`
  produces split parts plus `SHA256SUMS.txt` and `MANIFEST.txt`.
