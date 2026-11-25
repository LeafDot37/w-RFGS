# Main result — cross-transmitter generalization

One model, 144 training transmitter configurations, evaluated on six held-out axes
(all 210 holdout instances, 64 views each; `metrics.json` aggregates every instance).

- `metrics.json` — per-axis and pooled CBF/MVDR PSNR, SSIM, LPIPS.
- `figures/fig_axes_qual.{pdf,png}` — qualitative grid: one instance per axis, two views,
  physical image / CBF GT / CBF ours / MVDR GT / MVDR ours.
- `figures/fig_w_main.{pdf,png}` — per-axis bars with instance spread.

Reproduce with `python pipeline.py --stage all --out results/main_cross_tx`
(after downloading the datasets and the optical seed checkpoint).
