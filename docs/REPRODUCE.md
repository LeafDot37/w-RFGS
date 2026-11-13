# Reproducing the paper

Every number in the paper comes from a per-instance metric table produced by the scripts below.
Nothing is hand-transcribed: the LaTeX figures and tables read those files directly
(`paper/make_figs.py`, `paper/make_fig_w.py`).

All commands assume the repository root as the working directory and the datasets from
[docs/DATASETS.md](DATASETS.md) unpacked under `datasets/`. `PY` is your Python interpreter.

---

## 0. Metric protocol (read this before comparing anything)

| Rule | Why |
|---|---|
| **Jet domain only.** Ground truth is exported through the same 256-entry LUT the renderer uses (`--export_gt`). Never compare a grayscale `spectra/*.png` against a jet RGB render — they differ by 2–3 dB. | Two encodings of the same field exist in this project. |
| **`--skip_flat_gt` is mandatory.** | 0.2 % of test views have an all-zero ground-truth field, which becomes a fully flat `jet(0)` image. Any model beats it trivially and it dominates the mean. |
| **Per-instance tables, then aggregate.** Report mean ± population std over instances, plus paired Δ/SE when comparing two arms. | Per-image PSNR saturates at 99 dB on near-identical images and silently distorts pooled means. |
| **All three metrics, always.** | PSNR, SSIM and LPIPS disagree in this task at the 0.1 dB scale; LPIPS is the perceptual score of record. |
| **Do not compare across datasets.** | `RF-3DGS_cbf_params` is 300×200; the legacy RF-3DGS release has native 231×154/300×200 variants and a different transmitter. |

---

## 1. Table: main reading (unseen position, orientation and carrier)

Evaluated on instance `orig` (id 317): the author transmitter position, zero boresight offset,
2.40 GHz — none of the three values appears in training.

```bash
cd code
PY=python  # your interpreter

# train (this is the model behind the 180 k row; the 60 k row is the 60000 checkpoint)
$PY train_rrf_angular.py \
    -s ../datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
    -m ../results/run_wrfgs \
    --spectra_root ../datasets/RF-3DGS_cbf_params \
    --meta         ../datasets/RF-3DGS_cbf_params/meta.json \
    --start_checkpoint ../datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
    --iterations 180000 --views_per_step 2 \
    --lr_schedule cosine --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 180000 \
    --init_u 0.05 --embed_lr 2.5e-3 \
    --use_tx_orient 1 --use_tx_local_dir 1 --use_tx_freq 1 \
    --train_roles train --train_views train --train_view_random 640 --train_view_seed 0 \
    --save_iterations 60000 180000 --log_every 500

# evaluate the orig instance (ground truth + render + three metrics)
cd ..
ROOT=results/run_wrfgs STEP=180000 ROLES="orig" bash code/tools/eval_all_instances.sh
cat results/run_wrfgs/test_inst/metrics.txt
```

`eval_all_instances.sh` exports the jet ground truth once per role, renders, and calls
`tools/eval_instances.py --skip_missing_gt`; the last line of `tools/orig_vs_baseline.py`'s output
prints the main reading next to the released RF-3DGS numbers.

## 2. Table: held-out positions (52 instances)

Same checkpoint, different role — no retraining:

```bash
ROOT=results/run_wrfgs STEP=180000 ROLES="pos_holdout" bash code/tools/eval_all_instances.sh
```

Aggregate the per-instance table (and get the paired 60 k vs 180 k statistics):

```bash
$PY code/tools/agg_instances.py --a results/run_wrfgs_60k/test_inst/metrics.txt \
                                --b results/run_wrfgs/test_inst/metrics.txt \
                                --meta datasets/RF-3DGS_cbf_params/meta.json \
                                --out results/holdout_60k_vs_180k.txt
```

Only 13 distinct positions back the 52 instances (four parameter draws each). When testing a
trend, average the four draws first and treat the position as the unit of analysis — the paper
reports both.

## 3. Table: beamforming weights w (36 held-out weight vectors)

The w dataset uses the regularized MVDR operator and a 4×4 transmit array. All arms share the
frozen geometry, schedule and hyperparameters; the only difference is the input switch.

```bash
# two arms: w-conditioned (complex w) and w-blind control
cd /path/to/w-RFGS
ITERS=60000 SAVES="60000" ARMS="w blind" bash code/tools/run_w_arms.sh
# perceptual tail (the +LPIPS row): rerun arm "w" with LPW=1.2 (λ is a *threshold* under
# sign-Adam, not a proportion — see code/tools/run_w_arms.sh header)
ITERS=60000 SAVES="60000" ARMS="w" LPW=1.2 bash code/tools/run_w_arms.sh
```

Evaluation uses the w metadata, only the `w_holdout` role:

```bash
ROOT=results/_w_arms/w STEP=60000 \
SPC=$PWD/datasets/RF-3DGS_mvdr_w_params/spectra \
META=$PWD/datasets/RF-3DGS_mvdr_w_params/meta.json \
ROLES="w_holdout" bash code/tools/eval_all_instances.sh
```

The ground-truth floors in that table (same-position training-w prior, sibling instance,
same-position median) are computed from ground truth only, with
[`code/tools/eval_same_position_prior.py`](../code/tools/eval_same_position_prior.py) and the
sibling pairing in [`code/tools/make_w_pair_instances.py`](../code/tools/make_w_pair_instances.py).

The w-swap probe (74.50 vs 2.02 / 3.10 grey levels) renders the same instance twice, once with its
own weights and once with the base weights:

```bash
# own w: plain render of the w_holdout role
$PY code/render_rrf_angular.py ... --role w_holdout --out_root results/swap_own
# swapped w: every held-out instance is rendered with the w of its paired base instance
# (same position, orientation and carrier; only w changes)
$PY code/render_rrf_angular.py ... --role w_holdout --param_mode sibling_w \
    --out_root results/swap_swapped
$PY code/tools/eval_dir_pairs.py --a results/swap_own --b results/swap_swapped  # RMS, grey levels
```

⚠ Use `--role`, not `--instance`, for this probe. The single-instance branch silently falls back
to the instance's own w, which turns the swap control into a self-comparison (RMS 0).

---

## 4. Fine-tuning on a few target views (test-time adaptation)

The model is amortised, so it can also be *adapted* to one instance with a handful of its views.
This is not used for any headline number; it is the cheapest way to move a single deployment
site from "zero-shot" to "well fitted".

* Adaptation set: `k` views of the target instance, sampled **at random** (not strided, not the
  first `k`). Our measurements showed strided/prefix sampling is biased by position × orientation
  and can lose up to 2.5 dB; random sampling gains 1.4–2.8 dB.
* Held-out set: the remaining `640 − k` views, scored with `--skip_flat_gt`.
* Optimiser: same schedule as training, run for 8 k–23 k steps on the single instance.

```bash
# adapt: resume from the amortised checkpoint and restrict training to one instance
# and a random view budget of the target views
$PY code/train_rrf_angular.py -s datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
    --spectra_root datasets/RF-3DGS_mvdr_params/spectra \
    --meta datasets/RF-3DGS_mvdr_params/meta.json \
    -m results/tta150 --start_checkpoint datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
    --resume_angular results/run_wrfgs/angular_180000.pth \
    --train_roles pos_holdout --train_instance_ids 150 --train_views test \
    --train_view_random 320 --train_view_seed 0 --iterations 8000 \
    --use_tx_orient 1 --use_tx_local_dir 1 --use_tx_freq 1 --init_u 0.05 \
    --lr_schedule cosine --lr_max_steps 8000 --lr_delay_steps 0 \
    --mlp_lr_init 2e-3 --mlp_lr_final 1e-5 --opacity_lr 0 --embed_lr 3e-4

# render all 640 views of the target instance; with --export_gt the jet ground truth is written
# into the same output tree. Then score only the views that were not used for adaptation.
$PY code/render_rrf_angular.py ... --instance 150 --limit 640 --export_gt \
    --out_root results/tta150/renders
$PY code/tools/tta_eval.py --renders results/tta150/renders/inst150/renders \
    --gt results/tta150/renders/inst150/gt \
    --poses_root datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
    --random 320 --seed 0 --skip_flat_gt
```

Measured ladder on the 53-instance **delivery protocol** (isotropic-transmitter generation, a
different protocol from the tables above — do not mix the rows across sections):

| Setting | LPIPS ↓ |
|---|---|
| Zero-shot amortised (no target views) | 0.3957 |
| + 320 target views, adapted | 0.3440 |
| In-sample oracle (sees the evaluation views, 23 k steps) | 0.2247 |

So a few hundred target views buy ≈ 0.05 LPIPS, and the remaining 0.12 is *not* a capacity limit
of the head — it is the price of amortisation.

---

## 5. Regenerating the figures and the paper

```bash
$PY paper/make_figs.py     # figs 1-7, all numbers read from results/
$PY paper/make_fig_w.py    # fig 8 (w channel), reads results/_w_arms/*/metrics_*.txt
bash paper/build.sh        # pdflatex + bibtex x3 → paper/main.pdf
```

`make_figs.py` and `make_fig_w.py` expect the directory names used by the run scripts
(`results/run_wrfgs`, `results/_w_arms/...`). If your runs live elsewhere, edit the constants at
the top of each script — they are the only place paths appear.
