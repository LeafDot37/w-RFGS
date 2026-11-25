#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w-RFGS one-command pipeline: train -> evaluate -> figures.

Stages
------
    python pipeline.py --stage train   ...   # fit the CSI head (default 100k steps)
    python pipeline.py --stage eval    ...   # CBF + MVDR on held-out instances -> metrics.json
    python pipeline.py --stage figures ...   # bar chart + qualitative grid
    python pipeline.py --stage all     ...   # all three

Layout expected by default (override any path with the flags below):
    datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100/   # COLMAP + optical ckpt
    datasets/RF-3DGS_cbf_w_params/spectra/                        # per-instance GT u images
    datasets/_cir_train/                                          # optional CIR labels
    datasets_meta/mvdr_w_params/meta.json                         # instance table

Example (after downloading the released archives):
    python pipeline.py --stage all --out results/main_cross_tx
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AXES = ["pos_holdout", "yaw_holdout", "pitch_holdout",
        "freq_holdout", "double_holdout", "w_holdout"]
AXIS_NAME = {"pos_holdout": "Position", "yaw_holdout": "Yaw",
             "pitch_holdout": "Pitch", "freq_holdout": "Carrier",
             "double_holdout": "Joint", "w_holdout": "Weights"}


def run(cmd, log_path=None):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    if log_path is None:
        return subprocess.run([str(x) for x in cmd], check=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        return subprocess.run([str(x) for x in cmd], stdout=fh,
                              stderr=subprocess.STDOUT, check=True)


def parse_stdout(path: Path):
    """Parse eval_csi_cbf.py's per-instance lines into {iid: metrics}."""
    pat = re.compile(r"inst(\d+)\s+role=(\S+)\s+n=(\d+)\s+PSNR\s+([\d.]+)\s+.*?"
                     r"SSIM\s+([\d.]+)\s+LPIPS\s+([\d.]+)")
    out = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            out[int(m.group(1))] = {"role": m.group(2), "n": int(m.group(3)),
                                    "psnr": float(m.group(4)),
                                    "ssim": float(m.group(5)),
                                    "lpips": float(m.group(6))}
    return out


def select_ids(meta_path: Path, per_axis: int):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    by_axis = {a: [] for a in AXES}
    for inst in meta["instances"]:
        role = inst.get("role")
        if role in by_axis:
            by_axis[role].append(int(inst["id"]))
    ids = []
    for a in AXES:
        ids += by_axis[a][:per_axis]
    return ids, by_axis


def stage_train(a):
    has_cir = bool(a.cir_root) and Path(a.cir_root).exists()
    cmd = [sys.executable, "code/rfgs/csi/train_csi_cbf.py",
           "-s", a.source, "--spectra_root", a.spectra, "--meta", a.meta,
           "--start_checkpoint", a.start,
           "--model_path", a.out, "--head", "bisg", "--train_op", "cbf",
           "--tx_nr", "4", "--tx_nc", "4", "--w_input", "none", "--T", "256",
           "--iterations", a.iterations, "--save_every", a.save_every,
           "--log_every", 2000,
           "--w_img", "0.5", "--w_mvdr_img", "0.5",
           "--w_cir", "1.0" if has_cir else "0.0",
           "--mvdr_stride", "1"]
    if has_cir:
        cmd += ["--cir_root", a.cir_root]
    else:
        print("[pipeline] CIR labels not found; training with image + MVDR "
              "supervision only.", flush=True)
    run(cmd, Path(a.out) / "logs" / "train.log")


def stage_eval(a):
    ids, by_axis = select_ids(Path(a.meta), a.per_axis)
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    metrics = {}
    for op in ("cbf", "mvdr"):
        cmd = [sys.executable, "code/rfgs/evaluation/eval_csi_cbf.py",
               "-s", a.source, "--spectra_root", a.spectra, "--meta", a.meta,
               "--start_checkpoint", a.start,
               "--checkpoint", a.checkpoint, "--ids", *ids,
               "--views", str(a.views), "--op", op,
               "--tx_nr", "4", "--tx_nc", "4"]
        if op == "mvdr":
            cmd += ["--dt_ns", "0.1", "--T", "600"]
        log = out / "logs" / f"eval_{op}.log"
        run(cmd, log)
        metrics[op] = parse_stdout(log)
    # aggregate per axis
    summary = {}
    for op in ("cbf", "mvdr"):
        for axis in AXES:
            vals = [metrics[op][i] for i in by_axis[axis][:a.per_axis]
                    if i in metrics[op]]
            if not vals:
                continue
            summary.setdefault(AXIS_NAME[axis], {})[op] = {
                "psnr": float(np.mean([v["psnr"] for v in vals])),
                "ssim": float(np.mean([v["ssim"] for v in vals])),
                "lpips": float(np.mean([v["lpips"] for v in vals])),
                "n": len(vals)}
    (out / "metrics.json").write_text(
        json.dumps({"per_axis": summary, "per_instance": metrics},
                   indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary, by_axis


def stage_figures(a):
    out = Path(a.out)
    data = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    summary = data["per_axis"]
    names = [AXIS_NAME[x] for x in AXES if AXIS_NAME[x] in summary]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # bar chart
    for metric, fname in (("psnr", "bars_psnr.pdf"), ("lpips", "bars_lpips.pdf")):
        x = np.arange(len(names)); w = 0.36
        fig, ax = plt.subplots(figsize=(7.0, 2.4), dpi=300)
        for k, (op, lab) in enumerate((("cbf", "CBF"), ("mvdr", "MVDR"))):
            v = [summary[n][op][metric] for n in names]
            ax.bar(x + (k - 0.5) * w, v, w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel(metric.upper())
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(figdir / fname)
        fig.savefig(figdir / fname.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)

    # qualitative grid: one instance per axis, two views
    qual_ids = a.qual_ids or []
    if qual_ids:
        import matplotlib.pyplot as plt
        from PIL import Image
        ops = ("cbf", "mvdr")
        grid = {}
        for iid in qual_ids:
            for op in ops:
                d = out / "qual" / f"inst{iid:03d}_{op}"
                if not d.exists():
                    run([sys.executable, "code/rfgs/evaluation/eval_csi_cbf.py",
                         "-s", a.source, "--spectra_root", a.spectra,
                         "--meta", a.meta, "--start_checkpoint", a.start,
                         "--checkpoint", a.checkpoint, "--ids", str(iid),
                         "--views", "0", "--view_list", a.qual_views,
                         "--op", op, "--tx_nr", "4", "--tx_nc", "4",
                         "--save_dir", d], out / "logs" / f"qual_{iid}_{op}.log")
                grid[(iid, op)] = d
        views = a.qual_views.split(",")
        cols = [("CBF GT", "gt", "cbf"), ("CBF ours", "pred", "cbf"),
                ("MVDR GT", "gt", "mvdr"), ("MVDR ours", "pred", "mvdr")]
        fig, axes = plt.subplots(len(qual_ids), len(cols), squeeze=False,
                                 figsize=(7.0, 1.15 * len(qual_ids)), dpi=300)
        for r, iid in enumerate(qual_ids):
            for c, (lab, tag, op) in enumerate(cols):
                ax = axes[r, c]
                p = grid[(iid, op)] / f"{views[0]}_{tag}.png"
                if p.exists():
                    ax.imshow(np.asarray(Image.open(p).convert("RGB")))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(lab, fontsize=6)
            axes[r, 0].set_ylabel(str(iid), fontsize=6)
        fig.tight_layout(pad=0.1)
        fig.savefig(figdir / "qual_grid.pdf")
        fig.savefig(figdir / "qual_grid.png", dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["train", "eval", "figures", "all"], default="all")
    ap.add_argument("--out", default="results/main_cross_tx")
    ap.add_argument("--source", default="datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100")
    ap.add_argument("--spectra", default="datasets/RF-3DGS_cbf_w_params/spectra")
    ap.add_argument("--meta", default="datasets_meta/mvdr_w_params/meta.json")
    ap.add_argument("--start", default="datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth")
    ap.add_argument("--cir-root", default="datasets/_cir_train")
    ap.add_argument("--iterations", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--per-axis", type=int, default=6)
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--qual-ids", type=int, nargs="*", default=None)
    ap.add_argument("--qual-views", default="00098,00285")
    a = ap.parse_args()
    a.source = str(Path(a.source).resolve())
    a.spectra = str(Path(a.spectra).resolve())
    a.meta = str(Path(a.meta).resolve())
    a.start = str(Path(a.start).resolve())
    a.cir_root = str(Path(a.cir_root).resolve()) if a.cir_root else ""
    a.out = str(Path(a.out).resolve())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.checkpoint:
        a.checkpoint = str(Path(a.out) / f"csi_{a.iterations}.pth")
    os.environ.setdefault("PYTHONPATH", str(ROOT / "code"))
    if a.stage in ("train", "all"):
        stage_train(a)
    if a.stage in ("eval", "all"):
        stage_eval(a)
    if a.stage in ("figures", "all"):
        stage_figures(a)


if __name__ == "__main__":
    main()
