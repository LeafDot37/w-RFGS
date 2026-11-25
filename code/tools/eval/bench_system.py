# -*- coding: utf-8 -*-
"""系统级基准：单视角推理时延、显存、参数量、checkpoint 体积、新 TX 摊销成本。

输出 `results/_system/bench.json`，供论文的部署/复杂度表使用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                     # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw    # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry          # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--id", type=int, default=318)
    ap.add_argument("--view", default="")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="results/_system/bench.json")
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(R / "datasets/RF-3DGS_mvdr_w_params/meta.json",
                          encoding="utf-8"))
    inst = {int(i["id"]): i for i in meta["instances"]}[args.id]
    cams, _, test = load_cameras(
        str(R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"))
    vn = args.view or test[0]
    g = GaussianModel(3)
    load_visual_geometry(
        str(R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"), g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
    grid = CBFGrid()
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    from sim import w_codebook
    wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                         dtype=torch.complex64, device=dev)
    x = None
    for _ in range(args.warmup):
        x = render_cir(head, embed, g, cams[vn], inst, 256, 0.5, grid,
                       w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        x = render_cir(head, embed, g, cams[vn], inst, 256, 0.5, grid,
                       w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    t_spec = []
    for _ in range(args.reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = spec_raw(x, grid)
        torch.cuda.synchronize(); t_spec.append((time.perf_counter() - t0) * 1e3)
    n_head = sum(p.numel() for p in head.parameters())
    n_embed = int(embed.numel())
    n_rf = sum(p.numel() for p in head.rf_alpha.parameters()) if hasattr(head, "rf_alpha") else 0
    row = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_mb": round(os.path.getsize(args.checkpoint) / 1e6, 1),
        "gaussians": int(g.get_xyz.shape[0]),
        "head_params": int(n_head), "embed_params": n_embed, "rf_alpha_params": int(n_rf),
        "view_ms_median": float(np.median(ts)), "view_ms_p90": float(np.quantile(ts, 0.9)),
        "spec_ms_median": float(np.median(t_spec)),
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "views_640_s": round(float(np.median(ts)) * 640 / 1e3, 1),
    }
    out = R / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = []
    if out.exists():
        prev = json.load(open(out, encoding="utf-8"))
        if not isinstance(prev, list):
            prev = [prev]
    prev.append(row)
    out.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
