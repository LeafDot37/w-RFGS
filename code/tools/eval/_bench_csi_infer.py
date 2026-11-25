# -*- coding: utf-8 -*-
"""测 CSI 头单视角推理时延：CIR 渲染 + 解析 CBF（不含指标）。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import sys
import time
from pathlib import Path

import torch

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_from_cir        # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

CKPT = R / "results/_csi_cbf/arm20k/csi_20000.pth"
SEED = R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"
POSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
META = R / "datasets/RF-3DGS_cbf_params/meta.json"

import json
dev = "cuda"
meta = json.load(open(META, encoding="utf-8"))
inst = [i for i in meta["instances"] if int(i["id"]) == 317][0]
cams, _, test = load_cameras(str(POSE))
g = GaussianModel(3)
load_visual_geometry(str(SEED), g, 3)
head, embed, op, cfg, it = load_tx_state(str(CKPT), device=dev)
if op is not None:
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                    requires_grad=False)
grid = CBFGrid()
T = int(cfg.get("csi_T", 128)); dt = float(cfg.get("csi_dt_ns", 0.5))
views = test[:20]
with torch.no_grad():
    for vn in views[:3]:
        render_cir(head, embed, g, cams[vn], inst, T, dt, grid)
    torch.cuda.synchronize()
    t0 = time.time()
    for vn in views:
        x = render_cir(head, embed, g, cams[vn], inst, T, dt, grid)
        spec_from_cir(x, grid)
    torch.cuda.synchronize()
    el = time.time() - t0
print(f"CSI 1x1: {len(views)} views in {el:.3f}s -> {1000*el/len(views):.1f} ms/view")
