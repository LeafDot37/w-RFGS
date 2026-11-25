# -*- coding: utf-8 -*-
"""inst318 四个定性视角的「同位置 4w 先验」图（GT-only 参考行）。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from splat.color_scalar import apply_jet, make_jet_lut      # noqa: E402

SPC = R / "datasets/RF-3DGS_cbf_w_params/spectra"
META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
OUT = R / "results/_csi_cbfw/prior/inst318"
VIEWS = ["00214", "00213", "00216", "00188"]

meta = json.load(open(META, encoding="utf-8"))
ins = meta["instances"]
sib = [int(i["id"]) for i in ins
       if i.get("role") == "train" and int(i["position_id"]) == 0]
print("siblings", sib)
dev = "cuda"
lut = make_jet_lut(256).to(dev)
OUT.mkdir(parents=True, exist_ok=True)
for vn in VIEWS:
    us = []
    for s in sib:
        p = SPC / f"inst{s:03d}" / f"{vn}.png"
        if p.is_file():
            us.append(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
    if not us:
        print("[warn] no sibling for", vn)
        continue
    u = np.mean(np.stack(us, 0), 0).astype(np.float32)
    rgb = apply_jet(torch.as_tensor(u, device=dev), lut)
    a = (rgb.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(a).save(OUT / f"{vn}_prior.png")
    print("saved", vn, "from", len(us))
