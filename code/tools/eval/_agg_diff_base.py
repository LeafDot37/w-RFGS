# -*- coding: utf-8 -*-
"""统一域（灰度→jet）聚合漫散射基线 SH/CORF 与 F20k/G1 的 CBF 指标。"""
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
from splat.color_scalar import apply_jet, make_jet_lut          # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS               # noqa: E402
from splat.utils.image_utils import psnr                         # noqa: E402
from splat.utils.loss_utils import ssim                          # noqa: E402

IDS = [144, 196, 232, 268, 304, 318]
dev = "cuda" if torch.cuda.is_available() else "cpu"
lut = make_jet_lut(256).to(dev)
lp = LPIPS("vgg", "0.1").to(dev).eval()
for p in lp.parameters():
    p.requires_grad_(False)


def one(renders: Path, gts: Path):
    fs = sorted(renders.glob("*.png"))
    gs = sorted(gts.glob("*.png"))
    ps, ss, ls = [], [], []
    for f, g in zip(fs, gs):
        a = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
        b = np.asarray(Image.open(g).convert("L"), np.float32) / 255.0
        pr = apply_jet(torch.as_tensor(a, device=dev).unsqueeze(0), lut).unsqueeze(0)
        gt = apply_jet(torch.as_tensor(b, device=dev).unsqueeze(0), lut).unsqueeze(0)
        with torch.no_grad():
            ps.append(float(psnr(pr, gt).mean()))
            ss.append(float(ssim(pr, gt).mean()))
            ls.append(float(lp(pr, gt).mean()))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls)), len(ps)


def main():
    rows = {}
    for tag, root, it in (("SH", "results/_baseline_big/sh_cbf_inst%03d/test/ours_130000", 130000),
                          ("CORF", "results/_baseline_big/corf_cbf_inst%03d/rf_eval/test/ours_100000", 100000)):
        vals = []
        for i in IDS:
            d = R / (root % i)
            if not (d / "renders").exists():
                print("missing", tag, i, flush=True)
                continue
            vals.append(one(d / "renders", d / "gt"))
        rows[tag] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]),
                     np.mean([v[2] for v in vals]), sum(v[3] for v in vals))
    # ours
    for tag, jf in (("F20k", "results/_down_check/F20k_diff.jsonl"),
                    ("G1", "results/_down_check/G1_diff.jsonl")):
        p = R / jf
        if not p.exists():
            continue
        rr = [json.loads(l) for l in open(p) if l.strip()]
        sel = [r for r in rr if int(r["id"]) in IDS]
        if sel:
            rows[tag] = (np.mean([r["cbf_psnr"] for r in sel]),
                         np.mean([r["cbf_ssim"] for r in sel]),
                         np.mean([r["cbf_lpips"] for r in sel]), len(sel))
    print("\n=== diffuse CBF, same grayscale->jet domain, 6 reps ===", flush=True)
    for k, v in rows.items():
        print("%-5s n=%3d  PSNR %.3f  SSIM %.4f  LPIPS %.4f" % (k, v[3], v[0], v[1], v[2]), flush=True)


if __name__ == "__main__":
    main()
