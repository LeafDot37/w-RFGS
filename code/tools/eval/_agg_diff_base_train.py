# -*- coding: utf-8 -*-
"""漫散射逐场景基线（48/16 划分，100k）的统一域聚合：灰度→jet 的 CBF 三指标。

读 `results/_diff_base_train/{sh,corf}_instXXX` 的 test 渲染，与同一批 GT 比较；
同时读 ours 的 D1/G1/F20k JSONL（若存在），在 6 个代表实例上并列。
"""
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


def one(rd: Path, gd: Path):
    fs = sorted(rd.glob("*.png"))
    gs = sorted(gd.glob("*.png"))
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
    return np.mean(ps), np.mean(ss), np.mean(ls), len(ps)


def main():
    roots = {
        "SH": "results/_diff_base_train/sh_inst%03d/test/ours_100000",
        "CORF": "results/_diff_base_train/corf_inst%03d/rf_eval/test/ours_100000",
    }
    rows = {}
    per = {}
    for tag, root in roots.items():
        vals = []
        per[tag] = {}
        for i in IDS:
            d = R / (root % i)
            if not (d / "renders").exists():
                print("missing", tag, i, flush=True)
                continue
            v = one(d / "renders", d / "gt")
            vals.append(v)
            per[tag][i] = {"psnr": v[0], "ssim": v[1], "lpips": v[2], "n": v[3]}
        if vals:
            rows[tag] = (float(np.mean([v[0] for v in vals])),
                         float(np.mean([v[1] for v in vals])),
                         float(np.mean([v[2] for v in vals])),
                         int(sum(v[3] for v in vals)))
    for tag, jf in (("D1", "results/_down_check/D1_diff.jsonl"),
                    ("G1", "results/_down_check/G1_diff.jsonl"),
                    ("F20k", "results/_down_check/F20k_diff.jsonl")):
        p = R / jf
        if not p.exists():
            continue
        rr = [json.loads(l) for l in open(p) if l.strip()]
        sel = [r for r in rr if int(r["id"]) in IDS]
        if sel:
            rows[tag] = (float(np.mean([r["cbf_psnr"] for r in sel])),
                         float(np.mean([r["cbf_ssim"] for r in sel])),
                         float(np.mean([r["cbf_lpips"] for r in sel])),
                         len(sel))
    out = {"ids": IDS, "summary": {k: list(v) for k, v in rows.items()},
           "per_instance": per}
    (R / "results/_diff_base_train/summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== diffuse 48/16 (6 reps), grayscale->jet CBF ===", flush=True)
    for k, v in rows.items():
        print("%-5s n=%3d  PSNR %.3f  SSIM %.4f  LPIPS %.4f"
              % (k, v[3], v[0], v[1], v[2]), flush=True)


if __name__ == "__main__":
    main()
