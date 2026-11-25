# -*- coding: utf-8 -*-
"""位置留出（48/16）漫散射基线聚合：SH/CORF 渲染 + 我们的 loc16 JSONL。"""
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
    fs, gs = sorted(rd.glob("*.png")), sorted(gd.glob("*.png"))
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/_diff_loc_big")
    ap.add_argument("--ours_suffix", default="loc16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    roots = {
        "SH": f"{a.root}/sh_inst%03d/test/ours_100000",
        "CORF": f"{a.root}/corf_inst%03d/rf_eval/test/ours_100000",
    }
    rows, per = {}, {}
    for tag, root in roots.items():
        vals, per[tag] = [], {}
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
    for tag in ("F20k", "G1", "D1", "D2"):
        p = R / f"results/_down_check/{tag}_{a.ours_suffix}.jsonl"
        if not p.exists():
            continue
        rr = [json.loads(l) for l in open(p) if l.strip()]
        rr = [r for r in rr if int(r["id"]) in IDS]
        if rr:
            rows[tag] = (float(np.mean([r["cbf_psnr"] for r in rr])),
                         float(np.mean([r["cbf_ssim"] for r in rr])),
                         float(np.mean([r["cbf_lpips"] for r in rr])),
                         len(rr))
    out = {"ids": IDS, "split": "location-disjoint 48/16", "summary": {k: list(v) for k, v in rows.items()},
           "per_instance": per}
    outp = a.out or f"{a.root}/summary.json"
    (R / outp).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("\n=== diffuse location-disjoint 48/16 (6 reps) ===", flush=True)
    for k, v in rows.items():
        print("%-5s n=%3d  PSNR %.3f  SSIM %.4f  LPIPS %.4f"
              % (k, v[3], v[0], v[1], v[2]), flush=True)


if __name__ == "__main__":
    main()
