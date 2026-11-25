# -*- coding: utf-8 -*-
"""带 CIR/MVDR 监督的 K 视角适配曲线聚合（与仅图像曲线对照）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/rf")
IDS = [144, 196, 232, 268, 304, 318]
KS = [4, 8, 16]


def main() -> None:
    rows = {}
    for k in KS:
        vals = []
        for i in IDS:
            p = R / f"results/_down_check/adaptcir_{i}_k{k}.jsonl"
            if not p.exists():
                continue
            rr = [json.loads(l) for l in open(p) if l.strip()]
            rr = [r for r in rr if int(r["id"]) == i]
            if rr:
                vals.append(rr[0])
        if not vals:
            continue
        rows[k] = {
            "n": len(vals),
            "cbf_psnr": float(np.mean([r["cbf_psnr"] for r in vals])),
            "cbf_ssim": float(np.mean([r["cbf_ssim"] for r in vals])),
            "cbf_lpips": float(np.mean([r["cbf_lpips"] for r in vals])),
            "mvdr_psnr": float(np.mean([r["mvdr_psnr"] for r in vals])),
            "pdp": float(np.mean([r["pdp_corr"] for r in vals])),
        }
    # K=0 = G1 loc16
    g = R / "results/_down_check/G1_loc16.jsonl"
    if g.exists():
        rr = [json.loads(l) for l in open(g) if l.strip()]
        rr = [r for r in rr if int(r["id"]) in IDS]
        rows[0] = {"n": len(rr),
                   "cbf_psnr": float(np.mean([r["cbf_psnr"] for r in rr])),
                   "cbf_ssim": float(np.mean([r["cbf_ssim"] for r in rr])),
                   "cbf_lpips": float(np.mean([r["cbf_lpips"] for r in rr])),
                   "mvdr_psnr": float(np.mean([r["mvdr_psnr"] for r in rr])),
                   "pdp": float(np.mean([r["pdp_corr"] for r in rr]))}
    out = R / "results/_down_check/adapt_cir_curve.json"
    out.write_text(json.dumps({str(k): rows[k] for k in sorted(rows)}, indent=1))
    print("K   n  CBF_PSNR  SSIM    LPIPS   MVDR    PDP", flush=True)
    for k in sorted(rows):
        v = rows[k]
        print("%-3d %d  %.3f    %.4f  %.4f  %.3f  %.3f"
              % (k, v["n"], v["cbf_psnr"], v["cbf_ssim"], v["cbf_lpips"],
                 v["mvdr_psnr"], v["pdp"]), flush=True)


if __name__ == "__main__":
    main()
