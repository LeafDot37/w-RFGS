# -*- coding: utf-8 -*-
"""稀疏视角适配曲线聚合：K=0 取 G1 的 loc16 结果，K>0 取 adapt_<i>_k<K>。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/rf")
IDS = [144, 196, 232, 268, 304, 318]
KS = [0, 4, 8, 16, 32]


def mean_of(path: Path, ids):
    rr = [json.loads(l) for l in open(path) if l.strip()]
    rr = [r for r in rr if int(r["id"]) in ids]
    if not rr:
        return None
    return (len(rr),
            float(np.mean([r["cbf_psnr"] for r in rr])),
            float(np.mean([r["cbf_ssim"] for r in rr])),
            float(np.mean([r["cbf_lpips"] for r in rr])),
            float(np.mean([r["mvdr_psnr"] for r in rr])),
            float(np.mean([r["pdp_corr"] for r in rr])))


def main():
    rows = {}
    g1 = R / "results/_down_check/G1_loc16.jsonl"
    if g1.exists():
        rows[0] = mean_of(g1, IDS)
    for k in KS[1:]:
        vals = []
        for i in IDS:
            p = R / f"results/_down_check/adapt_{i}_k{k}.jsonl"
            if p.exists():
                m = mean_of(p, [i])
                if m:
                    vals.append(m)
        if vals:
            rows[k] = (len(vals),
                       float(np.mean([v[1] for v in vals])),
                       float(np.mean([v[2] for v in vals])),
                       float(np.mean([v[3] for v in vals])),
                       float(np.mean([v[4] for v in vals])),
                       float(np.mean([v[5] for v in vals])))
    print("\n=== sparse-view adaptation (G1 init, image-only, 400 steps) ===", flush=True)
    print("K   n  CBF_PSNR  SSIM    LPIPS   MVDR    PDP", flush=True)
    for k in sorted(rows):
        n, p, s, l, mv, pd = rows[k]
        print("%-3d %d  %.3f    %.4f  %.4f  %.3f  %.3f"
              % (k, n, p, s, l, mv, pd), flush=True)
    (R / "results/_down_check/sparse_curve.json").write_text(
        json.dumps({str(k): rows[k] for k in sorted(rows)}, indent=1))


if __name__ == "__main__":
    main()
