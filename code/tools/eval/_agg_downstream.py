# -*- coding: utf-8 -*-
"""汇总 eval_csi_downstream 的 jsonl：三算子 × 各臂的三指标。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import glob
import json

import numpy as np

for p in sorted(glob.glob("/root/autodl-tmp/rf/results/_downstream/*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"== {p.split('/')[-1]}  instances={len(rows)}")
    for op in ("cbf", "mvdr", "delay"):
        if rows and f"{op}_psnr" in rows[0]:
            ps = np.array([r[op+'_psnr'] for r in rows], float)
            ss = np.array([r[op+'_ssim'] for r in rows], float)
            lp = np.array([r[op+'_lpips'] for r in rows], float)
            fin = np.isfinite(ps)
            print(f"  {op:5s} PSNR {np.mean(ps[fin]):7.3f} (finite {fin.sum()}/{len(ps)})"
                  f"  SSIM {np.mean(ss):6.4f}  LPIPS {np.mean(lp):6.4f}")
