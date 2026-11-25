# -*- coding: utf-8 -*-
"""从 eval_csi_cbf 的日志里汇总三指标（逐实例均值再平均）。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import re
import sys

import numpy as np

for p in sys.argv[1:]:
    rows = []
    for ln in open(p, encoding="utf-8", errors="ignore"):
        m = re.search(r"inst(\d+) .*PSNR ([\d.]+).*SSIM ([\d.]+)\s+LPIPS ([\d.]+)", ln)
        if m:
            rows.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if rows:
        a = np.array(rows)
        print(f"{p.split('/')[-1]}: n={len(rows)} PSNR {a[:,0].mean():.3f} "
              f"SSIM {a[:,1].mean():.4f} LPIPS {a[:,2].mean():.4f}")
