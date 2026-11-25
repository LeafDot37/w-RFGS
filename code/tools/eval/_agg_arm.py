# -*- coding: utf-8 -*-
"""汇总 w-in-model 各臂（A0/A1/A2）在 36 个留出 w × 64 视角上的三指标。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import re
import statistics as st
import sys

for fn in sys.argv[1:]:
    v = []
    for ln in open(fn, errors="ignore"):
        m = re.search(r"PSNR ([\d.]+).*SSIM ([\d.]+)\s+LPIPS ([\d.]+)", ln)
        if m and "inst" in ln:
            v.append(tuple(float(x) for x in m.groups()))
    if not v:
        print(fn, "no rows")
        continue
    print(f"{fn}  n={len(v)}")
    for i, k in enumerate(["PSNR", "SSIM", "LPIPS"]):
        a = [x[i] for x in v]
        print(f"  {k:5s} {st.mean(a):.4f} ± {st.pstdev(a):.4f}")
