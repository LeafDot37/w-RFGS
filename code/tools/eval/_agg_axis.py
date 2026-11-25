# -*- coding: utf-8 -*-
"""汇总单参数留出块（yaw/pitch/freq/double）的逐实例三指标。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import re
import statistics as st
import sys
from collections import defaultdict

fn = sys.argv[1]
pat = re.compile(r"inst(\d+)\s+role=(\w+).*PSNR ([\d.]+).*SSIM ([\d.]+)\s+LPIPS ([\d.]+)")
g = defaultdict(list)
for ln in open(fn, errors="ignore"):
    m = pat.search(ln)
    if m:
        g[m.group(2)].append(tuple(float(x) for x in m.groups()[2:]))
for role in ("yaw_holdout", "pitch_holdout", "freq_holdout",
             "double_holdout", "pos_holdout", "train", "orig"):
    if role not in g:
        continue
    v = g[role]
    print(f"{role:14s} n={len(v):3d}  "
          f"LPIPS {st.mean(x[2] for x in v):.4f}±{st.pstdev(x[2] for x in v):.4f}  "
          f"SSIM {st.mean(x[1] for x in v):.4f}±{st.pstdev(x[1] for x in v):.4f}  "
          f"PSNR {st.mean(x[0] for x in v):.4f}±{st.pstdev(x[0] for x in v):.4f}")
