# -*- coding: utf-8 -*-
"""汇总 per-instance RF-3DGS 自拟合/替换与训练耗时。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])  # results/_perinst dir (downloaded)
rows = []
for f in sorted(root.glob("eval_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    rows.append(d)
if not rows:
    raise SystemExit("no eval json")
for key, label in (("own", "own fit"), ("sib", "substitution")):
    for m in ("lpips", "ssim", "psnr"):
        v = [r[key][m] for r in rows]
        print(f"{label:12s} {m:5s} {st.mean(v):.4f} ± {st.pstdev(v):.4f}")
    print(f"{label:12s} ms/view {st.mean(r['ms_per_view'] for r in rows):.1f}")
print(f"n positions {len(rows)}")
for r in rows:
    print(f"  inst{r['own']} own LPIPS {r['own']['lpips']:.3f} "
          f"sib LPIPS {r['sib']['lpips']:.3f}  "
          f"own PSNR {r['own']['psnr']:.2f} sib {r['sib']['psnr']:.2f}")
