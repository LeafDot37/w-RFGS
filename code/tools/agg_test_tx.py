# -*- coding: utf-8 -*-
"""把 _eval_all_test_tx.sh 的 metrics.txt 汇总成「13 个留出源」的 mean±std。

用法：python code/tools/_agg_test_tx.py <metrics.txt>
每行形如：`TX3 留出  n=640  PSNR 13.1234  SSIM 0.7456  LPIPS 0.4123`
"""
import re
import statistics as st
import sys
from pathlib import Path

pat = re.compile(r"^TX(\d+)\s+.*?PSNR\s+([\d.]+)\s+SSIM\s+([\d.]+)\s+LPIPS\s+([\d.]+)")
rows = []
for ln in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore").splitlines():
    m = pat.match(ln.strip())
    if m:
        rows.append((int(m.group(1)), *map(float, m.groups()[1:])))
if not rows:
    sys.exit("没有解析到任何行")
print(f"{'TX':>4} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
for r in rows:
    print(f"{r[0]:>4} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f}")
print(f"n={len(rows)}")
for i, name in enumerate(["PSNR", "SSIM", "LPIPS"], start=1):
    v = [r[i] for r in rows]
    print(f"{name}: mean {st.mean(v):.4f}  std {st.pstdev(v):.4f}  "
          f"min {min(v):.4f}  max {max(v):.4f}")
