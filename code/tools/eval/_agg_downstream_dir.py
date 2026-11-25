# -*- coding: utf-8 -*-
"""汇总指定目录下的下游 jsonl（三算子三指标）。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import glob, json, sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1])
for p in sorted(d.glob('*.jsonl')):
    rows=[json.loads(l) for l in open(p,encoding='utf-8')]
    print(f"== {p.name} n={len(rows)}")
    for op in ('cbf','mvdr','delay'):
        if rows and op+'_psnr' in rows[0]:
            ps=np.array([r[op+'_psnr'] for r in rows],float)
            ss=np.array([r[op+'_ssim'] for r in rows],float)
            lp=np.array([r[op+'_lpips'] for r in rows],float)
            f=np.isfinite(ps)
            print(f"  {op:5s} PSNR {ps[f].mean():7.3f} (fin {f.sum()})  SSIM {ss.mean():6.4f}  LPIPS {lp.mean():6.4f}")
