# -*- coding: utf-8 -*-
"""评连续 w′：generator 生成的 custom scales vs 58 方向 oracle。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import glob
import json
import sys

import numpy as np

R = "/root/autodl-tmp/rf"
iid = int(sys.argv[1])
vals = {}
for f in glob.glob(f"{R}/results/_cbf_beam_sweep/scales_*_inst{iid:03d}.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        vals.setdefault(int(r["view"]), []).append(float(r["pmax"]))
custom = {}
for ln in open(f"{R}/results/_wprime_gen/inst{iid:03d}/scales_inst{iid:03d}.jsonl"):
    r = json.loads(ln)
    custom[int(r["view"])] = float(r["pmax"])
rat, rr = [], []
for v, a in vals.items():
    if v in custom:
        rat.append(custom[v] / max(a))
        rr.append(np.mean(a) / max(a))
print(f"inst{iid} n={len(rat)} wprime/oracle {np.mean(rat):.4f} "
      f"random {np.mean(rr):.4f} min {min(rat):.3f} max {max(rat):.3f}")
