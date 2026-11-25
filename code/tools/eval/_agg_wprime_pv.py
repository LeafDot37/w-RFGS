# -*- coding: utf-8 -*-
"""汇总逐视角 w′（区域口径弃用）：逐实例比例、均值±std、胜率。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
import sys
from pathlib import Path

import numpy as np

SW = Path("/root/autodl-tmp/rf/results/_cbf_beam_sweep")
root = Path(sys.argv[1])
IDS = list(range(318, 354))


def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        d[int(r["view"])] = float(r["pmax"])
    return d


labels = sorted({p.name.split("scales_")[1].rsplit("_inst", 1)[0]
                 for p in SW.glob("scales_*_inst*.jsonl")})
rows = []
for iid in IDS:
    f = root / "perview" / f"scales_inst{iid:03d}.jsonl"
    if not f.exists():
        continue
    pv = load(f)
    orc = {l: load(SW / f"scales_{l}_inst{iid:03d}.jsonl") for l in labels
           if (SW / f"scales_{l}_inst{iid:03d}.jsonl").exists()}
    vs = [v for v in pv if all(v in o for o in orc.values())]
    om = np.array([max(o[v] for o in orc.values()) for v in vs], float)
    p = np.array([pv[v] for v in vs], float)
    ok = (om > 0) & (p > 0)
    if ok.sum() < 8:
        continue
    r_pv = float(np.mean(p[ok] / om[ok]))
    lab = []
    for l in orc:
        a = np.array([orc[l][v] for v in vs], float)
        lab.append(float(np.mean(a[ok] / om[ok])))
    rows.append((iid, r_pv, max(lab), float(np.mean(lab))))
    print(f"inst{iid:03d} n={int(ok.sum())} perview {r_pv:.4f} "
          f"codebook {max(lab):.4f} random {np.mean(lab):.4f}", flush=True)

a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
c = np.array([r[3] for r in rows])
print(f"\nN={len(rows)} perview {a.mean():.4f} ± {a.std():.4f} "
      f"(median {np.median(a):.4f})  codebook {b.mean():.4f}  "
      f"random {c.mean():.4f}  >codebook {int((a>b).sum())}/{len(rows)}")
