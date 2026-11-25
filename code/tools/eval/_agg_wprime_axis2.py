# -*- coding: utf-8 -*-
"""汇总每个轴子集的逐视角 w′（修正 instXXX/perview.json 的 id 解析）。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])


def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        d[int(r["view"])] = float(r["pmax"])
    return d


labels = sorted({p.name.split("scales_")[1].rsplit("_inst", 1)[0]
                 for p in (root / "oracle").glob("scales_*.jsonl")})
ids = sorted(int(p.parent.name.split("inst")[1]) for p in root.glob("inst*/perview.json"))
rows = []
for iid in ids:
    f = root / "perview" / f"scales_inst{iid:03d}.jsonl"
    if not f.exists():
        continue
    pv = load(f)
    orc = {l: load(root / "oracle" / f"scales_{l}_inst{iid:03d}.jsonl")
           for l in labels
           if (root / "oracle" / f"scales_{l}_inst{iid:03d}.jsonl").exists()}
    vs = [v for v in pv if all(v in o for o in orc.values())]
    if len(vs) < 8:
        continue
    om = np.array([max(o[v] for o in orc.values()) for v in vs], float)
    p = np.array([pv[v] for v in vs], float)
    ok = (om > 0) & (p > 0)
    r = float(np.mean(p[ok] / om[ok]))
    lab = [float(np.mean(np.array([orc[l][v] for v in vs], float)[ok] / om[ok]))
           for l in orc]
    rows.append((iid, r, max(lab), float(np.mean(lab))))
    print(f"inst{iid:03d} n={int(ok.sum())} perview {r:.4f} "
          f"codebook {max(lab):.4f} random {np.mean(lab):.4f}", flush=True)
a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows]); c = np.array([r[3] for r in rows])
print(f"\nN={len(rows)} perview {a.mean():.4f} ± {a.std():.4f} "
      f"(median {np.median(a):.4f}) codebook {b.mean():.4f} "
      f"random {c.mean():.4f} >codebook {int((a > b).sum())}/{len(rows)}")
json.dump(dict(n=len(rows), perview=float(a.mean()), perview_std=float(a.std()),
               median=float(np.median(a)), codebook=float(b.mean()),
               random=float(c.mean()), beats=int((a > b).sum()), rows=rows),
          open(root / "summary.json", "w"), indent=1)
