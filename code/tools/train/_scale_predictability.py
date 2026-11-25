# -*- coding: utf-8 -*-
"""G1 第一刀：把被 min-max 抹掉的量级取出来，看它主要随什么变化。

输出（dB，10log10 功率量纲）：
  - 逐实例的 range / pmax 的均值与**视角间标准差**（这一项就是「尺度头必须
    显式依赖视角」的下界）；
  - 去掉逐实例均值后剩下的残差（= 一个「逐实例常数尺度」模型能解释多少）；
  - 全样本方差分解：实例间 vs 实例内。
判据：实例内 std < 1 dB ⇒ 尺度几乎只随发射源实例变，头可以很小；
      > 3 dB ⇒ 必须显式依赖视角几何。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/_scale_probe"


def db(x):
    return 10.0 * math.log10(max(x, 1e-30))


rows = {}
for f in sorted(glob.glob(str(D / "scales_inst*.jsonl"))):
    iid = int(Path(f).stem.split("inst")[1])
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    rng = np.array([db(r["pmax"] - r["pmin"]) for r in recs])
    pk = np.array([db(r["pmax"]) for r in recs])
    rows[iid] = (rng, pk)

print(f"{'inst':>6} {'n':>4} {'range mean':>11} {'range std':>10} "
      f"{'pmax mean':>10} {'pmax std':>9}")
all_res = []
for iid, (rng, pk) in rows.items():
    print(f"{iid:>6} {len(rng):>4} {rng.mean():>11.2f} {rng.std(ddof=1):>10.2f} "
          f"{pk.mean():>10.2f} {pk.std(ddof=1):>9.2f}")
    all_res.append(rng - rng.mean())

inst_means = np.array([rng.mean() for rng, _ in rows.values()])
within = np.concatenate(all_res)
print("\n--- 分解（range，dB） ---")
print(f"实例间 std      : {inst_means.std(ddof=1):8.2f}")
print(f"实例内 std(合并): {within.std(ddof=1):8.2f}")
print(f"总 std          : {np.concatenate([r for r, _ in rows.values()]).std(ddof=1):8.2f}")
print(f"逐实例常数模型的残差 P90: {np.percentile(np.abs(within), 90):6.2f} dB"
      f"   max: {np.abs(within).max():5.2f} dB")
