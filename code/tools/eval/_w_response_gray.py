# -*- coding: utf-8 -*-
"""换 w 的**灰阶响应**：模型把图改动了多少 vs 真值因 w 改动了多少。

为什么不能只看 ΔPSNR
--------------------------------------------------------------------------------
`_w_swap_report.py` 给的是「换成错的 w 后指标差多少」。那是一个**奖惩**读数：
只要两张图都离 GT 很远，ΔPSNR 可以接近 0，但头可能其实动了很多像素（或一动
没动）。计划 §2 的判据 3 要的是**响应幅度**：

    模型响应   = mean |render(as_is) − render(wswap)|        （灰阶 0–255）
    GT 响应    = mean |gt(i) − gt(j)|，同位置、不同 w 的训练实例两两之间
    响应比     = 模型响应 / GT 响应

历史基线（180k 波束条件臂）：GT 74.5 灰阶，模型 2.0–3.1 灰阶 ⇒ 2.7–4.2%。

用法（服务器）
    python code/tools/eval/_w_response_gray.py \
        --root results/run_.../训练/秩4 --stride 16 \
        --trainprobe results/run_.../训练/秩4/test_trainprobe \
        --meta datasets/RF-3DGS_mvdr_w_params/meta.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def gray(p):
    return np.asarray(Image.open(p).convert("L"), np.float32)


def mean_abs(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="臂目录（含 test_inst/ 与 test_wswap/）")
    ap.add_argument("--trainprobe", required=True, help="含 inst*/gt 的同位置 4 实例")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stride", type=int, default=16, help="视角抽样步长（1=全量，很慢）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    meta = {int(i["id"]): i for i in json.load(open(args.meta, encoding="utf-8"))["instances"]}

    # ① 模型响应：同一实例同一视角，as_is 与 wswap 两张渲染
    vals = []
    for d in sorted((root / "test_inst").glob("inst*")):
        iid = int(d.name[4:])
        w = root / "test_wswap" / d.name / "renders"
        if not w.is_dir():
            continue
        names = sorted(p.name for p in (d / "renders").glob("*.png"))[:: args.stride]
        for n in names:
            q = w / n
            if not q.exists():
                continue
            vals.append(mean_abs(gray(d / "renders" / n), gray(q)))
    if not vals:
        raise SystemExit(f"[FAIL] {root} 下没有可配对的 as_is/wswap 渲染")

    # ② GT 响应：同位置、不同 w 的 4 个训练实例，GT 两两之间
    dirs = sorted(Path(args.trainprobe).glob("inst*"))
    cache = {}
    for d in dirs:
        names = sorted(p.name for p in (d / "gt").glob("*.png"))[:: args.stride]
        cache[d.name] = {n: gray(d / "gt" / n) for n in names}
    gvals = []
    for x, y in itertools.combinations(sorted(cache), 2):
        common = sorted(set(cache[x]) & set(cache[y]))
        gvals.extend(mean_abs(cache[x][n], cache[y][n]) for n in common)

    m = statistics.mean(vals)
    g = statistics.mean(gvals) if gvals else float("nan")
    print(f"\n[{root.name}]  实例 {len({p.name for p in (root / 'test_inst').glob('inst*')})}"
          f"  抽样步长 {args.stride}  模型响应 n={len(vals)}   GT 响应 n={len(gvals)}")
    print(f"  模型响应（换 w 引起的灰阶变化）   {m:7.2f} 灰阶")
    print(f"  GT   响应（同位置不同 w 之间）    {g:7.2f} 灰阶")
    print(f"  响应比                          {m / g * 100.0:7.2f}%"
          if g == g and g > 0 else "  响应比                          nan")
    print("  历史基线：GT 74.5 灰阶、模型 2.0–3.1 灰阶 ⇒ 2.7–4.2%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "model_gray", "gt_gray", "ratio_pct", "n_model", "n_gt"])
            w.writerow([root.name, f"{m:.4f}", f"{g:.4f}",
                        f"{m / g * 100.0 if g and g == g else float('nan'):.4f}",
                        len(vals), len(gvals)])
        print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
