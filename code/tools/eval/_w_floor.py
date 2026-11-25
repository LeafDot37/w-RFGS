# -*- coding: utf-8 -*-
"""w 数据集的**常数图地板**：每个视角取所有训练实例 GT 的均值，作为「什么条件都不用」的基线。

为什么必须有这个数
--------------------------------------------------------------------------------
`w_holdout` 上的 12.05 dB 是绝对值，光看它没法判断模型到底学到了什么：
如果常数图地板就是 12.2 dB，那模型等于什么都没做；如果地板只有 9 dB，
12.05 就是实打实的增益。所以任何绝对 PSNR 都必须配套报地板。

口径
--------------------------------------------------------------------------------
- 地板图：对每个视角 v，取 144 个训练实例在该视角的 GT 的像素均值（灰度 0–255 → /255）。
  这也是 RF-3DGS 系列一直用的「常数图基线」口径。
- 读数：把地板图与 `w_holdout` 各实例的 GT 比，逐图 PSNR 后按实例求均值。
- `--stride` 只抽一部分视角（地板图是按视角定义的，抽样不影响口径，
  只是把 640 次读盘降到 640/stride 次）。

用法（服务器）：
    python code/tools/eval/_w_floor.py --stride 8
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import math
import statistics

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default="/root/autodl-tmp/rf/datasets/RF-3DGS_mvdr_w_params/spectra")
    ap.add_argument("--meta", default="/root/autodl-tmp/rf/datasets/RF-3DGS_mvdr_w_params/meta.json")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    m = json.load(open(args.meta, encoding="utf-8"))
    inst = {int(i["id"]): i for i in m["instances"]}
    train = [int(i["id"]) for i in m["instances"] if i.get("role") == "train"]
    hold = [int(i["id"]) for i in m["instances"] if i.get("role") == "w_holdout"]

    # 视角名取训练实例目录里的文件名（所有实例共用同一套 640 个 test 视角）
    from pathlib import Path
    names = sorted(p.name for p in Path(f"{args.spectra}/inst{train[0]:03d}").glob("*.png"))
    names = names[:: args.stride]
    print(f"训练实例 {len(train)} 个，留出实例 {len(hold)} 个，抽样视角 {len(names)} / 640")

    # 逐视角累加训练实例的 GT
    floor = {}
    for n in names:
        acc = None
        for i in train:
            im = load(f"{args.spectra}/inst{i:03d}/{n}")
            acc = im.copy() if acc is None else acc + im
        floor[n] = acc / len(train)

    # 地板 vs 每个留出实例的 GT
    per_inst = []
    for i in hold:
        vals = [psnr(floor[n], load(f"{args.spectra}/inst{i:03d}/{n}")) for n in names]
        per_inst.append(statistics.mean(vals))
        print(f"  inst{i:03d}  w={inst[i].get('w_label')}  地板 PSNR {per_inst[-1]:.4f}")

    mean = statistics.mean(per_inst)
    sd = statistics.pstdev(per_inst)
    print(f"\n常数图地板（w 留出组，实例级）  PSNR {mean:.4f} ± {sd:.4f}  "
          f"（{len(per_inst)} 个实例 × {len(names)} 视角）")
    if args.out:
        json.dump({"psnr_mean": mean, "psnr_pstdev": sd, "n_inst": len(per_inst),
                   "n_views": len(names), "stride": args.stride,
                   "per_instance": per_inst},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"已写 {args.out}")


if __name__ == "__main__":
    main()
