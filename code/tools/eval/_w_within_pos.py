# -*- coding: utf-8 -*-
"""同一位置内、不同 w 的实例两两对比：GT 差多少 vs 模型差多少。

配对方式与 `_wswap_gt_gate.py` 相同，只是把「留出 w ↔ 基准 w」换成
「同位置的 4 个训练实例两两之间」。两边的 GT 都来自各自实例自己的谱图，
所以差的唯一来源是 w（位置、视角集合完全相同）。

读数：
- `GT`    GT_i vs GT_j           —— w 物理上把图改变了多少；
- `MODEL` 渲染_i vs 渲染_j        —— 模型对 w 的响应幅度；
- 若 MODEL 远好于 GT，说明即使在**训练集内**、位置和 w 都见过，
  模型也没把 w 用起来 ⇒ 是表示能力问题，不是泛化问题。

用法（服务器）：
    python code/tools/eval/_w_within_pos.py --root <arm>/test_trainprobe
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import itertools
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def index_dir(d):
    return {p.name: str(p) for p in Path(d).glob("*.png")}


def dir_psnr(xs, ys):
    names = sorted(set(xs) & set(ys))
    if not names:
        return None
    return statistics.mean(psnr(load(xs[n]), load(ys[n])) for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[FAIL] 目录不存在 {root}")

    inst = sorted(p.name for p in root.glob("inst*") if p.is_dir())
    if len(inst) < 2:
        raise SystemExit(f"[FAIL] 只有 {len(inst)} 个实例，没法配对")

    print(f"目录 {root}  实例 {inst}")
    cache = {}
    for i in inst:
        cache[i] = {
            "gt": index_dir(root / i / "gt"),
            "rd": index_dir(root / i / "renders"),
        }

    gt_vals, md_vals = [], []
    print(f"\n{'配对':<20}{'GT':>9}{'MODEL':>9}{'MODEL−GT':>10}")
    for a, b in itertools.combinations(inst, 2):
        g = dir_psnr(cache[a]["gt"], cache[b]["gt"])
        m = dir_psnr(cache[a]["rd"], cache[b]["rd"])
        if g is None or m is None:
            print(f"{a}↔{b:<14} 缺图，跳过")
            continue
        gt_vals.append(g)
        md_vals.append(m)
        print(f"{a}↔{b:<14}{g:9.3f}{m:9.3f}{m - g:10.3f}")

    if not gt_vals:
        raise SystemExit("[FAIL] 没有算得出结果的配对")
    tag = f"[{args.label}] " if args.label else ""
    print(f"\n{tag}k={len(gt_vals)} 对")
    print(f"  GT    同位置不同 w 之间      {statistics.mean(gt_vals):7.3f} ± {statistics.pstdev(gt_vals):.3f}")
    print(f"  MODEL 同位置不同 w 之间      {statistics.mean(md_vals):7.3f} ± {statistics.pstdev(md_vals):.3f}")
    gap = statistics.mean(md_vals) - statistics.mean(gt_vals)
    # 能量占比 = 模型/GT 两两差的能量比；19.9% 这个历史基线就是按
    # 10^(-PSNR/10) 的比值算的（13.30 dB vs 20.32 dB ⇒ 0.199）。
    e_gt = statistics.mean(10.0 ** (-v / 10.0) for v in gt_vals)
    e_md = statistics.mean(10.0 ** (-v / 10.0) for v in md_vals)
    print(f"  MODEL − GT = {gap:+.3f} dB")
    print(f"  模型表达的能量占比 = {e_md / max(e_gt, 1e-12) * 100.0:.1f}%")
    if gap > 3.0:
        print("  ⇒ 训练集内模型也不随 w 变化 ⇒ 表示能力问题（不是泛化问题）。")
    elif gap > 0.5:
        print("  ⇒ 训练集内模型对 w 有响应但幅度衰减。")
    else:
        print("  ⇒ 训练集内模型能跟上 w 的变化 ⇒ 问题出在泛化到没见过的 w。")


if __name__ == "__main__":
    main()
