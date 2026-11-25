# -*- coding: utf-8 -*-
"""w 置换对照的配对报告：同一实例「原样条件」vs「只把 w 换成训练域 w」的差。

读数怎么解释（判据写在 `_eval_w_arms.sh` 的文件头）
--------------------------------------------------------------------------------
两条指标都对着**同一份 GT**（GT 只由实例的 w 决定，不随条件换）：
- `as_is`  = 实例自己的留出 w，是正确条件；
- `wswap`  = 把 w 换成 `base_instance` 的训练域 w，**错的条件**。

于是：
- 头真的在用 w ⇒ `as_is` 明显好于 `wswap`（Δ = wswap − as_is > 0）；
- 头根本没在用 w ⇒ 两次渲染逐位相同，Δ 全为 0、所有指标一字不差。

**Δ ≈ 0 是不可接受的阴性结果**（不是因为「w 不重要」，而是头绕过了这个条件，
那这个臂证明不了任何关于波束泛化的事）。Δ 很大但 as_is 本身很差，说明 w 被用上了
但没学到没见过的方向 —— 那是另一回事，要去看 `w_holdout` 的绝对指标。

跑法：
    $PY code/tools/eval/_w_swap_report.py --as_is <root>/test_inst/metrics.txt \
        --wswap <root>/test_wswap/metrics.txt --meta datasets/RF-3DGS_mvdr_w_params/meta.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import csv
import json
import statistics
from pathlib import Path

KEYS = ("psnr", "ssim", "lpips")


def read_metrics(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["id"])] = {k: float(r[k]) for k in ("psnr", "ssim", "lpips")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as_is", required=True)
    ap.add_argument("--wswap", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--per_instance", type=int, default=0,
                    help="打印前 K 个实例的明细（0=全部）")
    args = ap.parse_args()

    a = read_metrics(args.as_is)
    b = read_metrics(args.wswap)
    meta = {}
    if args.meta:
        m = json.load(open(args.meta, encoding="utf-8"))
        meta = {int(i["id"]): i for i in (m.get("instances") or [])}

    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("两份 metrics 没有共同实例号 —— 确认 wswap 目录渲染的是同一批实例")
    only = sorted(set(a) ^ set(b))
    if only:
        print(f"[warn] 只在一侧出现的实例 {only[:8]}（共 {len(only)} 个），已忽略")

    print(f"{'实例':<10}{'w(实例)':<14}{'w(基准)':<14}"
          f"{'PSNR as_is':>12}{'PSNR wswap':>12}{'ΔPSNR':>9}"
          f"{'ΔSSIM':>9}{'ΔLPIPS':>9}")
    rows = []
    shown = 0
    for i in ids:
        d = {k: b[i][k] - a[i][k] for k in KEYS}
        rows.append(d)
        if not args.per_instance or shown < args.per_instance:
            mi = meta.get(i, {})
            bi = int(mi.get("base_instance", -1))
            w_self = str(mi.get("w_label", "-"))
            w_base = str(meta.get(bi, {}).get("w_label", "-")) if bi >= 0 else "-"
            print(f"inst{i:03d}   {w_self:<14}{w_base:<14}"
                  f"{a[i]['psnr']:12.4f}{b[i]['psnr']:12.4f}{d['psnr']:9.4f}"
                  f"{d['ssim']:9.4f}{d['lpips']:9.4f}")
            shown += 1

    def mean(k):
        return statistics.mean(r[k] for r in rows)

    def sd(k):
        return statistics.pstdev(r[k] for r in rows) if len(rows) > 1 else 0.0

    n_zero = sum(1 for r in rows if abs(r["psnr"]) < 1e-9 and abs(r["lpips"]) < 1e-9)
    print(f"\nk={len(rows)}   Δ = wswap − as_is（wswap 是**错的条件**，所以 Δ 越正"
          f"越说明头在用 w）")
    print(f"  ΔPSNR  {mean('psnr'):+.4f} ± {sd('psnr'):.4f}")
    print(f"  ΔSSIM  {mean('ssim'):+.4f} ± {sd('ssim'):.4f}")
    print(f"  ΔLPIPS {mean('lpips'):+.4f} ± {sd('lpips'):.4f}")
    print(f"  两次渲染完全相同的实例：{n_zero} / {len(rows)}")
    if n_zero == len(rows):
        print("  ⇒ 头没有使用 w 条件（阴性）。这个臂证明不了波束泛化，先查输入是否真的进了头。")
    elif mean("psnr") <= 0:
        print("  ⇒ 换 w 之后**没有变差**：要么 w 的影响被别的条件补偿了，"
              "要么留出 w 与基准 w 的图像本身就很像。去查 w_codebook 的角度间隔。")


if __name__ == "__main__":
    main()
