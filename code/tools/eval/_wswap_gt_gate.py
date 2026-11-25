# -*- coding: utf-8 -*-
"""w 置换对照的**鉴别力闸**：先问「换 w 之后 GT 自己变多少」，再看模型变多少。

为什么必须做这一步
--------------------------------------------------------------------------------
`_w_swap_report.py` 报的是「把 w 换成错的 w 之后，模型指标掉了多少」。
但这条读数只有在 GT 本身对 w 敏感时才有分辨力。如果某个位置换个波束方向，
GT 只差 40 dB（几乎一样），那么模型就算完全无视 w，Δ 也会是 0 —— 这不是阴性，
是**这次对照没有鉴别力**。

反过来，如果 GT 对 w 敏感（比如只差 13 dB）而模型输出几乎不动，
那才是真的「头没有把 w 用起来」。

所以本脚本对每个 w_holdout 实例报三行量：

  GT   : spectra(实例自己的 w) vs spectra(base_instance 的训练域 w)
  MODEL: test_inst/renders vs test_wswap/renders（同一检查点，只换 w）
  AS_IS: 模型 vs GT（原样条件）—— 这是误差地板，用来判断上面的差是不是噪声级

判读：
- `GT` 与 `AS_IS` 同量级 ⇒ w 对图像的影响和模型误差相当 ⇒ 置换对照有鉴别力；
- `GT` 远好于 `AS_IS`（GT 对 w 几乎不敏感）⇒ 置换对照没有鉴别力，Δ≈0 不能读成阴性；
- `MODEL` 远好于 `GT` ⇒ 模型几乎不受 w 影响 ⇒ **头没有把 w 用起来**（真阴性）。

用法（服务器）：
    ROOT=<arm 目录> python code/tools/eval/_wswap_gt_gate.py
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
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def dir_psnr(xs, ys):
    """两组按文件名对齐的图，逐张 PSNR 后取均值。"""
    names = sorted(set(xs) & set(ys))
    if not names:
        return None, 0
    vals = [psnr(load(xs[n]), load(ys[n])) for n in names]
    return statistics.mean(vals), len(vals)


def index_dir(d):
    from pathlib import Path
    return {p.name: str(p) for p in Path(d).glob("*.png")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="臂目录（含 test_inst / test_wswap）")
    ap.add_argument("--spectra", default="/root/autodl-tmp/rf/datasets/RF-3DGS_mvdr_w_params/spectra")
    ap.add_argument("--meta", default="/root/autodl-tmp/rf/datasets/RF-3DGS_mvdr_w_params/meta.json")
    ap.add_argument("--limit", type=int, default=0, help="只前 K 个实例（0=全部）")
    args = ap.parse_args()

    m = json.load(open(args.meta, encoding="utf-8"))
    inst = {int(i["id"]): i for i in m["instances"]}
    wh = [int(i["id"]) for i in m["instances"] if i.get("role") == "w_holdout"]
    if args.limit:
        wh = wh[: args.limit]

    print(f"{'实例':<9}{'w(自己)':<13}{'w(基准)':<13}"
          f"{'GT':>9}{'MODEL':>9}{'AS_IS':>9}{'MODEL−GT':>10}")
    rows = []
    for iid in wh:
        base = inst[iid].get("base_instance")
        if base is None or int(base) < 0:
            print(f"inst{iid:<5} 没有 base_instance，跳过")
            continue
        base = int(base)
        g_dir = f"{args.spectra}/inst{iid:03d}"
        g_base = f"{args.spectra}/inst{base:03d}"
        gt_psnr, n_gt = dir_psnr(index_dir(g_dir), index_dir(g_base))

        r_as = index_dir(f"{args.root}/test_inst/inst{iid:03d}/renders")
        r_sw = index_dir(f"{args.root}/test_wswap/inst{iid:03d}/renders")
        md_psnr, n_md = dir_psnr(r_as, r_sw)

        gt_as = index_dir(f"{args.root}/test_inst/inst{iid:03d}/gt")
        as_psnr, n_as = dir_psnr(r_as, gt_as)

        if gt_psnr is None or md_psnr is None or as_psnr is None:
            print(f"inst{iid:<5} 有一侧缺图（GT {n_gt} / MODEL {n_md} / AS_IS {n_as}），跳过")
            continue
        rows.append((gt_psnr, md_psnr, as_psnr))
        print(f"inst{iid:<5}{str(inst[iid].get('w_label','-')):<13}"
              f"{str(inst[base].get('w_label','-')):<13}"
              f"{gt_psnr:9.3f}{md_psnr:9.3f}{as_psnr:9.3f}"
              f"{md_psnr - gt_psnr:10.3f}")

    if not rows:
        raise SystemExit("[FAIL] 没有任何实例算出结果")

    g = [r[0] for r in rows]
    md = [r[1] for r in rows]
    asc = [r[2] for r in rows]
    print(f"\nk={len(rows)}")
    print(f"  GT    GT(自己的 w) vs GT(基准 w)   {statistics.mean(g):7.3f} ± {statistics.pstdev(g):.3f}"
          f"   ← 「w 物理上把图改变了多少」")
    print(f"  MODEL 渲染(自己的 w) vs 渲染(基准 w) {statistics.mean(md):7.3f} ± {statistics.pstdev(md):.3f}"
          f"   ← 「模型对 w 的响应幅度」")
    print(f"  AS_IS 渲染 vs GT（原样条件）        {statistics.mean(asc):7.3f} ± {statistics.pstdev(asc):.3f}"
          f"   ← 误差地板")
    gap = statistics.mean(md) - statistics.mean(g)
    print(f"\n  模型对 w 的响应 − GT 对 w 的响应 = {gap:+.3f} dB")
    if statistics.mean(g) - statistics.mean(asc) < 1.0:
        print("  ⇒ 【闸不通过】GT 对 w 几乎不敏感（与误差地板同量级），"
              "这次置换对照没有鉴别力，Δ≈0 不能读成阴性。")
    elif gap > 3.0:
        print("  ⇒ 【真阴性】GT 明显随 w 变，而模型输出几乎不变 ⇒ 头没有把 w 用起来。")
    elif gap > 0.5:
        print("  ⇒ 【部分】模型对 w 有响应，但幅度明显小于 GT ⇒ 条件被用上了但衰减了。")
    else:
        print("  ⇒ 【有响应】模型对 w 的响应幅度与 GT 相当 ⇒ 条件确实驱动了输出。")


if __name__ == "__main__":
    main()
