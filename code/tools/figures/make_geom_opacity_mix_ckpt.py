# -*- coding: utf-8 -*-
"""make_geom_opacity_mix_ckpt.py —— 造「几何取自甲、opacity 取自乙」的混合 ckpt。

为什么需要它（本仓目前唯一能把「几何通道」与「opacity 通道」分开的实验）
======================================================================
斜率轴上最强的一条读数（**同点数**，见 results/run_20260915_视觉质量因果斜率/说明.md）：

    档 0   （P = 1,276,957）  MVDR 17.9245 / 0.79395 / 0.30876
    档 500 （P = 1,276,957）  MVDR 19.0363 / 0.81362 / 0.28621
    ⇒ 点数一字未动，差 **+1.1118 dB**（8.87× 抖动）

但「视觉短训改了起点」这件事，在 RRF 的入口处有**两个**可动的量：

    xyz / scaling / rotation   —— 几何。`train_rrf_scalar.py:81-87` 把它们冻结
    _opacity                   —— 不透明度。同处 `requires_grad_(True)`，**不冻结**

两者在 0→500 之间都被视觉短训改过。**光看 0 档与 500 档的对比，永远分不开是哪一半**
（几何变了 ⇒ 头读到的位置/形状变了；opacity 变了 ⇒ 光栅化的混合权重变了）。
本工具造出 2×2 因子的另外两格：

                 opacity_A（档 0）      opacity_B（档 500）
    几何_A（档0）   (A,A) = 档 0，已有     **(A,B) ← 本工具**
    几何_B（档500） **(B,A) ← 本工具**     (B,B) = 档 500，已有

    几何主效应     = mean(几何_B 两格) − mean(几何_A 两格)
    opacity 主效应 = mean(op_B 两格) − mean(op_A 两格)
    交互           = (B,B) − (B,A) − (A,B) + (A,A)

若「几何主效应」占掉那 1.11 dB 的绝大部分 ⇒ 通道是**几何**（那就有理由去压头的容量、
或给几何解冻）；若被 opacity 占掉 ⇒ 通道是**不透明度**，与「视觉几何质量」是两回事。
**这是唯一能把两者分开的实验，而且两个源 ckpt 已经存在，只要 2 个新节点。**

为什么这件事是干净的（不是又一次「往 RRF 里塞先验」）
====================================================
两个源 ckpt 的**点数与点序完全相同**（P 都是 1,276,957）：`densify_from_iter=500`
⇒ 致密化 / 剪枝 / opacity reset 在 500 步时**一次都没启动** ⇒ 第 i 个点对第 i 个点。
所以「取甲的几何 + 取乙的 opacity」是**逐点对齐的替换**，不抽样、不插值、不引入新点。
这与已被证伪的「协方差初始化」（注入**新的形状**）和「SH/上下文块」（加**新的输入列**）
都不同 —— 本实验**一个字节的新信息都不加**，只是把两个已有的量分开。

逐点对应怎么验（**这里有一个我先写错、后改掉的断言，留着当教训**）
==================================================================
我原本想用「排序后 ∞ 距离 ≤ 恒等 ∞ 距离」当**精确闸**：若恒等配对是对的，右边必然
不小于左边，反之就说明配对错了。**这个断言是恒真的，当闸等于没闸** —— bottleneck
匹配的定义就是「排序后的 ∞ 距离 = 所有配对里 ∞ 距离的**最小值**」，恒等配对只是其中
一个候选，所以 `排序 ≤ 恒等` 必然成立，与配对对不对无关。用它当闸会**永远放行**。

真正有判别力的是三件事，全部打印出来：

  ① **点数相等**（硬闸）。点数不等 ⇒ 「逐点替换」根本无从谈起。
  ② **排序后 ∞ 距离的绝对量级**（相对场景尺度）。它度量的是「两侧的点在空间上
     是不是同一套分布」。若某一侧已经致密化/剪枝过（点集真变了），这个数会显著
     变大。硬闸取场景尺度的 20% —— 门槛宽松是**故意**的：它要拦的是「根本不是
     同一个点云」，不是「点挪得多」。
  ③ **恒等 / 排序 的比值**（诊断）。恒等配对有多接近最优配对：≈1 ⇒ 恒等就是最优
     配对（点序一致）；远大于 1 ⇒ 点序对不上。这一项**不做硬闸**，因为比值大也
     可能只是点集的局部重排，需要人看一眼再决定。

真正的来源证据不在几何里，而在**存档时机**上：`densify_from_iter=500`
（`code/splat/arguments/__init__.py`）⇒ 500 步时致密化/剪枝/opacity reset 一次都
没启动，两侧点数都等于 `mvsplat_clouds/init.ply` 的点数。这条由驱动脚本对着
斜率轴的 `来源.txt` 逐项核对，比任何几何启发式都硬。

顺带把「视觉短训到底动了什么」打出来（逐点位移 / opacity / 尺度 / 朝向）
======================================================================
这是本工具的**副产品，但本身就是一条读数**：它直接回答「那 +1.11 dB 是哪儿来的」。
不额外花 GPU，也不用再写一个工具。

用法
====
    <PY> tools/make_geom_opacity_mix_ckpt.py \
        --geom_ckpt <档0>/chkpnt0.pth \
        --op_ckpt   <档500>/chkpnt500.pth \
        --out       <TOP>/点云/chkpnt_几何0_opacity500.pth

退出码 0 = 成功；非 0 = 拒绝（参数不合理 / 形状不符 / 对应性不成立），
**绝不静默写半成品** —— 写坏了会让后面两小时的训练从错误的起点出发，且不报任何错。
写完会**重新 load 一遍逐项核对**，不靠「我写对了」。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# GaussianModel.capture() 的 12 项（与 train_rrf_scalar.load_visual_geometry 同源）
N_PARAMS = 12
IDX_SH_DEGREE, IDX_XYZ, IDX_F_DC, IDX_F_REST = 0, 1, 2, 3
IDX_SCALING, IDX_ROTATION, IDX_OPACITY = 4, 5, 6
IDX_OPT_DICT, IDX_SLR = 10, 11
# 「几何」这一侧要取的项：xyz / scaling / rotation，加上不使用但必须成套带走的
# SH 外观与 active_sh_degree。SH 在本实验的头里 `use_sh=none` 不参与，
# 但成套取可以让产物与源 ckpt 的其余部分保持一致，避免留下半新半旧的字段。
GEOM_SIDE = {
    "active_sh_degree": IDX_SH_DEGREE,
    "_xyz": IDX_XYZ, "_features_dc": IDX_F_DC, "_features_rest": IDX_F_REST,
    "_scaling": IDX_SCALING, "_rotation": IDX_ROTATION,
}


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, (tuple, list)) and len(ck) == 2):
        raise SystemExit(f"[拒绝] {path} 不是 (model_params, first_iter) 二元组")
    mp, first_iter = ck
    if len(mp) != N_PARAMS:
        raise SystemExit(f"[拒绝] {path} 的 model_params 有 {len(mp)} 项，"
                         f"与 capture() 的 {N_PARAMS} 项不符")
    return mp, first_iter


def npy(t):
    """张量 → numpy（去梯度、float32 前先转 float64 免得百分位被量化）"""
    if not hasattr(t, "detach"):
        raise SystemExit(f"[拒绝] 期望张量，拿到 {type(t).__name__}")
    return t.detach().float().cpu().numpy()


def _norm_q(q):
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description="造「几何取自甲、opacity 取自乙」的混合 ckpt")
    ap.add_argument("--geom_ckpt", required=True, help="几何（xyz/scaling/rotation）来源")
    ap.add_argument("--op_ckpt", required=True, help="opacity 来源")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first_iter", type=int, default=0,
                    help="产物写的 first_iter。默认 0 —— RRF 的 LR 调度按**本地步**算"
                         "（train_rrf_scalar.py:288 `mlp_sched(iteration - first_iter)`），"
                         "所以 first_iter=0 + 本地 1 万步与档 500 节点的 first_iter=500 "
                         "+ 本地 1 万步是**同一条调度**，绝对步号不同而已。")
    ap.add_argument("--allow-order-mismatch", action="store_true",
                    help="跳过逐点对应性硬断言（**只在你已经知道点序对不上、"
                         "并愿意承担后果时用**；默认拒绝）")
    ap.add_argument("--expect_P", type=int, default=None,
                    help="两侧点数都必须等于这个值（来自斜率轴的 来源.txt）。"
                         "这是**来源证据**而不是几何启发式：densify_from_iter=500 "
                         "⇒ 500 步时致密化一次都没启动，两侧点数都应 = init.ply 的点数。"
                         "给了就硬断言，不给只打印。")
    args = ap.parse_args()

    if args.geom_ckpt == args.op_ckpt:
        raise SystemExit("[拒绝] --geom_ckpt 与 --op_ckpt 是同一个文件 ⇒ 那不是混合，是复制")

    mp_g, fi_g = load_ckpt(args.geom_ckpt)
    mp_o, fi_o = load_ckpt(args.op_ckpt)
    print(f"[读入] 几何侧  {args.geom_ckpt}")
    print(f"       first_iter = {fi_g}")
    print(f"[读入] opacity 侧 {args.op_ckpt}")
    print(f"       first_iter = {fi_o}")

    # ---- 形状与场景一致性 ----
    xyz_g, xyz_o = npy(mp_g[IDX_XYZ]), npy(mp_o[IDX_XYZ])
    if xyz_g.ndim != 2 or xyz_g.shape[1] != 3:
        raise SystemExit(f"[拒绝] 几何侧 _xyz 形状 {xyz_g.shape} != (P,3)")
    if xyz_g.shape != xyz_o.shape:
        raise SystemExit(f"[拒绝] 两侧点数/形状不等：几何 {xyz_g.shape} vs opacity {xyz_o.shape}\n"
                         "        点数不等 ⇒ 「逐点对齐替换」根本不成立，本实验无从谈起。")
    P = xyz_g.shape[0]
    print(f"       点数 P = {P}（两侧相同 ✓）")
    if args.expect_P is not None:
        if P != args.expect_P:
            raise SystemExit(f"[拒绝] 点数 P = {P} 与 --expect_P {args.expect_P} 不符。\n"
                             f"        两侧点数应当都等于 init.ply 的点数"
                             f"（densify_from_iter=500 ⇒ 致密化还没启动）。\n"
                             f"        对不上 ⇒ 至少一侧的存档点已经在致密化/剪枝之后，"
                             f"「逐点替换」不成立。")
        print(f"       与 --expect_P {args.expect_P} 相符 ✓（来源证据，不是几何启发式）")

    slr_g, slr_o = mp_g[IDX_SLR], mp_o[IDX_SLR]
    if slr_g is None or slr_o is None or float(slr_g) != float(slr_o):
        raise SystemExit(f"[拒绝] spatial_lr_scale 不一致（几何 {slr_g} vs opacity {slr_o}）。"
                         f"它是 A× 头 `--use_dist` 真的会读的量（_dist 的分母）"
                         f"⇒ 不一致就说明两侧不是同一个场景，不许混。")
    print(f"       spatial_lr_scale = {float(slr_g)}（两侧相同 ✓）")

    # ---- 逐点对应性（见文件头：不要用「排序 ≤ 恒等」当闸，那是恒真的）----
    d = xyz_g.astype(np.float64) - xyz_o.astype(np.float64)
    disp = np.linalg.norm(d, axis=1)
    ident = float(np.abs(d).max())
    sorted_worst = max(float(np.abs(np.sort(xyz_g[:, k].astype(np.float64))
                                    - np.sort(xyz_o[:, k].astype(np.float64))).max())
                       for k in range(3))
    ext = float(np.percentile(np.linalg.norm(xyz_g.astype(np.float64)
                                             - np.median(xyz_g, 0), axis=1), 95))
    # 比值：恒等配对距离 / 最优（排序后）配对距离。≈1 ⇒ 恒等就是最优配对。
    ratio = (ident / sorted_worst) if sorted_worst > 0 else float("inf")
    print()
    print("  [逐点对应性]")
    print(f"       场景尺度 P95|xyz−中位|        = {ext:.6g}")
    print(f"       恒等配对 max|Δxyz|∞           = {ident:.6g}"
          f"（{ident / ext:.2%} 场景尺度）")
    print(f"       排序后   max|Δsort|∞          = {sorted_worst:.6g}"
          f"（{sorted_worst / ext:.2%} 场景尺度）")
    print(f"       恒等/排序 比值                = {ratio:.4g}（≈1 = 恒等就是最优配对）")
    bad = []
    if sorted_worst > 0.20 * ext:
        bad.append(f"排序后距离 {sorted_worst:.4g} 超过场景尺度的 20% ⇒ 两侧的**点集分布不同**"
                   f"（多半是某一侧已致密化/剪枝过）⇒ 逐点替换不成立")
    if ratio > 3.0:
        bad.append(f"恒等/排序 比值 {ratio:.4g} > 3 ⇒ 恒等配对远非最优 ⇒ **点序对不上**")
    if bad:
        msg = "[拒绝] 逐点对应性不成立：\n        - " + "\n        - ".join(bad)
        if not args.allow_order_mismatch:
            raise SystemExit(msg + "\n        （真要用 --allow-order-mismatch 放行，请先看一眼上面三个数）")
        print("       ⚠ " + msg.replace("[拒绝] ", "") + "  ← 被 --allow-order-mismatch 放行")
    else:
        print("       ✓ 通过：点集分布一致、恒等配对近似最优")

    # ---- 顺带读数：视觉短训到底把起点改了什么 ----
    sc_g, sc_o = npy(mp_g[IDX_SCALING]), npy(mp_o[IDX_SCALING])
    ro_g, ro_o = npy(mp_g[IDX_ROTATION]), npy(mp_o[IDX_ROTATION])
    op_g = npy(mp_g[IDX_OPACITY]).reshape(-1)
    op_o = npy(mp_o[IDX_OPACITY]).reshape(-1)

    def q(k, v, fmt=".6g"):
        # 注意是 `format(v, fmt)` 不是 `fmt.format(v)` —— 后者对 ".6g" 这种没有占位符的
        # 格式串只会把字符串原样返回，打出来就是字面量 ".6g"，把数字全吞掉。
        print(f"          {k:<22} {format(v, fmt)}")

    print()
    print("  [几何侧差异]（几何取自第一份，opacity 取自第二份；下面是两份**原始**的差）")
    for nm, arr in (("|Δxyz| 中位", np.median(disp)),
                    ("|Δxyz| p90", np.percentile(disp, 90)),
                    ("|Δxyz| p99", np.percentile(disp, 99)),
                    ("|Δxyz| max", disp.max())):
        q(nm, float(arr))
    # scaling 在 3DGS 里是 log 尺度 ⇒ 差就是对数比
    dsc = (sc_o - sc_g)
    print(f"          {'Δlog 尺 中位':<22} {np.median(dsc):.6g}"
          f"（= 尺度 ×{np.exp(np.median(dsc)):.4f}）")
    print(f"          {'Δlog 尺 |中位|':<22} {np.median(np.abs(dsc)):.6g}"
          f"（= 尺度 ×{np.exp(np.median(np.abs(dsc))):.4f}）")
    ang = 2.0 * np.degrees(np.arccos(np.clip(
        np.abs((_norm_q(ro_g.astype(np.float64)) * _norm_q(ro_o.astype(np.float64))).sum(1)),
        0.0, 1.0)))
    print(f"          {'Δ朝向 中位(度)':<22} {np.median(ang):.6g}")
    print(f"          {'Δ朝向 p90(度)':<22} {np.percentile(ang, 90):.6g}")

    sg, so = 1 / (1 + np.exp(-op_g)), 1 / (1 + np.exp(-op_o))
    print(f"          {'opacity(σ) 均值 甲':<22} {sg.mean():.6g}"
          f"   乙 {so.mean():.6g}")
    print(f"          {'|Δopacity(σ)| 中位':<22} {np.median(np.abs(so - sg)):.6g}")
    print(f"          {'|Δopacity(σ)| p90':<22} {np.percentile(np.abs(so - sg), 90):.6g}")
    n_up = int((so > sg).sum())
    print(f"          {'σ 上升的点占比':<22} {n_up / P:.4%}（{n_up}/{P}）")
    if np.allclose(op_g, op_o):
        raise SystemExit("[拒绝] 两侧 opacity **逐位相同** ⇒ 「换 opacity」是个空操作，"
                         "这个节点花两小时只会得到与 0 档一样的结果")

    # ---- 组装产物 ----
    new_mp = list(mp_g)
    new_mp[IDX_OPACITY] = mp_o[IDX_OPACITY]
    new_mp[IDX_OPT_DICT] = None
    print()
    print(f"  _opt_dict -> None（load_visual_geometry 把它丢进 `_`，根本不读；"
          f"保留原样会是一个形状对不上的雷 —— 与 make_step0_ckpt / make_count_matched_ckpt 同口径）")
    print(f"  first_iter -> {args.first_iter}")

    out = Path(args.out)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    torch.save((new_mp, args.first_iter), out)

    # 自证一旦抛异常，产物也必须删掉（make_count_matched_ckpt.py 栽过：文件写完、
    # 校验抛出 ⇒ 盘上留了一份从未被验证的 ckpt）
    try:
        _selfcheck(out, mp_g, mp_o, args.first_iter, P)
    except Exception:
        out.unlink(missing_ok=True)
        print("[混合] 自证过程本身抛异常 ⇒ 已删除产物，不留未验证的半成品")
        raise
    print(f"[write] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _selfcheck(out, mp_g, mp_o, first_iter, P):
    """重新 load 回读，逐项核对：几何项必须与几何侧**逐位相同**、opacity 必须与
    opacity 侧**逐位相同**、其余项一致、first_iter 一致。不靠「我写对了」。"""
    mp2, fi2 = load_ckpt(str(out))
    if fi2 != first_iter:
        raise SystemExit(f"[拒绝] 回读 first_iter {fi2} != {first_iter}")
    ok = True
    for name, i in GEOM_SIDE.items():
        a, b = mp_g[i], mp2[i]
        if name == "active_sh_degree":
            same = (a == b)
        else:
            same = torch.equal(a.detach().cpu(), b.detach().cpu())
        print(f"       回读 {name:<20} 与几何侧逐位相同：{'✓' if same else '✗'}")
        ok &= bool(same)
    for name, i, src in (("_opacity", IDX_OPACITY, mp_o),
                         ("max_radii2D", 7, mp_g), ("xyz_grad_acc", 8, mp_g),
                         ("denom", 9, mp_g)):
        same = torch.equal(src[i].detach().cpu(), mp2[i].detach().cpu())
        print(f"       回读 {name:<20} 与预期来源逐位相同：{'✓' if same else '✗'}")
        ok &= bool(same)
    if mp2[IDX_OPT_DICT] is not None:
        print("       回读 _opt_dict               应为 None：✗")
        ok = False
    if mp2[IDX_OPACITY].reshape(-1).shape[0] != P:
        print(f"       回读 _opacity 长度 {mp2[IDX_OPACITY].reshape(-1).shape[0]} != P {P}：✗")
        ok = False
    if mp2[IDX_SLR] is not None and mp_g[IDX_SLR] is not None \
            and float(mp2[IDX_SLR]) != float(mp_g[IDX_SLR]):
        print("       回读 spatial_lr_scale 与来源不一致：✗")
        ok = False
    if not ok:
        raise SystemExit("[混合] 自证失败")
    print("       自证全部通过 ✓")


if __name__ == "__main__":
    sys.exit(main())
