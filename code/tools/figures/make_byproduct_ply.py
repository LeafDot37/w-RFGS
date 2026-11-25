# -*- coding: utf-8 -*-
"""make_byproduct_ply.py —— 把**物理阶段短训 ckpt 里的次生产物**导出成一个可诊断的 ply。

为什么要它
==========
`GaussianModel.capture()` 存了 12 项，其中三项是短训过程**顺带产生**的逐点统计，
从来没有任何代码读过它们：

    max_radii2D        (P,)     该点投到屏幕上的最大半径 —— 「它对渲染的贡献有多大」
    xyz_gradient_accum (P,1)    位置梯度累积 —— 3DGS 用它决定 split/clone
    denom              (P,1)    被观测次数 —— 3DGS 的梯度归一化分母

`denom` / `max_radii2D` 是**实际光栅化统计出来的**，不是模型预测的（对比 MVSplat 的
opacity 是网络学出来的表面密度），所以它们是「硬」信息。

⚠ 但有一个**存档时机**的坑，必须先说清楚（实测，非推断）
--------------------------------------------------------
`gaussian_model.py:345-347` 的 `densify_and_prune` 每次调用都会把这三项**清零**：

    self.xyz_gradient_accum = torch.zeros(...)
    self.denom               = torch.zeros(...)
    self.max_radii2D         = torch.zeros(...)

而 `densification_interval = 100`、`densify_until_iter = 15000`。短训轴上的存档点
500/1000/2000/4000/8000 **全是 100 的倍数** ⇒ 每次都正好存到「刚被清零」的状态 ⇒
实测短档这三项 **100% 是零**（1430603/1430603 个零点），**不是没信息，是被清掉了**。

只有 30000 档（> densify_until_iter，致密化早停）才留着真值：实测
`denom` 零值 0/1261959、唯一值 48（值域 1..48）；`max_radii2D` 零值 0/1261959、max 6894。

⇒ 所以本工具**只对 30000 档有意义**，且导出前会**硬断言非常数**，把「导出一个全零的
伪特征去跑诊断」这种事故挡在门外。

口径
====
字段名与 `tools/conf_error_probe.py` 的 `--conf-field` 直接对接：
    --conf-field denom         或
    --conf-field max_radii2D
`probe` 的 `load_cloud()` 只要求有 x/y/z + 该字段；没有 f_dc_* 时它会 warn 并跳过
RGB 相关性那一半（本 ply 不含颜色，属正常）。

用法
====
    <PY> tools/make_byproduct_ply.py \
        --ckpt <30k基座>/rf-3dgs_1/chkpnt30000.pth \
        --out  <诊断目录>/点云/denom.ply
退出码非 0 = 拒绝（三项全是零 / 形状不符），**绝不静默写半成品**。
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

from plyfile import PlyData, PlyElement

# capture() 的 12 项顺序（与 train_rrf_scalar.load_visual_geometry 同源）
IDX_MAX_RADII2D, IDX_GRAD_ACC, IDX_DENOM = 7, 8, 9
BYPRODUCTS = [("max_radii2D", IDX_MAX_RADII2D),
              ("grad_accum", IDX_GRAD_ACC),
              ("denom", IDX_DENOM)]


def main():
    ap = argparse.ArgumentParser(description="导出短训 ckpt 的逐点次生产物为 ply")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-constant", action="store_true",
                    help="允许导出常数特征（默认拒绝 —— 那种输入跑诊断必假）")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, (tuple, list)) and len(ck) == 2):
        raise SystemExit(f"[拒绝] {args.ckpt} 不是 (model_params, first_iter) 二元组")
    mp, first_iter = ck
    if len(mp) != 12:
        raise SystemExit(f"[拒绝] model_params 有 {len(mp)} 项，与 capture() 的 12 项不符")

    xyz = mp[1].detach().float().cpu().numpy()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SystemExit(f"[拒绝] _xyz 形状 {xyz.shape} != (P,3)")
    P = xyz.shape[0]
    print(f"[读入] {args.ckpt}")
    print(f"       first_iter={first_iter}  点数 P={P}")

    cols, verdict = {}, {}
    for name, i in BYPRODUCTS:
        t = mp[i]
        if not hasattr(t, "shape"):
            raise SystemExit(f"[拒绝] 第 {i} 项（{name}）不是张量：{type(t).__name__}")
        v = t.detach().float().cpu().numpy().reshape(-1)
        if v.shape[0] != P:
            raise SystemExit(f"[拒绝] {name} 长度 {v.shape[0]} != P {P}")
        # 「非退化」= 不是常数。常数输入跑相关性诊断必然得到 nan 或假读数。
        # 短档（500~8000）就是全零，会在这里被拦下。
        is_const = bool(np.ptp(v) == 0)
        nz = int((v == 0).sum())
        uniq = int(np.unique(v).size)
        verdict[name] = is_const
        print(f"       {name:<14} 零值 {nz}/{P}  唯一值 {uniq:>8}  "
              f"min {v.min():.4g}  max {v.max():.4g}  "
              f"{'⚠ 常数' if is_const else 'ok'}")
        if is_const and not args.allow_constant:
            print(f"       ⇒ **这一项是常数，导出去也诊断不出东西**。"
                  f"若三个都常数，多半是取到了致密化刚清零的存档点（见文件头）。")
        cols[name] = v

    if all(verdict.values()) and not args.allow_constant:
        raise SystemExit(
            "[拒绝] 三项次生产物**全是常数** ⇒ 这个 ckpt 的存档点落在致密化周期上"
            "（densify_and_prune 刚把三项清零，见文件头）。\n"
            "        短档 500/1000/2000/4000/8000 都是这种情况；只有 30000 档有真值。")

    arr = np.zeros(P, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
                             + [(n, "f4") for n, _ in BYPRODUCTS])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, _ in BYPRODUCTS:
        arr[name] = cols[name].astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out))
    print(f"[write] {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # ---- 自证：读回来逐项核对，不靠「我写对了」----
    back = PlyData.read(str(out))["vertex"]
    names = {p.name for p in back.properties}
    need = {"x", "y", "z"} | {n for n, _ in BYPRODUCTS}
    if not need <= names:
        raise SystemExit(f"[拒绝] 回读缺字段：{sorted(need - names)}")
    if len(back["x"]) != P:
        raise SystemExit(f"[拒绝] 回读点数 {len(back['x'])} != {P}")
    for name, _ in BYPRODUCTS:
        a, b = cols[name], np.asarray(back[name], np.float32)
        if not np.array_equal(a.astype(np.float32), b):
            raise SystemExit(f"[拒绝] 回读后 {name} 与内存不逐位相同")
    print(f"[ok] 回读逐位一致，字段 = {sorted(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
