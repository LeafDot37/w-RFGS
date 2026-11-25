# -*- coding: utf-8 -*-
"""make_step0_ckpt.py —— 把 MVSplat 前馈点云直接做成 **first_iter = 0** 的 RRF 物理起点。

为什么需要它
============
「视觉质量 → RF 质量」的因果斜率，最左端必须是 **零视觉优化** 那一档：
只用 MVSplat 前馈吐出来的几何，一步 3DGS 都不训。服务器上有 500/1000/…/30000 步的
ckpt（或能现跑），但**没有第 0 步**：`train.py` 的循环是 `range(first_iter+1, iterations+1)`，
`--iterations 0` 一步都不走，`--iterations 1` 又已经吃过一次梯度 ⇒ 都拿不到真正的第 0 步。

所以本工具**复刻 `train.py` 在进循环之前做的那件事**（`GaussianModel.create_from_pcd`），
把结果按 `capture()` 的顺序打包成一个 ckpt。用的全是引擎自己的代码 ⇒
尺度/朝向/opacity 的约定**不是照抄的，是同一个函数产出的**。

`create_from_pcd` 吃的是**简单格式** ply（x/y/z/nx/ny/nz/rgb，无 scale/rot/opacity），
即 `mvsplat_clouds/init.ply`；它会用 `distCUDA2`(CUDA) 生成各向同性尺度、单位四元数朝向、
`inverse_sigmoid(0.1)` 的 opacity。**所以这一档的协方差是引擎给的球，不是 MVSplat 预测的**
—— 这与历史所有臂的起点一致（协方差那条通道在本仓从未接通过，见记忆「协方差初始化判决」）。

`spatial_lr_scale` 为什么必须从**别的 ckpt** 拿
==============================================
它是 `capture()` 的第 12 项，`load_visual_geometry` 会存进 `gaussians.spatial_lr_scale`，
而 RRF 的 A× 标量头 `use_dist=True` ⇒ `_dist(xyz, cam_center, spatial_lr_scale)` **真的读它**。
同一个场景同一批相机，这个数是常量（实测 8.437305）。自己重算（`getNerfppNorm`）等于
引入一条可能分叉的路径，所以默认**从参考 ckpt 里直接取**：`--ref_ckpt`。
两臂拿不到同一个数，后面比较的就不是同一件事。

`_opt_dict` 为什么写 None
========================
`capture()` 第 11 项是 `optimizer.state_dict()`，但此刻还没 `training_setup`（没有优化器）。
`load_visual_geometry` 把这一项丢进 `_`，**根本不读**。`make_count_matched_ckpt.py`
对同一件事有先例，同样写 None。文件头那里还留了更详细的说明。

用法
====
    <PY> tools/make_step0_ckpt.py \
        --init_ply /data/yuexf/3dgs/results/mvsplat_clouds/init.ply \
        --ref_ckpt /data/yuexf/3dgs/results/run_20260907_185429_parity档_视觉30k基座/rf-3dgs_1/chkpnt30000.pth \
        --sh_degree 3 --out <TOP>/物理起点/visstep0/chkpnt0.pth

退出码 0 = 成功；非 0 = 拒绝（参数不合理 / 形状不符），**绝不静默写半成品**。
写完会**重新 load 一遍并逐项打形状**自证，不靠「我写对了」。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import os
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import torch                                                          # noqa: E402

from splat.scene.gaussian_model import GaussianModel                  # noqa: E402
from splat.scene.dataset_readers import fetchPly                      # noqa: E402


def read_ref_spatial_lr_scale(ref_ckpt: Path):
    """从参考 ckpt 取 spatial_lr_scale（capture() 第 12 项）与它的 first_iter。"""
    ckpt = torch.load(str(ref_ckpt), map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, (tuple, list)) and len(ckpt) == 2):
        raise SystemExit(f"[拒绝] {ref_ckpt} 不是 (model_params, first_iter) 二元组")
    model_params, first_iter = ckpt
    if len(model_params) != 12:
        raise SystemExit(f"[拒绝] 参考 ckpt 的 model_params 有 {len(model_params)} 项，"
                         f"与 capture() 的 12 项不符 —— 引擎改过吗？")
    return model_params[11], model_params[1].shape[0], first_iter


def main():
    ap = argparse.ArgumentParser(description="由 MVSplat 前馈点云构造 first_iter=0 的 RRF 物理起点")
    ap.add_argument("--init_ply", required=True, help="简单格式 ply（x/y/z/nx/ny/nz/rgb）")
    ap.add_argument("--out", required=True, help="输出 ckpt 路径（.pth）")
    ap.add_argument("--ref_ckpt", default="",
                    help="参考 ckpt：从它取 spatial_lr_scale（同场景同相机必须一致）")
    ap.add_argument("--spatial_lr_scale", type=float, default=None,
                    help="不传 --ref_ckpt 时的显式值（传了就以此为准，并在日志里写明来源）")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--expect_points", type=int, default=0,
                    help=">0 时断言初试点数等于它（防止喂错 init.ply）")
    ap.add_argument("--device", default="cuda", help="distCUDA2 要 CUDA")
    args = ap.parse_args()

    init_ply = Path(args.init_ply)
    if not init_ply.is_file():
        raise SystemExit(f"[拒绝] init_ply 不存在：{init_ply}")

    # ---- spatial_lr_scale：只能有一个来源，取不到就报错，不许猜 ----
    s_lr, ref_P, ref_it = None, None, None
    if args.ref_ckpt:
        s_lr, ref_P, ref_it = read_ref_spatial_lr_scale(Path(args.ref_ckpt))
        print(f"[ref] {args.ref_ckpt}")
        print(f"      参考臂 first_iter={ref_it}  点数={ref_P}  spatial_lr_scale={s_lr!r}")
    if args.spatial_lr_scale is not None:
        if s_lr is not None and abs(s_lr - args.spatial_lr_scale) > 1e-9:
            raise SystemExit(
                f"[拒绝] --spatial_lr_scale={args.spatial_lr_scale} 与参考 ckpt 里的 "
                f"{s_lr} 不一致。两者不一致说明场景/相机不同，用它建的第 0 档"
                f"与其它档不是同一个坐标系 ⇒ 斜率无意义。")
        s_lr = args.spatial_lr_scale
        if not args.ref_ckpt:
            print(f"[warn] spatial_lr_scale={s_lr} 来自命令行，**没有参考 ckpt 佐证**")
    if s_lr is None:
        raise SystemExit("[拒绝] 必须给 --ref_ckpt 或 --spatial_lr_scale —— "
                         "该值会经 _dist() 真的进入头部的输入，不许默认成 0/1。")

    # ---- 复刻 train.py 进循环前的那一步 ----
    pcd = fetchPly(str(init_ply))
    if pcd is None:
        raise SystemExit(f"[拒绝] fetchPly 读不出 {init_ply}（简单格式？）")
    P0 = len(pcd.points)
    print(f"[init] {init_ply} 点数 = {P0}")
    if args.expect_points and P0 != args.expect_points:
        raise SystemExit(f"[拒绝] 初试点数 {P0} != --expect_points {args.expect_points}"
                         "（多半喂错了 init.ply）")

    g = GaussianModel(args.sh_degree)
    g.create_from_pcd(pcd, s_lr)
    P = g.get_xyz.shape[0]
    if P != P0:
        raise SystemExit(f"[拒绝] create_from_pcd 后的点数 {P} != {P0}")

    # xyz_gradient_accum / denom 平时在 training_setup() 里建；capture() 会读，这里补上。
    # （它们只影响致密化统计，RRF 阶段 --densify_until_iter 0 ⇒ 不参与，但形状要对。）
    g.xyz_gradient_accum = torch.zeros((P, 1), device=args.device)
    g.denom = torch.zeros((P, 1), device=args.device)

    model_params = (
        g.active_sh_degree,
        g._xyz, g._features_dc, g._features_rest,
        g._scaling, g._rotation, g._opacity,
        g.max_radii2D, g.xyz_gradient_accum, g.denom,
        None,                     # _opt_dict：load_visual_geometry 不读（见文件头）
        g.spatial_lr_scale,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save((model_params, 0), str(out))       # first_iter = 0 ⇒ 零视觉优化
    print(f"[write] {out}  ({out.stat().st_size / 1e6:.1f} MB)  first_iter=0")

    # ---- 自证：重新 load 一遍，逐项打形状 ----
    back, fi = torch.load(str(out), map_location="cpu", weights_only=False)
    names = ["active_sh_degree", "_xyz", "_features_dc", "_features_rest", "_scaling",
             "_rotation", "_opacity", "max_radii2D", "xyz_gradient_accum", "denom",
             "_opt_dict", "spatial_lr_scale"]
    if fi != 0:
        raise SystemExit(f"[拒绝] 回读的 first_iter={fi} != 0")
    bad = []
    for n, t in zip(names, back):
        shp = tuple(t.shape) if hasattr(t, "shape") else type(t).__name__
        print(f"    {n:<20} {shp}")
        # 只对**有行**的项查行数。`spatial_lr_scale` 是 np.float32（shape=()），
        # `active_sh_degree` 是 int，`_opt_dict` 是 None —— 三者都没有第 0 维，
        # 直接取 t.shape[0] 会 IndexError（第一版就栽在这）。
        if hasattr(t, "shape") and len(t.shape) > 0 and t.shape[0] != P:
            bad.append(n)
    if bad:
        raise SystemExit(f"[拒绝] 回读后行数不是 {P} 的项：{bad}")
    print(f"[ok] 第 0 档物理起点建好：P={P}  spatial_lr_scale={back[11]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
