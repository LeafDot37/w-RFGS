# -*- coding: utf-8 -*-
"""把阶段 1「冻结几何」老师的逐点射频属性**匹配到新点云**上，供前馈学生做属性蒸馏。

为什么需要匹配
--------------------------------------------------------------------------------
老师的点序是 MVSplat 前馈点云（1 277 062 点），学生用的是光学短训点云
（1 435 598 点），两者**不是逐点同序** ⇒ 不能像 `train_coupled_student.py` 那样
按同序号回归。两份点云是同一场景的两次重建（实测最近邻中位距离 0.054），
所以用三维最近邻把老师的属性搬过来。

空间对齐的物理含义：老师优化出来的射频外观是**空间场**上的取值，
在学生自己的点上取值，仍然是同一个场。

处理
--------------------------------------------------------------------------------
- 只保留最近邻距离 ≤ `--dist_thresh` 的点（学生比老师密，多出来的点落在
  未被老师覆盖的区域，硬搬会引入错误监督）。
- 老师的 |SH| 上界 16.7（极少数退化点），而学生的输出被 tanh 限制在 ±`--sh_bound`。
  目标先钳到同一个界内 —— 否则是在要求一个学生**不可能**表达的目标。
- 存 fp16（属性本身量级小，fp16 足够；省一半磁盘与内存）。

产出（`--out`）：
    opacity_rf [P,1]  features_rf_dc [P,1,3]  features_rf_rest [P,15,3]   （fp16）
    dist [P] fp32   mask [P] bool   meta dict（含学生 xyz 的点数，用于对齐校验）
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splat.scene.gaussian_model import GaussianModel    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEYS = ("opacity_rf", "features_rf_dc", "features_rf_rest")


def teacher_xyz(ckpt):
    """老师的几何 = 它的起点种子检查点（config.start_checkpoint）。"""
    cfg = ckpt.get("config", {})
    seed = cfg.get("start_checkpoint", "")
    if not seed or not os.path.isfile(seed):
        raise SystemExit(f"老师 config 里的 start_checkpoint 不可用：{seed!r}")
    m, _ = torch.load(seed, map_location="cpu")
    xyz = m[1]                                  # 解包顺序见 train_student.load_seed
    return xyz.detach().float().cpu(), os.path.abspath(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--cloud_ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist_thresh", type=float, default=0.5)
    ap.add_argument("--sh_bound", type=float, default=2.0,
                    help="球谐目标的钳位；学生的 SH 出口是 sh_bound·tanh，超过就不可达")
    ap.add_argument("--op_clip", type=float, default=8.0,
                    help="不透明度 logit 的钳位。学生出口是 sigmoid（无界），"
                         "不能用 SH 的界；老师的极端值 -17.9 只是退化点")
    args = ap.parse_args()

    t = torch.load(args.teacher_ckpt, map_location="cpu")
    t_xyz, seed = teacher_xyz(t)
    for k in KEYS:
        if k not in t:
            raise SystemExit(f"老师缺 {k}")
        if t[k].shape[0] != t_xyz.shape[0]:
            raise SystemExit(f"老师 {k} 点数 {t[k].shape[0]} != 几何 {t_xyz.shape[0]}")
    print(f"[match] 老师 P={t_xyz.shape[0]}  种子 {os.path.basename(seed)}")

    gm = GaussianModel(3)
    gm.load_ply(args.cloud_ply)
    s_xyz = gm.get_xyz.detach().float().cpu()
    print(f"[match] 学生 P={s_xyz.shape[0]}")

    print("[match] 建 KD 树 + 查询 …")
    tree = cKDTree(t_xyz.numpy().astype(np.float64))
    dist, idx = tree.query(s_xyz.numpy().astype(np.float64), k=1, workers=-1)
    dist = torch.from_numpy(dist).float()
    idx = torch.from_numpy(idx).long()
    mask = dist <= args.dist_thresh
    q = np.percentile(dist.numpy(), [50, 90, 95, 99])
    print(f"[match] 距离 中位 {q[0]:.4f} p90 {q[1]:.4f} p95 {q[2]:.4f} p99 {q[3]:.4f}")
    print(f"[match] 保留 {int(mask.sum())}/{mask.numel()} "
          f"({float(mask.float().mean()) * 100:.1f}%)  阈值 {args.dist_thresh}")

    out = {"dist": dist, "mask": mask,
           "meta": {"teacher_ckpt": os.path.abspath(args.teacher_ckpt),
                    "teacher_seed": seed, "teacher_points": int(t_xyz.shape[0]),
                    "student_cloud": os.path.abspath(args.cloud_ply),
                    "student_points": int(s_xyz.shape[0]),
                    "dist_thresh": args.dist_thresh,
                    "sh_bound_clip": args.sh_bound,
                    "op_clip": args.op_clip}}
    for k in KEYS:
        bound = args.op_clip if k == "opacity_rf" else args.sh_bound
        v = t[k].float()[idx]
        out[k + "_raw"] = v.half()          # 未钳位：诊断用（老师属性贴到学生点上渲一遍）
        before = float(v.abs().max())
        n_cut = int((v.abs() > bound).sum())
        v = v.clamp(-bound, bound)
        out[k] = v.half()
        print(f"[match]   {k} {tuple(out[k].shape)}  |x|max {before:.3f} -> "
              f"{float(v.abs().max()):.3f}（钳到 ±{bound}，影响 {n_cut} 点）")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[match] 写出 {args.out}  ({os.path.getsize(args.out) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
