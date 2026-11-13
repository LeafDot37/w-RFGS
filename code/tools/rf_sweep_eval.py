#!/usr/bin/env python
"""
rf_sweep_eval.py — 评估 short_sweep 各迭代数候选，判断 RRF stage-2 的最佳短训步数。

读取 short_sweep.log 的 loss 曲线，以及每个 iteration_* 的 point_cloud.ply：
  - 高斯数 / 文件大小
  - bbox（点云空间范围）
  - "空间外点"比例：超出场景有效边界（MVSplat init 云 bbox 外扩 5% 容差）的点占比
并与原 7000 步（blender_visual_trained）和 RRF stage-2 输出（3dgs_*_100）对比。

用法（WSL，mvsplat env）：
  python /path/to/w-RFGS/rf_sweep_eval.py
"""
import os
import re
import glob
import json
import numpy as np
from plyfile import PlyData

ROOT = "/path/to/RF-3DGS/output"
LOG = os.path.join(ROOT, "short_sweep.log")

# 场景有效边界：来自 MVSplat init 云 bbox（验证时 [-21.2..26, -21.7..20.4, -9.8..9.8]），外扩 5%
SCENE_BBOX = np.array([[-21.2, -21.7, -9.8], [26.0, 20.4, 9.8]])
PAD = 1.05  # 5% 容差（高斯有一定半径）

def scene_bounds():
    c = (SCENE_BBOX[0] + SCENE_BBOX[1]) / 2.0
    h = (SCENE_BBOX[1] - SCENE_BBOX[0]) / 2.0 * PAD
    return np.array([c - h, c + h])

def load_cloud(path):
    d = PlyData.read(path)
    v = d["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    return xyz, len(xyz.data) if hasattr(xyz, "data") else len(xyz)

def analyze(path):
    d = PlyData.read(path)
    v = d["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]])
    n = xyz.shape[0]
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    b = scene_bounds()
    inside = np.all((xyz >= b[0]) & (xyz <= b[1]), axis=1)
    frac_out = 1.0 - inside.mean()
    return {
        "pts": n,
        "MB": os.path.getsize(path) / 1e6,
        "bbox": f"[{lo[0]:.1f}..{hi[0]:.1f}, {lo[1]:.1f}..{hi[1]:.1f}, {lo[2]:.1f}..{hi[2]:.1f}]",
        "out_frac": frac_out,
    }

def loss_at(iteration, log_text):
    # 找最后一个 "iter/5000 [.. Loss=xxx]" 且进度 >= iteration 的 Loss
    best = None
    for m in re.finditer(r"(\d+)/(5000) \[[^\]]*?Loss=([0-9.e+-]+)", log_text):
        if int(m.group(1)) >= iteration:
            best = float(m.group(3))
    return best

def main():
    log_text = open(LOG, encoding="utf-8", errors="replace").read() if os.path.exists(LOG) else ""

    print("=== short_sweep 各迭代候选 ===")
    rows = []
    for it in (1000, 2000, 3000, 4000, 5000):
        p = os.path.join(ROOT, "short_sweep", "point_cloud", f"iteration_{it}", "point_cloud.ply")
        if not os.path.exists(p):
            rows.append((it, None, None))
            continue
        a = analyze(p)
        loss = loss_at(it, log_text)
        rows.append((it, a, loss))
        print(f"iter {it:>5}: {a['pts']:>8,} pts  {a['MB']:6.1f}MB  bbox{a['bbox']}  "
              f"空间外点 {a['out_frac']*100:5.1f}%  loss@1000={loss:.4f}" if loss is not None else
              f"iter {it:>5}: {a['pts']:>8,} pts  {a['MB']:6.1f}MB  bbox{a['bbox']}  空间外点 {a['out_frac']*100:5.1f}%  loss=?")

    print("\n=== 参考：短训 500/1000（先前单独跑的）===")
    for name, it in (("short_500", 500), ("short_1000", 1000)):
        p = os.path.join(ROOT, name, "point_cloud", f"iteration_{it}", "point_cloud.ply")
        if os.path.exists(p):
            a = analyze(p)
            print(f"{name}: {a['pts']:>8,} pts  {a['MB']:6.1f}MB  bbox{a['bbox']}  空间外点 {a['out_frac']*100:5.1f}%")

    print("\n=== 参考：原 7000 步（blender_visual_trained）===")
    for it in (7000,):
        p = f"/path/to/w-RFGS/datasets/RF-3DGS_dataset/blender_visual_trained/point_cloud/iteration_{it}/point_cloud.ply"
        if os.path.exists(p):
            a = analyze(p)
            print(f"原7000步: {a['pts']:>8,} pts  {a['MB']:6.1f}MB  bbox{a['bbox']}  空间外点 {a['out_frac']*100:5.1f}%")

    print("\n=== 参考：RRF stage-2 输出（3dgs_MPC_100 等，iteration_40000）===")
    for f in sorted(glob.glob("/path/to/w-RFGS/datasets/RF-3DGS_dataset/RF-3DGS_trained_RRF/3dgs_*/point_cloud/iteration_40000/point_cloud.ply")):
        name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        a = analyze(f)
        print(f"{name}: {a['pts']:>8,} pts  {a['MB']:6.1f}MB  bbox{a['bbox']}  空间外点 {a['out_frac']*100:5.1f}%")

if __name__ == "__main__":
    main()
