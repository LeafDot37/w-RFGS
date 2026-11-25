# -*- coding: utf-8 -*-
"""预计算前馈学生的**视觉支路**特征。

阶段 2 的射频头输入里，视觉那一支是「把每个高斯投到光学相机上采样、再聚合」。
几何钉死、相机固定 ⇒ 这一支是**常量**，一次性算完存盘，训练循环里不再碰投影。

产出（`--out`，一个 .pt）：
    xyz      [P, 3]  float32   世界坐标（也存一份，省得每次重读检查点）
    vis      [P, 12] float16   视觉支路特征（`corf/rf_head.py:VIS_CHANNELS`）
    meta     dict              来源与口径，用于溯源

投影口径**已在光学相机上对拍验证过**（`code/tools/diag/probe_projection.py`，
A 正确 r=+0.583 ≫ B y 翻转 +0.235 ≫ C 主点偏 20% +0.108）。

内存策略：逐相机流式累积（和 / 最大 / 平方和），**不**把 [P, V, 3] 整块放进内存
（P=1.01e6、V=809 ⇒ 9.7 GB，会爆）。

只写 `--out` 指定的一个文件，不改任何数据集与历史产物。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import os
import sys
import time
from argparse import Namespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))

from corf.rf_head import VIS_CHANNELS, project_and_sample   # noqa: E402

C0 = 0.28209479177387814


def load_seed(ckpt_path, device):
    """取种子的 xyz 与**激活后的**光学 DC 颜色。解包顺序与 `train_rrf_scalar.py:76-78` 逐字相同。

    ⚠ 返回的第二项是 RGB（`f_dc·C0 + 0.5` 再钳到 [0,1]），**不是**原始球谐系数。
    下游一律叫它 `opt_dc`（`corf/rf_head.py:visual_stats`）—— 名字是历史遗留。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model_params, first_iter = ckpt
    (active_sh_degree, xyz, f_dc, f_rest, scaling, rotation, opacity,
     _max_radii2D, _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params
    rgb = (f_dc.reshape(-1, 3) * C0 + 0.5).clamp(0.0, 1.0)
    return (xyz.detach().float().contiguous(), rgb.detach().float().contiguous(),
            int(first_iter))


def load_optical_cameras(source_path, resolution, eval_split, limit=None):
    """用**引擎自己的 Scene 装配路径**拿光学相机，避免另写一套口径。

    ⚠ Scene 在 `load_iteration=None` 时会 `create_from_pcd`；为免它往数据集目录里
    写 ply（`dataset_readers._external_pointcloud_path` 的兜底分支），先把
    `RF_POINTCLOUD` 指到已有的点云上。

    ⚠ **必须去重**，否则聚合出来的视觉特征带偏。本数据集的
    `transforms_test.json` 里那 161 帧的 `file_path` **指回 `train/`**
    （实测是 `train/0005.tif`、`train/0010.tif` … 每 5 张取一张），
    而 `eval=False` 时读者走 `train_cam_infos.extend(test_cam_infos)`
    （`dataset_readers.py:348-349`）⇒ 会返回 809 + 161 = **970** 个相机，
    但**只有 809 个唯一视点**，其中 161 个被数了两遍
    （mean/max/std 于是偏向那一小撮视点）。
    按 `image_name` 去重后回到 809。两条臂共用同一份缓存 ⇒ 修这个**不改变**
    「有视觉 / 无视觉」的比较，只是让描述子本身是对的。
    """
    from splat.scene import Scene
    from splat.scene.gaussian_model import GaussianModel

    os.environ.setdefault("RF_POINTCLOUD", os.path.join(
        ROOT, "results/mvsplat_clouds/init.ply"))
    a = Namespace(source_path=source_path, images="images", eval=eval_split,
                  resolution=resolution, data_device="cpu", sh_degree=3,
                  model_path=os.path.join(ROOT, "results/_student_feat"), white_background=False)
    os.makedirs(a.model_path, exist_ok=True)
    scene = Scene(a, GaussianModel(3), shuffle=False)
    cams = scene.getTrainCameras()

    n_raw = len(cams)
    seen, uniq = set(), []
    for c in cams:
        if c.image_name in seen:
            continue
        seen.add(c.image_name)
        uniq.append(c)
    if len(uniq) != n_raw:
        print(f"[feat] 相机去重 {n_raw} -> {len(uniq)}（{n_raw - len(uniq)} 个重复视点，"
              f"来自 transforms_test.json 指回 train/ 的那批）")
    return uniq if limit is None else uniq[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--source_path", default=os.path.join(
        ROOT, "datasets/RF-3DGS_dataset/blender_visual_dataset"))
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--eval_split", action="store_true",
                    help="用 llffhold 留出测试视角；默认 False = 全部视角当输入（与训练同源）")
    ap.add_argument("--n_cams", type=int, default=0, help="0 = 全部")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    t0 = time.time()
    xyz, opt_dc, first_iter = load_seed(args.ckpt, dev)
    p = xyz.shape[0]
    print(f"[feat] 种子 {os.path.basename(args.ckpt)}  绝对迭代 {first_iter}  P={p}")

    cams = load_optical_cameras(args.source_path, args.resolution, args.eval_split,
                                args.n_cams or None)
    print(f"[feat] 光学相机 {len(cams)} 个  "
          f"{cams[0].image_width}×{cams[0].image_height}  "
          f"（{time.time() - t0:.1f}s）")

    # 流式累积：[P,3] 的和 / 最大 / 平方和，外加可见计数
    s1 = torch.zeros(p, 3, device=dev)
    smax = torch.full((p, 3), float("-inf"), device=dev)
    s2 = torch.zeros(p, 3, device=dev)
    cnt = torch.zeros(p, 1, device=dev)

    t1 = time.time()
    for i, cam in enumerate(cams):
        img = cam.original_image[:3].float().to(dev).unsqueeze(0)      # [1,3,H,W]
        for a in range(0, p, args.chunk):
            b = min(a + args.chunk, p)
            samp, vis = project_and_sample(xyz[a:b], [cam], img)       # [n,1,3], [n,1]
            v = vis[:, 0]
            col = samp[:, 0, :]                                        # [n,3]
            m = v.unsqueeze(-1).float()
            s1[a:b] += col * m
            s2[a:b] += (col ** 2) * m
            smax[a:b] = torch.maximum(smax[a:b], torch.where(m > 0, col, smax[a:b]))
            cnt[a:b] += m
        if (i + 1) % 100 == 0 or i + 1 == len(cams):
            el = time.time() - t1
            print(f"[feat] {i + 1:4d}/{len(cams)}  {el:.0f}s  "
                  f"（预计还需 {el / (i + 1) * (len(cams) - i - 1):.0f}s）", flush=True)

    denom = cnt.clamp_min(1.0)
    mean = s1 / denom
    var = (s2 / denom) - mean ** 2
    std = var.clamp_min(0).sqrt()
    mx = torch.where(torch.isfinite(smax), smax, torch.zeros_like(smax))
    # 一个相机都没看见的点：三组统计全填 0，并在 vis_mask 里标出来（训练时可据此报「孤儿点」比例）
    seen = (cnt[:, 0] > 0)
    for t in (mean, mx, std):
        t[~seen] = 0.0

    vis_feat = torch.cat([mean, mx, std, opt_dc], dim=-1)              # [P,12]
    assert vis_feat.shape[1] == VIS_CHANNELS, (vis_feat.shape, VIS_CHANNELS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "xyz": xyz.cpu().float(),
        "vis": vis_feat.cpu().half(),
        "seen_mask": seen.cpu(),
        "meta": {
            "ckpt": os.path.abspath(args.ckpt), "first_iter": first_iter,
            "n_points": p, "n_cams": len(cams),
            "source_path": os.path.abspath(args.source_path),
            "resolution": args.resolution, "eval_split": bool(args.eval_split),
            "image_size": [int(cams[0].image_height), int(cams[0].image_width)],
            "vis_channels": VIS_CHANNELS,
        },
    }, args.out)

    print(f"[feat] 写出 {args.out}  ({os.path.getsize(args.out) / 2**20:.1f} MB)"
          f"  用时 {time.time() - t0:.0f}s")
    print(f"[feat] 至少被一个相机看到的点 {int(seen.sum())}/{p} "
          f"({float(seen.float().mean()) * 100:.1f}%)")
    print(f"[feat] 特征范数中位 {vis_feat.norm(dim=-1).median().item():.4f}  "
          f"（若 ≈0 说明投影/图像都没接上，别拿去训）")


if __name__ == "__main__":
    main()
