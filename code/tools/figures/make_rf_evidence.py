# -*- coding: utf-8 -*-
"""预计算耦合学生的**射频证据支路**（一次性，不进训练循环）。

为什么需要这一支
--------------------------------------------------------------------------------
用户要的最终模型输入是**可变的**：只给光学、只给射频、或两者都给。
光学证据已经有了（`make_student_features.py` 的 12 通道视觉统计）。
射频这边，实测数据形态是：

    800 个接收位置 × 每位置 4 个朝向 = 3200 张 300×200 频谱图
    train = 640 位置 × 4，test = 160 位置 × 4   ⇒ **按位置留出**

所以本脚本做的是「**空间近邻证据编码**」：
  1. 把每张频谱图压成一个固定维向量（自适应池化 + 在 2560 张训练图上拟合 PCA）；
  2. 对每个高斯点，取空间上最近的 K 个**训练位置**，把它们的 4 朝向向量做
     反距离加权平均 ⇒ 每个点得到 4·d 维射频证据 + K 个距离。

⚠ **不是**「把点投到频谱图上采样」。那条路已被判死两次
（`corf/rf_head.py:15-19`、`docs/RF前向模型规格.md:149`）：频谱图的像素与投影方向
之间没有可验证的几何约束。本脚本只用了**相机位置**这一条几何信息，
而相机位置是 `sparse/0` 里逐张存在的确定性量。

⚠ 只用 **train 位置**做证据 ⇒ test 位置（160 个）在证据里不出现，评估是干净的。

产出（`--out`，一个 .pt）:
    xyz       [P,3]        float32   与特征缓存逐点同序（脚本会断言）
    rf_emb    [P,4*d]      float16   反距离加权的 4 朝向证据
    rf_dist   [P,K]        float32   到 K 个近邻位置的距离（米）
    rf_mask   [P]          bool      证据是否有效（恒 True；留字段给将来稀疏化）
    meta      dict

用法：
    python make_rf_evidence.py \
        --rf_source_path datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 \
        --feature_cache results/student_feats/vis_mvsplat_full_r2.pt \
        --out results/student_feats/rf_ev_mvsplat.pt --pca_dim 24 --k 4
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

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))

from corf.dual_scene import load_rf_cameras   # noqa: E402


def cam_centers(cams, device="cpu"):
    """相机中心 C = -R·T。

    `Camera.R` 是 camera-to-world，`Camera.T = -Rᵀ·C`
    （`splat/utils/graphics_utils.py:38-49` 的 `getWorld2View2`）⇒ C = -R·T。
    这条约定在本仓已用光学相机对拍验证过（`corf/rf_head.py:194-203`）。
    """
    out = []
    for c in cams:
        R = torch.as_tensor(c.R, dtype=torch.float64, device=device)
        T = torch.as_tensor(c.T, dtype=torch.float64, device=device)
        out.append(-R @ T)
    return torch.stack(out)                       # [V,3]


def encode_images(cams, size=(8, 12), device="cpu"):
    """每张频谱图 -> 8×12×3 = 288 维。低频结构足够：频谱图是大块色带，不是纹理图。"""
    h, w = size
    feats = []
    for c in cams:
        img = c.original_image[:3].float().to(device)          # [3,H,W]
        x = F.adaptive_avg_pool2d(img.unsqueeze(0), (h, w))    # [1,3,h,w]
        feats.append(x.flatten())
    return torch.stack(feats)                                  # [V,288]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--feature_cache", required=True,
                    help="只用来取 xyz（与训练/渲染逐点同序）与断点校验")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca_dim", type=int, default=24)
    ap.add_argument("--k", type=int, default=4, help="每个点取多少个近邻位置")
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cache = torch.load(args.feature_cache, map_location="cpu")
    xyz = cache["xyz"].float().contiguous()
    p = xyz.shape[0]
    print(f"[rf-ev] 点云 P={p}  ({args.feature_cache})")

    train_cams, test_cams = load_rf_cameras(
        args.rf_source_path, args.rf_images, -1, "cpu", verbose=not args.quiet)

    # ---------------------------------------------------------------- 位置分组
    centers = cam_centers(train_cams)                          # [2560,3]
    key = np.round(centers.numpy(), 5)
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, key)):
        groups.setdefault(k, []).append(i)
    sizes = sorted(len(v) for v in groups.values())
    print(f"[rf-ev] train 相机 {len(train_cams)} ⇒ {len(groups)} 个位置，"
          f"每位置图数 {sizes[0]}..{sizes[-1]}（中位 {sizes[len(sizes)//2]}）")
    if sizes[0] != sizes[-1]:
        raise SystemExit(
            f"位置分组大小不齐（{sizes[0]}..{sizes[-1]}）—— 与实测的「每位置 4 朝向」不符，"
            f"先查 sparse/0/images.txt，别硬套。")
    n_orient = sizes[0]
    pos_keys = sorted(groups)                                  # 确定性顺序
    pos_xyz = torch.tensor(pos_keys, dtype=torch.float64)      # [M,3]
    m = len(pos_keys)
    print(f"[rf-ev] 位置数 M={m}，每位置朝向数={n_orient}")

    # ---------------------------------------------------------------- 图像编码 + PCA
    enc = encode_images(train_cams, device=args.device)        # [V,288]
    enc = enc / enc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # PCA（SVD 口径）：减均值后取前 d 个右奇异向量
    mu = enc.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(enc - mu, full_matrices=False)
    d = min(args.pca_dim, Vh.shape[0])
    comps = Vh[:d]                                             # [d,288]
    proj = (enc - mu) @ comps.t()                              # [V,d]
    ev = (S ** 2).sum()
    ev_kept = (S[:d] ** 2).sum() / ev.clamp_min(1e-12)
    print(f"[rf-ev] PCA d={d}  解释方差 {float(ev_kept) * 100:.2f}%")

    # 按位置聚成 [M, n_orient, d]（同一位置内的朝向按 images.txt 出现顺序）
    per_pos = torch.zeros(m, n_orient, d)
    for j, k in enumerate(pos_keys):
        idx = groups[k]
        per_pos[j] = proj[idx]

    # ---------------------------------------------------------------- 逐点 kNN
    emb = torch.zeros(p, n_orient * d, dtype=torch.float16)
    dist = torch.zeros(p, args.k, dtype=torch.float32)
    xyz64 = xyz.double()
    t1 = time.time()
    for a in range(0, p, args.chunk):
        b = min(a + args.chunk, p)
        dd = torch.cdist(xyz64[a:b], pos_xyz)                  # [n,M]
        dv, di = torch.topk(dd, k=args.k, dim=1, largest=False)
        w = 1.0 / (dv + 1e-6) ** 2
        w = w / w.sum(dim=1, keepdim=True)                     # [n,k]
        sel = per_pos[di]                                      # [n,k,n_orient,d]
        e = (sel * w[:, :, None, None]).sum(dim=1)             # [n,n_orient*d]
        emb[a:b] = e.reshape(e.shape[0], -1).half()
        dist[a:b] = dv.float()
        if (a // args.chunk) % 10 == 9:
            el = time.time() - t1
            print(f"[rf-ev] {b:8d}/{p}  {el:.0f}s", flush=True)
    print(f"[rf-ev] kNN 用时 {time.time() - t1:.0f}s")

    # 诊断：近邻距离与证据范数 —— 若距离远大于场景尺度，证据就是噪声
    dmin = dist[:, 0]
    print(f"[rf-ev] 最近位置距离  min {dmin.min():.3f}  p50 {dmin.median():.3f}  "
          f"p95 {dmin.quantile(0.95):.3f}  max {dmin.max():.3f}（场景 xyz 跨度 "
          f"{float(xyz.max(0).values.sub(xyz.min(0).values).max()):.3f}）")
    print(f"[rf-ev] 证据范数中位 {float(emb.float().norm(dim=-1).median()):.4f}  "
          f"（若 ≈0 说明编码/分组没接上，别拿去训）")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "xyz": xyz.cpu().float(),
        "rf_emb": emb,
        "rf_dist": dist,
        "rf_mask": torch.ones(p, dtype=torch.bool),
        "meta": {
            "rf_source_path": os.path.abspath(args.rf_source_path),
            "feature_cache": os.path.abspath(args.feature_cache),
            "n_points": p, "n_train_cams": len(train_cams), "n_test_cams": len(test_cams),
            "n_positions_train": m, "n_orient": n_orient,
            "pca_dim": d, "pca_explained": float(ev_kept), "k": args.k,
            "pool_size": [8, 12], "enc_dim_raw": 288,
            "rf_emb_dim": n_orient * d,
        },
    }, args.out)
    print(f"[rf-ev] 写出 {args.out}  "
          f"({os.path.getsize(args.out) / 2**20:.1f} MB)  用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
