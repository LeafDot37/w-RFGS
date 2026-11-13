# -*- coding: utf-8 -*-
"""在 NIST 数据集点云的自由空间里采样接收位姿候选（只依赖点云，不需要 Sionna）。

场只与 (点云, TX) 有关，接收位姿只用于评测渲染 ⇒ 换一批评测位姿不必重跑仿真。
输出的位置交给 `code/tools/select_nist_rx.py` 做「6 个朝向都有几何」筛选。

用法：
    python code/sim/sample_nist_rx.py --data_root datasets/NIST_lobby_spatial \
        --n 96 --out datasets/NIST_lobby_spatial/rx_candidates.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def fps_select(pts, n, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n, len(pts))
    sel = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(1, n):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets/NIST_lobby_spatial")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--clear", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_root, "meta.json"), encoding="utf-8"))
    roi = meta["roi"]
    v = PlyData.read(os.path.join(args.data_root, "cloud.ply"))["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

    lo = np.array([roi["x"][0] + 0.5, roi["y"][0] + 0.5, -0.70], np.float32)
    hi = np.array([roi["x"][1] - 0.5, roi["y"][1] - 0.5, 0.50], np.float32)
    rng = np.random.default_rng(args.seed)
    p = rng.uniform(lo, hi, size=(400_000, 3)).astype(np.float32)
    d, _ = cKDTree(pts).query(p)
    p = p[d > args.clear]
    p = p[fps_select(p, args.n, args.seed)]
    out = args.out or os.path.join(args.data_root, "rx_candidates.json")
    json.dump({"positions": p.tolist(), "clear": args.clear, "seed": args.seed},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[rx] 候选 {len(p)} -> {out}")


if __name__ == "__main__":
    main()
