"""Check dominant-gaze pool sizes per cluster for a given K.

For K clusters (position kmeans over the 809-frame trajectory), each cluster
picks its dominant compass sector; pool = frames in that sector. We need pool
>= V for a num_views=V forward (fewer crashes the rearrange). This tells us the
safe (K, V) combos.

Usage:  RF_NUM_CLUSTERS=16 python rf_gaze_poolsize.py   (Windows)
"""
import json
import os
from pathlib import Path

import numpy as np

SRC = Path("datasets/RF-3DGS_dataset/blender_visual_dataset")


def quantize_gaze(look):
    ang = np.arctan2(look[:, 1], look[:, 0])
    return np.round(ang / (np.pi / 4)).astype(int) % 8


def kmeans(points, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        d2 = np.stack([((points - c) ** 2).sum(1) for c in centers]).min(0)
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.stack(centers)
    for _ in range(iters):
        lab = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
        for i in range(k):
            m = lab == i
            if m.any():
                centers[i] = points[m].mean(0)
    return lab, centers


def main():
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    look = c2w[:, :3, 2]
    pos = c2w[:, :3, 3]
    K = int(os.environ.get("RF_NUM_CLUSTERS", 16))

    labels, _ = kmeans(pos, K, seed=0)
    gazes = quantize_gaze(look)
    pools = []
    top2 = []
    for i in range(K):
        idx = np.where(labels == i)[0]
        k = gazes[idx]
        counts = np.bincount(k, minlength=8)
        dom = counts.argmax()
        pool = idx[k == dom]
        pools.append(len(pool))
        top2.append(int(np.sort(counts)[-2]))   # 2nd-largest sector pool
    pools = np.array(pools)
    top2 = np.array(top2)
    print(f"K={K}: dominant-gaze pool sizes per cluster: {pools.tolist()}")
    print(f"  min {pools.min()}  median {int(np.median(pools))}  "
          f"frac >= 4: {(pools >= 4).mean()*100:.0f}%  frac >= 6: "
          f"{(pools >= 6).mean()*100:.0f}%")
    print(f"  2nd-gaze pools: {top2.tolist()}  "
          f"min {top2.min()}  frac >= 4: {(top2 >= 4).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
