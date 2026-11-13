"""Diagnose cluster gaze selection: texture per compass sector per cluster.

A cluster's dominant-gaze selection (init_cloud.py) picks the most common of
the 8 compass sectors. But in this dataset every station has all 8 gazes, so
counts are ~equal and the 'dominant' is effectively arbitrary. If it lands on a
void direction (smooth sky/ground beyond the scene) MVSplat emits a far shell.

Here we measure, per cluster and per compass sector, the mean grayscale image
variance of the frames looking that way. High variance = textured content; low
= void. Compare against what init_cloud actually selected.

Usage (Windows, numpy+PIL only):  python rf_gaze_texture.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("datasets/RF-3DGS_dataset/blender_visual_dataset")


def quantize_gaze(look):
    ang = np.arctan2(look[:, 1], look[:, 0])
    return np.round(ang / (np.pi / 4)).astype(int) % 8


def variance(path, step=4):
    """Grayscale variance of a downsampled image (cheap texture proxy)."""
    im = np.asarray(Image.open(path).convert("RGB"), np.float32)
    g = im[::step, ::step].mean(2)
    return g.var()


def main():
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    look = c2w[:, :3, 2]
    pos = c2w[:, :3, 3]
    paths = [SRC / f["file_path"] for f in frames]

    # station membership: frames are grouped station-major (8 gazes per station)
    # infer stations by grouping identical camera positions
    keys, inv = np.unique(np.round(pos, 4), axis=0, return_inverse=True)
    n_sta = len(keys)
    print(f"{len(frames)} frames -> {n_sta} stations (frames/station "
          f"{len(frames) / n_sta:.0f})")
    gazes = quantize_gaze(look)

    var = np.zeros((n_sta, 8))
    for st in range(n_sta):
        fr = np.where(inv == st)[0]
        for f in fr:
            var[st, gazes[f]] = variance(paths[f])

    gmean = var.mean(0)
    print(f"global mean texture per gaze (N=0 NE=1 E=2 SE=3 S=4 SW=5 W=6 NW=7):")
    print("   " + "  ".join(f"{g:6.1f}" for g in gmean))

    # per-station: is texture uniform, or does one direction dominate?
    dom_frac = var.max(1) / var.sum(1)
    print(f"per-station dominant-gaze fraction of texture: "
          f"p10 {np.percentile(dom_frac, 10):.2f} median {np.median(dom_frac):.2f} "
          f"p90 {np.percentile(dom_frac, 90):.2f}  (1.0 => all texture in one dir)")

    # now emulate the cluster selection and check chosen gaze texture rank
    import os
    K = int(os.environ.get("RF_NUM_CLUSTERS", 8))

    def kmeans(points, k, seed=0, iters=60):  # copy from init_cloud (no hydra here)
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

    labels, centers = kmeans(pos, K, seed=0)
    print(f"\ncluster dominant-gaze vs texture-based pick (texture rank 0=best):")
    for i in range(K):
        idx = np.where(labels == i)[0]
        k = gazes[idx]
        dom = np.bincount(k).argmax()
        t = var[inv[idx]].mean(0)          # mean texture per sector in cluster
        best = int(np.argmax(t))
        rank_of_dom = int(np.where(np.argsort(-t) == dom)[0][0])
        print(f"  cluster {i:2d} (span {np.ptp(pos[idx], axis=0).max():.2f}): "
              f"dominant gaze {dom} (texture rank {rank_of_dom}/7, val {t[dom]:.0f}) | "
              f"best-texture gaze {best} (val {t[best]:.0f}) | "
              f"sector vals {np.round(t).astype(int).tolist()}")


if __name__ == "__main__":
    main()
