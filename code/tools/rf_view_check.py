"""Diagnostic: project the init cloud into RF cameras and compare to the real images.

Answers "is the MVSplat init geometry correct and in the right world frame?" by
projecting the cloud's points into a few RF cameras (full 1600x900 intrinsics)
and saving both the point projection and the actual composite image for visual
comparison.

Usage (WSL, from repo root):
    python rf_view_check.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

_ROOT = Path(__file__).resolve().parents[2]
SRC = _ROOT / "datasets/RF-3DGS_dataset/blender_visual_dataset"
OUT = _ROOT / "code/outputs/rf_init"   # init_cloud 阶段 A 从 code/ 运行后的点云落点
CAMS = [0, 200, 400, 600]
FX = FY = 800.0
CX, CY = 800.0, 450.0


def load_cloud(path):
    v = PlyData.read(path)["vertex"]
    pos = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    return pos, rgb


def load_poses():
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    c2w[:, :3, 1:3] *= -1  # same flip as init_cloud.py
    return c2w


def project(pos, rgb, c2w):
    """Pinhole project world points -> (u, v, z, rgb_subset) (y down, z>0 front)."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (pos - t) @ R                      # p_cam = R^T (p - t), rows
    z = pc[:, 2]
    m = z > 0.1
    u = FX * pc[:, 0] / z + CX
    v = FY * pc[:, 1] / z + CY
    return u[m], v[m], z[m], rgb[m]


def zrender(u, v, z, rgb, H=900, W=1600):
    """Z-buffered nearest-wins point render at full res."""
    img = np.zeros((H, W, 3), np.uint8)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() == 0:
        return img, 0
    xs = u[inb].astype(np.int64)
    ys = v[inb].astype(np.int64)
    zs = z[inb]
    keys = xs * H + ys
    order = np.lexsort((zs, keys))
    _, first = np.unique(keys[order], return_index=True)
    near = order[first]
    img[ys[near], xs[near]] = rgb[inb][near]
    return img, inb.sum()


def composite_tif(path):
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    out = im[..., :3] * im[..., 3:4]                      # onto black
    return (out * 255).astype(np.uint8)


def main():
    pos, rgb = load_cloud(OUT / "init.ply")
    c2w = load_poses()
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    print(f"cloud {pos.shape[0]} pts  bbox {np.round(np.vstack([pos.min(0), pos.max(0)]), 1).tolist()}")

    for cam in CAMS:
        u, v, z, rgb_sub = project(pos, rgb, c2w[cam])
        img, npix = zrender(u, v, z, rgb_sub)
        Image.fromarray(img).save(OUT / f"proj_cam{cam}.png")
        # real image for the same camera
        real = composite_tif(SRC / frames[cam]["file_path"])
        Image.fromarray(real).save(OUT / f"real_cam{cam}.png")

        # rough alignment: fraction of projected pixels that hit a non-black
        # pixel of the real image (object region)
        obj = real.sum(2) > 30
        hit = obj[img.sum(2) > 0]
        frac = hit.mean() if len(hit) else float("nan")
        print(f"cam {cam}: {len(u)} pts in front, {npix} px projected, "
              f"{len(hit)} on non-black real px, object-hit {frac:.2f}")

    print(f"\nsaved proj_cam*.png / real_cam*.png -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
