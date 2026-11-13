"""Model-free pose check: photometric warping between two RF views.

If the c2w convention + intrinsics are right, then for the two cameras A->B
there is some scene depth where view A's pixels, back-projected and re-projected
into B, align with B (high NCC). Wrong pose convention => no depth gives
alignment. This is independent of MVSplat.

Usage:  python rf_warp_check.py   (Windows, numpy+PIL only)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("datasets/RF-3DGS_dataset/blender_visual_dataset")
A, B = 87, 425
W, H = 256, 144
S = 256 / 1600.0


def load_gray(path):
    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    g = im.mean(2)  # h w
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                   np.float32) / 255.0
    return g


def warp(ga, gb, RA, tA, RB, tB, K, depth):
    """Back-project A pixels at `depth`, project into B, bilinear-sample gb."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    Xc = (xs - K[0, 2]) / K[0, 0] * depth          # camera-A coords (x = right)
    Yc = (ys - K[1, 2]) / K[1, 1] * depth
    Zc = np.full_like(Xc, depth)
    Pc = np.stack([Xc, Yc, Zc], -1)                # h w 3
    Pw = Pc @ RA.T + tA                            # camera A -> world
    Pb = (Pw - tB) @ RB                            # world -> camera B
    zb = Pb[..., 2]
    valid = zb > 0.05
    u = K[0, 0] * Pb[..., 0] / np.where(valid, zb, 1) + K[0, 2]
    v = K[1, 1] * Pb[..., 1] / np.where(valid, zb, 1) + K[1, 2]
    inb = valid & (u > 1) & (u < W - 2) & (v > 1) & (v < H - 2)
    if inb.mean() < 0.05:
        return -1.0, 0.0
    # bilinear sample
    u0, v0 = np.floor(u[inb]).astype(int), np.floor(v[inb]).astype(int)
    fu, fv = u[inb] - u0, v[inb] - v0
    s = (gb[v0, u0] * (1 - fu) * (1 - fv)
         + gb[v0, u0 + 1] * fu * (1 - fv)
         + gb[v0 + 1, u0] * (1 - fu) * fv
         + gb[v0 + 1, u0 + 1] * fu * fv)
    ga_m, gb_m = ga[inb], s
    ncc = np.corrcoef(ga_m, gb_m)[0, 1]
    return ncc, inb.mean()


def main():
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])

    ga = load_gray(SRC / frames[A]["file_path"])
    gb = load_gray(SRC / frames[B]["file_path"])
    K = np.array([[800 * S, 0, 128.0],
                  [0, 800 * S, 72.0],
                  [0, 0, 1.0]])

    for tag, flip in (("no-flip", False), ("flip 1:3", True)):
        c = c2w.copy()
        if flip:
            c[:, :3, 1:3] *= -1
        RA, tA = c[A][:3, :3], c[A][:3, 3]
        RB, tB = c[B][:3, :3], c[B][:3, 3]
        print(f"[{tag:9s}]  (A={A},B={B} baseline "
              f"{np.linalg.norm(tA - tB):.2f} units)")
        best = (-1, None)
        for d in (1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15):
            ncc, cov = warp(ga, gb, RA, tA, RB, tB, K, d)
            if ncc > best[0]:
                best = (ncc, (d, cov))
            print(f"    depth {d:5.1f}: NCC {ncc:+.3f}  coverage {cov:.2f}")
        print(f"    BEST depth {best[1][0]}: NCC {best[0]:+.3f}\n")


if __name__ == "__main__":
    main()
