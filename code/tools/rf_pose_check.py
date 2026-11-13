"""Cheap geometric sanity checks on the RF camera poses (no GPU, numpy only).

Answers two questions that could explain the "lines of dots in a rectangle"
cloud without any rendering:

1. Are the cameras spread out enough (real baseline) to triangulate a scene?
2. Do all cameras' principal rays converge on a common point?
   - If yes for the *flipped* convention -> MVSplat's world frame is consistent
     with the cameras, so the cloud SHOULD be a coherent object.
   - If no -> the flip convention (c2w[:3,1:3] *= -1) is wrong, and every view's
     depth sheet lands in a different world frame => scattered lines of dots.

Usage:  python rf_pose_check.py   (Windows or WSL, numpy only)
"""
import json
from pathlib import Path

import numpy as np

SRC = Path("datasets/RF-3DGS_dataset/blender_visual_dataset")
CX, CY = 800.0, 450.0


def ray_conv(centers, dirs, idx):
    """Least-squares point closest to a set of rays; return point + per-ray dist."""
    # each ray: c + t*d. Solve for p minimizing sum |p - (c_i + t_i d_i)|^2
    # => p = (sum (I - d_i d_i^T))^{-1} sum (I - d_i d_i^T) c_i
    ii = np.tile(np.eye(3), (len(idx), 1, 1))
    P = ii - dirs[idx][:, :, None] * dirs[idx][:, None, :]
    A = P.sum(0)
    b = (P @ centers[idx][:, :, None]).sum(0)
    p = np.linalg.solve(A, b)[:, 0]
    # distance from each ray to p
    pc = p[None, :] - centers[idx]
    d = np.linalg.norm(pc - (pc * dirs[idx]).sum(1)[:, None] * dirs[idx], axis=1)
    return p, d


def main():
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w_raw = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    n = len(c2w_raw)
    print(f"n_cameras = {n}")

    # camera centers are unchanged by the column flip
    centers = c2w_raw[:, :3, 3]
    span = centers.max(0) - centers.min(0)
    print(f"camera centers bbox (world units): {np.round(span, 3)}  "
          f"(spread {np.linalg.norm(span):.3f})")
    print(f"center-of-mass: {np.round(centers.mean(0), 3)}  "
          f"max|offset| {np.linalg.norm(centers - centers.mean(0), axis=1).max():.3f}")

    step = max(1, n // 80)                      # sample ~80 cameras
    idx = np.arange(0, n, step)

    for tag, c2w in (("no-flip", c2w_raw),
                     ("flip 1:3", c2w_raw.copy())):
        c2w = c2w.copy()
        if tag != "no-flip":
            c2w[:, :3, 1:3] *= -1
        fwd = c2w[:, :3, 2]                     # camera +Z axis (forward)
        up = c2w[:, :3, 1]
        p, d = ray_conv(centers, fwd, idx)
        # up-direction consistency: dot of each camera up with world +Y
        updot = (up @ np.array([0.0, 1.0, 0.0]))
        pair = np.array([0, min(399, n - 1)])   # wide-baseline pair
        pp, dd = ray_conv(centers, fwd, pair)
        print(f"[{tag:9s}] ray-convergence: median dist {np.median(d):7.3f} "
              f"units | 90th {np.percentile(d, 90):7.3f} | "
              f"wide-pair dist {dd.mean():7.3f}")
        print(f"        median up-dot(world +Y): {np.median(updot):+.3f} "
              f"(+1 => cameras upright like Blender, -1 => flipped upside-down)")
        if tag == "flip 1:3":
            print(f"        convergence point: {np.round(p, 3)}  "
                  f"(compare to camera COM {np.round(centers.mean(0), 3)})")

    # nearest-neighbour baseline histogram
    cd = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(cd, np.inf)
    nn = cd.min(1)
    print(f"\nadjacent-camera baseline: min {nn.min():.3f}  "
          f"median {np.median(nn):.3f}  max {nn.max():.3f} units")


if __name__ == "__main__":
    main()
