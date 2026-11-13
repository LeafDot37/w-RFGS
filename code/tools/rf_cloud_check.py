"""Verify the exported init cloud against real images.

Projects the cloud into real cameras (full 1600x900 K), z-buffers it, and
checks that at each projected pixel the CLOUD color matches the REAL image
color. High correlation => cloud sits at the right place with the right colors.
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
W, H = 1600, 900


def main():
    v = PlyData.read(OUT / "init.ply")["vertex"]
    pos = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    c2w[:, :3, 1:3] *= -1

    print(f"cloud {pos.shape[0]} pts  bbox {np.round(np.vstack([pos.min(0), pos.max(0)]),1).tolist()}")

    for cam in CAMS:
        R, t = c2w[cam][:3, :3], c2w[cam][:3, 3]
        pc = (pos - t) @ R
        z = pc[:, 2]
        m = z > 0.1
        u = FX * pc[m, 0] / z[m] + CX
        vv = FY * pc[m, 1] / z[m] + CY
        inb = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        # z-buffer nearest-wins
        xs, ys = u[inb].astype(int), vv[inb].astype(int)
        keys = xs * H + ys
        order = np.lexsort((z[m][inb], keys))
        _, first = np.unique(keys[order], return_index=True)
        near = order[first]
        cl = np.zeros((H, W, 3), np.float32)
        cl[ys[near], xs[near]] = rgb[m][inb][near] / 255.0
        real = np.asarray(Image.open(SRC / frames[cam]["file_path"]).convert("RGB"),
                          np.float32) / 255.0

        mask = cl.sum(2) > 0
        hit = int(mask.sum())
        # colour agreement on hits: correlation of cloud vs real per channel
        cc = [np.corrcoef(cl[..., k][mask], real[..., k][mask])[0, 1]
              for k in range(3)]
        # mean abs error on hits
        mae = np.abs(cl[mask] - real[mask]).mean()
        print(f"cam {cam:3d}: {hit:7d} px hit ({hit/1.44e6*100:4.1f}%)  "
              f"color corr R/G/B {cc[0]:+.2f}/{cc[1]:+.2f}/{cc[2]:+.2f}  "
              f"MAE {mae:.3f}")


if __name__ == "__main__":
    main()
