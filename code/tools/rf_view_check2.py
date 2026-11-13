"""Numeric alignment check: projected cloud mask vs real object mask per camera."""
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


def bb(mask):
    ys, xs = np.where(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def main():
    v = PlyData.read(OUT / "init.ply")["vertex"]
    pos = np.vstack([v["x"], v["y"], v["z"]]).T
    frames = json.load(open(SRC / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    c2w[:, :3, 1:3] *= -1

    for cam in CAMS:
        R, t = c2w[cam][:3, :3], c2w[cam][:3, 3]

        # sanity: point 5 units along the optical axis must project to image centre
        p_axis = t + R @ np.array([0.0, 0.0, 5.0])
        pc_a = (p_axis - t) @ R
        u_a = FX * pc_a[0] / pc_a[2] + CX
        v_a = FY * pc_a[1] / pc_a[2] + CY

        # project cloud
        pc = (pos - t) @ R
        z = pc[:, 2]
        m = z > 0.1
        u = FX * pc[m, 0] / z[m] + CX
        v = FY * pc[m, 1] / z[m] + CY
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        proj = np.zeros((H, W), bool)
        if inb.sum():
            xs, ys = u[inb].astype(int), v[inb].astype(int)
            proj[ys, xs] = True

        # real object mask (non-transparent, non-black)
        real = np.asarray(Image.open(SRC / frames[cam]["file_path"]).convert("RGBA"),
                          np.float32) / 255.0
        obj = (real[..., 3] > 0.5) & (real[..., :3].sum(2) > 0.1)

        inter = (proj & obj).sum()
        union = (proj | obj).sum()
        iou = inter / union if union else 0.0
        rec = inter / obj.sum() if obj.sum() else 0.0
        prec = inter / proj.sum() if proj.sum() else 0.0

        print(f"cam {cam:3d}: axis px=({u_a:.0f},{v_a:.0f}) expect (800,450) | "
              f"proj px={int(proj.sum()):6d} obj px={int(obj.sum()):6d} | "
              f"IoU={iou:.3f} rec={rec:.3f} prec={prec:.3f}")
        print(f"        real bbox {bb(obj)}  proj bbox {bb(proj)}")


if __name__ == "__main__":
    main()
