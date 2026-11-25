# -*- coding: utf-8 -*-
"""把测试集的 GT 图**预渲一次**，供所有实验臂复用。

GT 只由 (TX, 接收位姿, 朝向) 决定，与模型/基线无关；每个臂都现渲一遍是
纯浪费（每臂 3600 帧 × ~33 ms ≈ 2 分钟）。命名与 `render_geo2rf_spatial.py`
完全一致：`{tx:05d}_{v:03d}.png`。

用法：
    python code/rfgs/geo2rf/render_spatial_gt.py --data_root datasets/SionnaRoom \
        --split test --out results/spatial/gt_test
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
import sys
from argparse import ArgumentParser

import torch
from PIL import Image

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from corf.spatial_data import SpatialScene                    # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut         # noqa: E402
from rfgs.coupled.train_student import _Pipe                                # noqa: E402
from rfgs.geo2rf.render_geo2rf_spatial import to_uint8                     # noqa: E402


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/SionnaRoom")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    os.makedirs(args.out, exist_ok=True)
    sc = SpatialScene(args.data_root)
    lut = make_jet_lut(256).cuda()
    pipe = _Pipe()
    bg = torch.zeros(3, device="cuda")
    ids = list(sc.meta["split"][args.split])
    rx_ids = list(sc.meta["rx_test"] if args.split == "test" else sc.meta["rx_train"])
    views = [(r, f) for r in rx_ids for f in range(sc.n_face)]

    n = 0
    with torch.no_grad():
        for j in ids:
            u_gt = sc.u(j)
            for v, (r, f) in enumerate(views):
                cam = sc.camera(r, f)
                u_img = sc.render_scalar(cam, u_gt, pipe, bg)
                Image.fromarray(to_uint8(apply_jet(u_img, lut))).save(
                    os.path.join(args.out, f"{j:05d}_{v:03d}.png"))
                n += 1
    print(f"[gt] {args.split} {len(ids)}TX × {len(views)}视角 = {n} 帧 -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
