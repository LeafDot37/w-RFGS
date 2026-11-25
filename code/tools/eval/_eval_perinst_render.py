# -*- coding: utf-8 -*-
"""评一个 per-instance RF-3DGS：在它自己的 test 视角上，
分别对 (a) 自己这次绘制 的 GT、(b) 同位置另一组参数 (sibling) 的 GT 算三指标。

渲染图是 3 通道 u（灰度复制），先取均值 → jet → 与 jet(GT) 比。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from splat.gaussian_renderer import render                      # noqa: E402
from splat.scene import GaussianModel, Scene                    # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut          # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS              # noqa: E402
from splat.utils.image_utils import psnr                        # noqa: E402
from splat.utils.loss_utils import ssim                         # noqa: E402

SPC = R / "datasets/RF-3DGS_cbf_params/spectra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--own", type=int, required=True)
    ap.add_argument("--sib", type=int, required=True)
    ap.add_argument("--iteration", type=int, default=40000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    a = Namespace(source_path=args.source, model_path=args.model,
                  images="images", eval=True, sh_degree=3, resolution=-1,
                  white_background=False, data_device="cuda")
    g = GaussianModel(3)
    scene = Scene(a, g, load_iteration=args.iteration, shuffle=False)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False)
    bg = torch.zeros(3, dtype=torch.float32, device=dev)
    cams = scene.getTestCameras()
    print("test cams", len(cams))
    lut = make_jet_lut(256).to(dev)
    lp = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    def gt_jet(iid, name):
        if not name.lower().endswith(".png"):
            name = name + ".png"
        p = SPC / f"inst{iid:03d}" / name
        u = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
        flat = float(u.max() - u.min()) <= 1e-6
        return torch.as_tensor(u, device=dev), flat

    vals = {"own": [], "sib": []}
    t0 = time.time()
    with torch.no_grad():
        for cam in cams:
            rd = render(cam, g, pipe, bg)["render"]           # (3,H,W)
            u = rd.mean(0).clamp(0, 1)
            pred = apply_jet(u, lut)[None]
            for key, iid in (("own", args.own), ("sib", args.sib)):
                uu, flat = gt_jet(iid, cam.image_name)
                if flat:
                    continue
                gt = apply_jet(uu, lut)[None]
                vals[key].append((float(psnr(pred, gt).mean()),
                                  float(ssim(pred, gt).mean()),
                                  float(lp(pred, gt).mean())))
    el = time.time() - t0
    out = {"model": args.model, "own": args.own, "sib": args.sib,
           "n": len(vals["own"]), "seconds": el,
           "ms_per_view": 1000 * el / max(len(cams), 1)}
    for key in ("own", "sib"):
        v = np.array(vals[key])
        out[key] = dict(psnr=float(v[:, 0].mean()), ssim=float(v[:, 1].mean()),
                        lpips=float(v[:, 2].mean()),
                        psnr_std=float(v[:, 0].std()))
    print(json.dumps(out, ensure_ascii=False))
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
