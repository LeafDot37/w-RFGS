# -*- coding: utf-8 -*-
"""原版 RF-3DGS 协议（3dgs_CBF_100）上的外部评测：CBF 图 PSNR/SSIM/LPIPS。

与作者发布模型同口径：
- 同一份 test_index.txt（640 视角），GT 直接用原始 jet PNG；
- 模型输出标量 u -> 逐图 min-max -> `make_jet_lut(256)` -> RGB；
- 指标函数与 3DGS 评测一致（psnr / ssim / LPIPS("vgg","0.1")）。

协议差异：TX 1x1 iso、RX 8x8 tr38901、60 GHz（由调用方通过 --M/--tx_pattern 指定）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut             # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS                  # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from splat.utils.image_utils import psnr                            # noqa: E402
from splat.utils.loss_utils import ssim                             # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw             # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

SRC = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
META = R / "results/_orig_cbf/meta.json"


def norm_u(sp: torch.Tensor) -> torch.Tensor:
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--tx_pattern", default="iso")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--views", type=int, default=0, help="0=全部 640 个测试视角")
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(META, encoding="utf-8"))
    inst = meta["instances"][0]
    cams, _, test = load_cameras(str(SRC))
    g = GaussianModel(3)
    load_visual_geometry(
        str(R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"), g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
    grid = CBFGrid(M=args.M)
    lut = make_jet_lut(256).to(dev)
    lp = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    if args.views:
        test = test[:args.views]
    print(f"[orig] ckpt={Path(args.checkpoint).name} views={len(test)} "
          f"M={args.M} tx={args.tx_pattern} iter={it}", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    pv = open(Path(args.out).with_suffix(".per_view.jsonl"), "a", encoding="utf-8")
    ps, ss, lps = [], [], []
    for vn in test:
        with torch.no_grad():
            x = render_cir(head, embed, g, cams[vn], inst, args.T, args.dt, grid,
                           tx_nr=1, tx_nc=1, dir_out=dir_out,
                           tx_pattern=args.tx_pattern)
            pr = apply_jet(norm_u(spec_raw(x, grid)).unsqueeze(0), lut).unsqueeze(0)
            # GT：服务器上只有标量 u（原版 jet 图在本机），用同一张 jet 表重建；
            # 重建与原图逐像素差 <= 1/255（见 _orig_jet_to_scalar 的量化检查）。
            gu = np.asarray(Image.open(R / "results/_orig_cbf/scalar/inst000"
                                       / f"{vn}.png").convert("L"), np.float32) / 255.0
            gt = apply_jet(torch.as_tensor(gu, device=dev).unsqueeze(0),
                           lut).unsqueeze(0)
            ps.append(float(psnr(pr, gt).mean()))
            ss.append(float(ssim(pr, gt).mean()))
            lps.append(float(lp(pr, gt).mean()))
            pv.write(json.dumps({"name": vn, "psnr": ps[-1], "ssim": ss[-1],
                                 "lpips": lps[-1]}) + "\n")
        if len(ps) % 50 == 0:
            print(f"  {len(ps)}/{len(test)}  psnr={np.mean(ps):.3f}", flush=True)
    row = {"n": len(test), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
           "lpips": float(np.mean(lps))}
    out.write(json.dumps(row) + "\n"); out.close(); pv.close()
    print(f"[orig] RESULT n={row['n']} PSNR={row['psnr']:.3f} "
          f"SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
