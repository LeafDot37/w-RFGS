# -*- coding: utf-8 -*-
"""Ours-full 在漫散射子集上的零样本评测：CBF/MVDR/ H NMSE / PDP / w' SE。"""
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
from sim import w_codebook                                          # noqa: E402
from rfgs.csi.train_csi_cbf import (CBFGrid, render_cir, spec_from_cir_mvdr,  # noqa: E402
                           spec_raw)
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

HELD = [144, 152, 160, 168, 176, 184, 196, 202, 208, 214, 220, 226,
        232, 238, 244, 250, 256, 262, 268, 274, 280, 286, 292, 298,
        304, 306, 308, 310, 312, 314, 318, 319, 320, 321, 322, 323]


def norm_u(sp: torch.Tensor) -> torch.Tensor:
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", type=int, nargs="+", default=HELD)
    ap.add_argument("--views", type=int, default=64,
                    help="0=全部；配合 --view_list 时按名单过滤")
    ap.add_argument("--view_list", default="",
                    help="逗号分隔的视角名，如 00005,00049；用于与逐场景基线同测试集")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(R / "datasets/RF-3DGS_mvdr_w_params/meta.json",
                          encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    cams, _, test = load_cameras(
        str(R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"))
    g = GaussianModel(3)
    load_visual_geometry(str(R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"),
                         g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
    grid = CBFGrid(); lut = make_jet_lut(256).to(dev)
    lp = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    T, DT = 256, 0.5
    Pt, N0, BW = 0.01, 10 ** (-17.4), 1e8
    print(f"[diff] ckpt={Path(args.checkpoint).name} ids={len(args.ids)} "
          f"views={args.views}", flush=True)

    allm = {k: [] for k in ("cbf_psnr", "cbf_ssim", "cbf_lpips",
                            "mvdr_psnr", "mvdr_ssim", "mvdr_lpips",
                            "h_nmse", "pdp_corr")}
    out = open(args.out, "a", encoding="utf-8")
    for iid in args.ids:
        inst = by[iid]
        wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                             dtype=torch.complex64, device=dev)
        d = R / f"results/_diff_subset/inst{iid:03d}"
        cir = d / "cir.npz"
        if not cir.exists():
            print(f"  inst{iid:03d} 缺 cir.npz，跳过", flush=True)
            continue
        z = np.load(cir)
        vals = {k: [] for k in allm}
        nv = 0
        if args.view_list:
            _want = set(args.view_list.split(","))
            _views = [v for v in test if v in _want]
        else:
            _views = test[:args.views] if args.views else test
        for vn in _views:
            png = d / f"{int(vn):05d}.png"
            ka = f"v{int(vn):05d}_h"
            if not png.exists() or ka not in z.files:
                continue
            with torch.no_grad():
                x = render_cir(head, embed, g, cams[vn], inst, T, DT, grid,
                               w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
                xg = torch.as_tensor(z[ka], device=dev)
                gtu = np.asarray(Image.open(png).convert("L"), np.float32) / 255.0
                gtu = torch.as_tensor(gtu, device=dev)
                pr = apply_jet(norm_u(spec_raw(x, grid)).unsqueeze(0), lut).unsqueeze(0)
                gt = apply_jet(norm_u(gtu).unsqueeze(0), lut).unsqueeze(0)
                vals["cbf_psnr"].append(float(psnr(pr, gt).mean()))
                vals["cbf_ssim"].append(float(ssim(pr, gt).mean()))
                vals["cbf_lpips"].append(float(lp(pr, gt).mean()))
                pm = apply_jet(spec_from_cir_mvdr(x, grid).unsqueeze(0), lut).unsqueeze(0)
                gm = apply_jet(spec_from_cir_mvdr(xg, grid).unsqueeze(0), lut).unsqueeze(0)
                vals["mvdr_psnr"].append(float(psnr(pm, gm).mean()))
                vals["mvdr_ssim"].append(float(ssim(pm, gm).mean()))
                vals["mvdr_lpips"].append(float(lp(pm, gm).mean()))
                den = (x.conj() * x).real.sum().clamp_min(1e-12)
                al = (x.conj() * xg).sum() / den
                vals["h_nmse"].append(float(((xg - al * x).abs() ** 2).sum()
                                            / (xg.abs() ** 2).sum()))
                pe = (x.abs() ** 2).sum(0); pe = pe / pe.sum().clamp_min(1e-30)
                pg = (xg.abs() ** 2).sum(0); pg = pg / pg.sum().clamp_min(1e-30)
                vals["pdp_corr"].append(float(torch.corrcoef(
                    torch.stack([pe, pg]))[0, 1]))
            nv += 1
        if nv == 0:
            continue
        row = {"id": iid, "n": nv}
        for k in allm:
            row[k] = float(np.mean(vals[k]))
            allm[k].append(row[k])
        out.write(json.dumps(row) + "\n"); out.flush()
        print("  inst%-3d n=%d cbf %.2f/%.3f mvdr %.2f/%.3f hNMSE %.3f pdp %.3f"
              % (iid, nv, row["cbf_psnr"], row["cbf_lpips"],
                 row["mvdr_psnr"], row["mvdr_lpips"], row["h_nmse"], row["pdp_corr"]),
              flush=True)
    print("\n=== SUMMARY (diffuse zero-shot) ===", flush=True)
    for k in allm:
        print("%s %.4f" % (k, float(np.mean(allm[k])) if allm[k] else float("nan")),
              flush=True)
    out.close()


if __name__ == "__main__":
    main()
