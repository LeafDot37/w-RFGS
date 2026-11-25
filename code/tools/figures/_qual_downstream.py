# -*- coding: utf-8 -*-
"""定性对比：同一视角下 GT / image-only / CIR-only 的 CBF、MVDR、Delay。"""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut              # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from splat.utils.image_utils import psnr                            # noqa: E402
from splat.utils.loss_utils import ssim                             # noqa: E402
from sim import w_codebook                                          # noqa: E402
from sim.delay_spec import delay_map                                # noqa: E402
from rfgs.csi.train_csi_cbf import (CBFGrid, render_cir, spec_from_cir_mvdr,  # noqa: E402
                           spec_raw)
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

POSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
SPEC = R / "datasets/RF-3DGS_cbf_w_params/spectra"
META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
SEED = R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"


def gt_cir(npz, vn, T, dt):
    z = np.load(npz)
    a = z[f"v{vn}_a"]; tau = z[f"v{vn}_tau"]; val = z[f"v{vn}_valid"]
    tap = np.clip(np.floor(tau[val].astype(np.float64) / dt).astype(np.int64), 0, T - 1)
    x = np.zeros((a.shape[0], T), np.complex64)
    np.add.at(x, (slice(None), tap), a[:, val])
    return torch.as_tensor(x)


def norm_u(sp):
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def jet(u, lut, dev):
    return (apply_jet(torch.as_tensor(u, device=dev).unsqueeze(0), lut)
            .permute(1, 2, 0).clamp(0, 1).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[318, 321, 324])
    ap.add_argument("--views", nargs="+", default=["00005", "00006"])
    ap.add_argument("--out", default=str(R / "results/_downstream/qual"))
    ap.add_argument("--a_ckpt", default=str(
        R / "results/_csi_cbfw/arm20k_T256/csi_20000.pth"))
    ap.add_argument("--b_ckpt", default=str(
        R / "results/_csi_cbfw/arm20k_T256_bisg_3sup/csi_20000.pth"))
    ap.add_argument("--a_label", default="old main (img256)")
    ap.add_argument("--b_label", default="Bi-SG + 3sup")
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(META, encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    cams, _, _ = load_cameras(str(POSE))
    lut = make_jet_lut(256).to(dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def load_model(ckpt):
        g = GaussianModel(3); load_visual_geometry(str(SEED), g, 3)
        head, embed, op, cfg, it = load_tx_state(ckpt, device=dev)
        if op is not None:
            g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                            requires_grad=False)
        return g, head, embed, cfg

    gA, hA, eA, cfgA = load_model(args.a_ckpt)
    gB, hB, eB, cfgB = load_model(args.b_ckpt)
    dirA = str(cfgA.get("head_type", "")).endswith("bisg")
    dirB = str(cfgB.get("head_type", "")).endswith("bisg")
    T = int(cfgA.get("csi_T", 128)); dt = float(cfgA.get("csi_dt_ns", 0.5))
    grid = CBFGrid(); a_np = grid.a_mat.detach().cpu().numpy()

    for iid in args.ids:
        inst = by[iid]
        wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                             dtype=torch.complex64, device=dev)
        npz = R / f"results/_cir_w/inst{iid:03d}.npz"
        for vn in args.views:
            v = f"{int(vn):05d}"
            with torch.no_grad():
                xg = gt_cir(npz, vn, T, dt).to(dev)
                xA = render_cir(hA, eA, gA, cams[v], inst, T, dt, grid,
                                w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dirA)
                xB = render_cir(hB, eB, gB, cams[v], inst, T, dt, grid,
                                w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dirB)
                gt_cbf = torch.as_tensor(np.asarray(
                    Image.open(SPEC / f"inst{iid:03d}" / f"{vn}.png")
                    .convert("L"), np.float32) / 255.0, device=dev)
                maps = {}
                maps["cbf"] = [gt_cbf, norm_u(spec_raw(xA, grid)),
                               norm_u(spec_raw(xB, grid))]
                maps["mvdr"] = [spec_from_cir_mvdr(xg, grid),
                                spec_from_cir_mvdr(xA, grid),
                                spec_from_cir_mvdr(xB, grid)]
                maps["delay"] = [
                    torch.as_tensor(delay_map(xg.cpu().numpy(), a_np, dt, grid.hw)[0], device=dev),
                    torch.as_tensor(delay_map(xA.cpu().numpy(), a_np, dt, grid.hw)[0], device=dev),
                    torch.as_tensor(delay_map(xB.cpu().numpy(), a_np, dt, grid.hw)[0], device=dev),
                ]
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            cols = ["GT", args.a_label, args.b_label]
            for r, op in enumerate(["cbf", "mvdr", "delay"]):
                ref = torch.as_tensor(jet(maps[op][0], lut, dev), device=dev
                                      ).permute(2, 0, 1).unsqueeze(0)
                for c, name in enumerate(cols):
                    u = maps[op][c]
                    rgb = jet(u, lut, dev)
                    t = torch.as_tensor(rgb, device=dev).permute(2, 0, 1).unsqueeze(0)
                    p = float(psnr(t, ref).mean()); s = float(ssim(t, ref).mean())
                    axes[r, c].imshow(rgb)
                    _t = name if c == 0 else (f"{name}\n"
                                              f"PSNR {p:.2f}  SSIM {s:.3f}")
                    axes[r, c].set_title(_t, fontsize=8)
                    print(f"METRIC {iid} {vn} {op} {name} {p:.3f} {s:.4f}",
                          flush=True)
                    axes[r, c].axis("off")
                axes[r, 0].set_ylabel(op.upper(), fontsize=11)
            fig.suptitle(f"inst{iid:03d} view {vn}", fontsize=11)
            fig.tight_layout()
            p = out / f"inst{iid:03d}_v{int(v):05d}.png"
            fig.savefig(p, dpi=150); plt.close(fig)
            print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
