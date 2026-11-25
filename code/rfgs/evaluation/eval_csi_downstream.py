# -*- coding: utf-8 -*-
"""同一份 H 的三个下游算子：CBF / MVDR / Delay。

口径
--------------------------------------------------------------------------------
- 输入：CSI 头检查点（T/dt 从 cfg 读），逐实例的 w 由 `w_label` 解析；
- CBF：`spec_raw`，GT 用数据集里的 8bit 灰度图；
- MVDR：`spec_from_cir_mvdr`（loading=0.01，db=True），GT 由同一批路径的 CIR 现算；
- Delay：`sim.delay_spec.delay_map`，逐方向相对首达、0–200 ns，GT 同算；
- 三张图都查同一张 jet LUT 后算 PSNR/SSIM/LPIPS。

用法：
    python eval_csi_downstream.py --checkpoint ... --ids 318 ... --views 64 \
        --cir_dir results/_cir_w --out results/_downstream/csionly.jsonl
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

_CODE = str(Path(__file__).resolve().parent)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut              # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS                  # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from splat.utils.image_utils import psnr                            # noqa: E402
from splat.utils.loss_utils import ssim                             # noqa: E402
from sim import w_codebook                                          # noqa: E402
from sim.delay_spec import delay_map                                # noqa: E402
from rfgs.csi.train_csi_cbf import (CBFGrid, render_cir, spec_from_cir_mvdr,  # noqa: E402
                           spec_raw)
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402


def gt_cir(path_npz: Path, vn: str, T: int, dt: float) -> torch.Tensor:
    z = np.load(path_npz)
    a = z[f"v{vn}_a"]; tau = z[f"v{vn}_tau"]; val = z[f"v{vn}_valid"]
    tap = np.clip(np.floor(tau[val].astype(np.float64) / dt).astype(np.int64),
                  0, T - 1)
    x = np.zeros((a.shape[0], T), np.complex64)
    np.add.at(x, (slice(None), tap), a[:, val])
    return torch.as_tensor(x)


def norm_u(sp: torch.Tensor) -> torch.Tensor:
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path",
                    default="/root/autodl-tmp/rf/datasets/RF-3DGS_dataset/"
                            "training-rf-spectrum/3dgs_CBF_100")
    ap.add_argument("--spectra_root",
                    default="/root/autodl-tmp/rf/datasets/RF-3DGS_cbf_w_params/spectra")
    ap.add_argument("--meta",
                    default="/root/autodl-tmp/rf/datasets/RF-3DGS_mvdr_w_params/meta.json")
    ap.add_argument("--seed",
                    default="/root/autodl-tmp/rf/datasets/RF-3DGS_dataset/"
                            "blender_visual_trained/chkpnt30000.pth")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cir_dir", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", nargs="+", default=["cbf", "mvdr", "delay"])
    ap.add_argument("--max_delay_ns", type=float, default=200.0)
    ap.add_argument("--dtau", default="", help="时延残差旁车文件（dtau_*.pt）")
    args = ap.parse_args()

    dev = "cuda"
    meta = json.load(open(args.meta, encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    cams, _, test = load_cameras(args.source_path)
    views = test[: args.views]
    g = GaussianModel(3)
    load_visual_geometry(args.seed, g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    dtau = (torch.load(args.dtau, map_location=dev) if args.dtau else None)
    if op is not None:
        g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                        requires_grad=False)
    grid = CBFGrid()
    T = int(cfg.get("csi_T", 128)); dt = float(cfg.get("csi_dt_ns", 0.5))
    lut = make_jet_lut(256).to(dev)
    lp = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    a_np = grid.a_mat.detach().cpu().numpy()
    print(f"[down] ckpt={Path(args.checkpoint).name} T={T} dt={dt} "
          f"ids={len(args.ids)} views={len(views)}", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    allvals = {k: {"psnr": [], "ssim": [], "lpips": []} for k in args.ops}
    for iid in args.ids:
        inst = by[iid]
        wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                             dtype=torch.complex64, device=dev)
        npz = Path(args.cir_dir) / f"inst{iid:03d}.npz"
        vals = {k: {"psnr": [], "ssim": [], "lpips": []} for k in args.ops}
        for vn in views:
            vname = str(vn)
            fmt = f"{int(vn):05d}"
            with torch.no_grad():
                x = render_cir(head, embed, g, cams[vname], inst, T, dt, grid,
                               dtau=dtau, w_vec=wv, tx_nr=4, tx_nc=4,
                               dir_out=dir_out)
                xg = gt_cir(npz, fmt, T, dt).to(dev)
                pairs = {}
                if "cbf" in args.ops:
                    gtu = np.asarray(Image.open(
                        Path(args.spectra_root) / f"inst{iid:03d}" /
                        f"{fmt}.png").convert("L"), np.float32) / 255.0
                    pairs["cbf"] = (norm_u(spec_raw(x, grid)),
                                    torch.as_tensor(gtu, device=dev))
                if "mvdr" in args.ops:
                    pairs["mvdr"] = (spec_from_cir_mvdr(x, grid),
                                     spec_from_cir_mvdr(xg, grid))
                if "delay" in args.ops:
                    um, _ = delay_map(x.detach().cpu().numpy(), a_np, dt,
                                      grid.hw, max_delay_ns=args.max_delay_ns)
                    ug, _ = delay_map(xg.detach().cpu().numpy(), a_np, dt,
                                      grid.hw, max_delay_ns=args.max_delay_ns)
                    pairs["delay"] = (torch.as_tensor(um, device=dev),
                                      torch.as_tensor(ug, device=dev))
                for k, (u, ug) in pairs.items():
                    pr = apply_jet(u.unsqueeze(0), lut)
                    gt = apply_jet(ug.unsqueeze(0), lut)
                    pr4, gt4 = pr.unsqueeze(0), gt.unsqueeze(0)
                    vals[k]["psnr"].append(float(psnr(pr4, gt4).mean()))
                    vals[k]["ssim"].append(float(ssim(pr4, gt4).mean()))
                    vals[k]["lpips"].append(float(lp(pr4, gt4).mean()))
        row = {"id": iid, "n": len(views)}
        for k in args.ops:
            for m in ("psnr", "ssim", "lpips"):
                row[f"{k}_{m}"] = float(np.mean(vals[k][m]))
                allvals[k][m].append(row[f"{k}_{m}"])
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"  inst{iid:03d} " + " ".join(
            f"{k} {row[f'{k}_psnr']:.2f}/{row[f'{k}_lpips']:.3f}"
            for k in args.ops), flush=True)
    print("\n=== SUMMARY ===", flush=True)
    for k in args.ops:
        print(f"{k}: PSNR {np.mean(allvals[k]['psnr']):.3f}  "
              f"SSIM {np.mean(allvals[k]['ssim']):.4f}  "
              f"LPIPS {np.mean(allvals[k]['lpips']):.4f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
