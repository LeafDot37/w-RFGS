# -*- coding: utf-8 -*-
"""评测 CSI 头：逐高斯复散射 → CIR → 解析 CBF → jet → 三指标。

与 `eval_instances.py` 的口径一致：GT 是数据集灰度 u，先查同一张 jet LUT 再比；
逐视角算，再对实例取 mean ± pstdev。

跑法：
    python eval_csi_cbf.py --checkpoint <csi_*.pth> --start_checkpoint <seed.pth> \
        --spectra_root <spc> --meta <meta> --ids 317 318 ... --views 64
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import math
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
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_from_cir        # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--spectra_root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--start_checkpoint", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--views", type=int, default=0, help="每实例最多评多少视角（0=全部）")
    ap.add_argument("--view_list", default="", help="逗号分隔的视角名白名单")
    ap.add_argument("--dtau", default="", help="时延残差旁车文件（训练端 dtau_*.pt）")
    ap.add_argument("--tx_nr", type=int, default=1)
    ap.add_argument("--tx_nc", type=int, default=1)
    ap.add_argument("--w_uniform", type=int, default=0,
                    help="1=阵列渲染一律用 uniform w（w-盲对照评测）")
    ap.add_argument("--w_input", choices=["none", "dir", "complex"], default="none",
                    help="dir/complex: w 进头、渲染器 uniform（A1/A2 臂）")
    ap.add_argument("--save_dir", default="", help="保存逐视角 pred/gt 的 jet PNG")
    ap.add_argument("--tsv", default="", help="逐视角三指标 TSV")
    ap.add_argument("--op", choices=["cbf", "mvdr"], default="cbf")
    ap.add_argument("--dt_ns", type=float, default=0.0, help="0=用检查点里的；MVDR 建议 0.1")
    ap.add_argument("--T", type=int, default=0, help="0=用检查点里的；MVDR 建议 600")
    ap.add_argument("--loading", type=float, default=0.01)
    ap.add_argument("--db", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda"
    meta = json.load(open(args.meta, encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    if cfg.get("use_tx_w"):
        args.w_input = "dir"
    elif cfg.get("use_tx_w_complex"):
        args.w_input = "complex"
    log_scale = float(cfg.get("log_scale", 0.0))
    gaussians = GaussianModel(3)
    load_visual_geometry(args.start_checkpoint, gaussians, 3)
    if op is not None:
        gaussians._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                                requires_grad=False)
    cams, train_names, test_names = load_cameras(args.source_path)
    views = test_names if not args.views else test_names[: args.views]
    if args.view_list:
        want = set(args.view_list.split(","))
        views = [v for v in views if v in want]
    grid = CBFGrid()
    T = int(args.T or cfg.get("csi_T", 128))
    dt = float(args.dt_ns or cfg.get("csi_dt_ns", 0.5))
    dtau = (torch.load(args.dtau, map_location=dev, weights_only=False)
            if args.dtau else None)
    lut = make_jet_lut(256).to(dev)
    lpips_fn = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    print(f"[eval] ckpt iter={it}  T={T} dt={dt}  ids={args.ids} views={len(views)}")
    for iid in args.ids:
        inst = by[iid]
        vals = {"psnr": [], "ssim": [], "lpips": []}
        rows_tsv = []
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for vn in views:
                cam = cams[vn]
                w_vec, w_feat = None, None
                if args.tx_nr * args.tx_nc > 1:
                    from sim import w_codebook as _wb
                    _lab = ("uniform" if args.w_uniform
                            else str(inst.get("w_label", "uniform")))
                    if args.w_input == "dir":
                        _tp = _wb.parse_label(_lab)
                        w_feat = ([0.0, 0.0, 1.0] if _tp is None
                                  else [float(_tp[0]), float(_tp[1]), 0.0])
                        w_vec = torch.as_tensor(
                            _wb.uniform_weight(args.tx_nr, args.tx_nc),
                            dtype=torch.complex64, device=dev)
                    elif args.w_input == "complex":
                        _wv = _wb.vector_for(_lab, args.tx_nr, args.tx_nc)
                        w_feat = list(np.concatenate([_wv.real, _wv.imag]).astype(float))
                        w_vec = torch.as_tensor(
                            _wb.uniform_weight(args.tx_nr, args.tx_nc),
                            dtype=torch.complex64, device=dev)
                    else:
                        w_vec = torch.as_tensor(
                            _wb.vector_for(_lab, args.tx_nr, args.tx_nc),
                            dtype=torch.complex64, device=dev)
                x = render_cir(head, embed, gaussians, cam, inst, T, dt, grid,
                               dtau=dtau, w_vec=w_vec,
                               tx_nr=args.tx_nr, tx_nc=args.tx_nc,
                               w_feat=w_feat, dir_out=dir_out)
                x = x * math.exp(log_scale)
                if args.op == "cbf":
                    u = spec_from_cir(x, grid)
                else:
                    import scipy.linalg as sla
                    from sim.spectrum_gen import array_manifold_vector
                    a_mat = torch.as_tensor(array_manifold_vector(
                        10, grid.theta, grid.phi).reshape(100, -1),
                        dtype=torch.complex64, device=x.device)
                    xx = x.to(torch.complex128)
                    Rm = xx @ xx.conj().T
                    Rm = Rm + args.loading * (torch.trace(Rm).real / 100.0) * torch.eye(100, dtype=Rm.dtype, device=x.device)
                    Rn = Rm.cpu().numpy()
                    L = sla.cholesky(Rn, lower=True)
                    v = sla.solve_triangular(L, a_mat.cpu().numpy(), lower=True)
                    denom = (v.real ** 2 + v.imag ** 2).sum(0)
                    p = 1.0 / np.maximum(denom, 1e-30)
                    if args.db:
                        p = 10.0 * np.log10(p)
                    sp = torch.as_tensor(p.reshape(grid.hw), dtype=torch.float32, device=x.device)
                    mn, mx = sp.min(), sp.max()
                    u = (sp - mn) / torch.clamp(mx - mn, min=1e-12)
                pred = apply_jet(u.unsqueeze(0), lut)
                gt_u = np.asarray(Image.open(Path(args.spectra_root) /
                                             f"inst{iid:03d}" / f"{vn}.png")
                                  .convert("L"), np.float32) / 255.0
                gt_u = torch.as_tensor(gt_u, device=dev)
                gt = apply_jet(gt_u.unsqueeze(0), lut)
                pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
                vals["psnr"].append(float(psnr(pred, gt).mean()))
                vals["ssim"].append(float(ssim(pred, gt).mean()))
                vals["lpips"].append(float(lpips_fn(pred, gt).mean()))
                rows_tsv.append((vn, vals["psnr"][-1], vals["ssim"][-1],
                                 vals["lpips"][-1]))
                if args.save_dir:
                    for tag, t in (("pred", pred), ("gt", gt)):
                        a = (t[0].permute(1, 2, 0).clamp(0, 1) * 255).round()
                        Image.fromarray(a.to(torch.uint8).cpu().numpy()).save(
                            Path(args.save_dir) / f"{vn}_{tag}.png")
        if args.tsv:
            with open(args.tsv, "w", encoding="utf-8") as fh:
                fh.write(chr(9).join(["view", "psnr", "ssim", "lpips"]) + chr(10))
                for r in rows_tsv:
                    fh.write(chr(9).join([r[0], f"{r[1]:.4f}", f"{r[2]:.6f}", f"{r[3]:.6f}"]) + chr(10))

        import statistics as st
        print(f"inst{iid:03d} role={inst.get('role')} n={len(views)}  "
              f"PSNR {st.mean(vals['psnr']):.4f} ± {st.pstdev(vals['psnr']):.4f}  "
              f"SSIM {st.mean(vals['ssim']):.4f}  LPIPS {st.mean(vals['lpips']):.4f}")


if __name__ == "__main__":
    main()
