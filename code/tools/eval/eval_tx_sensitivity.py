# -*- coding: utf-8 -*-
"""TX 参数敏感度：位置 / yaw / pitch / 载频有误差时，CBF 图掉多少。

对应部署场景：TX 坐标来自标定或 GPS，存在误差；载频有偏移。
对每个 (实例,视角) 用真值参数渲染一次作为基线，再逐项扰动，比较 jet 域 PSNR/SSIM。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                     # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw    # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry          # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut              # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from splat.utils.image_utils import psnr                            # noqa: E402
from splat.utils.loss_utils import ssim                             # noqa: E402
from sim import w_codebook                                          # noqa: E402

SPEC = R / "datasets/RF-3DGS_cbf_w_params/spectra"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", type=int, nargs="+",
                    default=[144, 196, 232, 268, 304, 318])
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out", default="results/_system/tx_sensitivity.json")
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(R / "datasets/RF-3DGS_mvdr_w_params/meta.json",
                          encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    cams, _, test = load_cameras(
        str(R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"))
    g = GaussianModel(3)
    load_visual_geometry(
        str(R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"), g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
    grid = CBFGrid(); lut = make_jet_lut(256).to(dev)
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    u_dir = np.array([1.0, 0.3, 0.2]); u_dir = u_dir / np.linalg.norm(u_dir)
    levels = [("pos_0.1m", "pos", 0.1), ("pos_0.2m", "pos", 0.2),
              ("pos_0.3m", "pos", 0.3),
              ("yaw_2deg", "yaw", 2.0), ("yaw_5deg", "yaw", 5.0),
              ("pitch_2deg", "pitch", 2.0), ("pitch_5deg", "pitch", 5.0),
              ("freq_1pct", "freq", 0.01), ("freq_3pct", "freq", 0.03)]
    res = {name: {"dpsnr": [], "dssim": []} for name, _, _ in levels}

    def render(inst, vn):
        wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                             dtype=torch.complex64, device=dev)
        x = render_cir(head, embed, g, cams[vn], inst, 256, 0.5, grid,
                       w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
        sp = spec_raw(x, grid)
        mn, mx = sp.min(), sp.max()
        u = (sp - mn) / torch.clamp(mx - mn, min=1e-12)
        return apply_jet(u.unsqueeze(0), lut).unsqueeze(0)

    n = 0
    for iid in args.ids:
        inst = by[iid]
        for vn in test[:args.views]:
            p = SPEC / f"inst{iid:03d}" / f"{vn}.png"
            if not p.exists():
                continue
            gu = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
            gt = apply_jet(torch.as_tensor(gu, device=dev).unsqueeze(0),
                           lut).unsqueeze(0)
            with torch.no_grad():
                base = render(inst, vn)
                b_p = float(psnr(base, gt).mean()); b_s = float(ssim(base, gt).mean())
                for name, kind, val in levels:
                    pert = copy.deepcopy(inst)
                    if kind == "pos":
                        pert["xyz"] = [inst["xyz"][k] + val * u_dir[k] for k in range(3)]
                    elif kind == "yaw":
                        pert["yaw_abs_rad"] = float(inst["yaw_abs_rad"]) + np.deg2rad(val)
                    elif kind == "pitch":
                        pert["pitch_abs_rad"] = float(inst["pitch_abs_rad"]) + np.deg2rad(val)
                    else:
                        pert["freq_hz"] = float(inst["freq_hz"]) * (1.0 + val)
                    pr = render(pert, vn)
                    res[name]["dpsnr"].append(float(psnr(pr, gt).mean()) - b_p)
                    res[name]["dssim"].append(float(ssim(pr, gt).mean()) - b_s)
            n += 1
    out = {k: {"dpsnr_mean": float(np.mean(v["dpsnr"])),
               "dpsnr_std": float(np.std(v["dpsnr"])),
               "dpsnr_median": float(np.median(v["dpsnr"])),
               "dpsnr_p10": float(np.quantile(v["dpsnr"], 0.10)),
               "frac_drop3db": float(np.mean(np.array(v["dpsnr"]) < -3.0)),
               "dssim_mean": float(np.mean(v["dssim"])),
               "dssim_median": float(np.median(v["dssim"])),
               "n": len(v["dpsnr"]),
               "raw": [float(x) for x in v["dpsnr"]]} for k, v in res.items()}
    path = R / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_views": n, "levels": out}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"[txsens] n={n} views", flush=True)
    for k, v in out.items():
        print(f"  {k:12s} ΔPSNR {v['dpsnr_mean']:+.3f} ± {v['dpsnr_std']:.3f}  "
              f"ΔSSIM {v['dssim_mean']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
