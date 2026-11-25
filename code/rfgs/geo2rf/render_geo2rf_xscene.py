# -*- coding: utf-8 -*-
"""跨场景实验的评测渲染：输出 `<out_root>/<split_name>/ours_<it>/{renders,gt}`，
供 `tools/metrics_fast.py` 直接算 PSNR/SSIM/LPIPS。

两种模式：
    --mode head   读检查点，逐 TX 前向 → 光栅化 → jet
    --mode const  无模型基线：每个视角输出该视角 GT 标量图的均值（jet 后为常数色）

用法：
    python code/rfgs/geo2rf/render_geo2rf_xscene.py --mode head \
        --checkpoint results/xscene/D14_zero/head_20000.pth \
        --scene D14 --split test --out_root results/xscene/D14_zero
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import math
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splat.color_scalar import (apply_jet, combine_bands, make_jet_lut,  # noqa: E402
                                to_override_color)
from splat.gaussian_renderer import render                               # noqa: E402
from rfgs.coupled.train_student import _Pipe                                          # noqa: E402
from rfgs.geo2rf.train_geo2rf_xscene import build_head, load_scene                   # noqa: E402


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/GRaF-sim_processed")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--mode", choices=("head", "const"), default="head")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split_name", default="", help="输出子目录名（默认同 --split）")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help=">0 只渲前 N 个 TX（调试）")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="合成发射功率增益：GT 与渲染都按 u→clip(g·u) 走")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    sdir = os.path.join(args.data_root, args.scene)
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    ids = list(meta["split"][args.split])
    if args.subset > 0:
        ids = ids[:args.subset]
    nv = int(meta["views"]["n"])

    gain = float(args.gain)
    lut = make_jet_lut(256).cuda()
    if args.mode == "head":
        if not args.checkpoint:
            raise SystemExit("--mode head 需要 --checkpoint")
        ck = torch.load(args.checkpoint, map_location="cuda")
        cfg = ck["config"]
        it = args.iteration or int(ck["iteration"])
        head = build_head(cfg)
        head.load_state_dict(ck["head"])
        head.eval()
        scene = load_scene(args.data_root, args.scene, cfg["tx_bands"])
        power = bool(cfg.get("power_aug", 0))
        pipe = _Pipe()
        bg = torch.zeros(3, device="cuda")
    else:
        it = args.iteration or 1
        power = False

    H = W = int(meta["views"]["img"])
    split_name = args.split_name or args.split
    odir = os.path.join(args.out_root, split_name, f"ours_{it}")
    rd, gd = os.path.join(odir, "renders"), os.path.join(odir, "gt")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(gd, exist_ok=True)

    def gt_rgb(j, k):
        """增益后的 GT：g=1 直接用现成图；否则从标量域重查 jet（与训练同一条链）。"""
        name = f"{j:05d}_{k}.png"
        if gain == 1.0:
            return np.array(Image.open(os.path.join(sdir, "gt", name)).convert("RGB"),
                            dtype=np.uint8)
        sp = np.asarray(Image.open(os.path.join(sdir, "scalar", name)).convert("L"),
                        dtype=np.float32) / 255.0
        u = torch.from_numpy(sp).cuda()
        rgb = apply_jet((u * gain).clamp(0, 1), lut)
        return (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                * 255).round().astype(np.uint8)

    with torch.no_grad():
        for j in ids:
            for k in range(nv):
                name = f"{j:05d}_{k}.png"
                if args.mode == "head":
                    P = scene.xyz.shape[0]
                    app = scene.tx_app[j].unsqueeze(0).expand(P, -1)
                    if power:
                        app = torch.cat([app, torch.full(
                            (P, 1), math.log(gain), device="cuda")], -1)
                    op, attr = head(scene.xyz, app)
                    scene.view.set_rf_scalar(op)
                    mixed = render(scene.cams[k], scene.view, pipe, bg,
                                   override_color=to_override_color(attr))["render"]
                    img = apply_jet(combine_bands(mixed, head), lut)
                    arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                           * 255).round().astype(np.uint8)
                    Image.fromarray(arr).save(os.path.join(rd, name))
                else:
                    sp = np.asarray(Image.open(os.path.join(
                        sdir, "scalar", name)).convert("L"), dtype=np.float32) / 255.0
                    # ⚠ 必须是 H×W 的常数图：`apply_jet` 只认 (H,W)/(1,H,W)，
                    # 给它一个 1×1 标量会真的输出 1×1 的 PNG（LPIPS 随后直接崩）。
                    u_ref = np.clip(sp * gain, 0.0, 1.0)
                    u = torch.full((H, W), float(u_ref.mean()), device="cuda")
                    rgb = apply_jet(u, lut)
                    arr = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                           * 255).round().astype(np.uint8)
                    Image.fromarray(arr).save(os.path.join(rd, name))
                Image.fromarray(gt_rgb(j, k)).save(os.path.join(gd, name))
    print(f"[xrender] {args.scene} {split_name} 模式={args.mode} gain={gain:.2f} "
          f"{len(ids)}×{nv} 帧 -> {odir}")


if __name__ == "__main__":
    main()
