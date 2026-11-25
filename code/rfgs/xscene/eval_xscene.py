# -*- coding: utf-8 -*-
"""跨场景评测：留出场景上的前馈 / few-shot / scratch 三档。

- `material` 模式：头与特征都固定，测试场景零样本直接评测；
- `free` 模式：冻结共享头，只用 k 个训练视角拟合该场景的 embedding/opacity；
- `--scratch`：不加载共享头，随机初始化后用同样 k 个视角拟合同样步数（对照）。

指标与主实验同口径：灰度 u 查同一张 jet LUT 后算 PSNR/SSIM/LPIPS。
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

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splat.color_scalar import apply_jet, make_jet_lut              # noqa: E402
from splat.color_scalar_tx import head_from_cfg                     # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS                  # noqa: E402
from splat.utils.image_utils import psnr                            # noqa: E402
from splat.utils.loss_utils import ssim                             # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid                                   # noqa: E402
from rfgs.xscene.train_xscene import (XScene, load_gt, make_head, norm_spec,    # noqa: E402
                          render_cir_x)
import torch.nn as nn                                               # noqa: E402


def metrics(dev):
    lut = make_jet_lut(256).to(dev)
    lp = LPIPS("vgg", "0.1").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    return lut, lp


@torch.no_grad()
def evaluate(head, embed, op, scene, insts, views, T, dt, grid, lut, lp, dev):
    vals = {"psnr": [], "ssim": [], "lpips": []}
    for inst in insts:
        for vn in views:
            x = render_cir_x(head, embed, scene, scene.cams[vn], inst, T, dt,
                             grid, opacity=op)
            u = norm_spec(x, grid)
            gtu = load_gt(scene, inst, vn, dev)
            pred = apply_jet(u.unsqueeze(0), lut)
            gt = apply_jet(gtu.unsqueeze(0), lut)
            vals["psnr"].append(float(psnr(pred, gt).mean()))
            vals["ssim"].append(float(ssim(pred, gt).mean()))
            vals["lpips"].append(float(lp(pred, gt).mean()))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/rf/_external/xrooms")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--mode", choices=["free", "material"], default="free")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--scratch", type=int, default=0)
    ap.add_argument("--fit_views", type=int, default=0,
                    help="free 模式用多少个训练视角拟合（0=不拟合）")
    ap.add_argument("--fit_iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_lr", type=float, default=1e-3)
    ap.add_argument("--opacity_lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--dt_ns", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_views", type=int, default=0, help="0=全部 test 视角")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    names = args.scenes.split(",")
    hold = names[0]
    scene = XScene(hold, Path(args.root) / hold, dev)
    grid = CBFGrid()
    lut, lp = metrics(dev)

    if args.scratch:
        head = make_head(args.mode, args.hidden, args.depth, dev)
        from splat.color_mlp import set_scene_norm
        set_scene_norm(head, scene.xyz)
        embed = nn.Parameter(torch.zeros(scene.P, 17, device=dev).normal_(0, 0.02),
                             requires_grad=True)
        op = (nn.Parameter(torch.full((scene.P, 1), 2.2, device=dev),
                           requires_grad=True) if args.mode == "free" else None)
    else:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        head = head_from_cfg(ck["cfg"]).to(dev)
        head.load_state_dict(ck["head"])
        embed = ck["embed"][hold].to(dev)
        op = ck["opacity"].get(hold)
        op = op.reshape(-1, 1).to(dev) if op is not None else None
        if args.mode == "free":
            embed = nn.Parameter(embed.clone(), requires_grad=True)
            if op is not None:
                op = nn.Parameter(op.clone(), requires_grad=True)
        else:
            for p in head.parameters():
                p.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    if args.fit_views and args.mode == "free":
        k = min(args.fit_views, len(scene.train))
        fit_views = [scene.train[i] for i in
                     rng.choice(len(scene.train), size=k, replace=False)]
        params = []
        if isinstance(embed, nn.Parameter):
            params.append({"params": [embed], "lr": args.embed_lr})
        if isinstance(op, nn.Parameter):
            params.append({"params": [op], "lr": args.opacity_lr})
        if args.scratch:
            params.append({"params": list(head.parameters()), "lr": args.lr})
        opt = torch.optim.Adam(params)
        insts = scene.insts
        for it in range(1, args.fit_iters + 1):
            opt.zero_grad(set_to_none=True)
            inst = insts[int(rng.integers(0, len(insts)))]
            vn = fit_views[int(rng.integers(0, len(fit_views)))]
            x = render_cir_x(head, embed, scene, scene.cams[vn], inst,
                             args.T, args.dt_ns, grid, opacity=op)
            loss = (norm_spec(x, grid) - load_gt(scene, inst, vn, dev)).abs().mean()
            loss.backward()
            opt.step()
            if it % 500 == 0:
                print(f"[fit] {it}/{args.fit_iters} loss={float(loss):.4f}",
                      flush=True)

    views = scene.test if not args.eval_views else scene.test[: args.eval_views]
    vals = evaluate(head, embed, op, scene, scene.insts, views,
                    args.T, args.dt_ns, grid, lut, lp, dev)
    tag = ("scratch" if args.scratch else args.mode)
    res = {"scene": hold, "mode": tag, "fit_views": args.fit_views,
           "fit_iters": args.fit_iters,
           "n_views": len(views), "n_inst": len(scene.insts),
           "psnr": float(np.mean(vals["psnr"])),
           "psnr_std": float(np.std(vals["psnr"])),
           "ssim": float(np.mean(vals["ssim"])),
           "lpips": float(np.mean(vals["lpips"]))}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
