# -*- coding: utf-8 -*-
"""render_nerf2.py —— NeRF² 第二基线的渲染入口

产物布局与 `render.py:24-35` **逐字同构**：

    <out_dir>/test/ours_<iter>/renders/{k:05d}.png
    <out_dir>/test/ours_<iter>/gt/{k:05d}.png

这不是审美选择 —— `tools/metrics_fast.py:54-59` 是按这两条路径拼出目录再
`sorted(os.listdir(renders_dir))` 配对的。任何偏差（`%d` 而非 `%05d`、
放到 `renders_0/`、gt 数量与 renders 不等）都会让评测**静默**配错或直接
FileNotFoundError。G10 专门断言这件事。

`k` 的顺序 = `sorted(test_indices)` = `image_name` 升序 = `data["names_test"]` 的顺序
（`render.py` 走 `Scene(..., shuffle=False)`，而 `Scene` 默认 `shuffle=True` 会打乱 ——
见 `splat/scene/__init__.py:25-93`）。适配层已经把 test 侧排好并将全局序号映射
存进 `test_gidx`，两边用的是同一个序。

**网络与超参一律从 chkpnt 里读，不从命令行读**。命令行只给「在哪、渲哪些、渲到哪」。
理由：训练与渲染若从两处独立取超参，就多了一个可以静默不一致的地方，
而症状是「渲染出来的图与评测的对不上，但不报任何错」。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from baselines.nerf2 import _compat, data_adapter, heads


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_from_ckpt(ckpt, device):
    """按 chkpnt 里存的 cfg 原样重建网络与头。**不读命令行超参**。"""
    cfg = ckpt["cfg"]
    # **末端映射的形式必须从存档里读**（`heads.head_cfg()` 存它就是为了这一刻，
    # 见 `heads.py:351`）。三种形式的 alpha_raw / beta 参数名一字不差，
    # `make_head` 默认建 sigmoid ⇒ 给 leaky/clamp 存档装上 sigmoid 头
    # **不一定**当场报错，而是分成两类事故：
    #   * leaky 存档带 `sign` 这个 buffer、sigmoid 头没有它 ⇒ `load_state_dict`
    #     **直接抛** Unexpected key(s) "end.sign"（2026-09-17 实测：四个节点的
    #     RENDER 段全挂、`results.json` 一个都没产出、收尾哨兵 FAIL=1）。
    #   * clamp 存档与 sigmoid 头键完全相同 ⇒ **静默**装载成功、数值含义完全不同
    #     （一个是 softplus 后的 α，一个是带符号的 a）。探针工具
    #     `tools/probe_nerf2_n2_head.py:147-152` 就是被这一类咬过才写的注释。
    # 所以这里既修「抛错的那类」，也堵住「静默的那类」—— 与 train_nerf2.py:307 同一条闸。
    hd = cfg["head"]
    end = hd.get("end") or "sigmoid"
    head = heads.make_head(hd["mode"], lut_n=hd["lut_n"], device=device, end=end)
    net = _compat.build_network(D=cfg["net"]["D"], W=cfg["net"]["W"],
                                skips=tuple(cfg["net"]["skips"]),
                                n_channels=head.n_channels,
                                multires=cfg["net"]["multires"], device=device)
    net.load_state_dict(ckpt["net"])
    head.load_state_dict(ckpt["head"])
    net.eval()
    head.eval()
    return net, head, cfg


@torch.no_grad()
def render_one(head, renderer, data, g, chunk):
    """整张视角 → (3,H,W) ∈ [0,1] 的渲染图 与 同形状的 GT。

    分块是必须的：整图 60000 射线 × 64 采样 = 384 万点的中间激活是 31 GB
    （账在 `_compat.render_rays_chunked` 的 docstring 里）。分块只降显存峰值，
    不改变数值 —— 逐射线独立的运算，拼接顺序不变。
    """
    H, W = int(data["meta"]["H"]), int(data["meta"]["W"])
    n = H * W
    o = data["pos_t"][g].expand(n, 3)
    d = data["dirs_t"][data["orient_t"][g]].reshape(n, 3)
    tx = data["tx_t"].expand(n, 3)
    raw = _compat.render_rays_chunked(renderer, o, d, tx, chunk=chunk)
    rgb = head.value_to_rgb(head(raw), H, W)
    gt = data["targets_t"][g].reshape(H, W, 3).float().div_(255.0).permute(2, 0, 1)
    return rgb, gt


def render_split(out_dir, split, iteration, head, renderer, data, chunk, subset):
    rd = os.path.join(out_dir, split, "ours_%d" % iteration, "renders")
    gd = os.path.join(out_dir, split, "ours_%d" % iteration, "gt")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(gd, exist_ok=True)

    key = "test_gidx_t" if split == "test" else "train_gidx_t"
    gidx = data[key]
    n = int(gidx.numel())
    if subset and 0 < subset < n:
        n = subset

    t0 = time.time()
    psnrs, ssims = [], []
    from splat.utils.image_utils import psnr
    from splat.utils.loss_utils import ssim as ssim_fn

    for k in range(n):
        g = int(gidx[k])
        rgb, gt = render_one(head, renderer, data, g, chunk)
        torchvision.utils.save_image(rgb, os.path.join(rd, "%05d.png" % k))
        torchvision.utils.save_image(gt, os.path.join(gd, "%05d.png" % k))
        # 日志里的读数必须与 metrics_fast.py 对得上：那边读的是 **PNG**，
        # 即渲染图已被量化到 8 bit。这里先量化再算，否则日志会比 results.json
        # 高零点几 dB，看起来像 bug（GT 本来就是 uint8/255，量化是恒等的）。
        rq = (rgb.clamp(0, 1) * 255).round().div(255).unsqueeze(0)
        gq = (gt.clamp(0, 1) * 255).round().div(255).unsqueeze(0)
        psnrs.append(float(psnr(rq, gq)))
        ssims.append(float(ssim_fn(rq, gq)))
        if (k + 1) % 100 == 0 or k + 1 == n:
            log("[nerf2-render] %s %d/%d  PSNR %.3f  SSIM %.4f  %.1fs"
                % (split, k + 1, n, float(np.mean(psnrs)), float(np.mean(ssims)),
                   time.time() - t0))

    log("[nerf2-render] %s 完成 %d 张 → %s" % (split, n, rd))
    return {"split": split, "n_views": n, "iteration": iteration,
            "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)),
            "renders_dir": rd, "gt_dir": gd}


def main():
    ap = ArgumentParser(description="NeRF² 第二基线渲染（布局同 render.py）")
    ap.add_argument("--checkpoint", required=True, help="nerf2_iter_<i>.pth")
    ap.add_argument("--out_dir", required=True, help="节点目录（写入 <out_dir>/<split>/ours_<i>/）")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--which", choices=["test", "train", "both"], default="test",
                    help="默认只渲 test —— 训练集 2560 张对评测无用，白烧 6 分钟")
    ap.add_argument("--subset", type=int, default=0,
                    help="只渲前 N 张（G10 的秒级抽检用）；0 = 全部")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--iteration", type=int, default=-1,
                    help="目录名里的迭代号；-1 = 取 chkpnt 里记的 iter")
    args = ap.parse_args()

    _compat.assert_single_visible_device("nerf2-render")
    device = torch.device("cuda:0")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net, head, cfg = build_from_ckpt(ckpt, device)
    iteration = int(ckpt["iter"]) if args.iteration < 0 else args.iteration

    ccfg = data_adapter.Nerf2Cfg(data_dir=args.data_dir, method=cfg.get("method", ""),
                                 near=cfg["image"]["near"], far=cfg["image"]["far"],
                                 n_samples=cfg["sampling"]["n_samples"],
                                 lut_n=cfg["head"]["lut_n"])
    if args.cache and os.path.isfile(args.cache):
        data = data_adapter.load_cache(args.cache)
        for key, want in (("near", ccfg.near), ("far", ccfg.far)):
            if abs(float(data[key]) - float(want)) > 1e-9:
                raise ValueError("cache 的 %s=%g 与 chkpnt 记的 %g 不符 ⇒ 射线会与训练时不同"
                                 % (key, data[key], want))
    else:
        data = data_adapter.build(ccfg, verbose=True)

    # **必须过 to_device**：`load_cache` 解出来的是 numpy 侧的量，`*_t` 那一整族
    # （pos_t / dirs_t / orient_t / tx_t / targets_t / test_gidx_t / train_gidx_t）
    # 全在 `to_device` 里才建出来。少了这一步，`render_split` 取 `data["test_gidx_t"]`
    # 直接 KeyError —— 而**训练侧一直是对的**（`train_nerf2.py:268` 有这一行），
    # 所以这个 bug 只有在第一次真跑 RENDER 段时才会露出来（2026-09-17 实测：四个节点全挂）。
    data = data_adapter.to_device(data, device, fp16_targets=True)

    renderer = _compat.build_renderer(net, data["near"], data["far"],
                                      int(cfg["sampling"]["n_samples"]))

    log("[nerf2-render] 臂 %s | 迭代 %d | near=%.3f far=%.3f n_samples=%d | chunk=%d"
        % (head.describe(), iteration, data["near"], data["far"],
           int(cfg["sampling"]["n_samples"]), args.chunk))

    splits = ["test", "train"] if args.which == "both" else [args.which]
    out = []
    for sp in splits:
        out.append(render_split(args.out_dir, sp, iteration, head, renderer,
                                data, args.chunk, args.subset))

    man = {"checkpoint": os.path.abspath(args.checkpoint), "iteration": iteration,
           "arm": head.head_cfg(), "near": data["near"], "far": data["far"],
           "chunk": args.chunk, "splits": out}
    p = os.path.join(args.out_dir, "render_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    log("[nerf2-render] 清单 %s" % p)
    for r in out:
        log("[nerf2-render] %s: PSNR %.3f  SSIM %.4f  n=%d"
            % (r["split"], r["PSNR"], r["SSIM"], r["n_views"]))


if __name__ == "__main__":
    main()
