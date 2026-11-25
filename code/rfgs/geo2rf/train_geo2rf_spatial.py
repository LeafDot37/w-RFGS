# -*- coding: utf-8 -*-
"""空间射频场的前馈头：给定发射源位置，一次前向输出**整个房间表面**的 RRF 场。

与固定网关时代（`train_geo2rf_xscene.py`）的区别
--------------------------------------------------------------------------------
    固定网关   每个 TX 只有 4 张从网关看出去的频谱图，头学「TX → 4 张图」
    空间场     每个 TX 有 72000 个表面点的 dBm，及 36 个接收位姿 × 6 朝向
                ⇒ 头学「TX → 全场」，渲染只用来落地成任意接收机看到的图

损失用的是**逐点场误差**（全 72000 点，稠密梯度），外加若干张随机视角的
图像损失把光栅化口径对齐。只做图像损失时，被遮住的点拿不到梯度，低 N 下学不动。

发射源相对几何（`use_rel`）默认开：把 Δ=xyz−tx、距离、单位方向显式喂进去。
这是本数据集的主变化轴（自由空间里 u 几乎只由 |Δ| 决定）；该特征是否真的
带来收益由同预算消融臂回答，训练日志里不要写成结论。

用法：
    python code/rfgs/geo2rf/train_geo2rf_spatial.py --data_root datasets/SionnaRoom \
        --model_path results/spatial/N16 --n_tx 16 --iterations 20000
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from corf.geo_head import GeoToRfHead                       # noqa: E402
from corf.rf_head import xyz_norm_stats                    # noqa: E402
from corf.spatial_data import SpatialScene                 # noqa: E402
from corf.spatial_extra import (VIS_KEYS, build_extra, load_vis,   # noqa: E402
                                nearest_other)
from splat.color_scalar import apply_jet, make_jet_lut     # noqa: E402
from rfgs.coupled.train_student import _Pipe, rgb_loss                  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tx_features(tx: torch.Tensor, bands: int) -> torch.Tensor:
    """发射源位置 → 条件向量。与固定网关臂同一套（坐标、γ(x)、r、az/el、log r）。"""
    from splat.color_mlp import fourier_feat
    r = tx.norm(dim=-1, keepdim=True)
    d = tx / r.clamp_min(1e-9)
    az = torch.atan2(d[:, 1], d[:, 0])
    el = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    return torch.cat([
        tx, fourier_feat(tx, bands), r,
        torch.sin(az)[:, None], torch.cos(az)[:, None],
        torch.sin(el)[:, None], torch.cos(el)[:, None],
        torch.log1p(r),
    ], dim=-1)


def app_dim_of(bands: int) -> int:
    return 3 + 6 * int(bands) + 1 + 4 + 1


def build_head(cfg: dict) -> GeoToRfHead:
    return GeoToRfHead(
        app_dim=cfg["app_dim"], hidden=cfg["hidden"], n_layers=cfg["n_layers"],
        pe_bands=cfg["pe_bands"], chunk=cfg["chunk"],
        pe_mean=torch.tensor(cfg["pe_mean"], dtype=torch.float32),
        pe_std=torch.tensor(cfg["pe_std"], dtype=torch.float32),
        app_mean=torch.tensor(cfg["app_mean"], dtype=torch.float32),
        app_std=torch.tensor(cfg["app_std"], dtype=torch.float32),
        opacity_init=cfg["opacity_init"],
        out_kind=cfg["out_kind"], n_bands=cfg["n_bands"], init_u=cfg["init_u"],
        use_rel=cfg.get("use_rel", False),
        pe_rel_bands=cfg.get("pe_rel_bands", 0)).cuda()


def fps_order(pts: np.ndarray) -> list[int]:
    from tools.graf_prepare import fps_select
    return [int(i) for i in fps_select(pts, len(pts))]


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/SionnaRoom")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_tx", type=int, default=0, help="只用 FPS 顺序前 N 个训练 TX；0=全部")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--pe_bands", type=int, default=6)
    ap.add_argument("--tx_bands", type=int, default=4)
    ap.add_argument("--use_rel", type=int, default=1)
    ap.add_argument("--pe_rel_bands", type=int, default=4)
    # 逐点附加输入（见 corf/spatial_extra.py）：
    #   ref  最近训练源的整场 u —— 逐点场强记忆
    #   vis  光学几何可见性 6 通道 —— 遮挡
    ap.add_argument("--use_ref", type=int, default=0)
    ap.add_argument("--use_vis", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--img_weight", type=float, default=0.5,
                    help="图像损失权重；逐点 L1 的权重恒为 1")
    ap.add_argument("--views_per_step", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.9)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_iterations", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations 必须 > 0")
    from splat.utils.general_utils import safe_state
    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.model_path, exist_ok=True)

    t0 = time.time()
    sc = SpatialScene(args.data_root)
    tr_all = fps_order(sc.tx[sc.meta["split"]["train"]].numpy())
    tr_all = [sc.meta["split"]["train"][i] for i in tr_all]
    ids = tr_all[:args.n_tx] if args.n_tx > 0 else tr_all
    if len(ids) < (args.n_tx or 0):
        raise SystemExit(f"训练 TX 不足：要 {args.n_tx}，只有 {len(tr_all)}")
    rx_ids = list(sc.meta["rx_train"])
    log(f"[sp] P={sc.P}  TX 训 {len(ids)}/{len(tr_all)}  测试 TX {len(sc.meta['split']['test'])}  "
        f"RX 训 {len(rx_ids)}  面 {sc.n_face}  H=W={sc.H}")

    # 逐点标量场：一次全部载入 CPU（200×72000×4B ≈ 58 MB），用时再上 GPU
    u_all = torch.stack([sc.u(j, device="cpu") for j in range(sc.tx.shape[0])])  # [T,P]

    xyz_pool = sc.xyz.detach().cpu()[::10]
    pe_mean, pe_std = xyz_norm_stats(xyz_pool)
    # 逐点附加输入：参考场 + 可见性（没有就整段不拼，旧配置数值一字不变）
    ref_of = nearest_other(sc.tx.numpy(), ids) if args.use_ref else {}
    vis = load_vis(args.data_root, VIS_KEYS, ids) if args.use_vis else {}
    n_extra = (1 if args.use_ref else 0) + (len(VIS_KEYS) if args.use_vis else 0)

    def extras(j: int, pidx) -> list:
        out = []
        if args.use_ref:
            out.append(u_all[ref_of[j]][pidx].cuda().reshape(-1, 1))
        if args.use_vis:
            out.append(torch.from_numpy(vis[j][pidx]).cuda())
        return out

    sub = torch.arange(0, sc.P, max(1, sc.P // 4096))[:4096]
    rows = []
    for j in ids[:min(len(ids), 8)]:
        base = tx_features(sc.tx[[j]].cuda(), args.tx_bands)
        parts = [base.expand(len(sub), -1)] + extras(j, sub)
        rows.append(torch.cat(parts, 1))
    app_pool = torch.cat(rows, 0)
    a_mean = app_pool.mean(dim=0).cpu()
    # n_tx=1 时 unbiased std 是 NaN；常数维也要能活下来
    a_std = app_pool.std(dim=0, unbiased=False).cpu().nan_to_num(1.0).clamp_min(1e-6)
    log(f"[sp] 点云统计 mean={pe_mean.numpy().round(4).tolist()} "
        f"std={pe_std.numpy().round(4).tolist()}")
    log(f"[sp] app 维度 {app_pool.shape[1]}，逐维 std 中位 {float(a_std.median()):.3g}")

    # 初值取训练场的全局均值（u 空间），避免从 0.5 起步先学一个偏差
    u_mean = float(u_all[ids].mean())
    cfg = {
        "app_dim": int(a_mean.numel()), "hidden": args.hidden,
        "n_layers": args.n_layers, "pe_bands": args.pe_bands,
        "tx_bands": args.tx_bands, "chunk": args.chunk,
        "pe_mean": [float(v) for v in pe_mean],
        "pe_std": [float(v) for v in pe_std],
        "app_mean": [float(v) for v in a_mean],
        "app_std": [float(v) for v in a_std],
        "opacity_init": args.opacity_init, "out_kind": "scalar",
        "n_bands": 1, "init_u": u_mean,
        "use_rel": int(args.use_rel), "pe_rel_bands": args.pe_rel_bands,
        "use_ref": int(args.use_ref), "use_vis": int(args.use_vis),
        "vis_keys": list(VIS_KEYS) if args.use_vis else [],
        "ref_of": {str(k): int(v) for k, v in ref_of.items()},
        "data_root": args.data_root, "train_ids": [int(j) for j in ids],
        "iterations": args.iterations, "seed": args.seed,
        "n_tx": args.n_tx, "img_weight": args.img_weight,
        "views_per_step": args.views_per_step,
    }
    json.dump(cfg, open(os.path.join(args.model_path, "spatial_config.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    head = build_head(cfg)
    lut = make_jet_lut(256).cuda()
    pipe = _Pipe()
    bg = torch.zeros(3, device="cuda")
    optm = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-15)
    log(f"[sp] 头参数 {sum(p.numel() for p in head.parameters())}  "
        f"c_in={head.trunk[0].in_features}  u0={u_mean:.4f}  "
        f"use_rel={bool(args.use_rel)} γ_rel={args.pe_rel_bands}  （加载 {time.time()-t0:.0f}s）")

    xyz = sc.xyz
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    ema_p = ema_i = None
    # 逐点附加输入按源缓存到 GPU（48 源 × 7 通道 × P 在 209k 点时约 210 MB）
    _extra: dict[int, torch.Tensor] = {}

    def extra_full(j: int) -> torch.Tensor | None:
        if n_extra == 0:
            return None
        if j not in _extra:
            _extra[j] = torch.cat(extras(j, slice(None)), dim=1)
        return _extra[j]

    for it in range(1, args.iterations + 1):
        j = int(ids[rng.randint(len(ids))])
        txpos = sc.tx[j].cuda()
        parts = [tx_features(txpos[None], args.tx_bands).expand(sc.P, -1)]
        ex = extra_full(j)
        if ex is not None:
            parts.append(ex)
        app = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        op, attr = head(xyz, app, txpos=txpos)
        u_pred = attr[:, 0]
        u_gt = u_all[j].cuda()
        lp = (u_pred - u_gt).abs().mean()

        li = torch.zeros((), device="cuda")
        if args.img_weight > 0:
            for _ in range(max(1, args.views_per_step)):
                r = int(rx_ids[rng.randint(len(rx_ids))])
                f = int(rng.randint(sc.n_face))
                cam = sc.camera(r, f)
                u_img = sc.render_scalar(cam, u_pred, pipe, bg)
                gt_img = sc.render_scalar(cam, u_gt, pipe, bg)
                img = apply_jet(u_img, lut)
                gt = apply_jet(gt_img, lut)
                L, _ = rgb_loss(img, gt, args.lambda_dssim)
                li = li + L
            li = li / max(1, args.views_per_step)

        L = lp + args.img_weight * li
        if not L.requires_grad:
            raise SystemExit("损失没有梯度 —— 头没接上")
        L.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        with torch.no_grad():
            optm.step()
        optm.zero_grad(set_to_none=False)

        vp, vi = float(lp.item()), float(li.item())
        ema_p = vp if ema_p is None else 0.99 * ema_p + 0.01 * vp
        ema_i = vi if ema_i is None else 0.99 * ema_i + 0.01 * vi
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            with torch.no_grad():
                # 训练场上的点误差（dB）—— 只是优化过程读数，不是评测
                db = (u_pred - u_gt).abs().mean() * (sc.hi - sc.lo)
                log(f"[sp] {it:6d}/{args.iterations}  L1_u={vp:.5f}(ema {ema_p:.5f})  "
                    f"L1_img={vi:.5f}(ema {ema_i:.5f})  ≈{float(db):.2f} dB  "
                    f"u_std={float(u_pred.std()):.4f}  tx={j}  {el/it*1000:.1f}ms/it  "
                    f"剩 {(args.iterations-it)*el/it/60:.0f} 分")

        if it in args.save_iterations:
            torch.save({"iteration": it, "head": head.state_dict(), "config": cfg},
                       os.path.join(args.model_path, f"head_{it}.pth"))
            log(f"[sp] 存 -> head_{it}.pth")

    torch.save({"iteration": args.iterations, "head": head.state_dict(), "config": cfg},
               os.path.join(args.model_path, f"head_{args.iterations}.pth"))
    log(f"[sp] 完成，用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
