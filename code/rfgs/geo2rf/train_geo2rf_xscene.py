# -*- coding: utf-8 -*-
"""跨场景前馈射频头：在若干「房间表面点云」上训一个头，留出场景做零样本测试。

数据来自 `tools/graf_prepare.py`（GRaF-sim 改造），每个场景是：
    点云（网关局部归一化系的天花板 + 墙带，几何冻结）
    + 800 个 TX（每个带 4 个 96×96 的 jet 伪彩 GT 视角）
TX 条件通过头已有的 `app` 槽广播进来（逐点复制），**头代码一行不改**。

两条路：
    in-scene   在目标场景自己的 600 个 TX 上训，测它的 200 个 TX  ← 同场景上限
    zero-shot  在另外两个场景上训，直接测目标场景              ← 跨场景

用法：
    python code/rfgs/geo2rf/train_geo2rf_xscene.py --train_scenes D2 D8 \
        --model_path results/xscene/D14_zero --iterations 20000
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import math
import os
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corf.geo_head import GeoToRfHead                      # noqa: E402
from corf.rf_head import xyz_norm_stats                    # noqa: E402
from splat.color_mlp import fourier_feat                   # noqa: E402
from splat.color_scalar import (apply_jet, combine_bands,  # noqa: E402
                                make_jet_lut, to_override_color)
from splat.gaussian_renderer import render                 # noqa: E402
from splat.scene.cameras import Camera                     # noqa: E402
from splat.scene.gaussian_model import GaussianModel       # noqa: E402
from splat.utils.general_utils import safe_state           # noqa: E402
from rfgs.coupled.train_student import StudentView, _Pipe, rgb_loss     # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def freeze(gm: GaussianModel) -> None:
    for n in ("_xyz", "_scaling", "_rotation", "_features_dc", "_features_rest", "_opacity"):
        getattr(gm, n).requires_grad = False


# ---------------------------------------------------------------- TX 特征
def tx_features(tx: torch.Tensor, bands: int) -> torch.Tensor:
    """TX 位置 → 条件向量。归一化坐标、γ(位置)、到网关距离、方位/仰角三角量。

    与点云用同一个归一化尺度（meta 的 norm_unit），于是「相对房间的位置」
    在三个场景之间可比；绝对物理尺度被丢弃（见 说明.md 的口径说明）。
    """
    r = tx.norm(dim=-1, keepdim=True)
    d = tx / r.clamp_min(1e-9)
    az = torch.atan2(d[:, 1], d[:, 0])
    el = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    return torch.cat([
        tx,
        fourier_feat(tx, bands),
        r,
        torch.sin(az)[:, None], torch.cos(az)[:, None],
        torch.sin(el)[:, None], torch.cos(el)[:, None],
        torch.log1p(r),
    ], dim=-1)


def app_dim_of(bands: int) -> int:
    return 3 + 6 * int(bands) + 1 + 4 + 1


# ---------------------------------------------------------------- 场景
@dataclass
class SceneData:
    name: str
    gm: GaussianModel
    view: StudentView
    xyz: torch.Tensor
    tx_app: torch.Tensor
    sc_u8: torch.Tensor          # 标量域 GT（0..255）；RGB 目标每次由 jet 现查
    cams: list
    train_ids: list              # 已按 FPS 嵌套顺序排好：前 N 个就是 N-shot 子集
    test_ids: list
    meta: dict


def load_scene(root: str, name: str, tx_bands: int) -> SceneData:
    sdir = os.path.join(root, name)
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    gm = GaussianModel(3)
    gm.load_ply(os.path.join(sdir, "cloud.ply"))
    freeze(gm)
    xyz = gm.get_xyz.detach().float()

    tx = torch.tensor(meta["tx_local"], dtype=torch.float32)
    tx_app = tx_features(tx, tx_bands).cuda()

    v = meta["views"]
    nv, H, W = int(v["n"]), int(v["img"]), int(v["img"])
    n = len(meta["file_idx"])
    sc = torch.zeros(n, nv, H, W, dtype=torch.uint8)
    for j in range(n):
        for k in range(nv):
            sp = os.path.join(sdir, "scalar", f"{j:05d}_{k}.png")
            sc[j, k] = torch.from_numpy(np.array(Image.open(sp).convert("L"),
                                                  dtype=np.uint8))

    cams = []
    for k in range(nv):
        R = np.asarray(v["R"][k], dtype=np.float32)
        T = np.zeros(3, dtype=np.float32)
        cams.append(Camera(k, R, T, math.radians(float(v["fovx"])),
                           math.radians(float(v["fovy"])),
                           torch.zeros(3, H, W), None, f"{name}_v{k}", k))

    # few-shot 的嵌套子集：对**训练池**再做一次 FPS，前 N 个始终空间铺开且逐 N 包含
    tr = list(meta["split"]["train"])
    from tools.graf_prepare import fps_select
    order = fps_select(tx[tr].numpy(), len(tr))
    tr = [tr[i] for i in order]
    return SceneData(name, gm, StudentView(gm), xyz, tx_app, sc, cams,
                     tr, list(meta["split"]["test"]), meta)


# ---------------------------------------------------------------- 头
def build_head(cfg: dict) -> GeoToRfHead:
    return GeoToRfHead(
        app_dim=cfg["app_dim"], hidden=cfg["hidden"], n_layers=cfg["n_layers"],
        pe_bands=cfg["pe_bands"], chunk=cfg["chunk"],
        pe_mean=torch.tensor(cfg["pe_mean"], dtype=torch.float32),
        pe_std=torch.tensor(cfg["pe_std"], dtype=torch.float32),
        app_mean=torch.tensor(cfg["app_mean"], dtype=torch.float32),
        app_std=torch.tensor(cfg["app_std"], dtype=torch.float32),
        opacity_init=cfg["opacity_init"],
        out_kind=cfg["out_kind"], n_bands=cfg["n_bands"], init_u=cfg["init_u"]).cuda()


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/GRaF-sim_processed")
    ap.add_argument("--train_scenes", nargs="+", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--pe_bands", type=int, default=6, help="点云 xyz 的位置编码频带")
    ap.add_argument("--tx_bands", type=int, default=4, help="TX 位置的傅里叶频带")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--opacity_init", type=float, default=0.5)
    ap.add_argument("--init_u", type=float, default=0.5)
    ap.add_argument("--n_bands", type=int, default=1)
    ap.add_argument("--n_tx", type=int, default=0,
                    help="只取训练池的前 N 个（FPS 嵌套顺序）；0 = 全部 600")
    ap.add_argument("--power_aug", type=int, default=0,
                    help="1 = 头额外吃 log 功率增益，且每次采样随机增益缩放 GT")
    ap.add_argument("--gain_min", type=float, default=0.7)
    ap.add_argument("--gain_max", type=float, default=1.3)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_iterations", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations 必须 > 0")
    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.model_path, exist_ok=True)

    t0 = time.time()
    scenes = [load_scene(args.data_root, s, args.tx_bands) for s in args.train_scenes]
    if args.n_tx > 0:
        for s in scenes:
            s.train_ids = s.train_ids[:args.n_tx]
    for s in scenes:
        log(f"[xs] {s.name}: P={s.xyz.shape[0]}  TX {len(s.train_ids)} 训 / "
            f"{len(s.test_ids)} 测  视角 {len(s.cams)}")

    # 位置编码统计量：只由**训练场景**点云决定，测试场景不重估
    xyz_pool = torch.cat([s.xyz.detach().cpu()[::10] for s in scenes], dim=0)
    pe_mean, pe_std = xyz_norm_stats(xyz_pool)
    app_pool = torch.cat([s.tx_app[torch.tensor(s.train_ids)] for s in scenes], dim=0)
    a_mean = app_pool.mean(dim=0).cpu()
    a_std = app_pool.std(dim=0).cpu().clamp_min(1e-6)
    if args.power_aug:
        # 增益列（log g）的统计量按训练分布解析给出；**不能**用全 0 列，
        # 否则 std 被 clamp 到 1e-6，归一化后 g 的微小差异被放大成噪声。
        rs = np.random.RandomState(123)
        lg = np.log(rs.uniform(args.gain_min, args.gain_max, 8192))
        a_mean = torch.cat([a_mean, torch.tensor([float(lg.mean())])])
        a_std = torch.cat([a_std, torch.tensor([max(float(lg.std()), 1e-3)])])
        log(f"[xs] 功率轴：log g ~ U[{math.log(args.gain_min):.3f},"
            f"{math.log(args.gain_max):.3f}]  均值 {lg.mean():+.3f}  std {lg.std():.3f}")
    log(f"[xs] 点云统计（每 10 点抽样）mean={pe_mean.numpy().round(4).tolist()} "
        f"std={pe_std.numpy().round(4).tolist()}")
    log(f"[xs] app 维度 {app_pool.shape[1]}，逐维 std 中位 {float(a_std.median()):.3g}")

    cfg = {
        "app_dim": int(a_mean.numel()), "hidden": args.hidden,
        "n_layers": args.n_layers, "pe_bands": args.pe_bands,
        "tx_bands": args.tx_bands, "chunk": args.chunk,
        "pe_mean": [float(v) for v in pe_mean],
        "pe_std": [float(v) for v in pe_std],
        "app_mean": [float(v) for v in a_mean],
        "app_std": [float(v) for v in a_std],
        "opacity_init": args.opacity_init, "out_kind": "scalar",
        "n_bands": args.n_bands, "init_u": args.init_u,
        "train_scenes": list(args.train_scenes),
        "iterations": args.iterations, "seed": args.seed,
        "n_tx": args.n_tx, "power_aug": int(args.power_aug),
        "gain_min": args.gain_min, "gain_max": args.gain_max,
    }
    json.dump(cfg, open(os.path.join(args.model_path, "xscene_config.json"), "w"),
              indent=2, ensure_ascii=False)

    head = build_head(cfg)
    lut = make_jet_lut(256).cuda()
    pipe = _Pipe()
    background = torch.zeros(3, device="cuda")
    optm = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-15)
    n_par = sum(p.numel() for p in head.parameters())
    log(f"[xs] 头参数 {n_par}  c_in={head.trunk[0].in_features}  "
        f"训练场景 {args.train_scenes}  （加载 {time.time()-t0:.0f}s）")

    # 所有 (场景, TX, 视角) 组合；每个 epoch 重新洗牌
    combos = [(si, j, k) for si, s in enumerate(scenes)
              for j in s.train_ids for k in range(len(s.cams))]
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(combos))
    ptr = 0

    t0 = time.time()
    ema = None
    for it in range(1, args.iterations + 1):
        if ptr >= len(order):
            order = rng.permutation(len(combos))
            ptr = 0
        si, j, k = combos[int(order[ptr])]
        ptr += 1
        s = scenes[si]

        P = s.xyz.shape[0]
        app = s.tx_app[j].unsqueeze(0).expand(P, -1)
        if args.power_aug:
            g = float(np.exp(rng.uniform(math.log(args.gain_min),
                                         math.log(args.gain_max))))
            app = torch.cat([app, torch.full((P, 1), math.log(g), device="cuda")], -1)
        op, attr = head(s.xyz, app)
        cam = s.cams[k]
        u = s.sc_u8[j, k].float().cuda() / 255.0
        if args.power_aug:
            cam.original_image = apply_jet((u * g).clamp(0.0, 1.0), lut)
            clip_frac = float((u * g > 1.0).float().mean())
        else:
            g, clip_frac = 1.0, 0.0
            cam.original_image = apply_jet(u, lut)
        s.view.set_rf_scalar(op)
        mixed = render(cam, s.view, pipe, background,
                       override_color=to_override_color(attr))["render"]
        img = apply_jet(combine_bands(mixed, head), lut)
        L, l1 = rgb_loss(img, cam.original_image, args.lambda_dssim)
        if not L.requires_grad:
            raise SystemExit("损失没有梯度 —— 头没接上")

        L.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        with torch.no_grad():
            optm.step()
        optm.zero_grad(set_to_none=False)

        v = float(l1.item())
        ema = v if ema is None else 0.99 * ema + 0.01 * v
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            with torch.no_grad():
                extra = (f"s_mean={float(attr.mean()):.4f} "
                         f"s_std={float(attr.std()):.4f} "
                         f"op={float(torch.sigmoid(op).mean()):.4f}")
                if args.power_aug:
                    extra += f" g={g:.3f} clip={clip_frac*100:.1f}%"
            log(f"[xs] {it:6d}/{args.iterations}  L1={v:.5f}(ema {ema:.5f})  "
                f"{extra}  {s.name} tx={j} v={k}  {el/it:.3f}s/it  "
                f"剩 {(args.iterations-it)*el/it/60:.0f} 分")

        if it in args.save_iterations:
            torch.save({"iteration": it, "head": head.state_dict(), "config": cfg},
                       os.path.join(args.model_path, f"head_{it}.pth"))
            log(f"[xs] 存 -> head_{it}.pth")

    torch.save({"iteration": args.iterations, "head": head.state_dict(), "config": cfg},
               os.path.join(args.model_path, f"head_{args.iterations}.pth"))
    log(f"[xs] 完成，用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
