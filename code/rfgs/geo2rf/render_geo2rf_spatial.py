# -*- coding: utf-8 -*-
"""空间场的评测渲染：输出 `<out_root>/<split>/ours_<it>/{renders,gt}`，
命名 `{tx:05d}_{v:03d}.png`，v = 该 TX 下 (接收位姿, 朝向) 的局部序号，
可直接交给 `tools/metrics_xscene.py`。

三种模式：
    head   读检查点 → 逐 TX 一次前向得到全场 → 从每个测试接收位姿的 6 个朝向渲染
    knn    同预算基线：k 个最近训练 TX 的**整场**逐点等权平均
    const  无模型基线：训练场的全局均值（不是逐图像均值，避免用测试 GT）

除图像外，另写 `field_metrics.json`：逐点 dB 误差（把 u 反变换回 dBm 后比较）。
图像指标会掩盖被遮挡点的错误，逐点指标不会；两者都报。

用法：
    python code/rfgs/geo2rf/render_geo2rf_spatial.py --mode head \
        --checkpoint results/spatial/N16/head_20000.pth \
        --out_root results/spatial/N16 --iteration 20000
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from corf.spatial_data import SpatialScene                    # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut         # noqa: E402
from rfgs.coupled.train_student import _Pipe                                # noqa: E402
from rfgs.geo2rf.train_geo2rf_spatial import build_head, fps_order, tx_features  # noqa: E402
from corf.spatial_extra import VIS_KEYS, load_vis, nearest_in   # noqa: E402


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/SionnaRoom")
    ap.add_argument("--mode", choices=("head", "knn", "const"), default="head")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help=">0 只渲前 N 个测试 TX")
    ap.add_argument("--n_tx", type=int, default=0, help="knn/const 的训练源数（与头同预算）")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gt_dir", default="",
                    help="预渲好的 GT 目录（render_spatial_gt.py 的产物）；给了就复制，不现渲")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    sc = SpatialScene(args.data_root)
    lut = make_jet_lut(256).cuda()
    pipe = _Pipe()
    bg = torch.zeros(3, device="cuda")

    ids = list(sc.meta["split"][args.split])
    if args.subset > 0:
        ids = ids[:args.subset]
    rx_ids = list(sc.meta["rx_test"] if args.split == "test" else sc.meta["rx_train"])
    views = [(r, f) for r in rx_ids for f in range(sc.n_face)]

    tr_all = fps_order(sc.tx[sc.meta["split"]["train"]].numpy())
    tr_all = [sc.meta["split"]["train"][i] for i in tr_all]

    if args.mode == "head":
        if not args.checkpoint:
            raise SystemExit("--mode head 需要 --checkpoint")
        ck = torch.load(args.checkpoint, map_location="cuda")
        cfg = ck["config"]
        it = args.iteration or int(ck["iteration"])
        head = build_head(cfg)
        head.load_state_dict(ck["head"])
        head.eval()
        if args.n_tx <= 0:
            args.n_tx = len(cfg.get("train_ids", []))
    else:
        it = args.iteration or max(1, args.n_tx)
        if args.n_tx <= 0:
            raise SystemExit("--mode knn/const 需要 --n_tx（与头同一训练预算）")

    tr = tr_all[:args.n_tx]
    if len(tr) < args.n_tx:
        raise SystemExit(f"训练源不足：要 {args.n_tx}，只有 {len(tr_all)}")

    # 头的逐点附加输入（与训练端同一构造；cfg 里没有这些键 = 旧检查点，不拼）
    use_ref = bool(cfg.get("use_ref", 0)) if args.mode == "head" else False
    use_vis = bool(cfg.get("use_vis", 0)) if args.mode == "head" else False
    vis_all = (load_vis(args.data_root, cfg.get("vis_keys") or VIS_KEYS, range(len(sc.tx)))
               if use_vis else {})
    ref_map = ({int(k): int(v) for k, v in cfg.get("ref_of", {}).items()}
               if use_ref else {})
    tx_np = sc.tx.cpu().numpy()
    head_ids = [int(x) for x in cfg.get("train_ids", [])] if args.mode == "head" else []

    def head_app(j: int) -> torch.Tensor:
        txpos = sc.tx[j].cuda()
        parts = [tx_features(txpos[None], cfg["tx_bands"]).expand(sc.P, -1)]
        if use_ref:
            rid = ref_map[j] if j in ref_map else nearest_in(tx_np, j, head_ids)
            parts.append(sc.u(rid).cuda().reshape(-1, 1))
        if use_vis:
            parts.append(torch.from_numpy(vis_all[int(j)]).cuda())
        return torch.cat(parts, 1) if len(parts) > 1 else parts[0]
    tr_u = torch.stack([sc.u(j, device="cpu") for j in tr])       # [N,P]
    const_u = tr_u.mean(dim=0).cuda()
    tr_t = sc.tx[tr]

    odir = os.path.join(args.out_root, args.split, f"ours_{it}")
    rd, gd = os.path.join(odir, "renders"), os.path.join(odir, "gt")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(gd, exist_ok=True)

    field_rows = []
    with torch.no_grad():
        for j in ids:
            u_gt = sc.u(j)
            if args.mode == "head":
                txpos = sc.tx[j].cuda()
                _, attr = head(sc.xyz, head_app(j), txpos=txpos)
                u_pred = attr[:, 0]
            elif args.mode == "knn":
                d = torch.norm(tr_t - sc.tx[j], dim=1)
                kk = max(1, min(args.k, len(tr)))
                nb = torch.argsort(d)[:kk].tolist()
                u_pred = tr_u[nb].mean(dim=0).cuda()
            else:
                u_pred = const_u

            # 逐点 dB 误差（u 反变换；u 在两端被裁剪，误差不区分裁剪内外的来源）
            e_db = (sc.u_to_dbm(u_pred) - sc.u_to_dbm(u_gt)).abs()
            field_rows.append({"tx": int(j), "mae_db": float(e_db.mean()),
                               "rmse_db": float((e_db ** 2).mean().sqrt()),
                               "p95_db": float(torch.quantile(e_db, 0.95))})

            for v, (r, f) in enumerate(views):
                cam = sc.camera(r, f)
                u_img = sc.render_scalar(cam, u_pred, pipe, bg)
                name = f"{j:05d}_{v:03d}.png"
                Image.fromarray(to_uint8(apply_jet(u_img, lut))).save(os.path.join(rd, name))
                if args.gt_dir:
                    src = os.path.join(args.gt_dir, name)
                    if not os.path.isfile(src):
                        raise FileNotFoundError(f"预渲 GT 缺帧：{src}")
                    shutil.copyfile(src, os.path.join(gd, name))
                else:
                    gt_img = sc.render_scalar(cam, u_gt, pipe, bg)
                    Image.fromarray(to_uint8(apply_jet(gt_img, lut))).save(
                        os.path.join(gd, name))

    mae = float(np.mean([r["mae_db"] for r in field_rows]))
    rmse = float(np.mean([r["rmse_db"] for r in field_rows]))
    with open(os.path.join(odir, "field_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "iteration": it, "n_tx": args.n_tx,
                   "k": int(args.k) if args.mode == "knn" else 0,
                   "mae_db": mae, "rmse_db": rmse,
                   "per_tx": field_rows}, f, ensure_ascii=False, indent=2)
    print(f"[sprender] {args.mode} it={it}  {len(ids)}TX × {len(views)}视角  "
          f"逐点 MAE {mae:.3f} dB  RMSE {rmse:.3f} dB -> {odir}", flush=True)


if __name__ == "__main__":
    main()
