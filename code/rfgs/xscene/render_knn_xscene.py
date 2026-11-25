# -*- coding: utf-8 -*-
"""KNN 基线：用**同样 N 个训练源**的谱图做最近邻平均，预测测试源的 4 个视角。

为什么必须有这条臂：GRaF 论文的插值定理说明「目标源谱图 ≈ 近邻源谱图的加权和」，
其 KNN 基线在他们的场景上就有 15.16 dB。我们的头如果没有同预算的 KNN 对照，
就无法说明「显式几何 + 头」是否比纯谱图插值更强。

口径：
    训练源 = 与少样本实验完全相同的 FPS 嵌套顺序前 N 个；
    近邻 = 这 N 个里欧氏距离最近的 k 个（归一化源坐标）；
    预测 = 近邻标量图的等权平均，再查 jet 表（与训练/评测同一条链）；
    测试源不参与任何拟合。

用法：
    python code/rfgs/xscene/render_knn_xscene.py --scene D2 --n_tx 32 --k 4 \
        --out_root results/fewshot/D2_N32_knn
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splat.color_scalar import apply_jet, make_jet_lut   # noqa: E402


def load_scalar(sdir: str, j: int, k: int) -> torch.Tensor:
    p = os.path.join(sdir, "scalar", f"{j:05d}_{k}.png")
    a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(a)


def main():
    ap = ArgumentParser()
    ap.add_argument("--data_root", default="datasets/GRaF-sim_processed")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--n_tx", type=int, required=True, help="训练源数量（与少样本臂一致）")
    ap.add_argument("--k", type=int, default=4, help="近邻个数")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--iteration", type=int, default=0,
                    help="输出档位号（默认 = n_tx，便于与头的结果并列）")
    ap.add_argument("--subset", type=int, default=0, help=">0 只渲前 N 个测试源")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    sdir = os.path.join(args.data_root, args.scene)
    meta = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
    tx = torch.tensor(meta["tx_local"], dtype=torch.float32)
    nv = int(meta["views"]["n"])

    tr = list(meta["split"]["train"])
    from tools.graf_prepare import fps_select
    order = fps_select(tx[tr].numpy(), len(tr))
    tr = [tr[i] for i in order][:args.n_tx]
    if len(tr) < args.n_tx:
        raise SystemExit(f"训练源不足：要 {args.n_tx}，只有 {len(tr)}")

    ids = list(meta["split"][args.split])
    if args.subset > 0:
        ids = ids[:args.subset]

    it = args.iteration or args.n_tx
    odir = os.path.join(args.out_root, args.split, f"ours_{it}")
    rd, gd = os.path.join(odir, "renders"), os.path.join(odir, "gt")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(gd, exist_ok=True)

    lut = make_jet_lut(256).cuda()
    tr_t = tx[tr]
    kk = max(1, min(args.k, len(tr)))

    with torch.no_grad():
        for j in ids:
            d = torch.norm(tr_t - tx[j], dim=1)
            nb = torch.argsort(d)[:kk].tolist()
            for v in range(nv):
                us = torch.stack([load_scalar(sdir, tr[i], v).cuda() for i in nb])
                u = us.mean(0)                       # 等权平均（论文 KNN 口径）
                rgb = apply_jet(u, lut)
                arr = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                       * 255).round().astype(np.uint8)
                name = f"{j:05d}_{v}.png"
                Image.fromarray(arr).save(os.path.join(rd, name))
                Image.open(os.path.join(sdir, "gt", name)).save(os.path.join(gd, name))

    json.dump({"scene": args.scene, "split": args.split, "n_tx": args.n_tx,
               "k": kk, "iteration": it}, open(os.path.join(args.out_root, "knn_config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[knn] {args.scene} N={args.n_tx} k={kk} {len(ids)}×{nv} 帧 -> {odir}")


if __name__ == "__main__":
    main()
