# -*- coding: utf-8 -*-
"""测试时小样本适应的评测端：在**留出的那部分视角**上算三指标。

适应集的定义必须与训练端逐字一致，否则评的是拟合不是外推：
    适应集 = test_index 顺序下 `order[::stride][:skip]`
训练端对应 `--train_view_stride <stride> --train_view_limit <skip>`。

为什么不用 `tools/eval_instances.py`：它只支持「取前 N 张」，既不支持「跳过前 N 张」，
也不支持按步长挑。

视角顺序取作者 `test_index.txt` 的原始顺序 —— 与 `train_rrf_angular.py` 的
`test_names` 同一个来源、同一个顺序。

跑法（WSL mvsplat 环境）：
    $PY .tmp/_tta_eval.py --renders <dir> --gt <dir> \
        --poses_root datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
        --skip 32 --stride 10 --out <tsv>
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1] / "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.utils.image_utils import psnr            # noqa: E402
from splat.utils.loss_utils import ssim             # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--poses_root", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="适应集的抽样步长（与训练端 --train_view_stride 一致）")
    ap.add_argument("--random", type=int, default=0,
                    help="随机抽 N 个当适应集（与训练端 --train_view_random 一致）。"
                         "位置与朝向同时无偏")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机抽样种子（与训练端 --train_view_seed 一致）")
    ap.add_argument("--skip", type=int, default=0,
                    help="从 order[::stride] 里取前 N 个当适应集，评测时整批剔除")
    ap.add_argument("--exclude_file", default="",
                    help="逐行列出要剔除的视角名（去后缀）。用于把多个适应集的"
                         "并集一起剔掉，得到所有模型都没见过的公共评测集")
    ap.add_argument("--only_adapt", action="store_true",
                    help="只评适应集本身。用来分辨「拟合不了」与「过拟合」："
                         "适应集也没变好 ⇒ 头拟合不了；适应集大幅变好而其余变差 ⇒ 过拟合")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_flat_gt", action="store_true")
    args = ap.parse_args()

    order = [Path(x).stem for x in
             (Path(args.poses_root) / "test_index.txt").read_text().split() if x.strip()]
    if not order:
        raise SystemExit("test_index.txt 读空")

    if args.random > 0:
        # 与训练端 `--train_view_random` 逐字一致：同一个 generator、同一个种子、
        # 同一份 test_index 顺序，最后按索引排序还原原顺序。
        import numpy as _np
        idx = _np.random.default_rng(args.seed).choice(
            len(order), size=min(args.random, len(order)), replace=False)
        idx.sort()
        adapt = [order[i] for i in idx]
    else:
        adapt = order[::args.stride][: args.skip] if args.skip else []
    if args.only_adapt:
        order = list(adapt)
    else:
        ex = set(adapt)
        order = [n for n in order if n not in ex]
    if args.exclude_file:
        ex2 = set(Path(args.exclude_file).read_text().split())
        order = [n for n in order if n not in ex2]
    if not order:
        raise SystemExit("剔除适应集之后没有评测视角了")

    rd, gd = Path(args.renders), Path(args.gt)
    have = {p.stem for p in rd.glob("*.png")} & {p.stem for p in gd.glob("*.png")}
    names = [n for n in order if n in have]
    if not names:
        raise SystemExit(f"渲染/GT 里没有可评的视角（renders {rd} / gt {gd}）")
    if len(names) < len(order):
        print(f"[warn] {len(order) - len(names)} 个视角缺图，按 {len(names)} 张算",
              flush=True)

    crit = LPIPS("vgg", "0.1").to(args.device).eval()
    ps, ss, lps = [], [], []
    nflat = 0
    for n in names:
        r = tf.to_tensor(Image.open(rd / f"{n}.png")).unsqueeze(0)[:, :3].to(args.device)
        g = tf.to_tensor(Image.open(gd / f"{n}.png")).unsqueeze(0)[:, :3].to(args.device)
        flat = bool(float((g - g[:, :, :1, :1]).abs().max()) <= 1e-6)
        if flat:
            nflat += 1
            if args.skip_flat_gt:
                continue
        with torch.no_grad():
            _p = float(psnr(r, g).mean())
            ps.append(_p if math.isfinite(_p) else 99.0)
            ss.append(float(ssim(r, g).mean()))
            lps.append(float(crit(r, g).mean()))

    row = dict(skip=args.skip, stride=args.stride, n=len(ps), nflat=nflat,
               psnr=sum(ps) / len(ps), ssim=sum(ss) / len(ss),
               lpips=sum(lps) / len(lps))
    line = (f"stride={row['stride']:3d} skip={row['skip']:4d}  n={row['n']:4d}  "
            f"flat={nflat:2d}  PSNR {row['psnr']:7.4f}  SSIM {row['ssim']:.4f}  "
            f"LPIPS {row['lpips']:.4f}")
    print(line, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("stride\tskip\tn\tnflat\tpsnr\tssim\tlpips\n")
            f.write(f"{row['stride']}\t{row['skip']}\t{row['n']}\t{row['nflat']}\t"
                    f"{row['psnr']:.6f}\t{row['ssim']:.6f}\t{row['lpips']:.6f}\n")


if __name__ == "__main__":
    main()
