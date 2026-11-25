# -*- coding: utf-8 -*-
"""三指标地板：常数图（逐像素均值）在新旧两套 GT 上各得多少。

为什么必须做：逐像素均值就是 MSE 最优解，PSNR 天然偏袒它。主读数若只跟
「发布包 14.13」比，看不出模型是不是只学到了「一张平滑的平均图」。所以要在
**同一批视角、同一套指标**下把常数图也报出来，三指标一起看。

用法（WSL，torch + LPIPS 在 GPU）：
    $PY code/tools/eval/_floor_metrics.py \
        --gt_new results/<run>/test_inst/inst317/gt \
        --gt_iso datasets/RF-3DGS_cbf_multi/tx048 \
        --limit 0
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import math
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.utils.image_utils import psnr            # noqa: E402
from splat.utils.loss_utils import ssim             # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402


def load(dirp, names, device):
    out = []
    for n in names:
        p = Path(dirp) / n
        out.append(tf.to_tensor(Image.open(p)).unsqueeze(0)[:, :3].to(device))
    return out


def score(pred, gts, criterion):
    ps, ss, ls = [], [], []
    for g in gts:
        with torch.no_grad():
            _p = float(psnr(pred, g).mean())
            ps.append(_p if math.isfinite(_p) else 99.0)
            ss.append(float(ssim(pred, g).mean()))
            ls.append(float(criterion(pred, g).mean()))
    n = len(gts)
    return sum(ps) / n, sum(ss) / n, sum(ls) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_new", required=True, help="tr38901 协议的 640 张 GT")
    ap.add_argument("--gt_iso", required=True, help="原始 iso 协议的 tx048 目录（按同名挑）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    names = sorted(p.name for p in Path(args.gt_new).glob("*.png"))
    if args.limit:
        names = names[: args.limit]
    have = {p.name for p in Path(args.gt_iso).glob("*.png")}
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(f"iso 目录缺 {len(missing)} 张（前 3：{missing[:3]}）")

    criterion = LPIPS("vgg", "0.1").to(args.device).eval()
    print(f"视角 {len(names)} 张\n")
    print(f"{'口径':22s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
    for tag, d in (("tr38901（本次）", args.gt_new), ("iso（发布包口径）", args.gt_iso)):
        gts = load(d, names, args.device)
        mean = torch.stack(gts, 0).mean(0)
        p, s, l = score(mean, gts, criterion)
        print(f"{tag + ' 常数图':22s} {p:8.4f} {s:8.4f} {l:8.4f}")


if __name__ == "__main__":
    main()
