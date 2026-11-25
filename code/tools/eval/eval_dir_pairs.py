# -*- coding: utf-8 -*-
"""对一对 renders/gt 目录算 PSNR/SSIM/LPIPS（口径与 tools/metrics_fast.py 相同）。

用途：`metrics_fast.py` 只认 test/ 目录；本脚本用来量**训练视角**的读数，
       好把「训练拟合」与「留出位置泛化」摆在一起看。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.renders) if n.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"renders 无 png：{args.renders}")
    ps, ss, lps = [], [], []
    # 与 tools/metrics_fast.py 同口径：net_type="vgg"。
    # 注意：metrics_fast 的 lpips() 每张图都会重建网络（很慢）；这里只建一次。
    criterion = LPIPS("vgg", "0.1").cuda().eval()
    for n in names:
        r = tf.to_tensor(Image.open(os.path.join(args.renders, n))).unsqueeze(0)[:, :3].cuda()
        g = tf.to_tensor(Image.open(os.path.join(args.gt, n))).unsqueeze(0)[:, :3].cuda()
        with torch.no_grad():
            ps.append(float(psnr(r, g).mean()))
            ss.append(float(ssim(r, g).mean()))
            lps.append(float(criterion(r, g).mean()))
    print(f"{args.tag}  n={len(names)}  PSNR {sum(ps) / len(ps):.4f}  "
          f"SSIM {sum(ss) / len(ss):.4f}  LPIPS {sum(lps) / len(lps):.4f}")


if __name__ == "__main__":
    main()
