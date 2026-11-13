# -*- coding: utf-8 -*-
"""同位置先验：用 `base_instance` 的真值图当预测，量「不需要学习的基线」有多强。

回答的问题
----------
w 置换对照只回答「模型用没用 w」；它回答不了「模型离一个平凡预测器有多远」。
这里量后者：同一个位置、同一个视角，只把条件换成训练时见过的那个 w 的**真值图**。

对 CBF 尤其锋利——`synthetic_array=True` 下 w 只是闭式重加权，同位置不同 w 的
GT 本来就极像，所以「查表」几乎就是答案。若模型离它差很多，瓶颈就是基础保真度，
不是损失函数、也不是泛化。

合法性：`w_holdout` 的 `base_instance` 全部是训练实例、同位置、w 不同
（2026-11-08 核过 36/36）。先验用的是**训练集里的图**，不是留出真值。

为什么在内存里算：把 23k 张先验图写回 `/mnt/d` 再读回来约 2 小时（WSL→Windows
文件系统层 ~0.3 s/次写）。指标与渲染产物同源（都是 8bit），所以直接构造
`lut[u8]` 张量即可，数值上与落盘再读完全一致。指标函数与 `eval_instances.py` 逐字相同。

跑法（WSL）：
    $PY code/tools/eval_same_position_prior.py \
        --meta datasets/RF-3DGS_mvdr_w_params/meta.json \
        --spectra datasets/RF-3DGS_cbf_w_params/spectra \
        --gt_root results/<run>/w/test_inst \
        --out results/<run>/在先验_同位置/metrics.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.color_scalar import make_jet_lut            # noqa: E402
from splat.utils.image_utils import psnr              # noqa: E402
from splat.utils.loss_utils import ssim               # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--spectra", required=True, help="<root>/inst<id>/<view>.png（8bit 灰度 u）")
    ap.add_argument("--gt_root", required=True,
                    help="已导出的 jet GT 根目录，内含 inst<id>/gt/<view>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", default="w_holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="每实例最多算多少张视角")
    args = ap.parse_args()

    meta = json.load(open(args.meta, encoding="utf-8"))
    rows = [i for i in meta["instances"] if i.get("role") == args.role]
    if not rows:
        raise SystemExit(f"meta 里没有 role={args.role} 的实例")
    lut = (make_jet_lut(256).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    crit = LPIPS("vgg", "0.1").to(args.device).eval()

    out_rows = []
    for inst in rows:
        iid, src = int(inst["id"]), int(inst["base_instance"])
        gd = Path(args.gt_root) / f"inst{iid:03d}" / "gt"
        sd = Path(args.spectra) / f"inst{src:03d}"
        if not gd.is_dir():
            print(f"[skip] inst{iid:03d} 缺 GT 目录")
            continue
        names = sorted(p.name for p in gd.glob("*.png"))
        if args.limit:
            names = names[: args.limit]
        ps, ss, lps = [], [], []
        for nm in names:
            sp = sd / nm
            if not sp.exists():
                continue
            g = tf.to_tensor(Image.open(gd / nm)).unsqueeze(0)[:, :3].to(args.device)
            u8 = np.asarray(Image.open(sp).convert("L"))
            # contiguous 必须显式加：psnr() 里用 .view() 展平非连续张量会报错
            r = (torch.from_numpy(lut[u8]).permute(2, 0, 1).unsqueeze(0)
                 .float().div_(255.0).contiguous().to(args.device))
            with torch.no_grad():
                _p = float(psnr(r, g).mean())
                ps.append(_p if math.isfinite(_p) else 99.0)
                ss.append(float(ssim(r, g).mean()))
                lps.append(float(crit(r, g).mean()))
        if not ps:
            print(f"[skip] inst{iid:03d} 没有可用视角")
            continue
        row = dict(id=iid, n=len(ps), psnr=sum(ps) / len(ps),
                   ssim=sum(ss) / len(ss), lpips=sum(lps) / len(lps))
        out_rows.append(row)
        print(f"INST{iid:03d} ←inst{src:03d} n={row['n']:4d}  PSNR {row['psnr']:7.4f}  "
              f"SSIM {row['ssim']:.4f}  LPIPS {row['lpips']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("id\trole\tn\tpsnr\tssim\tlpips\n")
        for r in out_rows:
            f.write(f"{r['id']}\t{args.role}\t{r['n']}\t{r['psnr']:.6f}\t"
                    f"{r['ssim']:.6f}\t{r['lpips']:.6f}\n")
    print(f"\n写出 {args.out}（{len(out_rows)} 个实例）")
    print(f"  PSNR {statistics.mean(r['psnr'] for r in out_rows):.4f} ± "
          f"{statistics.pstdev(r['psnr'] for r in out_rows):.4f}  "
          f"SSIM {statistics.mean(r['ssim'] for r in out_rows):.4f}  "
          f"LPIPS {statistics.mean(r['lpips'] for r in out_rows):.4f}")


if __name__ == "__main__":
    main()
