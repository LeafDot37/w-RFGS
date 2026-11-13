# -*- coding: utf-8 -*-
"""
metrics_fast.py —— 收敛曲线用抽帧快速评测（渲染已全量，本脚本只算指定档的子集）
--------------------------------------------------------------------------------
metrics.py 对每个 test/ours_<i> 逐张跑 640 图 LPIPS（共享卡下 ~13min/档），对收敛曲线
的多个中间档太慢。本脚本复用已渲染的 test/ours_<i>/{renders,gt}，只对 **确定性子集**
（固定步长抽帧）算 PSNR/SSIM/LPIPS-VGG，输出与 metrics.py 相同的
    { "ours_<i>": { PSNR, SSIM, LPIPS }, ... }  结构（额外记 n_views），供 make_* 曲线脚本聚合。
评测口径：同 metrics.py（同一实现），仅视图数不同；末档另有全量 metrics.py 结果定标。

用法（服务器 code/ 目录；也可本地同构）：
    <PY> metrics_fast.py -m <model_dir> --iterations 35000 40000 ... --subset 48 \
        --out <model_dir>/curve.json
    # 若要某一档全量（subset=0 = 全部视图）：
    <PY> metrics_fast.py -m <model_dir> --iterations 70000 --subset 0 \
        --out <model_dir>/results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.utils.image_utils import psnr
from splat.utils.loss_utils import ssim
from splat.lpipsPyTorch import lpips


def load_pairs(renders_dir, gt_dir, subset):
    """确定性抽帧：对所有渲染帧按文件名排序，若 subset>0 则均匀取子集。"""
    names = sorted(os.listdir(renders_dir))
    names = [n for n in names if n.lower().endswith(".png")]
    if not names:
        raise RuntimeError(f"renders 目录无 png: {renders_dir}")
    if subset and subset > 0 and subset < len(names):
        step = max(1, len(names) // subset)
        names = names[::step][:subset]
    renders, gts = [], []
    for fname in names:
        renders.append(tf.to_tensor(Image.open(renders_dir / fname)).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(Image.open(gt_dir / fname)).unsqueeze(0)[:, :3, :, :].cuda())
    return renders, gts


def evaluate_iter(model_dir, iteration, subset):
    test_dir = Path(model_dir) / "test"
    odir = test_dir / f"ours_{iteration}"
    renders_dir, gt_dir = odir / "renders", odir / "gt"
    if not (renders_dir.is_dir() and gt_dir.is_dir()):
        raise FileNotFoundError(f"缺渲染档 test/ours_{iteration}: {odir}")

    renders, gts = load_pairs(renders_dir, gt_dir, subset)
    n = len(renders)
    ssims, psnrs, lpipss = [], [], []
    for r, g in zip(renders, gts):
        ssims.append(ssim(r, g).item())
        psnrs.append(psnr(r, g).item())
        lpipss.append(lpips(r, g, net_type="vgg").item())
    return {
        "PSNR": sum(psnrs) / n,
        "SSIM": sum(ssims) / n,
        "LPIPS": sum(lpipss) / n,
        "n_views": n,
    }


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ap = argparse.ArgumentParser(description="抽帧快速评测（收敛曲线）")
    ap.add_argument("-m", "--model", required=True, help="模型目录（含 test/ours_<i>）")
    ap.add_argument("--iterations", nargs="+", type=int, required=True,
                    help="要评测的绝对迭代档位")
    ap.add_argument("--subset", type=int, default=48,
                    help="每档抽帧数（0=全部视图，同全量 metrics.py）")
    ap.add_argument("--out", required=True, help="输出 json 路径")
    args = ap.parse_args()

    agg = {}
    for it in args.iterations:
        v = evaluate_iter(args.model, it, args.subset)
        agg[f"ours_{it}"] = v
        print(f"ours_{it}:  PSNR {v['PSNR']:.3f}  SSIM {v['SSIM']:.4f}  "
              f"LPIPS {v['LPIPS']:.4f}  n={v['n_views']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"写入 {args.out}（{len(agg)} 档）")


if __name__ == "__main__":
    main()
