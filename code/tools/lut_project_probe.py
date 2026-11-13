# -*- coding: utf-8 -*-
"""
lut_project_probe.py —— 色标投影探针：把已渲染的谱图投回「合法调色板」，看三指标怎么变
--------------------------------------------------------------------------------
背景：CBF/MVDR/TCBF 的谱图实测是 **恰好 253 色的 jet 查表**（色↔标量严格双射）；GT 像素
      100% 落在该调色板上，而 MLP 渲出的像素有相当比例漂在调色板外（MVDR V14 约 41%）。
问题：把输出强行约束回调色板，指标会变好还是变坏？——**首要看 LPIPS**。

做法（零训练、纯后处理，只读已有渲染）：
    调色板 P = 该臂 test/gt 全部图像的颜色并集（即真实 LUT），按到 jet@256 的最近档排序；
    对每张 render：每个像素取 P 中最近色 → 投影图。
    分别算 原图 / 投影图 的 PSNR、SSIM、LPIPS（同 metrics.py 实现），报 Δ。

用法（服务器 code/ 目录）：
    <PY> tools/lut_project_probe.py --root /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训 \
        --methods "CBF MVDR" --arms "v14 sh" --iteration 70000 --subset 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from scipy.spatial import cKDTree

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.utils.image_utils import psnr      # noqa: E402
from splat.utils.loss_utils import ssim       # noqa: E402
from splat.lpipsPyTorch import lpips          # noqa: E402

ARMS = {
    "v14": ("阶段2长训_位置编码_X5D3R1", "V14 长训（MLP+PE+dist）"),
    "sh":  ("阶段1b长训_SH头",            "SH@A* 长训（原版 SH）"),
}


def jet_palette_order(colours):
    """按到 jet@256 的最近档给调色板颜色排序（恢复色标的顺序）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jet = (np.asarray(plt.get_cmap("jet")(np.linspace(0, 1, 256)))[:, :3] * 255.0).astype(np.float32)
    d = ((colours[:, None, :] - jet[None, :, :]) ** 2).sum(-1).argmin(1)
    return np.argsort(d), d


def build_palette(gt_files, cap=400):
    acc = []
    for f in gt_files[:: max(1, len(gt_files) // cap)][:cap]:
        acc.append(np.unique(np.asarray(Image.open(f).convert("RGB")).reshape(-1, 3), axis=0))
    P = np.unique(np.concatenate(acc), axis=0).astype(np.float32)
    order, _ = jet_palette_order(P)
    return P[order]


def _nearest_vertex(flat, P, tree):
    """每像素最近的调色板格点索引。"""
    return tree.query(flat, k=1, workers=-1)[1]


def project_snap(img_u8, P, tree):
    """硬吸附：每像素取最近格点（会量化，仅作对照）。"""
    flat = img_u8.reshape(-1, 3).astype(np.float32)
    out = P[_nearest_vertex(flat, P, tree)]
    return np.clip(np.round(out), 0, 255).astype(np.uint8).reshape(img_u8.shape)


def project_curve(img_u8, P, tree):
    """投影到色标折线（允许落在两档之间）——对应架构里的可微 LUT 插值输出。
    先找最近格点 k，再在相邻两段 (k-1,k)、(k,k+1) 上取最近点。"""
    flat = img_u8.reshape(-1, 3).astype(np.float32)
    k = _nearest_vertex(flat, P, tree)
    best_d, best_p = None, None
    for shift in (-1, 0):
        ia = np.clip(k + shift, 0, len(P) - 1)
        ib = np.clip(k + shift + 1, 0, len(P) - 1)
        a, b = P[ia], P[ib]
        ab = b - a
        denom = (ab ** 2).sum(-1)
        denom[denom == 0] = 1.0
        t = np.clip(((flat - a) * ab).sum(-1) / denom, 0.0, 1.0)
        p = a + t[:, None] * ab
        d = ((flat - p) ** 2).sum(-1)
        if best_d is None:
            best_d, best_p = d, p
        else:
            m = d < best_d
            best_d = np.where(m, d, best_d)
            best_p = np.where(m[:, None], p, best_p)
    return np.clip(np.round(best_p), 0, 255).astype(np.uint8).reshape(img_u8.shape)


TAGS = ("orig", "snap", "curve")


def evaluate(model_dir, it, subset, device):
    odir = Path(model_dir) / "test" / f"ours_{it}"
    rf = sorted((odir / "renders").glob("*.png"))
    if subset and 0 < subset < len(rf):
        rf = rf[:: max(1, len(rf) // subset)][:subset]
    P = build_palette(sorted((odir / "gt").glob("*.png")))
    tree = cKDTree(P.astype(np.float64))
    res = {t: {"PSNR": [], "SSIM": [], "LPIPS": []} for t in TAGS}
    res["off"], res["n"] = [], len(rf)
    for f in rf:
        r0 = np.asarray(Image.open(f).convert("RGB"))
        g = np.asarray(Image.open(odir / "gt" / f.name).convert("RGB"))
        variants = {"orig": r0,
                    "snap": project_snap(r0, P, tree),
                    "curve": project_curve(r0, P, tree)}
        res["off"].append(100.0 * (np.abs(r0.astype(int) - variants["snap"].astype(int)).max(-1) > 4).mean())
        gt = tf.to_tensor(Image.fromarray(g)).unsqueeze(0)[:, :3, :, :].to(device)
        for tag, arr in variants.items():
            t = tf.to_tensor(Image.fromarray(arr)).unsqueeze(0)[:, :3, :, :].to(device)
            res[tag]["PSNR"].append(psnr(t, gt).item())
            res[tag]["SSIM"].append(ssim(t, gt).item())
            res[tag]["LPIPS"].append(lpips(t, gt, net_type="vgg").item())
    for tag in TAGS:
        for k in ("PSNR", "SSIM", "LPIPS"):
            res[tag][k] = float(np.mean(res[tag][k]))
    res["off"] = float(np.mean(res["off"]))
    res["palette_size"] = int(len(P))
    return res


def main():
    ap = argparse.ArgumentParser(description="色标投影探针（只读已有渲染）")
    ap.add_argument("--root", required=True, help="结果树根 run_mlp_v2_阶段1b阶段2与长训")
    ap.add_argument("--methods", default="CBF MVDR")
    ap.add_argument("--arms", default="v14 sh", help="v14 / sh / 两者")
    ap.add_argument("--iteration", type=int, default=70000)
    ap.add_argument("--subset", type=int, default=0, help="0 = 全部视图")
    ap.add_argument("--out", default=None, help="可选：写 json")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    allout = {}
    for M in args.methods.split():
        for arm in args.arms.split():
            sub, label = ARMS[arm]
            md = root / sub / M
            if not (md / "test" / f"ours_{args.iteration}").is_dir():
                print(f"[skip] {M}/{arm}: 无 {md}/test/ours_{args.iteration}")
                continue
            r = evaluate(md, args.iteration, args.subset, device)
            allout[f"{M}/{arm}"] = r
            print(f"\n=== {M}  {label}   (n={r['n']}, 调色板 {r['palette_size']} 色) ===")
            print(f"  漂在调色板外(>4 色阶)的像素: {r['off']:.1f}%")
            print(f"  {'指标':6s} {'原图':>9s} {'吸附格点':>10s} {'投影曲线':>10s}   "
                  f"{'Δ吸附':>9s} {'Δ曲线':>9s}")
            for k in ("LPIPS", "PSNR", "SSIM"):
                a, s, c = r["orig"][k], r["snap"][k], r["curve"][k]
                star = "  ← 曲线更优" if (k == "LPIPS" and c < a) else ""
                print(f"  {k:6s} {a:9.4f} {s:10.4f} {c:10.4f}   {s-a:+9.4f} {c-a:+9.4f}{star}")
    if args.out:
        Path(args.out).write_text(json.dumps(allout, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
