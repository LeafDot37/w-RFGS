# -*- coding: utf-8 -*-
"""
make_long_qualitative.py —— 阶段二「长训」定性对照图（每方法一张，PNG 本地/服务器同码）
--------------------------------------------------------------------------------
回答：长训到调度终点 t=1.0 之后，V14（MLP 头 + PE 全开）与 SH@A*（原版 SH 头 + A* LR）
      在**同一批帧**上到底差在哪？——柱状图（make_bars_long.py）给的是全局均值，
      本图给的是逐帧肉眼版：同一帧、同一切法，GT / V14 / SH 三行并排，每格标该帧 PSNR。

版式：3 行 × N 列（N 默认 4）
    第 1 行  GT              真值（不标 PSNR）
    第 2 行  V14 长训        阶段2长训_位置编码_X5D3R1/<M>/test/ours_<it>/renders
    第 3 行  SH@A* 长训      阶段1b长训_SH头/<M>/test/ours_<it>/renders
每格标题 = 该帧 PSNR（与 metrics.py 同一实现、同一 render↔gt 同名声配对）；
行标签另附该臂 **640 全量均值**（取自 results.json）便于与柱状图对表。

数据源（<root> = run_mlp_v2_阶段1b阶段2与长训；renders 与 gt **同文件名** 00000..00639）：
    V14  <root>/阶段2长训_位置编码_X5D3R1/<M>/test/ours_<it>/{renders,gt}/<idx>.png
    SH   <root>/阶段1b长训_SH头/<M>/test/ours_<it>/{renders,gt}/<idx>.png
    GT   取自 V14 的 gt/（两臂 gt 同源，同 index 即同帧）

**中文字体**：本图原本只在本地存在 → 当初就是本地出的。服务器（Linux）多半没有 CJK 字体，
    中文行标签会变豆腐块；因此约定**在本地跑**。只需要那几十张选定帧，不必整目录同步：
        for M in MPC CBF Delay MVDR; do
          for S in 阶段2长训_位置编码_X5D3R1 阶段1b长训_SH头; do
            for F in 00000 00130 00260 00390; do
              scp -r "yuexf@10.250.183.254:/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训/$S/$M/test/ours_70000/{renders,gt}/$F.png" \
                     "STAGE/$S/$M/test/ours_70000/"   # STAGE = 本地暂存根，保持 <root> 相对结构
            done
          done
        done
    （gt/ 只取一份即可，脚本会优先用 V14 的 gt/，缺失时回落到 SH 的 gt/）

用法：
    本地（mvsplat env；STAGE 见上）：
        python code/tools/figures/make_long_qualitative.py --root STAGE \
            --out results/run_mlp_v2_阶段1b阶段2与长训/阶段2长训_三指标
    服务器（有 CJK 字体时）：
        <PY> tools/make_long_qualitative.py --root /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训

输出（默认 <root>/阶段2长训_三指标/）：
    <M>/长训对照_<M>.png    3×N 定性对照图
    <M>/长训对照_<M>.json   该图用的帧号、逐帧 PSNR、各臂 640 均值（可入文档）
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cnfont  # noqa: E402
_cnfont.configure(plt)

import torch  # noqa: E402
import torchvision.transforms.functional as tf  # noqa: E402
from splat.utils.image_utils import psnr  # noqa: E402

ROWS = [  # (臂键, 行标签前缀, 源目录, 是否标 PSNR)
    ("gt",  "GT（真值）",                    None,                            False),
    ("v14", "V14 长训（MLP+PE+dist）",       "阶段2长训_位置编码_X5D3R1",     True),
    ("shA", "SH@A* 长训（原版 SH + A* LR）", "阶段1b长训_SH头",               True),
]
SRC = {k: d for k, _, d, _ in ROWS if d}


def _psnr_one(model_dir, it, idx):
    """单帧 PSNR（与 metrics.py 同实现；render↔gt 同名配对）。缺文件 → None。"""
    odir = model_dir / "test" / f"ours_{it}"
    rp, gp = odir / "renders" / f"{idx:05d}.png", odir / "gt" / f"{idx:05d}.png"
    if not (rp.is_file() and gp.is_file()):
        return None
    r = tf.to_tensor(Image.open(rp)).unsqueeze(0)[:, :3, :, :]
    g = tf.to_tensor(Image.open(gp)).unsqueeze(0)[:, :3, :, :]
    return psnr(r, g).item()


def _mean_640(model_dir, it):
    """该臂 640 全量均值（results.json，缺则 None）。"""
    p = model_dir / "results.json"
    if not p.is_file():
        return None
    return (json.load(open(p, encoding="utf-8")).get(f"ours_{it}") or {}).get("PSNR")


def _img(model_dir, it, idx, sub):
    p = model_dir / "test" / f"ours_{it}" / sub / f"{idx:05d}.png"
    return np.asarray(Image.open(p).convert("RGB")) if p.is_file() else None


def figure_method(root, method, it, frames, out_dir):
    dirs = {k: root / SRC[k] / method for k in SRC}
    if not dirs["v14"].is_dir() and not dirs["shA"].is_dir():
        print(f"[skip] {method}: 无渲染目录（{dirs['v14']} / {dirs['shA']}）")
        return None

    gtdir = dirs["v14"] if (dirs["v14"] / "test").is_dir() else dirs["shA"]
    n = len(frames)
    fig, axes = plt.subplots(len(ROWS), n, figsize=(3.15 * n, 2.55 * len(ROWS)), dpi=140)
    axes = np.atleast_2d(axes)
    if axes.shape != (len(ROWS), n):          # n==1 时 numpy 会退化
        axes = axes.reshape(len(ROWS), n)

    means = {k: _mean_640(dirs[k], it) for k in ("v14", "shA")}
    payload = {"method": method, "iteration": it, "frames": list(frames),
               "mean_640_PSNR": means, "per_frame_PSNR": {}, "source": {k: str(v) for k, v in dirs.items()}}

    for r, (key, label, _src, show_psnr) in enumerate(ROWS):
        md = gtdir if key == "gt" else dirs[key]
        m = means.get(key)
        row_label = label + (f"\n640 均值 {m:.3f} dB" if m is not None else "")
        for c, idx in enumerate(frames):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.8)
            img = _img(md, it, idx, "gt" if key == "gt" else "renders")
            if img is None:
                ax.text(0.5, 0.5, "缺图", ha="center", va="center", fontsize=9, color="#bbbbbb")
                ax.set_title(f"#{idx:05d}", fontsize=8, color="#888888")
                continue
            ax.imshow(img)
            if show_psnr:
                v = _psnr_one(md, it, idx)
                payload["per_frame_PSNR"].setdefault(key, {})[f"{idx:05d}"] = v
                ax.set_title(f"PSNR {v:.2f} dB" if v is not None else "PSNR —",
                             fontsize=9, color="#1a1a1a", pad=4)
            else:
                ax.set_title(f"帧 #{idx:05d}", fontsize=9, color="#1a1a1a", pad=4)
        axes[r][0].set_ylabel(row_label, fontsize=10, labelpad=8, va="center")

    m_v, m_s = means.get("v14"), means.get("shA")
    delta = f"   ΔPSNR(V14−SH@A*) = {m_v - m_s:+.3f} dB" if (m_v is not None and m_s is not None) else ""
    fig.suptitle(f"{method} 长训对照（RRF 4 万步 = abs{it}，640 全量 test）{delta}",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012,
             "V14 = MLP 颜色头 + PE(xyzL5 + dirL3) + 距离条件；SH@A* = 原版逐点 SH + A* exp LR。"
             "每格 PSNR 为该帧 render 与 gt（同名配对，同 metrics.py 实现）。",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0.012, 0.035, 1, 0.955))

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"长训对照_{method}.png"
    fig.savefig(png); plt.close(fig)
    json.dump(payload, open(out_dir / f"长训对照_{method}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[ok] {png}  " + "  ".join(
        f"{k}: {means[k]:.3f}" if means.get(k) is not None else f"{k}: —" for k in ("v14", "shA")))
    return payload


def main():
    ap = argparse.ArgumentParser(description="阶段二长训定性对照图（GT / V14 / SH@A*，3×N）")
    ap.add_argument("--root", default="/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训",
                    help="结果树根（本地跑时指向暂存根，保持同样的相对结构）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <root>/阶段2长训_三指标）")
    ap.add_argument("--methods", default="MPC CBF Delay MVDR", help="空格分隔的方法名")
    ap.add_argument("--iteration", type=int, default=70000, help="绝对迭代档位（长训末档）")
    ap.add_argument("--frames", default="0 130 260 390",
                    help="render 索引（空格分隔；与 gt/ 同文件名）")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "阶段2长训_三指标"
    frames = [int(x) for x in args.frames.split()]
    for m in args.methods.split():
        figure_method(root, m, args.iteration, frames, out / m)
    print(f"写入 {out}\n")


if __name__ == "__main__":
    main()
