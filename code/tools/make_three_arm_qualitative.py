# -*- coding: utf-8 -*-
"""
make_three_arm_qualitative.py —— 两组预算三臂对比定性图（每方法一张，4 行 × N 帧）
--------------------------------------------------------------------------------
回答：柱状图给的是 640 全量均值，本图给逐帧肉眼版 —— 同一帧、同一切法，
GT / A1 / A2 / A3 四行并排，每格标**该帧**的 PSNR。

    第 1 行  GT  真值（不标 PSNR）
    第 2 行  A1  原版物理 + 原版RRF
    第 3 行  A2  原版物理 + 新RRF
    第 4 行  A3  新物理   + 新RRF

数据源：**驱动 RENDER 段拷进本树的 4 帧**（基线保持只读，靠符号链接视图渲染）：
    <run>/<组>/阶段3_RRF重建/<臂>/<M>/定性抽帧/{renders,gt}/<idx>.png
哪一臂落在哪个目录、绝对迭代是多少，一律读 `清单/*.json`，不在这里硬编码 ——
复用臂（如组1 A1 直接用 run_rrf40k_sh 的 abs60000）是数据说了算。

每格 PSNR **本地从拉回的 renders 与 gt 现算**（与 metrics 同一实现、同一同名配对），
**绝不转述图里的数字** —— [[multiband-scalar-verdict]] 登记过这个教训：看图模型会
"读出"并不存在的数值。行标签另附该臂 640 全量均值（取自 results.json）便于对表柱状图。

用法（本地仓库根；中文字体只有本地有，**约定本地跑**）：
    python code/tools/make_three_arm_qualitative.py \
        --run 'results/run_20260912_两组预算_三臂对比' --group 组1_物理30000_RRF30000
    ... --methods "MPC CBF Delay MVDR" --frames "0 160 320 480"

输出（默认 <run>/<组>/图像对比/）：
    <M>/三臂对照_<M>.png
    <M>/三臂对照_<M>.json   帧号、逐帧 PSNR、各臂 640 均值、各臂来源与参数量
"""
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

GROUP_NAME = {1: "组1_物理30000_RRF30000", 2: "组2_物理10000_RRF20000"}
# 与 make_three_arm_bars.py 同表（臂键 → 短标签/行标签/目录名）
ARMS = [
    ("A1", "A1 原版物理+原版RRF", "A1_原版物理_原版RRF"),
    ("A2", "A2 原版物理+新RRF", "A2_原版物理_新RRF"),
    ("A3", "A3 新物理+新RRF", "A3_新物理_新RRF"),
]


def load_manifest(run: Path):
    out = {}
    for f in sorted((run / "清单").glob("*.json")):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] 清单读不动 {f}: {e}")
            continue
        out[(int(r["组"]), r["臂"], r["方法"])] = r
    return out


def _psnr_one(d, idx):
    """单帧 PSNR（与 metrics 同实现；render↔gt 同名配对）。缺文件 → None。"""
    rp, gp = d / "renders" / f"{idx:05d}.png", d / "gt" / f"{idx:05d}.png"
    if not (rp.is_file() and gp.is_file()):
        return None
    r = tf.to_tensor(Image.open(rp)).unsqueeze(0)[:, :3, :, :]
    g = tf.to_tensor(Image.open(gp)).unsqueeze(0)[:, :3, :, :]
    return psnr(r, g).item()


def _img(d, sub, idx):
    p = d / sub / f"{idx:05d}.png"
    return np.asarray(Image.open(p).convert("RGB")) if p.is_file() else None


def _mean_640(run: Path, rec):
    """该臂 640 全量均值（results.json，缺则 None）。"""
    p = run / rec["本树相对路径"] / "results.json"
    if not p.is_file():
        return None
    return (json.load(open(p, encoding="utf-8")).get(f"ours_{rec['绝对迭代']}") or {}).get("PSNR")


def figure_method(run, gno, method, frames, out_dir, mani):
    dirs, recs, notes = {}, {}, []
    for key, _label, dname in ARMS:
        rec = mani.get((gno, key, method))
        if rec is None:
            notes.append(f"{key} {method}: 清单里没有这个节点")
            dirs[key] = None
            continue
        d = run / rec["本树相对路径"] / "定性抽帧"
        if not d.is_dir():
            notes.append(f"{key} {method}: 无 定性抽帧/（RENDER 段没拷？）{d}")
            dirs[key] = None
            continue
        dirs[key], recs[key] = d, rec
    if not any(dirs.values()):
        print(f"[skip] {method}: 三臂都没有定性抽帧")
        return None

    means = {k: (_mean_640(run, recs[k]) if recs.get(k) else None) for k, *_ in ARMS}
    n = len(frames)
    nrow = 1 + len(ARMS)
    fig, axes = plt.subplots(nrow, n, figsize=(3.15 * n, 2.55 * nrow), dpi=140)
    axes = np.atleast_2d(axes)
    if axes.shape != (nrow, n):               # n==1 时 numpy 会退化
        axes = axes.reshape(nrow, n)

    payload = {"组": GROUP_NAME[gno], "方法": method, "帧": [f"{i:05d}" for i in frames],
               "逐帧PSNR": {}, "臂640均值PSNR": means, "缺数据说明": notes,
               "来源": {}}
    for k, *_ in ARMS:
        if recs.get(k):
            payload["来源"][k] = {kk: recs[k].get(kk) for kk in
                                  ("绝对迭代", "引擎", "源模型目录", "是否复用", "模型是否复用",
                                   "渲染是否复用", "复用说明",
                                   "高斯数", "头部参数", "头部参数说明", "渲染来源目录")}

    gtsrc = next((dirs[k] for k, *_ in ARMS if dirs.get(k) and (dirs[k] / "gt").is_dir()), None)
    for r in range(nrow):
        if r == 0:
            md, label, key, show = gtsrc, "GT（真值）", "gt", False
        else:
            key, lab, _d = ARMS[r - 1]
            md, label, show = dirs[key], lab, True
            m = means.get(key)
            if m is not None:
                label += f"\n640 均值 {m:.3f} dB"
            # 用**模型**复用标志，不用 `是否复用` —— 后者指的是"640 张渲染取自基线"，
            # 两者会不一致：组1/2 A2 的 MVDR/CBF 模型是只读复用的，但基线只有 48 张渲染，
            # 640 张是本树新渲的 ⇒ 只看 `是否复用` 会给这两臂漏标"（复用）"。
            _r = recs.get(key) or {}
            _mr = _r.get("模型是否复用", _r.get("是否复用"))
            if _mr:
                label += "（模型复用基线）"
        for c, idx in enumerate(frames):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.8)
            if md is None:
                ax.text(0.5, 0.5, "无此臂", ha="center", va="center",
                        fontsize=9, color="#bbbbbb")
                ax.set_title(f"#{idx:05d}", fontsize=8, color="#888888")
                continue
            img = _img(md, "gt" if r == 0 else "renders", idx)
            if img is None:
                ax.text(0.5, 0.5, "缺图", ha="center", va="center",
                        fontsize=9, color="#bbbbbb")
                ax.set_title(f"#{idx:05d}", fontsize=8, color="#888888")
                continue
            ax.imshow(img)
            if show:
                v = _psnr_one(md, idx)
                payload["逐帧PSNR"].setdefault(key, {})[f"{idx:05d}"] = v
                ax.set_title(f"PSNR {v:.2f} dB" if v is not None else "PSNR —",
                             fontsize=9, color="#1a1a1a", pad=4)
            else:
                ax.set_title(f"帧 #{idx:05d}", fontsize=9, color="#1a1a1a", pad=4)
        axes[r][0].set_ylabel(label, fontsize=10, labelpad=8, va="center")

    a1, a3 = means.get("A1"), means.get("A3")
    delta = f"   ΔPSNR(A3−A1) = {a3 - a1:+.3f} dB" if (a1 is not None and a3 is not None) else ""
    fig.suptitle(f"[{method}] {GROUP_NAME[gno]} 三臂定性对照（640 全量 test）{delta}",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012,
             "A1 原版物理+原版RRF（原版逐点 SH）；A2 原版物理+新RRF（MPC/Delay 用 MLP 头、"
             "MVDR/CBF 用 A× 标量头）；A3 新物理+新RRF（MVSplat init + 短训，颜色同 A2）。"
             "每格 PSNR 本地从 render 与 gt（同名配对）现算。",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0.012, 0.035, 1, 0.955))

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"三臂对照_{method}.png"
    fig.savefig(png); plt.close(fig)
    json.dump(payload, open(out_dir / f"三臂对照_{method}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[ok] {png}  " + "  ".join(
        f"{k}: {means[k]:.3f}" if means.get(k) is not None else f"{k}: —" for k, *_ in ARMS))
    for nt in notes:
        print(f"  [note] {nt}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="两组预算三臂定性对照图（GT/A1/A2/A3，4×N）")
    ap.add_argument("--run", required=True, help="本地 run 根")
    ap.add_argument("--group", default="", help="组目录名；空则 --group-no 决定")
    ap.add_argument("--group-no", type=int, default=1, choices=[1, 2])
    ap.add_argument("--methods", default="MPC CBF Delay MVDR")
    ap.add_argument("--frames", default="0 160 320 480",
                    help="render 索引（空格分隔；与 gt/ 同文件名）")
    ap.add_argument("--out", default="", help="输出目录（默认 <run>/<组>/图像对比）")
    args = ap.parse_args()

    run = Path(args.run)
    gno = args.group_no
    if args.group:
        if args.group not in GROUP_NAME.values():
            raise SystemExit(f"--group 不认: {args.group}（可选 {list(GROUP_NAME.values())}）")
        gno = next(k for k, v in GROUP_NAME.items() if v == args.group)
    out = Path(args.out) if args.out else run / GROUP_NAME[gno] / "图像对比"
    frames = [int(x) for x in args.frames.split()]

    mani = load_manifest(run)
    if not mani:
        raise SystemExit(f"清单为空：{run}/清单/*.json（先跑驱动 RENDER 段再 pull）")
    print(f"清单 {len(mani)} 节点；组 {GROUP_NAME[gno]}；帧 {frames}")
    for m in args.methods.split():
        figure_method(run, gno, m, frames, out / m, mani)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
