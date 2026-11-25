# -*- coding: utf-8 -*-
"""波束权 w 轴的交付图 + 指标表。

读 `_w_export.py` 导出的 `数据.json`（四个组合的逐实例读数、训练集探针两两 PSNR、
定性对照图），出五张图和一份 `指标.md`。所有数字都从 JSON 实算，不写估计值。

五张图各自回答一个问题：

1. `图1_三指标.png`      留出 w 上四个组合的绝对三指标 —— 只说明「差多少」。
2. `图2_配对散点.png`     同一实例上条件臂 vs 盲臂 —— 逐实例配对，躲开组均值的掩盖。
3. `图3_训练集探针.png`   GT 之间 vs 模型之间的两两 PSNR —— **表示层判据**。
4. `图4_置换对照.png`     换成错的 w 之后 ΔPSNR —— 头到底在不在用 w。
5. `图5_定性对照.png`     GT / 条件臂 / 换 w 后 / 盲臂 四联 —— 肉眼可读出「图没变」。

用法（服务器，matplotlib 需能取到中文字体）::

    /root/miniconda3/bin/python code/tools/figures/make_w_figs.py \
        --data results/<run>/_小件 --out results/<run>/图
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import _cnfont                                        # noqa: E402

_cnfont.configure(plt)
plt.rcParams["axes.unicode_minus"] = False

KEYS = ["CBF/w", "CBF/blind", "MVDR/w", "MVDR/blind"]
LABEL = {"CBF/w": "CBF·波束条件臂", "CBF/blind": "CBF·波束盲臂",
         "MVDR/w": "MVDR·波束条件臂", "MVDR/blind": "MVDR·波束盲臂"}
COL = {"CBF/w": "#2f6fbe", "CBF/blind": "#8fb2d9",
       "MVDR/w": "#b5739d", "MVDR/blind": "#d9b0c9"}
METRICS = [("psnr", "PSNR (dB)  ↑"), ("ssim", "SSIM  ↑"), ("lpips", "LPIPS  ↓")]
IDX = {"psnr": 3, "ssim": 4, "lpips": 5}


def agg(rows, role="w_holdout"):
    v = [r for r in rows if r[1] == role]
    if not v:
        return None
    return {k: st.mean(r[IDX[k]] for r in v) for k, _ in METRICS} | {"n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="_w_export.py 的 --out 目录")
    ap.add_argument("--out", required=True, help="出图目录")
    args = ap.parse_args()
    d = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    J = json.loads((d / "数据.json").read_text(encoding="utf-8"))
    C = J["combos"]

    stats = {k: agg(C[k]["metrics"]) for k in KEYS}
    orig = {k: agg(C[k]["metrics"], "orig") for k in KEYS}
    # 缺任何一个组合的留出 w 读数就报错退出 —— 宁可不出图，也不要出一张少一根柱子的图
    miss = [k for k in KEYS if not stats[k]]
    if miss:
        raise SystemExit(f"[FAIL] 这些组合没有 w_holdout 读数：{miss}")

    # ---------- 图1 三指标柱状图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=200)
    x = np.arange(len(KEYS))
    for ax, (m, title) in zip(axes, METRICS):
        vals = [stats[k][m] for k in KEYS]
        ax.bar(x, vals, color=[COL[k] for k in KEYS], width=0.62,
               edgecolor="black", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        lo = min(vals)
        ax.set_ylim(lo * 0.94 if m != "lpips" else lo * 0.94,
                    max(vals) * 1.025 if m != "lpips" else max(vals) * 1.06)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[k].replace("·", "\n") for k in KEYS], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("留出波束权 w 上的三指标（36 个实例 × 640 视角）", fontsize=13)
    fig.tight_layout()
    p1 = out / "图1_三指标.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图2 配对散点 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=200)
    for ax, op in zip(axes, ("MVDR", "CBF")):
        a = {r[0]: r for r in C[f"{op}/w"]["metrics"] if r[1] == "w_holdout"}
        b = {r[0]: r for r in C[f"{op}/blind"]["metrics"] if r[1] == "w_holdout"}
        ks = sorted(set(a) & set(b))
        xs = [b[k][3] for k in ks]
        ys = [a[k][3] for k in ks]
        ax.plot([min(xs + ys), max(xs + ys)], [min(xs + ys), max(xs + ys)],
                "--", color="#888", lw=1, label="y = x")
        ax.scatter(xs, ys, s=34, color=COL[f"{op}/w"], edgecolor="black",
                   linewidth=0.5, zorder=3)
        win = sum(1 for k in ks if a[k][3] > b[k][3])
        dp = [a[k][3] - b[k][3] for k in ks]
        ax.set_xlabel("波束盲臂 PSNR (dB)", fontsize=10)
        ax.set_ylabel("波束条件臂 PSNR (dB)", fontsize=10)
        ax.set_title(f"{op}：逐实例配对（n={len(ks)}）\n"
                     f"条件臂赢 {win}/{len(ks)}，ΔPSNR 均值 {st.mean(dp):+.4f} dB",
                     fontsize=10.5)
        ax.grid(alpha=0.3)
    fig.suptitle("条件臂 vs 盲臂：留出 w 上逐实例配对", fontsize=13)
    fig.tight_layout()
    p2 = out / "图2_配对散点.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图3 训练集探针 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=200)
    for ax, op in zip(axes, ("MVDR", "CBF")):
        pr_w = C[f"{op}/w"]["probe"]
        pr_b = C[f"{op}/blind"]["probe"]
        if not pr_w.get("ok"):
            ax.text(0.5, 0.5, f"探针缺失：{pr_w.get('reason')}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        labels, vals, cols = ["GT 之间"], [pr_w["mean_gt"]], ["#444444"]
        if pr_b.get("ok"):
            labels.append("模型之间\n（条件臂）"); vals.append(pr_w["mean_model"])
            cols.append(COL[f"{op}/w"])
            labels.append("模型之间\n（盲臂）"); vals.append(pr_b["mean_model"])
            cols.append(COL[f"{op}/blind"])
        else:
            labels.append("模型之间\n（条件臂）"); vals.append(pr_w["mean_model"])
            cols.append(COL[f"{op}/w"])
        xx = np.arange(len(vals))
        ax.bar(xx, vals, color=cols, width=0.6, edgecolor="black", linewidth=0.6)
        for xi, v in zip(xx, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_ylabel("同位置 4 组 w 两两 PSNR (dB)", fontsize=10)
        ratio = pr_w.get("energy_ratio", float("nan"))
        # 小于 1% 时必须保留一位小数，否则 CBF 的 0.1% 会被打平成「0%」，
        # 读起来像「完全没算」，而实际是「算了，只是极小」
        pct = f"{ratio * 100:.1f}%" if ratio < 0.01 else f"{ratio * 100:.0f}%"
        ax.set_title(f"{op}：训练集内探针\n模型只表达 GT 变化的 {pct}", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("同位置、见过 w 的 4 个训练实例：GT 差得多，模型几乎不差", fontsize=13)
    fig.tight_layout()
    p3 = out / "图3_训练集探针.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图4 置换对照 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=200)
    swap_stat = {}
    for ax, op in zip(axes, ("MVDR", "CBF")):
        a = {r[0]: r for r in C[f"{op}/w"]["metrics"] if r[1] == "w_holdout"}
        s = {r[0]: r for r in C[f"{op}/w"].get("wswap_metrics", [])}
        ks = sorted(set(a) & set(s))
        if not ks:
            ax.text(0.5, 0.5, "无 wswap 读数", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off(); continue
        dp = [s[k][3] - a[k][3] for k in ks]
        ax.axhline(0, color="#888", lw=1)
        ax.scatter(np.full(len(dp), 1.0), dp, s=36, color=COL[f"{op}/w"],
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar([1.0], [st.mean(dp)], yerr=[st.pstdev(dp)], fmt="_",
                    color="#c0392b", markersize=18, capsize=6, zorder=4)
        ax.set_xlim(0.4, 1.6); ax.set_xticks([1.0])
        ax.set_xticklabels([f"k={len(dp)}"], fontsize=10)
        ax.set_ylabel("ΔPSNR = 换成错的 w − 原条件 (dB)", fontsize=10)
        ax.set_title(f"{op}\n均值 {st.mean(dp):+.4f} ± {st.pstdev(dp):.4f} dB",
                     fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        swap_stat[op] = (st.mean(dp), st.pstdev(dp), len(dp))
    fig.suptitle("w 置换对照：换成错误的波束权，图应当明显变差", fontsize=13)
    fig.tight_layout()
    p4 = out / "图4_置换对照.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图5 定性对照 ----------
    quals = J.get("qual", {})
    ops = [o for o in ("MVDR", "CBF") if o in quals]
    panels = [("gt", "GT"), ("render", "条件臂渲染"),
              ("wswap", "条件臂（换成错的 w）"), ("blind_render", "盲臂渲染")]

    def _px_psnr(p, q):
        """两幅渲染之间的逐像素 PSNR：给「换 w 后图没变」一个不靠眼睛的数字。"""
        if not (p.exists() and q.exists()):
            return None
        a = np.asarray(Image.open(p).convert("L"), np.float64) / 255.0
        b = np.asarray(Image.open(q).convert("L"), np.float64) / 255.0
        mse = float(np.mean((a - b) ** 2))
        return 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)

    swap_px = {}
    for o in ops:
        qd = d / "qual" / o
        swap_px[o] = _px_psnr(qd / "render.png", qd / "wswap.png")
    if ops:
        fig, axes = plt.subplots(len(ops), len(panels),
                                 figsize=(3.3 * len(panels), 2.6 * len(ops)),
                                 dpi=200, squeeze=False)
        for i, op in enumerate(ops):
            qd = d / "qual" / op
            for j, (k, title) in enumerate(panels):
                ax = axes[i][j]
                p = qd / f"{k}.png"
                if p.exists():
                    ax.imshow(np.asarray(Image.open(p).convert("L")),
                              cmap="jet", vmin=0, vmax=255)
                else:
                    ax.text(0.5, 0.5, "缺", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(title, fontsize=10.5)
                if j == 1 and swap_px.get(op) is not None:
                    # 标在第二列下方：与右邻那一列的逐像素 PSNR
                    ax.text(0.5, -0.06, f"与右侧（换 w 后）逐像素 {swap_px[op]:.1f} dB",
                            ha="center", va="top", transform=ax.transAxes,
                            fontsize=9, color="#c0392b")
                if j == 0:
                    ax.set_ylabel(f"{op}\n实例 {quals[op]['instance']}", fontsize=10)
        fig.suptitle("定性对照：同一实例同一视角（留出 w）", fontsize=13)
        fig.tight_layout()
        p5 = out / "图5_定性对照.png"
        fig.savefig(p5, bbox_inches="tight")
        plt.close(fig)
    else:
        p5 = None

    # ---------- 指标.md ----------
    L = ["# 波束权 w 轴：指标汇总", "",
         "所有数字由 `_w_export.py` 导出的逐实例读数实算。"
         "「留出 w」= 36 个训练位置 × 一个没见过的 w，各 640 视角。", ""]
    L += ["## 1. 留出 w 上的绝对三指标", "",
          "| 组合 | 实例 | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for k in KEYS:
        s = stats[k]
        L.append(f"| {LABEL[k]} | {s['n']} | {s['psnr']:.4f} | "
                 f"{s['ssim']:.4f} | {s['lpips']:.4f} |")
    L += ["", "## 2. 条件臂 vs 盲臂（逐实例配对）", "",
          "| 算子 | n | 条件臂赢 | ΔPSNR | ΔSSIM | ΔLPIPS |", "|---|---|---|---|---|---|"]
    pair_stat = {}
    for op in ("MVDR", "CBF"):
        a = {r[0]: r for r in C[f"{op}/w"]["metrics"] if r[1] == "w_holdout"}
        b = {r[0]: r for r in C[f"{op}/blind"]["metrics"] if r[1] == "w_holdout"}
        ks = sorted(set(a) & set(b))
        win = sum(1 for k in ks if a[k][3] > b[k][3])
        dm = [st.mean(a[k][IDX[m]] - b[k][IDX[m]] for k in ks) for m, _ in METRICS]
        pair_stat[op] = (len(ks), win, dm)
        L.append(f"| {op} | {len(ks)} | {win}/{len(ks)} | {dm[0]:+.4f} | "
                 f"{dm[1]:+.4f} | {dm[2]:+.4f} |")
    L += ["", "同输入重跑的噪声地板是 **0.365 dB**，上表 PSNR 差若小于它即不可分辨。", "",
          "## 3. 训练集内探针（表示层判据）", "",
          "同一位置、4 组都在训练集里的 w。GT 之间差多少 vs 模型输出之间差多少。", "",
          "| 算子·臂 | GT 之间 | 模型之间 | 模型表达的能量占比 |", "|---|---|---|---|"]
    for k in KEYS:
        p = C[k]["probe"]
        if p.get("ok"):
            L.append(f"| {LABEL[k]} | {p['mean_gt']:.2f} dB | "
                     f"{p['mean_model']:.2f} dB | {p['energy_ratio'] * 100:.1f}% |")
        else:
            L.append(f"| {LABEL[k]} | — | — | 缺失：{p.get('reason')} |")
    L += ["", "## 4. w 置换对照", "",
          "换成**错的** w 之后，ΔPSNR 越正越说明头在用 w；≈0 说明头对 w 不响应。", "",
          "| 算子 | n | ΔPSNR 均值 ± 标准差 |", "|---|---|---|"]
    for op, (mu, sd, n) in swap_stat.items():
        L.append(f"| {op} | {n} | {mu:+.4f} ± {sd:.4f} |")
    if any(v is not None for v in swap_px.values()):
        L += ["", "定性对照那张图上，替代前后两幅渲染的逐像素 PSNR："]
        for op, v in swap_px.items():
            if v is not None:
                L.append(f"- {op}：**{v:.2f} dB**（越高越说明「换了 w 图也没变」）")
    L += ["", "## 5. 原发射源单实例", "",
          "| 组合 | PSNR | SSIM | LPIPS |", "|---|---|---|---|"]
    for k in KEYS:
        o = orig[k]
        if o:
            L.append(f"| {LABEL[k]} | {o['psnr']:.4f} | {o['ssim']:.4f} | "
                     f"{o['lpips']:.4f} |")
    L += ["", "## 图", "",
          f"- `{p1.name}` 三指标柱状图",
          f"- `{p2.name}` 留出 w 逐实例配对散点",
          f"- `{p3.name}` 训练集内探针",
          f"- `{p4.name}` w 置换对照",
          f"- `{p5.name}` 定性对照" if p5 else "- （定性对照缺失）", ""]
    md = out / "指标.md"
    md.write_text("\n".join(L), encoding="utf-8")

    for p in (p1, p2, p3, p4, p5):
        if p and p.exists():
            print(f"[fig] {p}  {p.stat().st_size // 1024} KB")
    print(f"[md]  {md}")


if __name__ == "__main__":
    main()
