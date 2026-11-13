# -*- coding: utf-8 -*-
"""make_figs_v2.py —— 论文第二轮的补充插图（示意图 + 三个成像算子的定性对照 + w·CBF 指标）。

与 make_figs.py 同一套约定：**所有数字从结果文件读**，本脚本只放出处路径与版式；
线图存 PDF、定性拼图存 PNG 300 dpi；单栏 W1=3.45 in、双栏 W2=7.16 in。

产物
----
fig2_gap.pdf          为什么不泛化就得重训（概念图，无数据）
fig9_w_cbf.pdf        w·CBF：条件臂 vs 盲臂三指标（读数来自 图/数据.json 的 combos.CBF/*）
fig10_qual_mvdr.png   单发射源 MVDR 定性：GT / RF-3DGS(MVDR) / 本方法
fig11_qual_w_cbf.png  阵列 w·CBF 定性：GT / 条件臂 / 盲臂（3 个留出 w 实例）
fig12_qual_w_mvdr.png 阵列 w·MVDR 定性：GT / 复值等权 / 复值感知 / 盲臂（2 实例 × 2 视角）

用法：python latex/make_figs_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(parents=True, exist_ok=True)

# —— 出处路径 ——
RUNM = ROOT / "results/run_20261027_服务器4090D_MVDR参数泛化"
QUAL_M = RUNM / "_xfer/qual_cbfviews/inst317"          # gt / renders / rf3dgs_mvdr
RUNW1 = ROOT / "results/run_20261029_服务器4090D_波束权泛化"
QUAL_W_CBF = RUNW1 / "_raw/qual"                        # {w,blind}/instXXX/{gt,renders}
DATA_W = RUNW1 / "图/数据.json"                          # 逐实例读数
RUNW2 = ROOT / "results/run_20261101_服务器4080_波束权复值场"
QUAL_W_MVDR = RUNW2 / "_图源"                            # GT_i###_vvvvv.png 等平铺命名

W1, W2 = 3.45, 7.16
C_OURS, C_REF, C_FLOOR, C_BLIND = "#d1495b", "#8c8c8c", "#2a9d8f", "#4a6fa5"
C_W2 = "#e07b39"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# fig2_gap —— 逐场景重训 vs 一步解耦
# --------------------------------------------------------------------------- #
def fig2_gap():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.62),
                            gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.10})
    for ax in axs:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, txt, fc, ec="#333333", fs=6.6, tc="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.5",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color="#333333", shrinkA=1, shrinkB=2))

    # (a) 逐场景：每个发射源配置各训一个场
    axs[0].set_title("(a) Per-scene radio radiance field\n(RF-3DGS, CORF-GS)",
                     fontsize=7.6, pad=3)
    ys = [7.2, 4.4, 1.6]
    for i, (y, tag) in enumerate(zip(ys, [r"$\theta_1$", r"$\theta_2$", r"$\theta_N$"])):
        box(axs[0], 0.2, y, 2.0, 1.6, "scene\n+ " + tag, "#eef2f7")
        box(axs[0], 3.1, y, 2.1, 1.6, "per-scene\noptimisation", "#f6e7e1")
        box(axs[0], 6.0, y, 2.3, 1.6, r"field $F_{\theta_%s}$" % ["1", "2", "N"][i], "#e8f0e8")
        arrow(axs[0], 2.25, y + 0.8, 3.05, y + 0.8)
        arrow(axs[0], 5.25, y + 0.8, 5.95, y + 0.8)
    axs[0].text(8.55, 5.0, "one training\nper transmitter\nconfiguration",
                ha="center", va="center", fontsize=6.3, color="#8a2d1e")
    axs[0].text(8.55, 4.15, r"$\propto\,|\{\theta\}|$", ha="center", va="center",
                fontsize=7.2, color="#8a2d1e")
    axs[0].plot([0.6, 8.2], [3.45, 3.45], ls=":", lw=0.6, color="#999999")
    axs[0].plot([0.6, 8.2], [6.3, 6.3], ls=":", lw=0.6, color="#999999")

    # (b) 本方法：一次训练，任意参数
    axs[1].set_title("(b) This work: transmitter parameters\ncondition a single feed-forward model",
                     fontsize=7.6, pad=3)
    box(axs[1], 0.1, 6.1, 2.0, 1.7, "scene", "#eef2f7")
    box(axs[1], 0.1, 1.6, 2.0, 1.9, "parameters\n" + r"$\theta=(\mathbf{p},\alpha,\beta,f,\mathbf{w})$",
        "#f6e7e1", fs=6.0)
    box(axs[1], 3.0, 3.5, 2.6, 2.6, "one model\ntrained once", "#d1495b", fs=6.6, tc="white")
    box(axs[1], 6.4, 4.0, 2.2, 1.6, r"RF field $F_\theta$", "#e8f0e8")
    arrow(axs[1], 2.15, 6.95, 2.95, 5.6)
    arrow(axs[1], 2.15, 2.55, 2.95, 4.0)
    arrow(axs[1], 5.65, 4.8, 6.35, 4.8)
    axs[1].text(7.5, 2.2, "any unseen\nconfiguration", ha="center", va="center",
                fontsize=6.4, color="#8a2d1e")
    axs[1].text(7.5, 1.35, "one forward pass", ha="center", va="center",
                fontsize=6.4, color="#8a2d1e")

    for a in axs:
        for s in ("top", "right", "bottom", "left"):
            a.spines[s].set_visible(True)
            a.spines[s].set_linewidth(0.5)
            a.spines[s].set_color("#bbbbbb")
    fig.savefig(FIG / "fig2_gap.pdf")
    fig.savefig(FIG / "fig2_gap.png", dpi=300)
    plt.close(fig)
    print("[fig2_gap] ok")


# --------------------------------------------------------------------------- #
# fig9 —— w·CBF 两臂三指标
# --------------------------------------------------------------------------- #
def fig9_w_cbf():
    d = jl(DATA_W)["combos"]
    arms = [("CBF/w", "w-conditioned", C_W2), ("CBF/blind", "w-blind", C_BLIND)]
    means = []
    for k, lab, c in arms:
        # 行格式：[id, role, n, psnr, ssim, lpips]；只取留出 w 的 36 行
        rows = [r for r in d[k]["metrics"] if r[1] == "w_holdout"]
        px = np.array([r[3] for r in rows])
        sx = np.array([r[4] for r in rows])
        lx = np.array([r[5] for r in rows])
        means.append((lab, c, float(px.mean()), float(sx.mean()), float(lx.mean()), len(px)))

    keys = [(2, "PSNR (dB) $\\uparrow$", ".3f"), (3, "SSIM $\\uparrow$", ".4f"),
            (4, "LPIPS $\\downarrow$", ".4f")]
    fig, axs = plt.subplots(1, 3, figsize=(W1 * 2.06, 1.62))
    xs = np.arange(len(means))
    for ax, (j, lab, fmt) in zip(axs, keys):
        vals = [m[j] for m in means]
        ax.bar(xs, vals, color=[m[1] for m in means], width=0.58,
               edgecolor="black", lw=0.4)
        for x, v in zip(xs, vals):
            ax.text(x, v, format(v, fmt), ha="center", va="bottom",
                    fontsize=6.0, rotation=90)
        lo, hi = min(vals), max(vals)
        pad = max(hi - lo, abs(hi) * 1e-3) * 0.6
        ax.set_ylim(lo - pad * 1.2, hi + pad * 4.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([m[0] for m in means], fontsize=6.6)
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.2, lw=0.4)
        ax.tick_params(length=2)
    axs[0].set_title("(a) PSNR", fontsize=7.5)
    axs[1].set_title("(b) SSIM", fontsize=7.5)
    axs[2].set_title("(c) LPIPS", fontsize=7.5)
    n = means[0][5]
    fig.savefig(FIG / "fig9_w_cbf.pdf")
    fig.savefig(FIG / "fig9_w_cbf.png", dpi=300)
    plt.close(fig)
    print(f"[fig9] n={n} | " + " | ".join(
        f"{m[0]} {m[2]:.3f}/{m[3]:.4f}/{m[4]:.4f}" for m in means))


# --------------------------------------------------------------------------- #
# 定性拼图（通用）
# --------------------------------------------------------------------------- #
def montage(cells, col_titles, row_labels, out, dpi=300):
    """cells: 行 × 列 的 PIL 图；col_titles: 列标题；row_labels: 行标签。"""
    nr, nc = len(cells), len(cells[0])
    h, w = cells[0][0].shape[:2]
    gap = 4
    width = W2 if nc >= 4 else W1 * 2.06
    fig, axs = plt.subplots(nr, nc,
                            figsize=(width, width * (nr * h + (nr - 1) * gap)
                                     / (nc * w + (nc - 1) * gap)),
                            gridspec_kw={"wspace": 0.012, "hspace": 0.03},
                            squeeze=False)
    for r in range(nr):
        for c in range(nc):
            axs[r, c].imshow(cells[r][c], aspect="auto")
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            for s in axs[r, c].spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                axs[r, c].set_title(col_titles[c], fontsize=7.0, pad=2.5)
    fig.subplots_adjust(left=0.055, right=0.999, top=0.93, bottom=0.005)
    placed = []
    for r in range(nr):
        pos = axs[r, 0].get_position()
        yc = (pos.y0 + pos.y1) / 2
        if r == 0 or row_labels[r] != row_labels[r - 1]:
            fig.text(0.014, yc, row_labels[r], rotation=90, ha="center",
                     va="center", fontsize=6.2)
    fig.savefig(FIG / out, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"[{out}] {nr}x{nc}")


def im(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"缺图：{p}")
    return np.asarray(Image.open(p).convert("RGB"))


def fig10_qual_mvdr():
    views = ["00005", "00649", "01501", "02328"]
    rows_m = [("Ground truth", QUAL_M / "gt"),
              ("RF-3DGS (released, MVDR)", QUAL_M / "rf3dgs_mvdr"),
              ("Ours (MVDR)", QUAL_M / "renders")]
    cells = []
    for _, d in rows_m:
        cells.append([im(d / f"{v}.png") for v in views])
    montage(cells, views, [t for t, _ in rows_m], "fig10_qual_mvdr.png")


def fig11_qual_w_cbf():
    insts = [331, 333, 349]
    cols = [("Ground truth", QUAL_W_CBF / "w" / "inst331" / "gt"),
            ("w-conditioned", QUAL_W_CBF / "w"),
            ("w-blind (control)", QUAL_W_CBF / "blind")]
    cells, labels = [], []
    for i in insts:
        row = []
        for k, (_, d) in enumerate(cols):
            base = d if k == 0 else d / f"inst{i:03d}" / "renders"
            row.append(im(base / "00005.png"))
        cells.append(row)
        labels.append(f"w-inst {i}")
    montage(cells, [t for t, _ in cols], labels, "fig11_qual_w_cbf.png")


def fig12_qual_w_mvdr():
    pairs = [(318, "00005"), (318, "00052"), (340, "00005"), (340, "00052")]
    names = [("GT", "GT"), ("complex, $\\lambda_p{=}0$", "复值_等权"),
             ("complex, +LPIPS", "复值_感知"), ("w-blind", "盲臂_感知")]
    cells, labels = [], []
    for i, v in pairs:
        cells.append([im(QUAL_W_MVDR / f"{pre}_i{i}_{v}.png") for _, pre in names])
        labels.append(f"w-inst {i}, view {v.lstrip('0')}")
    montage(cells, [t for t, _ in names], labels, "fig12_qual_w_mvdr.png")


def main():
    fig2_gap()
    fig9_w_cbf()
    fig10_qual_mvdr()
    fig11_qual_w_cbf()
    fig12_qual_w_mvdr()


if __name__ == "__main__":
    main()
