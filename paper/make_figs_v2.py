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
QUAL_M = RUNM / "全量评测/test_full/inst317"            # 我方协议：gt / renders（160 视角）
# 作者协议：发布渲染按 test_index.txt 的行号命名（00000.png …），我方视角号 → 行号
MV_AUTH = ROOT / ("datasets/RF-3DGS_dataset/RF-3DGS_trained_RRF/"
                  "3dgs_MVDR_100/test/ours_40000")       # gt / renders
MV_TEST = ROOT / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/test_index.txt"
RUNW1 = ROOT / "results/run_20261029_服务器4090D_波束权泛化"
QUAL_W_CBF = RUNW1 / "_raw/qual_sel"                     # wcbf_<inst>_<view>_{gt,w,blind}.png
DATA_W = RUNW1 / "图/数据.json"                          # 逐实例读数
RUNW2 = ROOT / "results/run_20261101_服务器4080_波束权复值场"
QUAL_W_MVDR = RUNW2 / "_图源_sel"                        # wmvdr_<inst>_<view>_{gt,eq,ps,bl}.png
BRIGHT = ROOT / "results/_brightness_analysis.json"      # 亮区分档 / 暖色覆盖率 / 增益标定
QUAL_TTA = RUNM / "小样本适应/定性_新选"                  # tta_<view>_{gt,zs,ad}.png

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
    """两协议并列：上两行是作者协议（GT + 其发布渲染），下两行是我方协议。

    两个 GT 生成器本身差 3–8 dB，所以**每行只允许与同协议的 GT 比**；本图的作用
    正是把这条断层摆出来。视角号是我方的（作者侧按 test_index.txt 行号换算）。
    """
    views = ["00006", "00100", "00380", "00424"]
    lines = [s.strip() for s in MV_TEST.read_text(encoding="utf-8").splitlines() if s.strip()]
    rows_m = [
        ("Their GT", lambda v: MV_AUTH / "gt" / f"{lines.index(v):05d}.png"),
        ("RF-3DGS (theirs)", lambda v: MV_AUTH / "renders" / f"{lines.index(v):05d}.png"),
        ("Our GT", lambda v: QUAL_M / "gt" / f"{v}.png"),
        ("Ours", lambda v: QUAL_M / "renders" / f"{v}.png"),
    ]
    cells = [[im(f(v)) for v in views] for _, f in rows_m]
    montage(cells, views, [t for t, _ in rows_m], "fig10_qual_mvdr.png")


def fig11_qual_w_cbf():
    cols = [(349, "02172"), (351, "01822"), (349, "01055"), (351, "02216")]
    rows_k = [("Ground truth", "gt"), ("w-conditioned", "w"), ("w-blind (control)", "blind")]
    cells = [[im(QUAL_W_CBF / f"wcbf_{i}_{v}_{k}.png") for i, v in cols] for _, k in rows_k]
    montage(cells, [f"w-inst {i} / view {int(v)}" for i, v in cols],
            [t for t, _ in rows_k], "fig11_qual_w_cbf.png")


def fig12_qual_w_mvdr():
    cols = [(349, "00098"), (350, "00118"), (350, "01925"), (343, "01232")]
    rows_k = [("Ground truth", "gt"), ("complex, $\\lambda_p{=}0$", "eq"),
              ("complex, +LPIPS", "ps"), ("w-blind (control)", "bl")]
    cells = [[im(QUAL_W_MVDR / f"wmvdr_{i}_{v}_{k}.png") for i, v in cols] for _, k in rows_k]
    montage(cells, [f"w-inst {i} / view {int(v)}" for i, v in cols],
            [t for t, _ in rows_k], "fig12_qual_w_mvdr.png")


def fig14_qual_tta():
    """微调定性：GT / 零样本 / 适应后，实例 150 的 4 个留出视角。"""
    views = ["01503", "00442", "00286", "02722"]
    rows_k = [("Ground truth", "gt"), ("Zero-shot", "zs"),
              ("+320 views, 20 k steps", "ad")]
    cells = [[im(QUAL_TTA / f"tta_{v}_{k}.png") for v in views] for _, k in rows_k]
    montage(cells, [f"view {int(v)}" for v in views],
            [t for t, _ in rows_k], "fig14_qual_tta.png")


def fig13_where():
    """亮区分档误差 + 暖色覆盖率：说明优势集中在暗视角、且 w 头到不了色标暖端。"""
    d = jl(BRIGHT)
    q = d["strat_cbf"]["quartiles"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 1.72),
                                   gridspec_kw={"width_ratios": [1.0, 1.18], "wspace": 0.30})
    x = np.arange(len(q))
    w = 0.38
    a = [r["ours"] for r in q]
    b = [r["ref"] for r in q]
    ax1.bar(x - w / 2, a, w, color=C_OURS, edgecolor="black", lw=0.4, label="Ours")
    ax1.bar(x + w / 2, b, w, color=C_REF, edgecolor="black", lw=0.4, label="RF-3DGS")
    for xi, va, vb in zip(x, a, b):
        ax1.text(xi - w / 2, va + 0.25, f"{va:.1f}", ha="center", va="bottom", fontsize=5.8)
        ax1.text(xi + w / 2, vb + 0.25, f"{vb:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r["q"]}\n{r["bright"]:.3f}' for r in q], fontsize=6.4)
    ax1.set_xlabel("Ground-truth bright-pixel fraction (quartile)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_ylim(0, max(max(a), max(b)) * 1.22)
    ax1.legend(loc="upper right", frameon=False, handlelength=1.0)
    ax1.grid(axis="y", alpha=0.2, lw=0.4)
    ax1.tick_params(length=2)
    ax1.set_title("(a) Single-transmitter CBF, 640 views", fontsize=7.2)

    wm = d["warm"]
    names = [r["arm"] for r in wm]
    gtv = [r["gt"] for r in wm]
    mv = [r["model"] for r in wm]
    y = np.arange(len(wm))[::-1]
    ax2.barh(y + w / 2, gtv, w, color="#bcbcbc", edgecolor="black", lw=0.4, label="Ground truth")
    ax2.barh(y - w / 2, mv, w, color=C_OURS, edgecolor="black", lw=0.4, label="Model")
    for yi, va, vb in zip(y, gtv, mv):
        ax2.text(va + 0.06, yi + w / 2, f"{va:.2f}", va="center", fontsize=5.6)
        ax2.text(max(vb, 0.02) + 0.06, yi - w / 2, f"{vb:.3f}", va="center", fontsize=5.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=6.2)
    ax2.set_xlabel("Warm-colour pixels (\\% of image)")
    ax2.set_xlim(0, max(gtv + mv) * 1.20)
    ax2.legend(loc="lower right", frameon=False, handlelength=1.0)
    ax2.grid(axis="x", alpha=0.2, lw=0.4)
    ax2.tick_params(length=2)
    ax2.set_title("(b) Coverage of the warm end of the palette", fontsize=7.2)
    fig.savefig(FIG / "fig13_where.pdf")
    fig.savefig(FIG / "fig13_where.png", dpi=300)
    plt.close(fig)
    print("[fig13_where] ok")


def main():
    fig2_gap()
    fig9_w_cbf()
    fig10_qual_mvdr()
    fig11_qual_w_cbf()
    fig12_qual_w_mvdr()
    fig13_where()
    fig14_qual_tta()


if __name__ == "__main__":
    main()
