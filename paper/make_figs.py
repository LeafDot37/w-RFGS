# -*- coding: utf-8 -*-
"""make_figs.py —— 论文（IEEE 双栏）全部插图，英文标注，矢量优先。

约定
----
* **所有数字都从结果文件读**，本脚本只放「出处路径」与版式，不放读数。
  唯一的例外是示意图（fig1）里的结构常量（层数、维度），它们直接来自 `color_scalar_tx.py`
  与 `angular_experiment.json`，并在注释里写明取处。
* 线图/柱图存 PDF（矢量），定性拼图存 PNG 300 dpi（本来就是栅格）。
* 版面宽度按 IEEEtran：单栏 `W1 = 3.45 in`，双栏 `W2 = 7.16 in`。

用法（Windows python，任意 cwd）：
    python latex/make_figs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(parents=True, exist_ok=True)

RUN = ROOT / "results/run_20261024_本机4060_发射源参数泛化"
RUN180 = RUN / "长训180k"
META = ROOT / "datasets/RF-3DGS_cbf_params/meta.json"
GATE = ROOT / "results/_gate_params/数据.json"
FIGDATA = RUN180 / "图_180k/数据.json"
AGG = RUN180 / "agg_180k_vs_60k.json"
M180 = RUN180 / "full/test_inst/metrics.txt"
M60 = RUN / "full/test_inst/metrics.txt"
EXPJSON = RUN180 / "full/angular_experiment.json"
AUTH = ROOT / ("final/前馈pipeline/运行/"
               "run_20261020_本机4060_角域前馈多发射源_起步点修复/rf3dgs_released_renders")
QUAL_INST = 317
# 第二个成像算子（正则化 MVDR）的产物：服务器 4090D 上训的 180k 臂。
# 读的是 make_mvdr_figs.py 写出的 payload（其中的 floor 由 _floor_metrics.py 实算）。
RUNM = ROOT / "results/run_20261027_服务器4090D_MVDR参数泛化"
FIGDATA_M = RUNM / "图/数据.json"

W1, W2 = 3.45, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

C_TRAIN, C_HOLD, C_GREY = "#1f6fb4", "#d1495b", "#8c8c8c"
C_OURS, C_REF, C_FLOOR = "#d1495b", "#8c8c8c", "#2a9d8f"


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_metrics(p):
    """metrics.txt → {id: {role, psnr, ssim, lpips}}（制表符分隔，无表头）。"""
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 6 or not row[0].isdigit():
                continue
            out[int(row[0])] = {"role": row[1], "n": int(row[2]),
                                "psnr": float(row[3]), "ssim": float(row[4]),
                                "lpips": float(row[5])}
    return out


# --------------------------------------------------------------------------- #
# Fig. 1 —— 方法总览示意图
# --------------------------------------------------------------------------- #
def fig1_pipeline():
    meta, exp = jl(META), jl(EXPJSON)
    P = exp["P"]
    fig, ax = plt.subplots(figsize=(W2, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#eef3f9", ec="#33475b", fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
                fontsize=fs, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=fs - 0.8)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-", color="#33475b", lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     linestyle=ls, color=color, lw=lw,
                                     mutation_scale=8, shrinkA=0, shrinkB=0))

    ytop, htop = 27, 12
    box(1.0, ytop, 15.5, htop, "Optical views", "2,560 posed RGB")
    box(20.0, ytop, 17.5, htop, "Frozen 3DGS", f"P = {P:,}")
    box(41.0, ytop, 18.0, htop, "TX-conditioned", "scalar head, 128×2")
    box(62.5, ytop, 16.0, htop, "$\\alpha$-blend", "rasterizer")
    box(82.0, ytop, 17.0, htop, "jet LUT", "300×200 spectrum")
    for x0, x1 in ((16.5, 20.0), (37.5, 41.0), (59.0, 62.5), (78.5, 82.0)):
        arrow(x0, ytop + htop / 2, x1, ytop + htop / 2)

    # 输入向量（下排）
    box(28.0, 2.0, 44.0, 14.5, "", fc="#fbf7ee", ec="#8a6d3b")
    lines = [
        r"$e_i\;(32)\;\cdot\;\gamma_5(x_i)\;\cdot\;d_i\;\cdot\;\gamma_3(d_i)\;\cdot\;r_i$",
        r"$\Delta_i/s,\;|\Delta_i|/s,\;\hat{\Delta}_i,\;\gamma_4(\Delta_i/s)"
        r"\;\cdot\;\hat{x}_t,\;\gamma_4(\hat{x}_t),\;|x_t|,\;\sin/\cos\,\mathrm{az},\mathrm{el}$",
        r"$\sin/\cos(\Delta\psi,\Delta\theta)\;\cdot\;R^{\top}\hat{\Delta}_i"
        r"\;\cdot\;\tilde{f}$",
    ]
    for k, s in enumerate(lines):
        ax.text(50.0, 12.6 - 4.3 * k, s, ha="center", va="center", fontsize=7.4)
    ax.text(28.8, 15.9, "input of the head, per point $i$", fontsize=6.8,
            color="#8a6d3b", ha="left", va="center")
    arrow(50.0, 16.5, 50.0, ytop, lw=0.9)

    # 发射源参数与监督
    box(1.0, 3.5, 23.0, 11.0, "Transmitter", r"$x_t,\;\Delta\psi,\;\Delta\theta,\;f$",
        fc="#f4f4f4", ec="#666666")
    arrow(24.0, 9.0, 28.0, 9.0)
    box(82.0, 2.0, 17.0, 11.0, "Ground truth", "8-bit, min–max",
        fc="#f9eef0", ec=C_OURS)
    arrow(90.5, ytop, 90.5, 13.0, ls=(0, (2.5, 1.5)), color=C_OURS)
    ax.text(91.6, 19.0, r"$\mathcal{L}_1$", fontsize=7.2, color=C_OURS, ha="left")

    ax.text(1.0, 44.0, "one forward pass per transmitter configuration, "
                        "amortized over 144 instances",
            fontsize=7.4, fontweight="bold", ha="left")
    fig.savefig(FIG / "fig1_pipeline.pdf"); fig.savefig(FIG / "fig1_pipeline.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 2 —— 数据设计：位置划分 + 参数空间
# --------------------------------------------------------------------------- #
def fig2_design():
    meta = jl(META)
    I = meta["instances"]
    ps = meta["param_space"]
    tr_pos = set(meta["split"]["train"])
    by_pos = {}
    for x in I:
        by_pos.setdefault(x["position_id"], []).append(x)

    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0),
                            gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # (a) 平面位置
    ax = axs[0]
    tw = np.array(meta["tx_world"])
    for pid, xy in enumerate(tw):
        tr = pid in tr_pos
        ax.scatter(xy[0], xy[1], s=26, marker="o" if tr else "s",
                   facecolor=C_TRAIN if tr else "none",
                   edgecolor=C_TRAIN if tr else C_HOLD, lw=0.8, zorder=3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) transmitter positions")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25, lw=0.4)
    ax.scatter([], [], marker="o", s=26, facecolor=C_TRAIN, edgecolor=C_TRAIN,
               label=f"train ({len(meta['split']['train'])})")
    ax.scatter([], [], marker="s", s=26, facecolor="none", edgecolor=C_HOLD,
               label=f"held-out ({len(meta['split']['test'])})")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)

    # (b) yaw / pitch 抽样
    ax = axs[1]
    seen_y = np.array(ps["yaw_off_deg"]["seen"])
    seen_p = np.array(ps["pitch_off_deg"]["seen"])
    hold_y = np.array(ps["yaw_off_deg"]["holdout"])
    hold_p = np.array(ps["pitch_off_deg"]["holdout"])
    for v in seen_y:
        ax.axvline(v, color=C_TRAIN, alpha=0.22, lw=0.7, zorder=0)
    for v in seen_p:
        ax.axhline(v, color=C_TRAIN, alpha=0.22, lw=0.7, zorder=0)
    ys = np.degrees([x["offset_yaw_rad"] for x in I if x["role"] == "train"])
    pt = np.degrees([x["offset_pitch_rad"] for x in I if x["role"] == "train"])
    ax.scatter(ys, pt, s=7, color=C_TRAIN, alpha=0.55, lw=0, zorder=2,
               label="sampled (144)")
    ax.scatter(hold_y, [0, 0, 0], s=30, marker="*", color=C_HOLD, zorder=3,
               label="held-out yaw")
    ax.scatter([0, 0, 0], hold_p, s=30, marker="*", color=C_HOLD, zorder=3)
    ax.axvspan(min(hold_y), max(hold_y), color=C_HOLD, alpha=0.07, lw=0)
    ax.axhspan(min(hold_p), max(hold_p), color=C_HOLD, alpha=0.07, lw=0)
    ax.set_xlabel(r"yaw offset $\Delta\psi$ ($^\circ$)")
    ax.set_ylabel(r"pitch offset $\Delta\theta$ ($^\circ$)")
    ax.set_title("(b) orientation draws")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, fontsize=6.2)
    ax.set_xlim(-70, 70); ax.set_ylim(-30, 30)

    # (c) 频率轴（单行刻度带）
    ax = axs[2]
    lo, hi = ps["freq_ghz"]["declared"]
    ax.axvspan(lo, hi, color="#000000", alpha=0.05, lw=0)
    fseen = ps["freq_ghz"]["seen"]
    fhold = ps["freq_ghz"]["holdout"]
    ax.scatter(fseen, np.zeros(len(fseen)), s=34, color=C_TRAIN, zorder=3,
               label="train")
    ax.scatter(fhold, np.zeros(len(fhold)), s=44, marker="*", color=C_HOLD,
               zorder=3, label="held-out")
    f_orig = [x["freq_hz"] / 1e9 for x in I if x["role"] == "orig"][0]
    ax.axvline(f_orig, color=C_OURS, lw=0.9, ls="--", zorder=2)
    ax.annotate("main reading\n%.2f GHz" % f_orig, xy=(f_orig, 0.30),
                xytext=(f_orig + 0.13, 0.62), fontsize=6.2, color=C_OURS,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_OURS))
    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.95)
    ax.set_xlim(lo - 0.12, hi + 0.12)
    ax.set_xlabel("carrier frequency (GHz)")
    ax.set_title("(c) carrier")
    ax.legend(loc="lower left", frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=0.8, fontsize=6.2)
    for a in axs:
        a.tick_params(length=2)
    fig.savefig(FIG / "fig2_design.pdf"); fig.savefig(FIG / "fig2_design.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 3 —— 阶段 0 可行性闸
# --------------------------------------------------------------------------- #
def fig3_gates():
    g = jl(GATE)["verdict"]
    floor = g["floor"]["psnr"]
    thr = floor + 3.0
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.05))

    ax = axs[0]
    yaw = np.array(g["yaw_curve"], float)
    pit = np.array(g["pitch_curve"], float)
    ax.plot(yaw[1:, 0], yaw[1:, 1], "o-", color=C_TRAIN, ms=3.2, label="yaw")
    ax.plot(pit[1:, 0], pit[1:, 1], "s--", color="#7a4fa3", ms=3.2, label="pitch")
    ax.axhline(floor, color=C_FLOOR, lw=0.9, ls=":")
    ax.text(150, floor + 1.2, "structure-destroyed floor", fontsize=6.4,
            color=C_FLOOR, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 45, 90, 180])
    ax.set_xticklabels(["2", "5", "10", "20", "45", "90", "180"])
    ax.set_xlabel(r"orientation perturbation ($^\circ$)")
    ax.set_ylabel("PSNR vs. reference spectrum (dB)")
    ax.set_title("(a) orientation is discriminative")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(30, 72)

    ax = axs[1]
    fc = np.array([(d[0] / 1e6, d[1]) for d in g["freq_curve"]], float)
    ax.plot(fc[:, 0], fc[:, 1], "o-", color="#c1662b", ms=3.2)
    ax.axhline(floor, color=C_FLOOR, lw=0.9, ls=":")
    ax.axhline(thr, color=C_FLOOR, lw=0.6, ls="--")
    ax.text(4.5, floor + 0.8, "floor", fontsize=6.4, color=C_FLOOR)
    ax.text(4.5, thr + 0.8, "floor + 3 dB", fontsize=6.4, color=C_FLOOR)
    ax.annotate(f"{fc[-1,1]:.1f} dB at {fc[-1,0]/1000:.1f} GHz",
                xy=(fc[-1, 0], fc[-1, 1]), xytext=(150, 46),
                fontsize=6.4, arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))
    ax.set_xscale("log")
    ax.set_xticks([5, 50, 300, 1000, 2100])
    ax.set_xticklabels(["5", "50", "300", "1000", "2100"])
    ax.set_xlabel("carrier offset $\\Delta f$ (MHz)")
    ax.set_title("(b) frequency stays correlated")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(40, 82)
    for a in axs:
        a.tick_params(length=2)
    fig.savefig(FIG / "fig3_gates.pdf"); fig.savefig(FIG / "fig3_gates.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 4 —— 定性对照
# --------------------------------------------------------------------------- #
def fig4_qual(views=("00005", "00649", "01501", "02328")):
    gdir = RUN180 / f"full/test_inst/inst{QUAL_INST:03d}/gt"
    rdir = RUN180 / f"full/test_inst/inst{QUAL_INST:03d}/renders"
    cols = [("Ground truth", gdir), ("RF-3DGS (released)", AUTH),
            ("Ours", rdir)]
    rows = []
    for v in views:
        row = []
        for _, d in cols:
            p = d / f"{v}.png"
            if not p.exists():
                raise FileNotFoundError(f"缺图：{p}")
            row.append(np.asarray(Image.open(p).convert("RGB")))
        rows.append(row)
    h, w = rows[0][0].shape[:2]
    nr, nc = len(rows), len(cols)
    gap = 4
    fig, axs = plt.subplots(nr, nc, figsize=(W2, W2 * (nr * h + (nr - 1) * gap)
                                            / (nc * w + (nc - 1) * gap)),
                            gridspec_kw={"wspace": 0.012, "hspace": 0.03})
    for r in range(nr):
        for c in range(nc):
            axs[r, c].imshow(rows[r][c], aspect="auto")
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            for s in axs[r, c].spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                axs[r, c].set_title(cols[c][0], fontsize=7.5, pad=2.5)
    # 视角标签用 fig.text 画在最左外侧，避免等宽 ylabel 被 tight bbox 裁掉首字
    fig.subplots_adjust(left=0.045, right=0.999, top=0.94, bottom=0.005)
    for r, v in enumerate(views):
        pos = axs[r, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, v, rotation=90, ha="center",
                 va="center", fontsize=6.4)
    fig.savefig(FIG / "fig4_qual.png", dpi=300, bbox_inches=None)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 5 —— 主读数
# --------------------------------------------------------------------------- #
def fig5_main():
    d = jl(FIGDATA)
    ours = d["ours_main_reading"]
    fl = d["floor_constant_image"]
    ref = {r["name"]: r for r in d["reference"]}
    corf = [r for r in d["reference_corf"]][0]
    bars = [
        ("Constant\nimage", fl["psnr"], fl["ssim"], fl["lpips"], "ref"),
        ("RF-3DGS\n(published)", ref["RF-3DGS 发布渲染"]["psnr"],
         ref["RF-3DGS 发布渲染"]["ssim"], ref["RF-3DGS 发布渲染"]["lpips"], "ref"),
        ("CORF-GS\n(coupled)", corf["psnr"], corf["ssim"], corf["lpips"], "ref"),
        ("Prior FF\n(position only)", ref["旧角域前馈（单源）"]["psnr"],
         ref["旧角域前馈（单源）"]["ssim"], ref["旧角域前馈（单源）"]["lpips"], "ref"),
        ("Ours\n(full params)", ours["psnr"], ours["ssim"], ours["lpips"], "ours"),
    ]
    keys = [("psnr", "PSNR (dB) $\\uparrow$", ".3f", 0),
            ("ssim", "SSIM $\\uparrow$", ".4f", 1),
            ("lpips", "LPIPS $\\downarrow$", ".4f", 2)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.95))
    xs = np.arange(len(bars))
    for ax, (k, lab, fmt, j) in zip(axs, keys):
        vals = [b[j + 1] for b in bars]
        cols = [C_GREY if b[4] == "ref" else C_OURS for b in bars]
        ax.bar(xs, vals, color=cols, width=0.68, edgecolor="black", lw=0.4)
        for x, v in zip(xs, vals):
            ax.text(x, v + (max(vals) - min(vals)) * 0.03, format(v, fmt),
                    ha="center", va="bottom", fontsize=6.0, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in bars], rotation=42, ha="right", fontsize=6.0)
        ax.set_ylabel(lab)
        lo = min(vals); hi = max(vals)
        ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.30)
        ax.grid(axis="y", alpha=0.2, lw=0.4)
        ax.tick_params(length=2)
    axs[0].set_title("(a) PSNR", fontsize=7.5)
    axs[1].set_title("(b) SSIM", fontsize=7.5)
    axs[2].set_title("(c) LPIPS", fontsize=7.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=C_GREY, ec="black", lw=0.4, label="reference"),
                        Patch(fc=C_OURS, ec="black", lw=0.4, label="ours")],
               loc="lower center", ncol=2, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.0, columnspacing=1.2)
    fig.savefig(FIG / "fig5_main.pdf"); fig.savefig(FIG / "fig5_main.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 6 —— 位置留出：逐实例与逐位置
# --------------------------------------------------------------------------- #
def fig6_holdout():
    m180, m60 = read_metrics(M180), read_metrics(M60)
    meta = jl(META)
    pos_of = {x["id"]: x["position_id"] for x in meta["instances"]}
    ids = sorted(i for i in m180 if m180[i]["role"] == "pos_holdout")
    a = np.array([m180[i]["psnr"] for i in ids])
    b = np.array([m60[i]["psnr"] for i in ids])
    pos = [pos_of[i] for i in ids]
    uniq = sorted(set(pos))

    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2),
                            gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axs[0]
    lo = min(a.min(), b.min()) - 0.15
    hi = max(a.max(), b.max()) + 0.15
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.7, ls="--")
    ax.text(hi - 0.05, lo + 0.08, "$y=x$", fontsize=6.2, color="#555555",
            ha="right", va="bottom")
    cmap = plt.get_cmap("tab20")
    for k, p in enumerate(uniq):
        sel = [j for j, q in enumerate(pos) if q == p]
        ax.scatter(b[sel], a[sel], s=11, color=cmap(k % 20), lw=0, alpha=0.9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("PSNR, 60k steps (dB)"); ax.set_ylabel("PSNR, 180k steps (dB)")
    ax.set_title("(a) 52 held-out instances")
    ax.text(0.03, 0.95, f"{int(np.sum(a > b))}/52 improve", transform=ax.transAxes,
            fontsize=6.6, va="top")
    ax.grid(alpha=0.2, lw=0.4); ax.tick_params(length=2)

    ax = axs[1]
    agg = jl(AGG)["roles"]["pos_holdout"]
    by = {}
    for i, p in zip(ids, pos):
        by.setdefault(p, []).append(m180[i]["psnr"] - m60[i]["psnr"])
    dp = np.array([np.mean(by[p]) for p in uniq])
    xs = np.arange(len(dp))
    ax.bar(xs, dp, color=[C_TRAIN if v >= 0 else C_HOLD for v in dp],
           width=0.7, edgecolor="black", lw=0.35)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"P{p}" for p in uniq], fontsize=6.2,
                                          rotation=90)
    ax.set_ylabel(r"$\Delta$PSNR per position (dB)")
    ax.set_title("(b) grouped by position")
    ax.set_ylim(min(-0.12, dp.min() - 0.05), max(dp.max() + 0.06, 0.25))
    ax.grid(axis="y", alpha=0.2, lw=0.4); ax.tick_params(length=2)
    fig.savefig(FIG / "fig6_holdout.pdf"); fig.savefig(FIG / "fig6_holdout.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 7 —— 第二个成像算子（正则化 MVDR）
# --------------------------------------------------------------------------- #
def fig7_mvdr():
    """三根柱：常数图地板 → 位置留出组 → 主读数。读数全部来自 payload，不在此硬编码。

    与 fig5 的差别：这里的参考列只有「常数图」——CBF 口径的发布包/CORF 读数与
    MVDR 不是同一个物理量，不能同轴。
    """
    d = jl(FIGDATA_M)
    fl, ho, mn = d["constant_floor_main_gt"], d["pos_holdout"], d["main_reading"]
    bars = [("Constant\nimage", fl["psnr"], fl["ssim"], fl["lpips"], "ref"),
            ("Held-out\npositions (52)", ho["psnr"], ho["ssim"], ho["lpips"], "ours"),
            ("Main\nreading", mn["psnr"], mn["ssim"], mn["lpips"], "ours")]
    keys = [("psnr", "PSNR (dB) $\\uparrow$", ".3f", 0),
            ("ssim", "SSIM $\\uparrow$", ".4f", 1),
            ("lpips", "LPIPS $\\downarrow$", ".4f", 2)]
    fig, axs = plt.subplots(1, 3, figsize=(W1 * 2.06, 1.72))
    xs = np.arange(len(bars))
    for ax, (k, lab, fmt, j) in zip(axs, keys):
        vals = [b[j + 1] for b in bars]
        ax.bar(xs, vals, color=[C_GREY if b[4] == "ref" else C_OURS for b in bars],
               width=0.62, edgecolor="black", lw=0.4)
        for x, v in zip(xs, vals):
            ax.text(x, v + (max(vals) - min(vals)) * 0.04, format(v, fmt),
                    ha="center", va="bottom", fontsize=5.8, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in bars], fontsize=5.6, linespacing=1.05)
        ax.set_ylabel(lab)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - (hi - lo) * 0.14, hi + (hi - lo) * 0.34)
        ax.grid(axis="y", alpha=0.2, lw=0.4)
        ax.tick_params(length=2)
    axs[0].set_title("(a) PSNR", fontsize=7.5)
    axs[1].set_title("(b) SSIM", fontsize=7.5)
    axs[2].set_title("(c) LPIPS", fontsize=7.5)
    fig.savefig(FIG / "fig7_mvdr.pdf"); fig.savefig(FIG / "fig7_mvdr.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    fig1_pipeline(); print("fig1 ok")
    fig2_design(); print("fig2 ok")
    fig3_gates(); print("fig3 ok")
    fig4_qual(); print("fig4 ok")
    fig5_main(); print("fig5 ok")
    fig6_holdout(); print("fig6 ok")
    fig7_mvdr(); print("fig7 ok")


if __name__ == "__main__":
    main()
