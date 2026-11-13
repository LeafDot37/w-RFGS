# -*- coding: utf-8 -*-
"""make_fig_w.py —— 波束权 w 那一节的插图（fig8_w_channel）。

所有数字都从结果文件读：
  * 三臂指标：`results/run_20261101_服务器4080_波束权复值场/_小件/metrics_<臂>.txt`
    （逐实例行，取 role == w_holdout 的 36 行求均值）；
  * 真值参照：同目录 `参考_天花板.txt`（三行：先验 / 兄弟实例 / 中位数）；
  * w 置换探针：同目录 `w_swap_probe.json`（数字的出处写在它的 _source 字段里）。

用法：python latex/make_fig_w.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figs"
WRUN = ROOT / "results/run_20261101_服务器4080_波束权复值场/_小件"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

W1 = 3.45
ARMS = [("复值_等权", "complex, $\\lambda_{p}=0$", "#c8c8c8"),
        ("复值_感知", "complex, perceptual", "#e07b39"),
        ("盲臂_感知", "w-blind (control)", "#4a6fa5")]


def read_holdout(name: str):
    rows = []
    for line in (WRUN / f"metrics_{name}.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[1] == "w_holdout":
            rows.append([float(x) for x in parts[3:6]])
    return np.array(rows)


def main():
    probe = json.loads((WRUN / "w_swap_probe.json").read_text(encoding="utf-8"))
    rms = probe["rms_grey"]

    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.55), gridspec_kw={"width_ratios": [1, 1.12]})

    # (a) w 置换后的响应幅度（横条，标签在 y 轴，避免拥挤）
    labels = ["GT", "complex, $\\lambda_p{=}0$", "complex, +LPIPS", "w-blind"]
    vals = [rms["GT"], rms["complex_equal"], rms["complex_perceptual"], rms["blind"]]
    cols = ["#3c3c3c", "#c8c8c8", "#e07b39", "#4a6fa5"]
    y = np.arange(4)[::-1]
    b = ax[0].barh(y, vals, color=cols, height=0.62, edgecolor="none")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("RMS change after swapping $w$  [grey levels]")
    ax[0].set_title("(a) Is the $w$ channel alive?", pad=3)
    for r, v, yy in zip(b, vals, y):
        txt = f"{v:.2f}" if 0 < v < 10 else f"{v:.0f}"
        ax[0].text(v, yy, "  " + txt, ha="left", va="center", fontsize=6.5)
    # 模型/真值比例（放到底部留白，避开柱端数值）
    ax[0].text(2, -0.62,
               f"model response = {rms['complex_equal'] / rms['GT'] * 100:.1f}% / "
               f"{rms['complex_perceptual'] / rms['GT'] * 100:.1f}% of GT",
               fontsize=6.2, color="#7a3b12", va="center")
    ax[0].set_xlim(0, 92)
    ax[0].set_ylim(-0.95, 3.45)

    # (b) 三臂 LPIPS 与真值参照地板
    lp = [read_holdout(n)[:, 2].mean() for n, _, _ in ARMS]
    x = np.arange(3)
    ax[1].bar(x, lp, color=[c for _, _, c in ARMS], width=0.6, edgecolor="none")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["complex\n$\\lambda_p{=}0$", "complex\nperceptual", "w-blind"], fontsize=6.5)
    ax[1].set_ylabel("LPIPS $\\downarrow$  (36 held-out $w$)")
    ax[1].set_title("(b) Held-out $w$: no arm beats the blind control", pad=3)
    for xi, v in zip(x, lp):
        ax[1].text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ref = (WRUN / "参考_天花板.txt").read_text(encoding="utf-8").split("\n")
    prior = float(ref[0].split()[2])
    ax[1].axhline(prior, ls="--", lw=0.8, color="#2e7d32")
    ax[1].text(2.42, prior, "no-w prior\n0.290", fontsize=6, color="#2e7d32",
               ha="right", va="bottom")
    ax[1].set_ylim(0, max(max(lp), prior) * 1.28)

    for a in ax:
        a.tick_params(width=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig8_w_channel.pdf")
    fig.savefig(FIG / "fig8_w_channel.png", dpi=300)
    print("[fig8] 三臂 LPIPS:", [f"{v:.4f}" for v in lp], "| no-w 先验地板", prior)


if __name__ == "__main__":
    main()
