# -*- coding: utf-8 -*-
"""MVDR 通道模型（第二个成像算子）的三张交付图 + 数值 payload。

与 `make_param_figs.py` 分开写，原因是**参考值不同轴**：那张图里的
RF-3DGS 发布 14.1315 / CORF 14.6326 / 常数图地板 14.2932 全是 **CBF** 口径，
把它们画进 MVDR 图里就是跨物理量比较（MVDR 是 Capon 自适应谱，与 CBF 不是同一个量）。
柱状图的三根柱一律取 MVDR 口径：原 RF-3DGS（作者发布包 `3dgs_MVDR_100`）、
CORF 完整耦合、本次（**只留最终臂**，不再画位置留出）。两条常数图地板画成横线，
因为它们是在两套数据上各自算的（作者 / 本仓），绝对值不能直接相减。

读数一律从 metrics.txt / floor 实算值来，不写估计值。

用法（WSL mvsplat 环境）::
    $PY code/tools/figures/make_mvdr_figs.py \
        --metrics results/<run>/metrics.txt \
        --meta datasets/RF-3DGS_mvdr_params/meta.json \
        --qual_root results/<run>/_xfer/qual_lean \
        --out results/<run>/图
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import _cnfont                                        # noqa: E402

METRICS = [("psnr", "PSNR (dB)  ↑", 1), ("ssim", "SSIM  ↑", 1),
           ("lpips", "LPIPS  ↓", -1)]
FMT = {"psnr": ".3f", "ssim": ".4f", "lpips": ".4f"}

COL_FLOOR = "#c0392b"
COL_HOLD = "#c0a16b"
COL_OURS = "#2f6fbe"
COL_REF = "#9aa0a6"
COL_CORF = "#b5739d"

# ---- 三根柱子里的两条外部参考（都是 MVDR 口径、都在作者原始数据上）----------------
# 出处：final/原pipeline/图像对比/MVDR/三臂对照_MVDR.json 的 A 臂
# （作者发布包 3dgs_MVDR_100/test/ours_40000，639 帧，`final/原pipeline/README.md` 同数）
RF3DGS_MVDR = ("原 RF-3DGS", 16.0188, 0.7309, 0.3911)
# 出处：final/归档_已清理历史阶段/run_20261008_本机4060_CORF老师三臂/说明.md
# （rf_results.json，640 测试视角全量）。只取「完整耦合」= 论文 Ours；
# 另两臂是消融阶梯（冻结几何 16.7780 / 射频调几何 19.5905），只进 数据.json 备查。
CORF_MVDR = ("CORF 完整耦合", 19.4438, 0.8034, 0.2866)
CORF_ABLATION_MVDR = [("CORF 冻结几何", 16.7780, 0.7513, 0.3632),
                      ("CORF 射频调几何", 19.5905, 0.8060, 0.2835),
                      ("CORF 完整耦合", 19.4438, 0.8034, 0.2866)]

# 两条常数图地板，各自在自己的 640 张测试 GT 上算，**不是同一条**：
#   ours —— 本仓 tr38901 + 0.1 ns 时延格 + 漫散射（code/tools/eval/_floor_metrics.py）
#   auth —— 作者原始 iso 阵数据（code/tools/archive/_floor_author_mvdr.sh，12.1642）
FLOOR_OURS = (11.3806, 0.4107, 0.5064)
FLOOR_AUTH = (12.1642, 0.6023, 0.4392)


def read_metrics(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path} 读空 —— 空表不许出图")
    return {int(r["id"]): dict(id=int(r["id"]), role=r["role"], n=int(r["n"]),
                               **{m: float(r[m]) for m in ("psnr", "ssim", "lpips")})
            for r in rows}


def fig_bars(main, floor_ours, out_dir, tag):
    """两条外部参考 + 本次最终臂；地板画成横线而不是柱子。

    屋顶只留一根柱子代表本方法（最终臂 = 留出发射源主读数），位置留出组
    挪去 `位置留出_逐实例_MVDR.png`，不在这张图里重复。

    两条地板必须都画：参考那两列是在**作者原始 MVDR 数据**上测的，常数图地板
    是 12.164，不是本仓的 11.381。少画一条，读者就会把 19.44 与 14.21 直接相减。
    """
    slots = [(RF3DGS_MVDR[0], RF3DGS_MVDR[1:], COL_REF, None),
             (CORF_MVDR[0], CORF_MVDR[1:], COL_CORF, "///"),
             ("本次（最终臂）", tuple(main), COL_OURS, None)]
    n = len(slots)
    idx = {"psnr": 0, "ssim": 1, "lpips": 2}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6))
    for ax, (key, ylab, _) in zip(axes, METRICS):
        vals = [s[1][idx[key]] for s in slots]
        bars = ax.bar(range(n), vals, color=[s[2] for s in slots],
                      hatch=[s[3] for s in slots], width=0.62,
                      edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, format(v, FMT[key]),
                    ha="center", va="bottom", fontsize=9)
        f_o, f_a = floor_ours[idx[key]], FLOOR_AUTH[idx[key]]
        lo = min(vals + [f_o, f_a])
        hi = max(vals + [f_o, f_a])
        pad = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - pad * 1.15, hi + pad * 0.95)
        # 两条地板进图例而不是贴在轴上：PSNR 里两条只差 0.8 dB、LPIPS 里作者地板
        # 又贴着最后一根柱顶的数值标签，文字压上去就读不清了。
        ax.axhline(f_o, ls="--", lw=1.1, color=COL_FLOOR, zorder=0,
                   label=f"本次数据地板 {f_o:.3f}")
        ax.axhline(f_a, ls=":", lw=1.3, color="#6b7280", zorder=0,
                   label=f"作者数据地板 {f_a:.4f}")
        ax.legend(loc="lower left", fontsize=6.8, framealpha=0.92,
                  handlelength=1.6, borderpad=0.35, labelspacing=0.3)
        ax.set_xticks(range(n))
        ax.set_xticklabels([s[0] for s in slots], fontsize=8.0, linespacing=1.2)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.5, -0.14, "逐场景优化（非前馈）", color=COL_CORF,
                fontsize=7.0, ha="center", va="top", linespacing=1.15,
                transform=ax.get_xaxis_transform())
    fig.suptitle(f"留出发射源（实例 317 / 作者 TX48）上的 MVDR 三指标；{tag}",
                 fontsize=11.5)
    fig.text(0.5, 0.012,
             "参考两列测在作者原始 MVDR 数据上（常数图地板 "
             f"{FLOOR_AUTH[0]:.3f} / {FLOOR_AUTH[1]:.4f} / {FLOOR_AUTH[2]:.4f}），"
             "本次测在本仓重建数据上（"
             f"{floor_ours[0]:.4f} / {floor_ours[1]:.4f} / {floor_ours[2]:.4f}）"
             "——两套生成协议不同，绝对值不能直接相减，只能各自对着自己的地板读。",
             ha="center", fontsize=7.0, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    p = out_dir / "三指标对比_MVDR.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_scatter(rows, meta, out_dir):
    pos = {int(i["id"]): int(i.get("position_id", -1)) for i in meta["instances"]}
    ids = [i for i, r in rows.items() if r["role"] == "pos_holdout"]
    if len(ids) < 2:
        raise SystemExit(f"pos_holdout 只有 {len(ids)} 个实例，画不了散点")
    groups = defaultdict(list)
    for i in ids:
        groups[pos[i]].append(i)
    pids = sorted(groups)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    for ax, (key, ylab, _) in zip(axes, METRICS):
        for j, pid in enumerate(pids):
            v = [rows[i][key] for i in sorted(groups[pid])]
            ax.scatter([j] * len(v), v, s=26, color=cmap(j % 20), edgecolor="w",
                       linewidth=0.4, zorder=3)
            ax.hlines(statistics.mean(v), j - 0.28, j + 0.28, color="k", lw=1.4,
                      zorder=4)
        ax.set_xticks(range(len(pids)))
        ax.set_xticklabels([f"P{p}" for p in pids], fontsize=7.5, rotation=90)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xlabel("未见发射源位置（每位置 4 组参数）", fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    v = [rows[i]["psnr"] for i in ids]
    fig.suptitle(f"位置留出组（MVDR）：{len(ids)} 个实例 / {len(pids)} 个未见位置"
                 f"（PSNR {statistics.mean(v):.2f} ± {statistics.pstdev(v):.2f} dB；"
                 f"黑横线 = 位置均值）", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "位置留出_逐实例_MVDR.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, len(ids), len(pids)


def fig_qual(root, inst, out_dir, extras=(), k=5):
    """GT | extras... | 本次渲染 —— **与 CBF 的 `定性对照.png` 同一构图**。

    改用单实例（留出发射源，即 RF-3DGS 对应的 TX48）而不是多实例并排：
    两臂的定性图要能并排看，列语义必须一致（左 GT、右本次），行是同一批视角。

    extras 每个目录都必须**恰好**含有所选视角的同名文件；缺一张就报错退出 ——
    少了不报错会画出一列空白，而空白列在缩略图上看不出来。
    """
    d = Path(root) / f"inst{inst:03d}"
    gd, rd = d / "gt", d / "renders"
    names = sorted(p.name for p in gd.glob("*.png"))
    if not names:
        raise SystemExit(f"{gd} 没有 GT")
    pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, k).round()]

    cols = [("GT", gd)] + list(extras) + [("本次渲染", rd)]
    for label, ed in cols:
        have = {p.name for p in Path(ed).glob("*.png")}
        miss = [n for n in names if n not in have]
        if miss:
            raise SystemExit(f"{label}：{ed} 缺 {len(miss)} 张（例如 {miss[:3]}），"
                             f"不与 GT 同名同集合，不出图")

    fig, axes = plt.subplots(k, len(cols), figsize=(2.6 * len(cols), 2.05 * k),
                             squeeze=False)
    for r, n in enumerate(pick):
        for c, (label, ed) in enumerate(cols):
            a = np.asarray(Image.open(Path(ed) / n).convert("RGB"))
            axes[r, c].imshow(a)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(label, fontsize=11)
            if c == 0:
                axes[r, c].set_ylabel(n.replace(".png", ""), fontsize=8)
    head = " / ".join(label for label, _ in cols)
    fig.suptitle(f"定性对照：留出发射源（{head}；视野 300×200；MVDR 通道模型）",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "定性对照_MVDR.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--floor", required=True,
                    help="常数图地板 psnr,ssim,lpips（实测值，见 _floor_metrics.py）")
    ap.add_argument("--qual_root", required=True)
    ap.add_argument("--qual_insts", default="317",
                    help="定性对照只取一个实例（默认 317 = 留出发射源 / RF-3DGS 的 TX48）")
    ap.add_argument("--qual_extras", nargs="*", default=[], metavar="LABEL=DIR",
                    help="插在 GT 与本次渲染之间的参考渲染列；本臂无外部参考，默认空")
    ap.add_argument("--tag", default="服务器 4090D，180 k 步")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(args.metrics)
    meta = json.load(open(args.meta, encoding="utf-8"))
    orig = [r for r in rows.values() if r["role"] == "orig"]
    if len(orig) != 1:
        raise SystemExit(f"orig 角色应恰好 1 个实例，实际 {len(orig)}")
    main_r = orig[0]
    hold = [r for r in rows.values() if r["role"] == "pos_holdout"]
    if not hold:
        raise SystemExit("没有 pos_holdout 行")
    f = [float(x) for x in args.floor.split(",")]
    if len(f) != 3:
        raise SystemExit("--floor 要 3 个数")

    main_v = (main_r["psnr"], main_r["ssim"], main_r["lpips"])
    hold_v = tuple(statistics.mean(r[m] for r in hold) for m in ("psnr", "ssim", "lpips"))
    hold_sd = tuple(statistics.pstdev([r[m] for r in hold]) for m in ("psnr", "ssim", "lpips"))

    if max(abs(a - b) for a, b in zip(f, FLOOR_OURS)) > 1e-4:
        raise SystemExit(f"--floor {f} 与图里写死的地板 {list(FLOOR_OURS)} 不一致；"
                         f"改了实测值就同时改 FLOOR_OURS 与图注")
    p1 = fig_bars(main_v, tuple(f), out, args.tag)
    p2, ni, npid = fig_scatter(rows, meta, out)
    inst = int(args.qual_insts.split(",")[0])
    extras = []
    for e in args.qual_extras:
        if "=" not in e:
            raise SystemExit(f"--qual_extras 要写成 LABEL=DIR，收到 {e!r}")
        lab, dd = e.split("=", 1)
        extras.append((lab, dd))
    p3, picks = fig_qual(args.qual_root, inst, out, extras=extras)

    payload = {
        "tag": args.tag,
        "channel_model": "regularized MVDR (Capon), loading=0.01, 0.1 ns grid, diffuse on",
        "constant_floor_main_gt": dict(psnr=f[0], ssim=f[1], lpips=f[2],
                                       n_views=main_r["n"], instance=int(main_r["id"])),
        "constant_floor_author_mvdr": dict(psnr=FLOOR_AUTH[0], ssim=FLOOR_AUTH[1],
                                           lpips=FLOOR_AUTH[2], n_views=640,
                                           source="code/tools/archive/_floor_author_mvdr.sh"),
        "reference": [dict(name=RF3DGS_MVDR[0], psnr=RF3DGS_MVDR[1], ssim=RF3DGS_MVDR[2],
                           lpips=RF3DGS_MVDR[3], n_views=639,
                           source="作者发布包 3dgs_MVDR_100/test/ours_40000"),
                      dict(name=CORF_MVDR[0], psnr=CORF_MVDR[1], ssim=CORF_MVDR[2],
                           lpips=CORF_MVDR[3], n_views=640,
                           source="run_20261008_本机4060_CORF老师三臂/完整耦合/MVDR/rf_results.json")],
        "corf_ablation": [dict(name=n, psnr=a, ssim=b, lpips=c)
                          for n, a, b, c in CORF_ABLATION_MVDR],
        "main_reading": dict(instance=int(main_r["id"]), n=main_r["n"],
                             psnr=main_v[0], ssim=main_v[1], lpips=main_v[2]),
        "pos_holdout": dict(n_inst=ni, n_pos=npid,
                            psnr=hold_v[0], ssim=hold_v[1], lpips=hold_v[2],
                            psnr_sd=hold_sd[0], ssim_sd=hold_sd[1], lpips_sd=hold_sd[2]),
        "notes": [
            "本图三条柱子全部是 **MVDR 口径**：原 RF-3DGS 16.0188、CORF 完整耦合 19.4438、"
            "本次 14.2093。CBF 口径的 14.1315 / 14.6326 不入本图（换个操作子就是换个物理量）。",
            "但**数据不是同一份**：前两条测在作者原始 MVDR 数据上（iso 阵、作者自己的时延格），"
            "本次测在本仓重建的 MVDR 数据上（tr38901 + 0.1 ns 时延格 + 漫散射）。"
            "两条常数图地板因此不同（作者 12.164 vs 本仓 11.381），只画一条会把 19.44 与 14.21 "
            "误读成可以直接相减。按各自的净余量读：CORF +7.28 dB、原 RF-3DGS +3.85 dB、"
            "本次 +2.83 dB。",
            "位置留出组（52 实例）仍在 `位置留出_逐实例_MVDR.png` 里，本图只留最终臂一根柱子。",
            "作者 output_metric.txt 里的 MVDR 读数（12.40/0.569/0.667 与 _self 18.71/0.793/0.352）"
            "自相矛盾，未采用。",
            "位置留出组的 ± 是实例级 pstdev（52 个实例来自 13 个位置，非独立样本）。",
        ],
        "figures": [str(p1), str(p2), str(p3)],
        "qual_instance": inst,
        "qual_columns": ["GT"] + [lab for lab, _ in extras] + ["本次渲染"],
        "qual_views": picks,
    }
    (out / "数据.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"写出：\n  {p1}\n  {p2}\n  {p3}\n  {out / '数据.json'}")
    print(f"主读数 {main_v[0]:.4f} / {main_v[1]:.4f} / {main_v[2]:.4f}；"
          f"位置留出 {hold_v[0]:.4f}±{hold_sd[0]:.4f} / {hold_v[1]:.4f}±{hold_sd[1]:.4f} / "
          f"{hold_v[2]:.4f}±{hold_sd[2]:.4f}")


if __name__ == "__main__":
    main()
