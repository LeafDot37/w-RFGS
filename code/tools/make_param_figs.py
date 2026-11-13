# -*- coding: utf-8 -*-
"""make_param_figs.py —— 发射源参数泛化的三张交付图 + 数值 payload。

   三指标对比.png      发布包 / 旧角域前馈 / CORF 完整耦合 / 本次，三面板柱状图；
                       虚线 = 同口径常数图地板

   位置留出_逐实例.png  13 个未见位置 × 4 组参数，三面板散点（同位置用同色）
   定性对照.png        GT | 任意多列参考渲染 | 本次渲染，5 个视角

数据一律从 `<run>/full/test_inst/metrics.txt` 读，不在这里硬编码本次读数；
外部参考值（发布包 / 旧角域前馈 / CORF 三臂）与常数图地板的出处写在注释里。

参考渲染列由 `--extra_qual 标签=目录` 给出（可重复），目录里必须是**同名**的 640 张
测试视角；名字对不上就报错，不静默跳过——静默跳过会画出一列空白还照样出图。

用法（WSL mvsplat 环境）：
    $PY code/tools/make_param_figs.py \
        --metrics results/<run>/full/test_inst/metrics.txt \
        --meta datasets/RF-3DGS_cbf_params/meta.json \
        --qual_inst 317 --tag "本次全参数臂 60k" --out results/<run>/图 \
        --extra_qual "原 RF-3DGS=final/.../rf3dgs_released_renders"
"""
from __future__ import annotations

import argparse
import csv
import json
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

METRICS = [("psnr", "PSNR (dB)  ↑", 1), ("ssim", "SSIM  ↑", 1), ("lpips", "LPIPS  ↓", -1)]
FMT = {"psnr": ".3f", "ssim": ".4f", "lpips": ".4f"}

# 出处：results/_日志/angular_console.log（2026-10-20，同 640 测试视角、同指标定义）
REF = [("RF-3DGS 发布渲染", 14.1315, 0.7233, 0.4457),
       ("旧角域前馈（单源）", 11.8472, 0.7379, 0.4404)]
# CORF 三臂（逐场景优化，非前馈）：取自
# final/归档_已清理历史阶段/run_20261008_本机4060_CORF老师三臂/说明.md 与各 rf_results.json。
# **只取 CBF 那一列**：MVDR 是另一份射频真值（不同仿真方法、同一 300×200 网格但物理量不同），
# 与本次的 CBF 真值不可同轴比较（见 memory: rf-dataset-resolution-differs）。
# 冻结几何 = 两阶段基线（几何与光学都不动，只训射频外观），正是本仓前馈管线所处的设定。
# 只留「完整耦合」这一臂（论文 Ours）。另外两臂（冻结几何 12.2791、射频调几何 14.6049）
# 是消融阶梯，不是外部对照，读数仍在 数据.json 的 corf_ablation 里备查，不进图。
#
# 论文 Table II 的 18.52 **没有**进图：那是 147 关键帧（128 训 / 19 测）、10 次随机顺序
# 平均的口径，与本图 640 测试视角不同集合；而且论文正文只有一张总表，没有 CBF/MVDR
# 分列——从「本仓 MVDR 冻结几何 16.78 对论文 RF-3DGS 16.92」这一步对得上来看，
# 论文那份数据是 MVDR 口径，和图里这条 CBF 的 14.63 不是同一个物理量。
CORF = [("CORF 完整耦合", 14.6326, 0.7554, 0.4029)]
# 完整消融阶梯，只写进 数据.json 备查，不画
CORF_ABLATION = [("CORF 冻结几何", 12.2791, 0.6754, 0.5009),
                 ("CORF 射频调几何", 14.6049, 0.7574, 0.4020),
                 ("CORF 完整耦合", 14.6326, 0.7554, 0.4029)]
# 出处：code/tools/_floor_metrics.py 在 inst317 的 640 张 GT 上算的常数图地板
FLOOR = dict(psnr=14.2932, ssim=0.4664, lpips=0.4326)

COL_REF = ["#9aa0a6", "#c0a16b"]
COL_CORF = "#b5739d"
COL_OURS = "#2f6fbe"


def read_metrics(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path} 读空 —— 空表不许出图")
    out = {}
    for r in rows:
        out[int(r["id"])] = dict(role=r["role"], n=int(r["n"]),
                                 **{m: float(r[m]) for m in ("psnr", "ssim", "lpips")})
    return out


def fig_bars(rows, meta, tag, out_dir):
    iid = int(meta["orig_instance_id"])
    if iid not in rows:
        raise SystemExit(f"metrics 里没有主读数实例 {iid}")
    ours = [rows[iid]["psnr"], rows[iid]["ssim"], rows[iid]["lpips"]]

    # 柱子依次是：两条外部参考 → CORF 完整耦合 → 本次。CORF 用异色+斜纹，
    # 下方写明「逐场景优化」，免得读者把绝对读数和前馈直接当同设定比。
    # 刻度名单独写死短名：靠 replace 拆长名会拆不干净（"RF-3DGS 发布渲染" 中间只有一个
    # 空格），挤在一格里就糊成一片。
    SHORT = {"RF-3DGS 发布渲染": "RF-3DGS\n发布", "旧角域前馈（单源）": "旧角域\n前馈"}
    # 每根：标签、三指标、填充色、斜纹、短刻度名
    slots = ([(n, (a, b, c), COL_REF[i], None, SHORT.get(n, n))
              for i, (n, a, b, c) in enumerate(REF)]
             + [(n, (a, b, c), COL_CORF, "///", n.replace("CORF ", "CORF\n"))
                for n, a, b, c in CORF]
             + [(tag, tuple(ours), COL_OURS, None,
                 tag.replace("本次全参数臂 ", "本次\n全参数臂"))])
    n = len(slots)
    idx = {"psnr": 0, "ssim": 1, "lpips": 2}
    tick = [s[4] for s in slots]
    corf_i = len(REF)                    # CORF 那根的下标

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6))
    for ax, (key, ylab, _) in zip(axes, METRICS):
        vals = [s[1][idx[key]] for s in slots]
        bars = ax.bar(range(n), vals, color=[s[2] for s in slots],
                      hatch=[s[3] for s in slots], width=0.62,
                      edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, format(v, FMT[key]),
                    ha="center", va="bottom", fontsize=9)
        lo = min(vals + [FLOOR[key]])
        hi = max(vals + [FLOOR[key]])
        pad = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - pad * 1.15, hi + pad * 0.9)
        ax.axhline(FLOOR[key], ls="--", lw=1.1, color="#c0392b", zorder=0)
        ax.text(n - 0.55, FLOOR[key], " 常数图地板（同口径）", color="#c0392b",
                fontsize=7.5, va="bottom", ha="right")
        ax.set_xticks(range(n))
        ax.set_xticklabels(tick, fontsize=7.8, linespacing=1.15)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # 只剩一根，不画分组括号，直接把「逐场景优化」挂在这根柱子下面
        ax.text(corf_i, -0.20, "逐场景优化\n（非前馈）", color=COL_CORF,
                fontsize=7.2, ha="center", va="top", linespacing=1.15,
                transform=ax.get_xaxis_transform())
    fig.suptitle(f"留出发射源 TX48 位置上的三指标（作者 640 测试视角；{tag}）", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    p = out_dir / "三指标对比.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, ours


def fig_scatter(rows, meta, tag, out_dir):
    pos = {int(i["id"]): int(i["position_id"]) for i in meta["instances"]}
    ids = [i for i, r in rows.items() if r["role"] == "pos_holdout"]
    if len(ids) < 2:
        raise SystemExit(f"pos_holdout 只有 {len(ids)} 个实例，画不了散点")
    groups = defaultdict(list)
    for i in ids:
        groups[pos[i]].append(i)
    pids = sorted(groups)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (key, ylab, _) in zip(axes, METRICS):
        for j, pid in enumerate(pids):
            g = sorted(groups[pid])
            v = [rows[i][key] for i in g]
            ax.scatter([j] * len(v), v, s=26, color=cmap(j % 20), edgecolor="w",
                       linewidth=0.4, zorder=3)
            ax.hlines(np.mean(v), j - 0.28, j + 0.28, color="k", lw=1.4, zorder=4)
        ax.set_xticks(range(len(pids)))
        ax.set_xticklabels([f"P{p}" for p in pids], fontsize=7.5, rotation=90)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xlabel("未见发射源位置（每位置 4 组参数）", fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    allv = [rows[i]["psnr"] for i in ids]
    fig.suptitle(f"位置留出组：{len(ids)} 个实例 / {len(pids)} 个未见位置"
                 f"（PSNR {np.mean(allv):.2f} ± {np.std(allv):.2f} dB；黑横线 = 位置均值）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "位置留出_逐实例.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, len(ids), len(pids)


def fig_qual(inst_dir, out_dir, extras=(), k=5):
    """GT | extras... | 本次渲染。

    extras 里每个目录都必须**恰好**含有所选视角的同名文件；缺一张就报错退出。
    少了不报错会画出一列空白，而空白列在缩略图上看不出来（空图闸的同类坑）。
    """
    rd, gd = inst_dir / "renders", inst_dir / "gt"
    names = sorted(p.name for p in gd.glob("*.png"))
    if not names:
        raise SystemExit(f"{gd} 没有 GT")
    pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, k).round()]

    cols = [("GT", gd)] + list(extras) + [("本次渲染", rd)]
    for label, d in extras:
        have = {p.name for p in Path(d).glob("*.png")}
        miss = [n for n in names if n not in have]
        if miss:
            raise SystemExit(f"{label}：{d} 缺 {len(miss)} 张（例如 {miss[:3]}），"
                             f"不与 GT 同名同集合，不出图")

    fig, axes = plt.subplots(k, len(cols), figsize=(2.6 * len(cols), 2.05 * k),
                             squeeze=False)
    for r, n in enumerate(pick):
        for c, (label, d) in enumerate(cols):
            a = np.asarray(Image.open(Path(d) / n).convert("RGB"))
            axes[r, c].imshow(a)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(label, fontsize=11)
            if c == 0:
                axes[r, c].set_ylabel(n.replace(".png", ""), fontsize=8)
    head = " / ".join(label for label, _ in cols)
    fig.suptitle(f"定性对照：留出发射源（{head}；视野 300×200）", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "定性对照.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--meta", default="datasets/RF-3DGS_cbf_params/meta.json")
    ap.add_argument("--qual_inst", type=int, default=317)
    ap.add_argument("--tag", default="本次全参数臂")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra_qual", action="append", default=[],
                    metavar="标签=目录",
                    help="定性对照里插在 GT 与本次渲染之间的参考渲染列，可重复；"
                         "目录内须有与 GT 同名的 640 张测试视角")
    args = ap.parse_args()

    extras = []
    for spec in args.extra_qual:
        if "=" not in spec:
            raise SystemExit(f"--extra_qual 要写成 标签=目录，收到：{spec}")
        label, d = spec.split("=", 1)
        if not Path(d).is_dir():
            raise SystemExit(f"--extra_qual 的目录不存在：{d}")
        extras.append((label, Path(d)))

    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(args.metrics)
    meta = json.load(open(args.meta, encoding="utf-8"))

    p1, ours = fig_bars(rows, meta, args.tag, out_dir)
    p2, ni, npid = fig_scatter(rows, meta, args.tag, out_dir)
    inst_dir = Path(args.metrics).parent / f"inst{args.qual_inst:03d}"
    p3, picks = fig_qual(inst_dir, out_dir, extras=extras, k=5)

    payload = {
        "tag": args.tag,
        "reference": [dict(name=n, psnr=a, ssim=b, lpips=c) for n, a, b, c in REF],
        "reference_corf": [dict(name=n, psnr=a, ssim=b, lpips=c) for n, a, b, c in CORF],
        "corf_ablation": [dict(name=n, psnr=a, ssim=b, lpips=c) for n, a, b, c in CORF_ABLATION],
        "corf_note": "图里只画 CORF 完整耦合（论文 Ours）。取 CBF 列，与本次同一份射频真值；"
                     "MVDR 是另一份真值，不同轴。该臂为逐场景优化，本次为前馈。"
                     "论文 Table II 的 18.52 未入图：那是 147 关键帧 / 19 测试 / 10 次平均的口径，"
                     "与 640 测试视角不同集合，且论文没有 CBF/MVDR 分列；由"
                     "「本仓 MVDR 冻结几何 16.78 对论文 RF-3DGS 16.92」判断其为 MVDR 口径。"
                     "渲染图已随归档清理，定性对照里暂无 CORF 列。",
        "qual_columns": [label for label, _ in extras],
        "floor_constant_image": FLOOR,
        "ours_main_reading": dict(instance=args.qual_inst, psnr=ours[0],
                                  ssim=ours[1], lpips=ours[2]),
        "pos_holdout": dict(n_inst=ni, n_pos=npid),
        "figures": [str(p1), str(p2), str(p3)],
        "qual_views": picks,
    }
    (out_dir / "数据.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"写出：\n  {p1}\n  {p2}\n  {p3}\n  {out_dir / '数据.json'}")
    print(f"主读数 {ours[0]:.4f} / {ours[1]:.4f} / {ours[2]:.4f}；"
          f"位置留出 {ni} 实例 / {npid} 位置")


if __name__ == "__main__":
    main()
