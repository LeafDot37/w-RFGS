# -*- coding: utf-8 -*-
"""make_vis_slope_figs.py —— 方案C 轴1 的出图：**视觉短训步数 → RF 质量** 的因果斜率。

两张图（每方法一份数据，图里两方法并列）：

  A `因果斜率_<M>.png`   1×3 面板（PSNR / SSIM / LPIPS）。
                        横轴 = 视觉短训步数（**等距分类轴**，不是线性轴：0 与 30000 放
                        同一个线性轴上会把 500–8000 挤成一团；分类轴上每个点的步数写在刻度里，
                        所以没有骗人）。每个点标出**该档的点数 P**。
                        灰带 = ±3×抖动（同 RRF 阶段同输入重跑实测），**判据就是"点落在带里"**。

  B `判据_首末档跨度.png`  每方法每指标一根柱：**首末档跨度**（|末档 − 首档|），
                        横向虚线 = 3×抖动。柱高在虚线之上 = 斜率显著，之下 = 平坦。

为什么点数 P 必须画在图上
-----------------------------------------------------------------------------
这是本设计**消不掉的混杂**：视觉步数不同 ⇒ 致密化/剪枝历史不同 ⇒ 终态点数不同，
而 A× 头的逐点 embedding 是 P×32 ⇒ **点数差就是容量差**。
第一版设计里我以为"只差视觉步数"是干净的单变量，那是错的：P 也在动。
把 P 印在每个点上，读者才能自己判断"RF 变好"是视觉质量的功劳还是点变多了。
它**不能**把两件事分开；能分开的做法是计数配平（`make_count_matched_ckpt.py`，另一条臂）。

数据源（逐项本地重算，不转述）
    <TOP>/RRF/vis<步>/<方法>/results.json            节点 ours_<步+RRF_ITERS>
    <TOP>/RRF/vis<步>/<方法>/point_cloud/iteration_*/point_cloud.ply   点数读 ply 头
抖动（**不是本图各臂实测**）
    MVDR 0.0418 / 0.00084 / 0.00178；CBF 0.2461 / 0.00128 / 0.00254
    来源 results/run_20260914_协方差初始化/说明.md（同 RRF 阶段、同输入重跑）

用法（本地仓库根，无需 GPU）
    python code/tools/figures/make_vis_slope_figs.py --top <TOP 的本地镜像> --out final/因果斜率
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import json
import re
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import _cnfont
    _cnfont.configure(plt)
except Exception:                                                            # noqa: BLE001
    pass

METRICS = [("PSNR", "PSNR (dB)  ↑", 1, ".3f"), ("SSIM", "SSIM  ↑", 1, ".4f"),
           ("LPIPS", "LPIPS (VGG)  ↓", -1, ".4f")]
MCOLOR = {"MVDR": "#1f77b4", "CBF": "#d62728"}
# 抖动（同 RRF 阶段同输入重跑实测）——只作读数尺度，**不是**本图各臂实测
JIT = {"MVDR": {"PSNR": 0.0418, "SSIM": 0.00084, "LPIPS": 0.00178},
       "CBF":  {"PSNR": 0.2461, "SSIM": 0.00128, "LPIPS": 0.00254}}
JIT_SRC = ("抖动 = 同 RRF 阶段（A× 标量头）的同输入重跑实测，"
           "来源 results/run_20260914_协方差初始化/说明.md；本图各臂**未**单独实测抖动")


def ply_count(p: Path):
    """从 ply 文件头读 `element vertex N` —— 只读前 1 KB，不把 300 MB 拉进内存。"""
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        head = f.read(1024)
    m = re.search(rb"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def load_arm(top: Path, step: int, method: str, rrf_iters: int):
    md = top / "RRF" / f"vis{step}" / method
    abs_it = step + rrf_iters
    rj = md / "results.json"
    if not rj.is_file():
        return None, f"{method} 档 {step}：缺 results.json（{rj}）"
    node = json.load(open(rj, encoding="utf-8")).get(f"ours_{abs_it}")
    if not isinstance(node, dict):
        return None, f"{method} 档 {step}：results.json 里没有 ours_{abs_it}"
    miss = [k for k, _t, _s, _f in METRICS if k not in node]
    if miss:
        return None, f"{method} 档 {step}：ours_{abs_it} 缺 {miss}"
    out = {k: node[k] for k, _t, _s, _f in METRICS}
    out["n_views"] = node.get("n_views")
    pc = md / "point_cloud" / f"iteration_{abs_it}" / "point_cloud.ply"
    out["P"] = ply_count(pc)
    out["abs_iter"] = abs_it
    out["渲染目录"] = str(pc.parent).replace("\\", "/")
    return out, None


def draw_slope(method, steps, vals, jit, out_png, notes):
    """3 面板：x=视觉步数（等距分类轴），y=指标；灰带=±3×抖动。"""
    n = len(steps)
    xs = list(range(n))
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.0))
    for ax, (m, title, sign, fmt) in zip(axes, METRICS):
        ys = [vals[s][m] for s in steps]
        j = jit[m]
        mu = sum(ys) / len(ys)
        # 灰带画在"全轴均值 ±3×抖动"上：判据是"首末档跨度是否超出这条带"
        ax.axhspan(mu - 3 * j, mu + 3 * j, color="#cccccc", alpha=.55, zorder=0)
        ax.axhline(mu, color="#888888", lw=1.0, ls=":", zorder=1)
        ax.plot(xs, ys, "-o", color=MCOLOR[method], lw=2.0, ms=6.5, zorder=3,
                label=f"{method}（{len(steps)} 档）")
        for x, s, y in zip(xs, steps, ys):
            P = vals[s].get("P")
            ax.annotate(f"{y:{fmt}}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.4,
                        fontweight="bold", color=MCOLOR[method])
            if P:
                ax.annotate(f"{P/1000:.0f}k 点", (x, y), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=7.2, color="#666666")
        span = abs(ys[-1] - ys[0])
        ax.set_title(f"{title}\n首末档跨度 {span:{fmt}}　3×抖动 {3*j:{fmt}}"
                     f"　⇒ {'显著' if span > 3*j else '平坦'}",
                     fontsize=10.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in steps], fontsize=8.6)
        ax.set_xlabel("视觉短训步数（0 = 纯 MVSplat 前馈，零视觉优化）", fontsize=9.4)
        ax.grid(axis="y", alpha=.3)
        ax.margins(x=.08, y=.22)
        # y 轴不从 0 起 —— 只在轴上写明，不偷偷放大
        ax.annotate("纵轴非零起点", (0.02, 0.02), xycoords="axes fraction",
                    fontsize=6.8, color="#888888")
    fig.suptitle(f"[{method}] 视觉短训步数 → RF 质量的因果斜率"
                 f"（同一 RRF 配置，本地 {vals[steps[0]]['abs_iter'] - steps[0]} 步，640 全量）",
                 fontsize=12.8, y=0.985)
    fig.text(.5, .032,
             "灰带 = 全轴均值 ±3×抖动。判据（预登记）：首末档跨度 > 3×抖动 ⇒ 斜率显著；"
             "否则平坦。每个点下方标出**该档的点数 P** —— 视觉步数不同 ⇒ 致密化历史不同 ⇒ "
             "P 不同，而 A× 头的逐点 embedding 是 P×32，**点数差就是容量差**，"
             "这是本轴消不掉的混杂，不能读成纯粹的「视觉质量」效应。",
             ha="center", fontsize=7.8, color="#333333")
    fig.text(.5, .012, JIT_SRC + ("；" + "；".join(notes) if notes else ""),
             ha="center", fontsize=7.2, color="#777777")
    fig.subplots_adjust(bottom=.20, top=.845, wspace=.20)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_criterion(method, steps, vals, jit, out_png):
    """一张判据图：每指标一根柱 = 首末档跨度，横线 = 3×抖动。"""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ms = [m for m, _t, _s, _f in METRICS]
    spans = [abs(vals[steps[-1]][m] - vals[steps[0]][m]) for m in ms]
    thrs = [3 * jit[m] for m in ms]
    xs = range(len(ms))
    ax.bar([x - .18 for x in xs], spans, .34, color="#4c72b0", label="首末档跨度")
    ax.bar([x + .18 for x in xs], thrs, .34, color="#bbbbbb", label="3× 抖动（判据线）")
    for x, s, t in zip(xs, spans, thrs):
        ax.annotate(f"{s:.4g}", (x - .18, s), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.6, fontweight="bold")
        ax.annotate(f"{t:.4g}", (x + .18, t), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.6, color="#666666")
        ax.annotate("显著" if s > t else "平坦", (x, max(s, t)), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=9.2,
                    fontweight="bold", color="#1a9850" if s > t else "#c0392b")
        # 三个指标量纲差 3 个数量级（0.0008 vs 5 dB）⇒ 跨度柱必须归一化看待
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{m}\n（跨度 {s:.4g} / 阈值 {t:.4g}）" for m, s, t in zip(ms, spans, thrs)],
                       fontsize=8.8)
    ax.set_ylabel("|末档 − 首档|（各指标自身量纲，**不可跨指标比柱高**）", fontsize=9.4)
    ax.set_title(f"[{method}] 判据：首末档跨度 vs 3×抖动\n"
                 f"（视觉步数 {steps[0]} → {steps[-1]}；柱高不可跨指标比较，只看是否越过灰柱）",
                 fontsize=11.4)
    ax.legend(fontsize=9.0)
    ax.grid(axis="y", alpha=.3)
    fig.text(.5, .015, "三个指标量纲差三个数量级 ⇒ 本图**只在同一指标内**比较两根柱的高度；"
                       "跨指标看柱高的高低没有意义。", ha="center", fontsize=7.4, color="#777777")
    fig.subplots_adjust(bottom=.16, top=.86)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="方案C 轴1 因果斜率出图")
    ap.add_argument("--top", required=True, help="run_20260915_视觉质量因果斜率 的本地镜像")
    ap.add_argument("--methods", nargs="+", default=["MVDR", "CBF"])
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 500, 1000, 2000, 4000, 8000, 30000])
    ap.add_argument("--rrf-iters", type=int, default=10000)
    ap.add_argument("--out", default="final/因果斜率")
    args = ap.parse_args()

    top = Path(args.top).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    payload, notes = {}, []
    for M in args.methods:
        vals, drop = {}, []
        for s in args.steps:
            v, err = load_arm(top, s, M, args.rrf_iters)
            if v is None:
                notes.append(err)
                drop.append(s)
                continue
            vals[s] = v
        keep = [s for s in args.steps if s in vals]
        payload[M] = {"读数": {str(s): vals[s] for s in keep}, "缺档": drop}
        if len(keep) < 2:
            print(f"  [skip] {M}：只剩 {len(keep)} 档，画不出斜率")
            continue
        draw_slope(M, keep, vals, JIT[M], out / M / f"因果斜率_{M}.png",
                   [n for n in notes if n.startswith(M + " ")])
        draw_criterion(M, keep, vals, JIT[M], out / M / f"判据_首末档跨度_{M}.png")
        print(f"  [{M}] 画出 {len(keep)} 档（{keep[0]} → {keep[-1]}）")
        for s in keep:
            v = vals[s]
            print("      %-6d abs %-6d P %-9s PSNR %8.4f  SSIM %8.5f  LPIPS %8.5f"
                  % (s, v["abs_iter"], v.get("P") or "—", v["PSNR"], v["SSIM"], v["LPIPS"]))
        for m, _t, _s, fmt in METRICS:
            span = abs(vals[keep[-1]][m] - vals[keep[0]][m])
            thr = 3 * JIT[M][m]
            print("      %-6s 首末档跨度 %s  3×抖动 %s  ⇒ %s"
                  % (m, f"{span:{fmt}}", f"{thr:{fmt}}", "显著" if span > thr else "平坦"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "数据.json").write_text(json.dumps(
        {"口径": "方案C 轴1：视觉短训步数 → RF 质量；同一 RRF 配置（A× 标量头，本地 1 万步）",
         "档位": args.steps, "抖动": JIT, "抖动来源": JIT_SRC,
         "读数": payload, "缺项": notes}, ensure_ascii=False, indent=2), encoding="utf-8")
    if notes:
        print("  缺项：")
        for n in notes:
            print(f"    - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
