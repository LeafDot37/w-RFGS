# -*- coding: utf-8 -*-
"""make_fewshot_figs.py —— 单场景多射频源泛化的结果表与图。

两种模式：
    curve  协议 A：测试指标 vs 训练源数 N（每场景一条），常数基线为虚线
    power  协议 B：测试指标 vs 合成增益 g（功率条件 / 功率盲 / 常数基线）

同时打印 Markdown 表（PSNR / SSIM / LPIPS），并做空图闸：每个面板的点数必须 > 0，
否则直接报错（图里有没有东西只能在出图端数）。

用法：
    <PY> code/tools/figures/make_fewshot_figs.py --mode curve --root results/fewshot
    <PY> code/tools/figures/make_fewshot_figs.py --mode power --root results/fewshot/power
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import _cnfont                            # noqa: E402

METRICS = [("PSNR", "PSNR / dB", True), ("SSIM", "SSIM", True),
           ("LPIPS", "LPIPS（越低越好）", False)]


def read_json(p: Path):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def read_one(d: Path):
    """`<dir>/results.json` 里的单个 ours_<it> 记录。"""
    j = read_json(d / "results.json")
    if not j:
        return None
    return list(j.values())[0]


def collect_curve(root: Path):
    """返回 {scene: {"Ns": [...], "runs": {N: rec}, "const": rec}}"""
    out = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(.+)_N(\d+)$", p.name)
        if m:
            rec = read_one(p)
            if rec:
                out.setdefault(m.group(1), {"runs": {}, "const": None})
                out[m.group(1)]["runs"][int(m.group(2))] = rec
            continue
        m = re.match(r"^(.+)_const$", p.name)
        if m:
            rec = read_one(p)
            if rec:
                out.setdefault(m.group(1), {"runs": {}, "const": None})
                out[m.group(1)]["const"] = rec
    # 同一个 N 出现多次时（重跑）取 results.json 所在者，已自然覆盖
    return out


def fmt_table_curve(data):
    lines = ["| 场景 | N | PSNR | SSIM | LPIPS | 帧数 |", "|---|---|---|---|---|---|"]
    for s, d in sorted(data.items()):
        if d["const"]:
            c = d["const"]
            lines.append(f"| {s} | 常数基线 | {c['PSNR']:.3f} | {c['SSIM']:.4f} | "
                         f"{c['LPIPS']:.4f} | {c['n_views']} |")
        for n in sorted(d["runs"]):
            r = d["runs"][n]
            lines.append(f"| {s} | {n} | {r['PSNR']:.3f} | {r['SSIM']:.4f} | "
                         f"{r['LPIPS']:.4f} | {r['n_views']} |")
    return "\n".join(lines)


def fig_curve(data, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    counts = []
    for ax, (key, ylabel, _higher) in zip(axes, METRICS):
        npts = 0
        for s, d in sorted(data.items()):
            ns = sorted(d["runs"])
            if not ns:
                continue
            ys = [d["runs"][n][key] for n in ns]
            ax.plot(ns, ys, marker="o", label=s)
            npts += len(ns)
            if d["const"]:
                ax.axhline(d["const"][key], ls="--", lw=1.0, color="gray",
                           label=f"{s} 常数基线")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("训练使用的源数 N")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        counts.append(npts)
    axes[0].legend(fontsize=7)
    fig.suptitle("单场景少样本：测试指标 vs 源数", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return counts


def collect_power(root: Path):
    """{scene: {arm: {gain: rec}}}，arm ∈ {功率条件, 功率盲, 常数基线}"""
    out = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(.+)_(cond|blind|const)$", p.name)
        if not m:
            continue
        scene, arm = m.group(1), {"cond": "功率条件", "blind": "功率盲",
                                  "const": "常数基线"}[m.group(2)]
        gdirs = {}
        for gd in sorted(p.iterdir()):
            mg = re.match(r"^gain_g(\d+)$", gd.name)
            if not mg:
                continue
            rec = read_one(gd)
            if rec:
                gdirs[int(mg.group(1)) / 100.0] = rec
        if gdirs:
            out.setdefault(scene, {})[arm] = gdirs
    return out


def fmt_table_power(data):
    lines = ["| 场景 | 臂 | 增益 | PSNR | SSIM | LPIPS | 帧数 |",
             "|---|---|---|---|---|---|---|"]
    for s, arms in sorted(data.items()):
        for arm, gs in arms.items():
            for g in sorted(gs):
                r = gs[g]
                lines.append(f"| {s} | {arm} | {g:.2f} | {r['PSNR']:.3f} | "
                             f"{r['SSIM']:.4f} | {r['LPIPS']:.4f} | {r['n_views']} |")
    return "\n".join(lines)


def fig_power(data, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    counts = []
    for ax, (key, ylabel, _higher) in zip(axes, METRICS):
        npts = 0
        for s, arms in sorted(data.items()):
            for arm, gs in arms.items():
                xs = sorted(gs)
                ys = [gs[g][key] for g in xs]
                ax.plot(xs, ys, marker="o", label=arm if len(data) == 1 else f"{s} {arm}")
                npts += len(xs)
        ax.set_xlabel("合成功率增益 g")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        counts.append(npts)
    axes[0].legend(fontsize=7)
    fig.suptitle("单场景功率条件化：测试指标 vs 增益", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("curve", "power"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--fig_dir", default="")
    args = ap.parse_args()

    _cnfont.configure(plt)
    root = Path(args.root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else root / "图"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "curve":
        data = collect_curve(root)
        if not data:
            raise SystemExit(f"没有可用的 *_N* 结果: {root}")
        print(fmt_table_curve(data))
        out_png = fig_dir / "少样本曲线.png"
        counts = fig_curve(data, out_png)
    else:
        data = collect_power(root)
        if not data:
            raise SystemExit(f"没有可用的 *_cond/_blind/_const 结果: {root}")
        print(fmt_table_power(data))
        out_png = fig_dir / "功率条件化.png"
        counts = fig_power(data, out_png)

    print(f"\n图 -> {out_png}")
    # 空图闸：每个面板必须有数据点
    if min(counts) <= 0:
        raise SystemExit(f"空图闸失败：各面板点数 {counts}")
    print(f"空图闸通过：各面板点数 {counts}")


if __name__ == "__main__":
    main()
