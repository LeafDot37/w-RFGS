# -*- coding: utf-8 -*-
"""阶段 3 的交付图：**样本效率曲线**。

横轴 = 监督用的频谱关键帧数 N，纵轴 = 三指标（**PSNR / SSIM / LPIPS 一律并报，LPIPS 为主判据**）。
每条线一个臂；**两条线唯一的差别就是射频头的输入有没有那 12 维视觉特征**。

数据来源：`<TOP>/<臂>/<N标签>/rf_results.json`（由 `metrics_fast.py` 产出，不手写）
         `<TOP>/<臂>/<N标签>/student_experiment.json`（取实际用到的 N 与关键帧索引）

用法：
    python make_student_curve.py --top <学生结果目录> --out <图.png> [--teacher <json>]

`--teacher` 给「逐场景优化」在 N = 全部时的三指标 json（阶段 1 的读数），画一条参考横线。
**没有它这条线就不要画** —— 不许拿别处的数字顶替。

纯 matplotlib、**0 GPU**，本机跑。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# 臂目录名 -> 图上标签。文字名，不用代号（全仓纪律）。
ARM_LABEL = {
    "有视觉": "有视觉耦合",
    "无视觉": "无视觉耦合",
    "阴性对照_特征打乱": "阴性对照（视觉特征打乱）",
}
ARM_ORDER = ["有视觉", "无视觉", "阴性对照_特征打乱"]
ARM_STYLE = {
    "有视觉": dict(color="#c0392b", marker="o", lw=2.4, ms=8),
    "无视觉": dict(color="#2980b9", marker="s", lw=2.4, ms=8),
    "阴性对照_特征打乱": dict(color="#7f8c8d", marker="^", lw=1.8, ms=7, ls="--"),
}
METRICS = [("PSNR", "PSNR (dB)  ↑", 1), ("SSIM", "SSIM  ↑", 1), ("LPIPS", "LPIPS (VGG)  ↓", -1)]


def load_runs(top):
    runs = []
    for arm in os.listdir(top):
        d_arm = os.path.join(top, arm)
        if not os.path.isdir(d_arm):
            continue
        for tag in os.listdir(d_arm):
            d = os.path.join(d_arm, tag)
            j = os.path.join(d, "rf_results.json")
            e = os.path.join(d, "student_experiment.json")
            if not (os.path.isfile(j) and os.path.isfile(e)):
                continue
            res = json.load(open(j))
            exp = json.load(open(e))
            # metrics_fast 的 json 是 {<iteration>: {...}} 或直接 {...}，两种都吃
            rec = next(iter(res.values())) if isinstance(next(iter(res.values())), dict) else res
            n_used = int(exp.get("n_keyframes_used", 0))
            n_all = int(exp.get("n_rf_cameras", 0))
            runs.append(dict(
                arm=arm, tag=tag, dir=d, n=(n_all if exp.get("n_keyframes", 0) <= 0 else n_used),
                n_used=n_used, n_all=n_all,
                psnr=float(rec["PSNR"]), ssim=float(rec["SSIM"]), lpips=float(rec["LPIPS"]),
                # metrics_fast.py:71 写的键是 n_views（不是 n / count）
                n_img=int(rec.get("n_views", 0)),
            ))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="", help="逐场景优化（N=全部）的 results.json，画参考横线")
    ap.add_argument("--teacher_label", default="逐场景优化（阶段 1，N=全部）")
    ap.add_argument("--suptitle", default="样本效率曲线：前馈学生（只训射频头）")
    ap.add_argument("--foot", default="")
    args = ap.parse_args()

    runs = load_runs(args.top)
    if not runs:
        print("[curve] 一个 rf_results.json 都没找到 —— **不出图**（不许拿空图顶）")
        return 2

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _cnfont
        _cnfont.configure(plt)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[curve] 中文字体未配置（{exc}），标签可能缺字形")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    for ax, (key, ylabel, sign) in zip(axes, METRICS):
        plotted = 0
        for arm in ARM_ORDER:
            pts = sorted([r for r in runs if r["arm"] == arm], key=lambda r: r["n"])
            if not pts:
                continue
            xs = [r["n"] for r in pts]
            ys = [r[key.lower()] for r in pts]
            ax.plot(xs, ys, label=ARM_LABEL[arm], **ARM_STYLE[arm])
            for r in pts:
                ax.annotate(f"{r[key.lower()]:.3f}" if key != "SSIM" else f"{r[key.lower()]:.4f}",
                            (r["n"], r[key.lower()]), textcoords="offset points",
                            xytext=(0, 9 if sign > 0 else -14), ha="center", fontsize=7.6,
                            color=ARM_STYLE[arm]["color"])
            plotted += len(pts)
        # 没被 ARM_ORDER 覆盖的臂（例如 可抄性）单独画，免得静默丢掉数据
        for arm in sorted({r["arm"] for r in runs} - set(ARM_ORDER)):
            pts = sorted([r for r in runs if r["arm"] == arm], key=lambda r: r["n"])
            ax.plot([r["n"] for r in pts], [r[key.lower()] for r in pts],
                    label=arm, color="#8e44ad", marker="D", lw=1.8, ms=7, ls=":")
            plotted += len(pts)

        if args.teacher and os.path.isfile(args.teacher):
            t = json.load(open(args.teacher))
            rec = next(iter(t.values())) if isinstance(next(iter(t.values())), dict) else t
            ax.axhline(float(rec[key]), color="#27ae60", ls="-.", lw=1.6,
                       label=args.teacher_label)
        elif args.teacher:
            print(f"[curve] 警告：--teacher 指向的文件不存在（{args.teacher}），不画参考线")

        ax.set_xlabel("监督用的频谱关键帧数 N", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=11.5)
        ax.grid(alpha=.28, ls=":")
        ns = sorted({r["n"] for r in runs})
        n_all = max(r["n_all"] for r in runs)
        ax.set_xticks(ns)
        ax.set_xticklabels([("全部" if v >= n_all else str(v)) for v in ns], fontsize=9.5)
        ax.set_title(f"{key}  （{plotted} 个配置）", fontsize=11.5)

    axes[0].legend(loc="lower right", fontsize=9.0, framealpha=.92)
    if args.foot:
        fig.text(.5, .012, args.foot, ha="center", fontsize=8.0, color="#555555", wrap=True)
    fig.suptitle(args.suptitle, fontsize=13.0, y=1.0)
    fig.subplots_adjust(bottom=.24, top=.87, wspace=.24)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[curve] -> {args.out}")

    print("\n[curve] 读数表（供文档引用）")
    print(f"{'臂':<22}{'N':>8}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}{'张数':>8}")
    for r in sorted(runs, key=lambda r: (r["arm"], r["n"])):
        print(f"{ARM_LABEL.get(r['arm'], r['arm']):<22}{r['n']:>8}"
              f"{r['psnr']:>10.3f}{r['ssim']:>10.4f}{r['lpips']:>10.4f}{r['n_img']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
