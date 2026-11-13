# -*- coding: utf-8 -*-
"""
make_bars_rrf40k.py —— 阶段二 RRF 拉长 4 万步：末档对比柱状图（MLP vs 原版SH，本地出图）
--------------------------------------------------------------------------------
双引擎 RRF 到绝对 70000（=RRF 4万步）后，服务器侧对末档做了 **全量 640 held-out** 评测
（tools/metrics_fast.py --subset 0 → results.json 的 ours_70000）。本工具按
MPC/CBF/Delay/MVDR 四方法、PSNR/SSIM/LPIPS 三指标画分组柱状图 + Δ(MLP−SH) 标注：
  - 原版 SH  = results/run_rrf40k_双引擎4万步/sh/<M>/results.json（ours_70000）
  - 我方 MLP = results/run_rrf40k_双引擎4万步/mlp/<M>/results.json（ours_70000）
末档全量口径与收敛曲线（抽帧48）分开锚定：本图数字可作文档/结论最终依据。

用法（WSL mvsplat env；本地仓库 /path/to/w-RFGS 视图，cwd=code/）：
    <PY> tools/make_bars_rrf40k.py --run /path/to/w-RFGS/results/run_rrf40k_双引擎4万步 \
                                   --out /path/to/w-RFGS/results/run_rrf40k_双引擎4万步/图

输出（默认 <run>/图）：
    对比柱状图_阶段二_RRF40k_MLP-vs-SH.png   3 面板分组柱状图（Δ 标在组上留白区）
    对比柱状图_阶段二_RRF40k_MLP-vs-SH.json 数值 payload（可入文档）
"""
import argparse
import json
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

_METHODS = ["MPC", "CBF", "Delay", "MVDR"]
_KEYS = ("PSNR", "SSIM", "LPIPS")
_ITER = 70000
_SIDES = [("mlp", "MLP", "embedding+MLP 颜色头", "#1f77b4"),
          ("sh", "SH", "原版 逐点 SH 颜色头", "#d62728")]


def load_results(run_dir, eng):
    """<run>/<eng>/<M>/results.json → {M: metrics(ours_70000)}。"""
    out = {}
    for m in _METHODS:
        rj = Path(run_dir) / eng / m / "results.json"
        if not rj.is_file():
            raise FileNotFoundError(f"缺末档结果: {rj}")
        v = json.load(open(rj, encoding="utf-8")).get(f"ours_{_ITER}")
        if not v:
            raise RuntimeError(f"{rj} 缺 ours_{_ITER}")
        out[m] = v
    return out


def draw_bars(mlp, sh, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import _cnfont
    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140

    titles = {"PSNR": "PSNR (dB) ↑", "SSIM": "SSIM ↑", "LPIPS": "LPIPS ↓"}
    fmt = {"PSNR": ".2f", "SSIM": ".3f", "LPIPS": ".3f"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    x = np.arange(len(_METHODS))
    w = 0.36
    for ax, (key, title) in zip(axes, titles.items()):
        mv = [mlp[m][key] for m in _METHODS]
        sv = [sh[m][key] for m in _METHODS]
        b1 = ax.bar(x - w / 2, mv, w, color=_SIDES[0][3], label=f"{_SIDES[0][2]}（ours_{_ITER}）")
        b2 = ax.bar(x + w / 2, sv, w, color=_SIDES[1][3], label=f"{_SIDES[1][2]}（ours_{_ITER}）")
        for b in list(b1) + list(b2):
            ax.annotate(f"{b.get_height():{fmt[key]}}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7.5)
        hi = max(max(mv), max(sv))
        ax.set_ylim(0, hi * 1.24)
        for m, a, c in zip(_METHODS, mv, sv):
            dv = a - c
            good = (dv > 0) if key != "LPIPS" else (dv < 0)
            ax.annotate(f"Δ{dv:+.3f}", (x[_METHODS.index(m)], hi),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color="#1a9850" if good else "#666666")
        ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(x); ax.set_xticklabels(_METHODS)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=.9)
    fig.suptitle(f"阶段二 RRF 拉长 4 万步：颜色表达单变量对比（同起点 orig_seed 视觉 30000，"
                 f"双引擎各整段重训 RRF→4万 = abs{_ITER}；全量 640 held-out，同 metrics.py）",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="阶段二 RRF40k 末档 MLP-vs-原版SH 对比柱状图（本地出图）")
    ap.add_argument("--run", required=True, help="本地结果目录（含 sh|mlp/<M>/results.json）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/图）")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "图"
    out.mkdir(parents=True, exist_ok=True)

    mlp = load_results(run, "mlp")
    sh = load_results(run, "sh")

    png = out / "对比柱状图_阶段二_RRF40k_MLP-vs-SH.png"
    draw_bars(mlp, sh, png)

    payload = {
        "协议": f"阶段二 RRF abs{_ITER}（orig_seed 视觉 30000 + RRF 4 万），双引擎各整段重训，"
                "640 held-out 全量，metrics.py。MLP=embedding+共享MLP颜色头；"
                "SH=原版逐点SH颜色头（本长跑重训，非 config 参考值）。",
        "MLP": {m: mlp[m] for m in _METHODS},
        "原版SH": {m: sh[m] for m in _METHODS},
        "Δ(MLP−SH)": {m: {k: round(mlp[m][k] - sh[m][k], 4) for k in _KEYS} for m in _METHODS},
    }
    jf = out / "对比柱状图_阶段二_RRF40k_MLP-vs-SH.json"
    json.dump(payload, open(jf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"  已生成: {png}\n          {jf}")
    for m in _METHODS:
        d = payload["Δ(MLP−SH)"][m]
        print(f"  [{m}]  ΔPSNR {d['PSNR']:+.3f}  ΔSSIM {d['SSIM']:+.4f}  ΔLPIPS {d['LPIPS']:+.4f}")


if __name__ == "__main__":
    main()
