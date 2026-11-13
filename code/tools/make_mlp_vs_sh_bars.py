# -*- coding: utf-8 -*-
"""
make_mlp_vs_sh_bars.py —— 阶段二颜色形式单变量对比柱状图（本地出图）
--------------------------------------------------------------------------------
embedding+MLP 颜色头（本实验，ours_40000，已拉回本地 results/run_mlp_rrf_full_MLP首轮全量/<M>/results.json）
vs 原版 RF-3DGS 逐点 SH 颜色（config.yaml eval.reference.stage2_orig，= 原版 recipe abs40000 参考值），
按 MPC/CBF/Delay/MVDR 四方法，PSNR / SSIM / LPIPS 三指标各一张分组柱状图 + 标注 Δ(MLP−SH)。

用法（WSL mvsplat env；本地仓库 /path/to/w-RFGS 视图）：
    <PY> tools/make_mlp_vs_sh_bars.py \
        --run /path/to/w-RFGS/results/run_mlp_rrf_full_MLP首轮全量 \
        --cfg  /path/to/w-RFGS/config.yaml \
        --out  /path/to/w-RFGS/results/run_mlp_rrf_full_MLP首轮全量/图

输出：
    图/对比柱状图_阶段二_MLP-vs-SH.png   3 面板分组柱状图
    图/对比柱状图_阶段二_MLP-vs-SH.json 数值 payload（可入文档）
"""
import argparse
import json
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

METHODS = ["MPC", "CBF", "Delay", "MVDR"]
KEYS = ["PSNR", "SSIM", "LPIPS"]
ITER = 40000


def load_local_results(run_dir):
    """我方 MLP 结果：run_dir/<M>/results.json 的 ours_40000。"""
    out = {}
    for m in METHODS:
        rj = Path(run_dir) / m / "results.json"
        if not rj.is_file():
            raise FileNotFoundError(f"缺本地 results.json: {rj}（先 scp 服务器结果）")
        d = json.load(open(rj, encoding="utf-8"))
        v = d.get(f"ours_{ITER}")
        if not v:
            raise RuntimeError(f"{m} results.json 缺 ours_{ITER}: {rj}")
        out[m] = v
    return out


def load_sh_reference(cfg_path):
    """原版 SH 参考：config.yaml eval.reference.stage2_orig（权威源）。"""
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    ref = (cfg.get("eval") or {}).get("reference") or {}
    stage2 = ref.get("stage2_orig") or {}
    if not all(m in stage2 for m in METHODS):
        raise RuntimeError(f"config.yaml stage2_orig 缺方法: {stage2.keys()}")
    return stage2


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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"MLP": "#1f77b4", "SH": "#d62728"}
    x = np.arange(len(METHODS))
    w = 0.36

    for ax, (key, title) in zip(axes, titles.items()):
        mv = [mlp[m][key] for m in METHODS]
        sv = [sh[m][key] for m in METHODS]
        b1 = ax.bar(x - w / 2, mv, w, color=colors["MLP"],
                    label="embedding+MLP（ours_40000）")
        b2 = ax.bar(x + w / 2, sv, w, color=colors["SH"],
                    label="原版 逐点SH（stage2_orig）")
        # 柱顶数值（每根柱正上方，offset 2pt）
        for b in list(b1) + list(b2):
            ax.annotate(f"{b.get_height():{fmt[key]}}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7.5)
        # 顶留白：给 Δ 标注让出空间（annotation 按 point 偏移不进 autoscale）
        hi = max(max(mv), max(sv))
        ax.set_ylim(0, hi * 1.24)
        # Δ(MLP−SH)：组中心、柱对上方留白区，避免与柱顶数值重叠
        for m, a, c in zip(METHODS, mv, sv):
            dv = a - c
            good = (dv > 0) if key != "LPIPS" else (dv < 0)
            ax.annotate(f"Δ{dv:+.3f}", (x[METHODS.index(m)], hi),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color="#1a9850" if good else "#666666")
        ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(x); ax.set_xticklabels(METHODS)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=.9)
    fig.suptitle("阶段二 RRF：颜色表达单变量对比（同起点 orig_seed 视觉 30000，RRF 1 万步，"
                 "abs40000，640 held-out，同 metrics.py）", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return mv, sv


def main():
    ap = argparse.ArgumentParser(description="阶段二 MLP-vs-SH 对比柱状图（本地出图）")
    ap.add_argument("--run", required=True, help="本地结果目录（含 <M>/results.json）")
    ap.add_argument("--cfg", required=True, help="config.yaml（读 stage2_orig 参考）")
    ap.add_argument("--out", required=True, help="输出目录（png/json）")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mlp = load_local_results(args.run)
    sh = load_sh_reference(args.cfg)

    png = out / "对比柱状图_阶段二_MLP-vs-SH.png"
    draw_bars(mlp, sh, png)

    payload = {
        "协议": f"阶段二 RRF abs{ITER}（orig_seed 视觉 30000 + RRF 1 万），640 held-out，metrics.py。"
                "MLP=embedding+共享MLP颜色头（ours_40000，本实验）；SH=config stage2_orig（原版参考）。",
        "MLP": {m: mlp[m] for m in METHODS},
        "SH参考": {m: sh[m] for m in METHODS},
        "Δ(MLP−SH)": {m: {k: round(mlp[m][k] - sh[m][k], 4) for k in KEYS} for m in METHODS},
    }
    jf = out / "对比柱状图_阶段二_MLP-vs-SH.json"
    json.dump(payload, open(jf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"  已生成: {png}\n          {jf}")
    for m in METHODS:
        d = payload["Δ(MLP−SH)"][m]
        print(f"  [{m}]  ΔPSNR {d['PSNR']:+.3f}  ΔSSIM {d['SSIM']:+.4f}  ΔLPIPS {d['LPIPS']:+.4f}")


if __name__ == "__main__":
    main()
