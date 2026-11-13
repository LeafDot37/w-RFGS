# -*- coding: utf-8 -*-
"""
make_curve_phase2.py —— Phase2 判据图：PE 5 臂(A*) vs 基准
--------------------------------------------------------------------------------
回答：在 MLP 颜色头上加位置编码(PE)+A* LR，能否让 MLP 超过
    SH@A*（原版SH + A*LR，阶段1b长训_SH头）与 SH 原版(固定)？
每方法画：SH@A*(实蓝) / MLP@A* 无PE基线(实橙) / SH 原版固定(虚蓝) 为底，
    + 5 条 PE 臂曲线（V10..V14）。末点另用 640 全量 results.json 判优
    （与 A* 选择同口径）；曲线用 48 子集 curve.json（同 phase1b 判据图）。
数据源（<root>=run_mlp_v2_阶段1b阶段2与长训）：
    SH@A*   <root>/阶段1b长训_SH头/<M>/{curve,results}.json
    MLP@A*  <root>/阶段1b_MLP头/<M>/{curve,results}.json     （无PE基线）
    SH 原版 <rrf40k>/run_rrf40k_sh/<M>/curve.json           （无 640，取曲线末点）
    PE 臂   <root>/阶段2_位置编码_<ENC>/<M>/{curve,results}.json
用法（需 matplotlib + 中文字体，服务器 gs3d env 已具备；--no-font 英文兜底）：
    <PY> tools/make_curve_phase2.py --root /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训 \
         --rrf40k /data/yuexf/3dgs/results --methods "MPC CBF Delay MVDR" --base 30000
输出（默认 <root>/阶段2_曲线/）：曲线图_<M>.png + 640柱状_<M>.png + json + 判据打印
"""
import argparse
import json
import os
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

ENC_INFO = [  # (目录编码, 图例名, V码)
    ("X5D0R0", "V10 PE(xyz)5", "X5D0R0"),
    ("X3D0R0", "V11 PE(xyz)3", "X3D0R0"),
    ("X0D3R0", "V12 PE(dir)3", "X0D3R0"),
    ("X5D3R0", "V13 PE(xyz)5+dir3", "X5D3R0"),
    ("X5D3R1", "V14 PE(xyz)5+dir3+dist", "X5D3R1"),
]


def load(src, method, iters):
    """返回 {iter: PSNR}；缺文件 → {}。"""
    out = {}
    p = Path(src) / method / "curve.json"
    if p.is_file():
        d = json.load(open(p, encoding="utf-8"))
        for it in iters:
            v = d.get(f"ours_{it}")
            if v and "PSNR" in v:
                out[it] = float(v["PSNR"])
    return out


def load640(src, method, it):
    p = Path(src) / method / "results.json"
    if p.is_file():
        d = json.load(open(p, encoding="utf-8"))
        v = d.get(f"ours_{it}")
        if v and "PSNR" in v:
            return float(v["PSNR"])
    return None


def draw_curves(name2ps, out_png, base, no_font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not no_font:
        try:
            import _cnfont
            _cnfont.configure(plt)
        except Exception:
            pass
    plt.rcParams["figure.dpi"] = 140
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    arm_colors = ["#2ca02c", "#bcbd22", "#9467bd", "#17becf", "#e377c2"]
    arm_mk = ["o", "v", "^", "D", "P"]
    styles = {
        "SH@A* (实线)": ("o-", "#1f77b4", 2.2),
        "MLP@A* 无PE基线 (实线)": ("s-", "#ff7f0e", 2.2),
        "SH 原版固定 (虚线)": ("o--", "#1f77b4", 2.0),
    }
    plotted = []
    for nm, (st, col, lw) in styles.items():
        ps = name2ps.get(nm)
        if not ps:
            continue
        xs = [(it - base) / 1000.0 for it in sorted(ps)]
        ys = [ps[it] for it in sorted(ps)]
        ax.plot(xs, ys, st, color=col, lw=lw, ms=7, label=nm)
        plotted.append(nm)
    for i, (enc, lbl, _v) in enumerate(ENC_INFO):
        ps = name2ps.get(enc)
        if not ps:
            continue
        xs = [(it - base) / 1000.0 for it in sorted(ps)]
        ys = [ps[it] for it in sorted(ps)]
        ax.plot(xs, ys, arm_mk[i % len(arm_mk)] + "-", color=arm_colors[i],
                lw=2.0, ms=6, label=lbl)
        plotted.append(enc)
    ax.set_xlabel(f"RRF 训练步数（千） [绝对迭代 = RRF + {base}]")
    ax.set_ylabel("PSNR (dB) ↑   [48 子集，同口径]")
    ax.set_title("Phase2 判据：MLP + PE × A* LR（虚线 = SH 固定基准）", fontsize=12)
    ax.grid(alpha=.3)
    leg = ax.legend(fontsize=8.5, loc="best", framealpha=.9,
                    handlelength=2.8, handleheight=1.4)
    for ln in leg.get_lines():
        ln.set_linewidth(2.6)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_bars(names, vals, out_png, no_font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not no_font:
        try:
            import _cnfont
            _cnfont.configure(plt)
        except Exception:
            pass
    plt.rcParams["figure.dpi"] = 140
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4",
              "#2ca02c", "#bcbd22", "#9467bd", "#17becf", "#e377c2"]
    names = names[:len(vals)]
    bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)], width=.62)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("PSNR (dB) ↑   [640 全量 @abs40000]")
    ax.set_title("Phase2 末点 640 全量对比（谁超过 SH@A* = 蓝第1柱）", fontsize=12)
    ax.grid(axis="y", alpha=.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .05, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="run_mlp_v2_阶段1b阶段2与长训 根（含 阶段1b_* 与 阶段2_*）")
    ap.add_argument("--rrf40k", required=True, help="rrf40k 根（含 run_rrf40k_sh）")
    ap.add_argument("--methods", default="MPC", help="方法名（空格分隔）")
    ap.add_argument("--iters", default="32000 35000 38000 40000")
    ap.add_argument("--base", type=int, default=30000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-font", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    rrf = Path(args.rrf40k)
    out = Path(args.out) if args.out else root / "阶段2_曲线"
    out.mkdir(parents=True, exist_ok=True)
    iters = [int(s) for s in args.iters.split()]

    for m in args.methods.split():
        # —— 48 子集曲线（PE 臂全、基准有 curve 即画）——
        name2ps = {}
        name2ps["SH@A* (实线)"] = load(root / "阶段1b长训_SH头", m, iters)
        name2ps["MLP@A* 无PE基线 (实线)"] = load(root / "阶段1b_MLP头", m, iters)
        name2ps["SH 原版固定 (虚线)"] = load(rrf / "run_rrf40k_sh", m, iters)
        for enc, _lbl, _v in ENC_INFO:
            name2ps[enc] = load(root / f"阶段2_位置编码_{enc}", m, iters)
        draw_curves(name2ps, out / f"曲线图_{m}.png", args.base, args.no_font)

        # —— 640 全量末点（判优口径；SH固定 rrf40k 无 results.json → 用曲线末点并注明）——
        fin = iters[-1]
        cats = []
        get640 = lambda src: load640(src, m, fin)
        shA = get640(root / "阶段1b长训_SH头")
        mlpA = get640(root / "阶段1b_MLP头")
        shFix640 = get640(rrf / "run_rrf40k_sh")
        shFix48 = load(rrf / "run_rrf40k_sh", m, [fin]).get(fin)
        ref_sh_fix = shFix640 if shFix640 is not None else shFix48
        names = ["SH@A*", "MLP@A*无PE", "SH原版"]
        vals = [shA, mlpA, ref_sh_fix]
        if shFix640 is None:
            names[-1] = "SH原版(48)"
        arm640, arm48 = {}, {}
        for enc, lbl, _v in ENC_INFO:
            a640 = get640(root / f"阶段2_位置编码_{enc}")
            a48 = load(root / f"阶段2_位置编码_{enc}", m, [fin]).get(fin)
            arm640[enc] = a640
            arm48[enc] = a48
            if a640 is not None:
                names.append(lbl)
                vals.append(a640)
            elif a48 is not None:
                names.append(lbl + "(48)")
                vals.append(a48)
            else:
                print(f"  warn[{m}] {enc}: 无曲线/无640")

        draw_bars(names, [v for v in vals if v is not None], out / f"640柱状_{m}.png",
                  args.no_font)
        json.dump({"base": args.base, "末点640": dict(zip(names, vals)),
                   "48子集": {k: v for k, v in name2ps.items() if v}},
                  open(out / f"判据_{m}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        # —— 判据打印 ——
        print(f"\n[{m}] 640全量 末点 PSNR@abs{fin}：")
        for nm, v in zip(names, vals):
            print(f"    {nm:22s} {v:.3f}" if v is not None else f"    {nm}: n/a")
        best = max(((v, nm) for nm, v in zip(names, vals) if v is not None))
        print(f"  最优: {best[1]} = {best[0]:.3f}")
        if shA:
            for enc, lbl, _v in ENC_INFO:
                a640 = arm640.get(enc)
                base64 = a640 if a640 is not None else arm48.get(enc)
                if a640 is not None:
                    print(f"    {lbl:24s} vs SH@A* {a640-shA:+.3f}  "
                          f"vs 无PE基线 {a640-mlpA:+.3f}  (640)")
                elif base64 is not None and mlpA:
                    print(f"    {lbl:24s} vs 无PE基线 {base64-mlpA:+.3f}  (48, 无640)")


if __name__ == "__main__":
    main()
