# -*- coding: utf-8 -*-
"""
make_curve_phase1b.py —— Phase1b 判据图：同 LR 协议(A*)下 SH 原版 vs MLP，叠固定-LR 参考
--------------------------------------------------------------------------------
回答 Phase1 A2E0 收益来源（换颜色表达 vs 纯换 LR 协议）。每方法画 4 条 PSNR 收敛线：
    SH@A*     = <root>/阶段1b长训_SH头/<M>/curve.json      （本实验：SH 颜色 + exp 8e-3->8e-5）
    MLP@A*    = <root>/阶段1b_MLP头/<M>/curve.json     （本实验：MLP 颜色 + exp 8e-3->8e-5）
    SH@fixed  = <rrf40k>/run_rrf40k_sh/<M>/curve.json （原版 SH + 引擎固定 feature_lr）
    MLP@fixed = <rrf40k>/run_rrf40k_mlp/<M>/curve.json
档位取 4 线共有的绝对迭代（默认 32000 35000 38000 40000，48 子集同口径）。
判据（x = RRF 本地步数 = 绝对迭代 - base）：
    S_exp vs M_exp 末点  = 同 LR 下颜色形式的净收益（形式假设）
    S_exp vs S_fixed     = exp LR 在 SH 内的收益；M_exp vs M_fixed = 在 MLP 内的收益
    S_exp vs M_fixed     = 交叉：MLP 若想超过原版 SH，须在"改了 LR"的基础上再说
用法（任一机器，需 matplotlib；_cnfont 处理中文，缺字体可 --no-font 改英文）：
    <PY> tools/make_curve_phase1b.py --root /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训 \
         --rrf40k /data/yuexf/3dgs/results --methods "MPC CBF Delay MVDR" --base 30000
输出（默认 <root>/阶段1b_曲线/）：判据曲线_<M>.png + json（4 侧数据与末点 Δ）
"""
import argparse
import json
import os
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def load_curve(root, method):
    p = Path(root) / method / "curve.json"
    if not p.is_file():
        raise FileNotFoundError(f"缺 curve.json: {p}")
    return json.load(open(p, encoding="utf-8"))


def collect(root, method, iters, name):
    d = load_curve(root, method)
    rows = []
    for it in iters:
        key = f"ours_{it}"
        v = d.get(key)
        if v and "PSNR" in v:
            rows.append({"绝对迭代": it, "PSNR": float(v["PSNR"]),
                         "SSIM": float(v["SSIM"]), "LPIPS": float(v["LPIPS"])})
    return {"名称": name, "rows": rows}


def draw(curves, out_png, base, no_font=False):
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
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # 线型规则：实线 = A* LR 协议(exp)；虚线 = 原固定 LR。同色不同形即 SH/MLP 各自两版。
    # 图例条目显式标注 实线/虚线，避免只看线样误读。
    styles = {
        "SH@A* (exp · 实线)":  ("o-", "#1f77b4"),
        "MLP@A* (exp · 实线)": ("s-", "#ff7f0e"),
        "SH 原版 (固定 · 虚线)": ("o--", "#1f77b4"),
        "MLP 旧 (固定 · 虚线)":  ("s--", "#ff7f0e"),
    }
    for c in curves:
        st, col = styles[c["名称"]]
        xs = [(r["绝对迭代"] - base) / 1000.0 for r in c["rows"]]
        ys = [r["PSNR"] for r in c["rows"]]
        ax.plot(xs, ys, st, color=col, lw=2, ms=7, label=c["名称"])
    ax.set_xlabel(f"RRF 训练步数（千） [绝对迭代 = RRF + {base}]")
    ax.set_ylabel("PSNR (dB) ↑   [48 子集，同口径]")
    ax.set_title("Phase1b 判据：颜色形式 × LR 协议（2×2）\n"
                 "实线 = A* exp LR；虚线 = 原固定 LR（同色两线 = 同形式的两个 LR 协议）",
                 fontsize=12)
    ax.grid(alpha=.3)
    leg = ax.legend(fontsize=9, loc="best", framealpha=.9,
                    handlelength=3.2, handleheight=1.6, borderpad=0.7)
    for ln in leg.get_lines():            # 加粗拉长图例线样，虚线段更明显
        ln.set_linewidth(3.0)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Phase1b 结果根目录（含 阶段1b长训_SH头 与 阶段1b_MLP头）")
    ap.add_argument("--rrf40k", required=True,
                    help="rrf40k 结果根目录（含 run_rrf40k_sh 与 run_rrf40k_mlp）")
    ap.add_argument("--methods", default="MPC", help="方法名（空格分隔）")
    ap.add_argument("--iters", default="32000 35000 38000 40000",
                    help="4 线共取的绝对迭代（空格分隔）")
    ap.add_argument("--base", type=int, default=30000, help="视觉基准步数（RRF=0 处）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <root>/阶段1b_曲线）")
    ap.add_argument("--no-font", action="store_true", help="不尝试中文字体（缺字体时用）")
    args = ap.parse_args()

    root = Path(args.root)
    rrf = Path(args.rrf40k)
    out = Path(args.out) if args.out else root / "阶段1b_曲线"
    out.mkdir(parents=True, exist_ok=True)
    iters = [int(s) for s in args.iters.split()]

    for m in args.methods.split():
        specs = [
            (root / f"阶段1b长训_SH头",  "SH@A* (exp · 实线)"),
            (root / f"阶段1b_MLP头", "MLP@A* (exp · 实线)"),
            (rrf / "run_rrf40k_sh",  "SH 原版 (固定 · 虚线)"),
            (rrf / "run_rrf40k_mlp", "MLP 旧 (固定 · 虚线)"),
        ]
        curves, warns = [], []
        for d, name in specs:
            try:
                c = collect(d, m, iters, name)
                if not c["rows"]:
                    warns.append(f"{name}: 无可用档位")
                curves.append(c)
            except FileNotFoundError as e:
                warns.append(str(e))
                curves.append({"名称": name, "rows": []})
        for c in curves:
            print(f"  [{m}] {c['名称']}: " + " | ".join(
                f"R{(row['绝对迭代']-args.base)//1000}k(abs{row['绝对迭代']}): {row['PSNR']:.3f}"
                for row in c["rows"]))
        draw(curves, out / f"判据曲线_{m}.png", args.base, args.no_font)
        payload = {
            "协议": "RF-3DGS 频谱 3dgs_{M}_100，48 子集同口径；同 RRF 梯度档位。"
                    f"A*=exp 8e-3->8e-5 max_steps 40000；固定=引擎默认 LR。",
            "base": args.base,
            "曲线": {c["名称"]: c["rows"] for c in curves},
        }
        jf = out / f"判据曲线_{m}.json"
        json.dump(payload, open(jf, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        for w in warns:
            print(f"  warn: {w}")

        # 判据（用非空侧）
        have = {c["名称"]: c["rows"] for c in curves if c["rows"]}
        def last(n):
            r = have.get(n)
            return r[-1]["PSNR"] if r else None
        s_exp, m_exp = last("SH@A* (exp)"), last("MLP@A* (exp)")
        s_fix, m_fix = last("SH 原版 (固定)"), last("MLP 旧 (固定)")
        print(f"\n[判据-{m}@RRF{iters[-1]-args.base}步]")
        if s_exp is not None and m_exp is not None:
            print(f"  同 LR(A*)下：SH@A* − MLP@A* = {s_exp-m_exp:+.3f} dB  "
                  f"→ {'SH 更好 ⇒ 形式假设不成立，纯 LR 是主因' if s_exp>=m_exp else 'MLP 更好 ⇒ 形式收益在匹配 LR 下仍成立'}")
        if s_exp is not None and s_fix is not None:
            print(f"  LR 收益(SH 内)：SH@A* − SH固定 = {s_exp-s_fix:+.3f} dB")
        if m_exp is not None and m_fix is not None:
            print(f"  LR 收益(MLP 内)：MLP@A* − MLP固定 = {m_exp-m_fix:+.3f} dB")
        if s_exp is not None and m_fix is not None:
            print(f"  交叉(MLP 要超过的基准)：SH@A* − MLP固定 = {s_exp-m_fix:+.3f} dB")
        print(f"  已生成: {out / ('判据曲线_'+m+'.png')}")


if __name__ == "__main__":
    main()
