# -*- coding: utf-8 -*-
"""
make_bars_long.py —— 阶段二「长训」颜色头三方对比柱状图（每方法一张，PNG 本地/服务器同码）
--------------------------------------------------------------------------------
回答：把 V14（MLP + PE 全开）推到调度终点 t=1.0 之后，它对 SH 与「加 PE 之前的 MLP」
      到底领先多少？——即长训结果的三方对照：

    ① SH@A*          = 原版逐点 SH 颜色头 + A* exp LR（受控 SH 对照）
    ② MLP@A* 无PE    = 逐点 embedding + 共享 MLP 颜色头，无任何 PE（PE 之前的 MLP）
    ③ V14 长训        = MLP 头 + PE(xyzL5 + dirL3) + 距离条件（PE 全开），长训到 t=1.0

**口径（重要，图内已标注）**：两个 x 组各自内部自洽，不跨组直接比数值——
    组 A「RRF 1万步 = abs40000（t=0.25）」：三条都是 **48 子集**（唯一三条齐全的口径）
    组 B「RRF 4万步 = abs70000（t=1.0）」：SH@A* 与 V14 都是 **640 全量**；
                                          MLP@A* 无PE **只跑到 t=0.25，此组为空位**
（48 子集与 640 的差是抽样噪声、非系统性偏移，实测 ±0.5 dB 以内；但为避免误读，两组不混。）

数据源（<root> = run_mlp_v2_阶段1b阶段2与长训；48 取 curve.json，640 取 results.json）：
    SH@A*       <root>/阶段1b长训_SH头/<M>/{curve,results}.json
    MLP@A*无PE  <root>/阶段1b_MLP头/<M>/{curve,results}.json
    V14 长训    <root>/阶段2长训_位置编码_X5D3R1/<M>/{curve,results}.json

用法（服务器 gs3d env 具备 matplotlib + 中文字体；本地 mvsplat env 亦可）：
    <PY> tools/make_bars_long.py --root /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训 \
         --out  /data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训/阶段2长训_柱状图 \
         --methods "MPC CBF Delay MVDR" --base 30000

输出（默认 <root>/阶段2长训_柱状图/）：
    <M>/三指标_长训对比_<M>.png    1×3 面板（PSNR / SSIM / LPIPS）分组柱状图
    <M>/数据.json                  该方法的全部数值 payload（可入文档）
    说明.md                        组级说明（改了什么 + 口径 + 数值表），只此一份
"""
import argparse
import json
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

METRICS = [  # (key, 面板标题, 方向)
    ("PSNR", "PSNR (dB)  ↑", 1),
    ("SSIM", "SSIM  ↑", 1),
    ("LPIPS", "LPIPS (VGG)  ↓", -1),
]
FMT = {"PSNR": ".3f", "SSIM": ".4f", "LPIPS": ".4f"}

# 横轴短标签（完整名字交给图例，避免旋转长标签互相挤压）
SHORT = {"shA": "SH@A*", "mlpA": "无PE MLP", "v14": "V14 长训"}

# 三条序列：key → (图例名, 颜色, 说明, 具体参数)
SERIES = [
    ("shA", "SH@A*（原版SH + A* LR）", "#1f77b4",
     "颜色表达 = 原版逐点 SH 不变；唯一改动 = LR 协议换成 A*（受控 SH 对照）",
     "train_rrf_sh.py；exp 8e-3 → 8e-5，lr_max_steps 40000"),
    ("mlpA", "MLP@A* 无PE（PE 之前的 MLP）", "#ff7f0e",
     "颜色表达换成 embedding + 共享 MLP 头，且用 A* LR；无任何位置编码（各 PE 臂的对照基线）",
     "train_rrf_mlp.py；embed 32 + MLP(h128,d2)；exp 8e-3 → 8e-5；无 --pe_* 开关"),
    ("v14", "V14 长训（PE 全开）", "#2ca02c",
     "在 MLP@A* 无PE 之上加 PE(xyzL5 + dirL3) + 距离条件，并延长训练到调度终点 t=1.0",
     "train_rrf_mlp.py；--pe_xyz_L 5 --pe_dir_L 3 --use_dist；exp 8e-3 → 8e-5，lr_max_steps 40000"),
]

# 两个 x 组：(组标题, abs 迭代, 口径标签, {series_key: 取数方式})
#   取数方式 "48" = curve.json ours_<it>；"640" = results.json ours_<it>；None = 该组无此臂
GROUPS = [
    ("RRF 1万步", 40000, "48 子集", {"shA": "48", "mlpA": "48", "v14": "48"}),
    ("RRF 4万步", 70000, "640 全量", {"shA": "640", "mlpA": None, "v14": "640"}),
]

SRC = {
    "shA": "阶段1b长训_SH头",
    "mlpA": "阶段1b_MLP头",
    "v14": "阶段2长训_位置编码_X5D3R1",
}


def _load(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.is_file() else None


def pick(root, key, method, it, mode):
    """按口径取三指标；缺 → None。"""
    d = root / SRC[key] / method
    if mode == "640":
        v = (_load(d / "results.json") or {}).get(f"ours_{it}")
    else:
        v = (_load(d / "curve.json") or {}).get(f"ours_{it}")
    if not v:
        return None
    return {k: v.get(k) for k, _t, _f in METRICS}


def collect(root, method, base):
    """→ {series_key: {"40k":[v,v,v], "70k":[...]}}，值为三指标 dict 或 None。"""
    out = {}
    for key, _n, _c, _d, _p in SERIES:
        out[key] = {}
        for _t, it, _s, modes in GROUPS:
            mode = modes.get(key)
            out[key][it] = pick(root, key, method, it, mode) if mode else None
    return out


def draw(method, data, out_png, no_font):
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

    n_ser = len(SERIES)
    w = 0.78
    # 组内条位置；组间留 1.4 个条宽的空隙
    centres, slots = [], []
    for gi, (_t, _it, _s, _m) in enumerate(GROUPS):
        c0 = gi * (n_ser + 1.4)
        pos = [c0 + i for i in range(n_ser)]
        slots.append(pos)
        centres.append(c0 + (n_ser - 1) / 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0))
    for ax, (key, title, sign) in zip(axes, METRICS):
        # 先收集全部数值，好定 y 轴上限（空位才画得出合适高度的虚线占位框）
        grid = [[(None if data[sk].get(it) is None else data[sk].get(it)[key])
                 for sk, _n, _c, _d, _p in SERIES]
                for _t, it, _s, _m in GROUPS]
        allv = [v for row in grid for v in row if v is not None]
        hi = max(allv) if allv else 1.0
        ax.set_ylim(0, hi * 1.30)

        for gi in range(len(GROUPS)):
            for i, (sk, name, col, _d, _p) in enumerate(SERIES):
                x = slots[gi][i]
                val = grid[gi][i]
                if val is None:
                    # 该组此臂无数据：画虚线空心占位框（不是 0 值柱），框内写明原因
                    ax.bar(x, hi * 0.52, w, facecolor="#f5f5f5",
                           edgecolor="#bbbbbb", linestyle="--", linewidth=1.0)
                    ax.annotate("未跑\n(t=0.25 止)", (x, hi * 0.26), ha="center",
                                va="center", fontsize=7, color="#888888",
                                linespacing=1.5)
                    continue
                b = ax.bar(x, val, w, color=col)
                ax.annotate(f"{val:{FMT[key]}}",
                            (b[0].get_x() + b[0].get_width() / 2, val),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", va="bottom", fontsize=7)

        # 组标题 + 组内 Δ
        for gi, (_t, it, sub, _m) in enumerate(GROUPS):
            gx = centres[gi]
            ax.annotate(f"{_t}\nabs{it}（t={0.25 if it == 40000 else 1.0}）\n{sub}",
                        (gx, 0), textcoords="offset points", xytext=(0, -46),
                        ha="center", va="top", fontsize=8.2, color="#333333")
            v14, shA = data["v14"].get(it), data["shA"].get(it)
            if v14 and shA:
                dv = v14[key] - shA[key]
                good = (dv > 0) if sign > 0 else (dv < 0)
                ax.annotate(f"Δ(V14−SH@A*) {dv:+.3f}", (gx, hi),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                            color="#1a9850" if good else "#c0392b")
            mlpA = data["mlpA"].get(it)
            if v14 and mlpA:
                dv = v14[key] - mlpA[key]
                good = (dv > 0) if sign > 0 else (dv < 0)
                ax.annotate(f"Δ(V14−无PE) {dv:+.3f}", (gx, hi),
                            textcoords="offset points", xytext=(0, 26),
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                            color="#1a9850" if good else "#c0392b")

        ax.set_xticks(slots[0] + slots[1])
        ax.set_xticklabels([SHORT[sk] for _g in GROUPS
                            for sk, _n, _c, _d, _p in SERIES], fontsize=8.4)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.3)
        ax.margins(x=.08)
    fig.subplots_adjust(bottom=.30)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _k, _n, c, _d, _p in SERIES]
    fig.legend(handles, [n for _k, n, _c, _d, _p in SERIES],
               loc="lower center", ncol=3, fontsize=8.6, framealpha=.9,
               bbox_to_anchor=(.5, .02))
    fig.text(.5, -.035,
             "两组口径各自内部自洽，勿跨组直接比数值；空位 = MLP@A* 无PE 只跑到 abs40000"
             "（Phase1b 判据终点），该组无数据、不做外插。",
             ha="center", fontsize=8, color="#555555")
    fig.suptitle(f"[{method}] 阶段二长训：颜色头三方对比（同起点 orig_seed 视觉 30000，"
                 f"冻结几何 / opacity 可训 / densify 0）", fontsize=12, y=1.03)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_readme(top, methods, base):
    L = []
    L.append("# 阶段二长训 —— 颜色头三方对比柱状图（PSNR / SSIM / LPIPS）")
    L.append("")
    L.append("> 本目录每个方法文件夹（" + " / ".join(methods) + "）内是同一组实验在一个频谱场景的"
             "**三指标_长训对比_<方法>.png**（1×3 面板分组柱状图）+ **数据.json**；本说明对各方法通用。")
    L.append("")
    L.append("## 三条序列（改了什么 + 具体参数）")
    L.append("")
    L.append("| 图例 | 改了什么 | 具体参数 |")
    L.append("|---|---|---|")
    for _k, name, _c, desc, param in SERIES:
        L.append(f"| {name} | {desc} | {param} |")
    L.append("")
    L.append("## 两个 x 组的口径（重要）")
    L.append("")
    L.append("图内两组**各自内部自洽**，请在同一组内读差值，不要跨组直接比数值：")
    L.append("")
    L.append("| x 组 | 绝对迭代 | 口径 | 有哪些臂 |")
    L.append("|---|---|---|---|")
    for _t, it, sub, modes in GROUPS:
        have = " / ".join(n.split("（")[0] for k, n, _c, _d, _p in SERIES if modes.get(k))
        miss = [n.split("（")[0] for k, n, _c, _d, _p in SERIES if not modes.get(k)]
        note = have + ("（" + "、".join(miss) + " 此组为空位）" if miss else "")
        L.append(f"| {_t} | abs{it} | {sub} | {note} |")
    L.append("")
    L.append("说明：**MLP@A* 无PE 只跑到 abs40000**（Phase1b 的判据终点），因此 RRF 4 万步组"
             "只有 SH@A* 与 V14 两条——图中该位置画成灰斜线并标「未跑」，不做任何外插。")
    L.append("48 子集与 640 全量的差是抽样噪声、非系统性偏移（实测 ±0.5 dB 内），"
             "但两组仍分开口径以免误读；V14 / SH@A* 在 abs70000 用 640 全量，"
             "与 docs §7.3 的判据表同源。")
    L.append("")
    L.append("## 读法（结论）")
    L.append("")
    L.append("- **MPC / CBF / Delay**：abs40000 组 V14 已同时高于 SH@A* 与无PE MLP；"
             "推到 abs70000（t=1.0）后对 SH@A* 的领先反而**扩大**——优势不是收敛速度差。")
    L.append("- **MVDR 是反例**：PE(xyz5) 对该场景是毒药，V14 在同组内垫底（低于无PE MLP）；"
             "SH@A* 反而领先。属于**表示问题**而非收敛问题（见 docs §7.3）。")
    L.append("- 附带：A* LR 对 SH 本身也有效（对比 docs 里的 SH 原版固定线），"
             "所以「SH@A*」已是加强过的 SH 对照。")
    L.append("")
    L.append("## 全源共同参数")
    L.append("")
    for c in [
        "数据：RF-3DGS 频谱训练集 3dgs_<场景>_100（本目录各方法文件夹 = 一个场景）；评测 640 张 held-out（独立相机位姿）",
        "起点：阶段一视觉 chkpnt30000（orig_seed_chkpnt30000.pth）",
        f"RRF：chkpnt30000 → 绝对迭代 40000（RRF 1 万步，t=0.25）或 70000（RRF 4 万步，t=1.0）；"
        "冻结 xyz/scaling/rotation，opacity 可训，densify_until_iter 0，loss = L1 + 0.2·(1−SSIM)",
        "A* LR 协议（凡 @A* 源）：exp 8e-3 → 8e-5，lr_max_steps 40000（RRF 本地步）⇒ abs70000 正好落在调度终点 t=1.0",
        "MLP 颜色头：embedding 32 维 + 共享 MLP（hidden 128, depth 2）→ sigmoid 出 [0,1] RGB",
        f"绝对迭代 = RRF 本地步 + {base}（阶段一视觉 30000 步）",
    ]:
        L.append(f"- {c}")
    L.append("")
    (top / "说明.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="阶段二长训颜色头三方对比柱状图")
    ap.add_argument("--root", required=True, help="run_mlp_v2_阶段1b阶段2与长训 根")
    ap.add_argument("--methods", default="MPC CBF Delay MVDR")
    ap.add_argument("--base", type=int, default=30000)
    ap.add_argument("--out", default=None, help="输出目录（默认 <root>/阶段2长训_柱状图）")
    ap.add_argument("--no-font", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    top = Path(args.out) if args.out else root / "阶段2长训_柱状图"
    top.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split()

    for m in methods:
        data = collect(root, m, args.base)
        mdir = top / m
        mdir.mkdir(parents=True, exist_ok=True)
        png = mdir / f"三指标_长训对比_{m}.png"
        draw(m, data, png, args.no_font)

        payload = {
            "方法": m,
            "协议": "阶段二 RRF：orig_seed 视觉 30000 起步；组A=abs40000(48子集)，组B=abs70000(640全量)。",
            "序列": {n: {"说明": d, "参数": p} for _k, n, _c, d, p in SERIES},
            "组": [{"名称": t, "绝对迭代": it, "口径": s,
                    "有": {n: data[k].get(it) for k, n, _c, _d, _p in SERIES}}
                   for t, it, s, _mo in GROUPS],
        }
        json.dump(payload, open(mdir / "数据.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"== {m}: 已生成 {png}")
        for t, it, s, _mo in GROUPS:
            print(f"   [{t} abs{it} · {s}]")
            for k, n, _c, _d, _p in SERIES:
                v = data[k].get(it)
                if v is None:
                    print(f"     {n:28s} 未跑")
                    continue
                print(f"     {n:28s} PSNR {v['PSNR']:.3f}  SSIM {v['SSIM']:.4f}  "
                      f"LPIPS {v['LPIPS']:.4f}")
            v14, shA = data["v14"].get(it), data["shA"].get(it)
            if v14 and shA:
                print("     Δ(V14−SH@A*)  PSNR %+.3f  SSIM %+.4f  LPIPS %+.4f"
                      % (v14["PSNR"] - shA["PSNR"], v14["SSIM"] - shA["SSIM"],
                         v14["LPIPS"] - shA["LPIPS"]))
            mlpA = data["mlpA"].get(it)
            if v14 and mlpA:
                print("     Δ(V14−无PE)   PSNR %+.3f  SSIM %+.4f  LPIPS %+.4f"
                      % (v14["PSNR"] - mlpA["PSNR"], v14["SSIM"] - mlpA["SSIM"],
                         v14["LPIPS"] - mlpA["LPIPS"]))
        # 报错式校验：三条齐全的组里 V14 不应缺
        for t, it, _s, mo in GROUPS:
            if mo.get("v14") and data["v14"].get(it) is None:
                raise RuntimeError(f"[{m}] {t} abs{it} 缺 V14 数据（路径或档位不对）")

    write_readme(top, methods, args.base)
    print(f"\n说明.md 已写至: {top / '说明.md'}\n输出根: {top}")


if __name__ == "__main__":
    main()
