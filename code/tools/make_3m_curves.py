# -*- coding: utf-8 -*-
"""
make_3m_curves.py —— 每方法一个文件夹，内含 PSNR / SSIM / LPIPS 三指标收敛曲线
--------------------------------------------------------------------------------
统一为 phase1b 与 phase2 两组画"三指标曲线"，输出结构：
    <out>/<M>/PSNR曲线.png  SSIM曲线.png  LPIPS曲线.png   （48 子集，多线叠加）
    <out>/<M>/末点三指标.txt                          （640 全量优先，缺则 48 末点）
    <out>/<M>/数据.json
    <out>/说明.md                                    （组级：每源改了什么 + 具体参数，对各方法通用）
组别 source 定义（48 子集 curve.json；640 取 results.json ours_<i>）：
    phase1b：SH@A*(阶段1b长训_SH头) / MLP@A*(阶段1b_MLP头) / SH原版固定(rrf40k_sh) /
             MLP旧固定(rrf40k_mlp)
    phase2 ：SH@A*(阶段1b长训_SH头) / MLP@A* 无PE基线(阶段1b_MLP头) / SH原版固定(rrf40k_sh)
             + 5 个 PE 臂 (阶段2_位置编码_X5D0R0..X5D3R1)（缺臂自动跳过，可中途跑）
指标方向：PSNR/SSIM ↑ 好，LPIPS ↓ 好（图内注明）。
用法（需 matplotlib + 中文字体；--no-font 英文兜底）：
    <PY> tools/make_3m_curves.py --group phase1b --root ... --rrf40k ... \
         --methods "MPC CBF Delay MVDR" --base 30000
    <PY> tools/make_3m_curves.py --group phase2  --root ... --rrf40k ... \
         --methods "MPC CBF Delay MVDR" --base 30000
"""
import argparse
import json
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

ENC_INFO = [  # (目录编码, 图例名, 线型标记, 颜色)
    ("X5D0R0", "V10 PE(xyz)5", "o", "#2ca02c"),
    ("X3D0R0", "V11 PE(xyz)3", "v", "#bcbd22"),
    ("X0D3R0", "V12 PE(dir)3", "^", "#9467bd"),
    ("X5D3R0", "V13 PE(xyz)5+dir3", "D", "#17becf"),
    ("X5D3R1", "V14 PE(xyz)5+dir3+dist", "P", "#e377c2"),
]
METRICS = [  # (key, 中文名, ylabel, 方向)
    ("PSNR", "PSNR", "PSNR (dB)  ↑ 好", 1),
    ("SSIM", "SSIM", "SSIM  ↑ 好", 1),
    ("LPIPS", "LPIPS", "LPIPS (VGG)  ↓ 好", -1),
]
PE_DESC = {  # 各 PE 臂的唯一改动（相对 MLP@A* 无PE 基线）= 下列 PE 开关
    "X5D0R0": "加 xyz 位置编码：--pe_xyz_L 5（γ 谐波 6L=30 维，xyz 按场景归一化中心/半径）",
    "X3D0R0": "加 xyz 位置编码：--pe_xyz_L 3（18 维，带宽更窄）",
    "X0D3R0": "加 视线方向 位置编码：--pe_dir_L 3（在恒有的 raw dir(3) 上再叠 6L=18 维）",
    "X5D3R0": "同时加 xyz 与 dir 位置编码：--pe_xyz_L 5 --pe_dir_L 3",
    "X5D3R1": "在 X5D3R0 基础上再开距离条件：--pe_xyz_L 5 --pe_dir_L 3 --use_dist（编码全开）",
}
GROUP_README = {
    "phase1b": {
        "标题": "Phase1b —— 颜色表达形式 × LR 协议（2×2 判据）",
        "线约定": "实线 = A* LR 协议（exp）；虚线 = 原固定 LR。同色两条 = 同一颜色表达在两个 LR 协议下。",
        "背景": "把 Phase1 锁定的 A* LR 协议分别装到 SH 原版颜色头（SH@A*）与共享 MLP 颜色头（MLP@A*）上，"
                "叠两条固定-LR 历史基准，判：A* LR 是否为收益主源、同 LR 下 MLP 共享头是否仍优于逐点 SH。"
                "结论（Phase1b）：A* LR 是主收益源且普适（SH 内全 4 方法为正）；MLP 形式假设在同 LR 下 2-2 不成立。",
        "共同参数": [
            "数据：RF-3DGS 频谱训练集 3dgs_<场景>_100（本目录各方法文件夹 = 一个场景）；评测 640 张 held-out（独立相机位姿）",
            "起点：阶段一视觉 chkpnt30000（orig_seed_chkpnt30000.pth）",
            "RRF 重训：chkpnt30000 → 绝对迭代 40000（= RRF 本地 1 万步，画线档 abs 32000/35000/38000/40000）；"
            "冻结 xyz/scaling/rotation，opacity 可训，densify_until_iter 0",
            "MLP 颜色头（凡 MLP 源）：embedding 32 维 + 共享 MLP（hidden 128, depth 2）→ sigmoid 出 [0,1] RGB",
            "A* LR 协议（凡 @A* 源）：颜色 LR 按 exp 8e-3 → 8e-5 退火，lr_max_steps 40000；embed LR 固定 2.5e-3（E0，不传 --embed_lr）",
            "固定 LR（历史基准源）：引擎默认 feature_lr，无 LR 调度",
            "画线口径：48 子集逐档 curve.json；判优口径：640 全量 results.json（* = rrf40k 源无 640，末点取 48 子集）",
        ],
    },
    "phase2": {
        "标题": "Phase2 —— MLP 颜色头 × 位置编码网格（5 臂，LR 锁 A*）",
        "线约定": "实线 = A* LR 协议；虚线 = SH 原版固定（参考）。绿/紫系彩色实线 = 各 PE 臂（V10..V14）。",
        "背景": "在 MLP@A*（无 PE）基线上逐档加位置编码（PE）：xyz-PE、dir-PE、距离条件，共 5 臂（V10..V14），"
                "回答 PE + A* LR 能否让 MLP 在落后方法（MPC/CBF）翻超 SH@A* 并压过 SH 原版。"
                "每臂训练配置与 阶段1b_MLP头（MLP@A* 无PE）完全一致，单变量 = 下表所加 PE 开关。",
        "共同参数": [
            "数据：RF-3DGS 频谱训练集 3dgs_<场景>_100（本目录各方法文件夹 = 一个场景）；评测 640 张 held-out（独立相机位姿）",
            "起点：阶段一视觉 chkpnt30000（orig_seed_chkpnt30000.pth）",
            "RRF 重训：chkpnt30000 → 绝对迭代 40000（= RRF 本地 1 万步，画线档 abs 32000/35000/38000/40000）；"
            "冻结 xyz/scaling/rotation，opacity 可训，densify_until_iter 0",
            "MLP 颜色头（凡 MLP 源）：embedding 32 维 + 共享 MLP（hidden 128, depth 2）→ sigmoid 出 [0,1] RGB",
            "A* LR 协议（凡 @A* 源）：mlp LR exp 8e-3 → 8e-5，lr_max_steps 40000；embed LR 固定 2.5e-3（E0）",
            "画线口径：48 子集逐档 curve.json；判优口径：640 全量 results.json（* = rrf40k 源无 640，末点取 48 子集）",
            "PE 臂 = 新训练（train_rrf_mlp.py + 下表 PE 开关）；三条基准（SH@A* / MLP@A* / SH 原版固定）复用 Phase1b/历史产物，不重跑",
        ],
    },
    "longall": {
        "标题": "长训对照（RRF 到 40000 = abs70000，调度终点 t=1.0）",
        "线约定": "实线 = 长训到 abs70000 的两条线（V14 臂 / SH@A*）；虚线 = SH 原版固定（已有到 abs70000 的历史曲线）。",
        "背景": "Phase2 的判据只跑到 RRF 1 万步 / 调度 t=0.25，V14 臂的曲线尚未压平，因此无法区分"
                "\"V14 的优势是真实的\" 与 \"只是两条线收敛速度不同\"。本组把每个方法延长到 RRF 40000"
                "（abs70000，调度终点 t=1.0）：V14（MLP+PE 全开）与 SH@A*（原版+exp LR）各跑一条，"
                "与已有的 SH 原版固定曲线同轴对比。MVDR 先跑（它趋势与其余方法不一致，V14 塌到 14.28，"
                "判定为表示问题而非收敛问题，见 docs §7.3）；其余方法随后用同一流程验证 t=0.25 的优势"
                "在 t=1.0 是否仍然成立。",
        "共同参数": [
            "数据：RF-3DGS 频谱训练集 3dgs_<场景>_100（本目录各方法文件夹 = 一个场景）；评测 640 张 held-out（独立相机位姿）",
            "起点：阶段一视觉 chkpnt30000（orig_seed_chkpnt30000.pth）",
            "RRF 重训：chkpnt30000 → 绝对迭代 70000（= RRF 本地 40000 步，画线档 abs 32000/35000/38000/40000/"
            "45000/50000/55000/60000/65000/70000）；冻结 xyz/scaling/rotation，opacity 可训，densify_until_iter 0",
            "A* LR 协议：exp 8e-3 → 8e-5，lr_max_steps 40000（RRF 本地步）⇒ 到 abs70000 正好落在调度终点 t=1.0",
            "V14 臂 = MLP 头（embed 32 + MLP(h128,d2)）+ PE(xyzL5 + dirL3) + 距离条件（--use_dist）",
            "SH 原版固定线：run_rrf40k_sh（引擎固定 LR），已有到 abs70000 的 48 子集曲线（无 640 → 末点带 *）",
            "画线口径：48 子集逐档 curve.json；判优口径：640 全量 results.json（仅长训两条线有 640）",
            "驱动：tools/run_long_methods.sh（METHODS=<方法列表>，占一卡串行；各段存在即跳过，可续跑）",
        ],
    },
}
GROUP_README["mvdrlong"] = GROUP_README["longall"]  # 旧名保留，指向同一组

GROUP_OUT = {  # 组键 → 默认输出目录名（磁盘上一律「阶段X_」前缀，与数据目录同一套）
    "phase1b": "阶段1b_三指标",
    "phase2": "阶段2_三指标",
    "longall": "阶段2长训_三指标",
    "mvdrlong": "阶段2长训_三指标",
}
def read_curve(path, iters):
    """返回 {iter: {PSNR, SSIM, LPIPS}}；缺文件 → {}。"""
    if not Path(path).is_file():
        return {}
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for it in iters:
        v = d.get(f"ours_{it}")
        if v:
            out[it] = {"PSNR": v.get("PSNR"), "SSIM": v.get("SSIM"),
                       "LPIPS": v.get("LPIPS")}
    return out


def read_640(src, method, it):
    """results.json ours_<it> 的三指标；缺 → None。"""
    p = Path(src) / method / "results.json"
    if p.is_file():
        v = json.load(open(p, encoding="utf-8")).get(f"ours_{it}")
        if v:
            return {k: v.get(k) for k in ("PSNR", "SSIM", "LPIPS")}
    return None


def source_dirs(root, rrf, group):
    """返回每个绘图源的元数据：名称 / curve 源目录 / 640 源目录 / 线型 / 颜色 / 改动说明(desc) / 具体参数(param)"""
    srcs = []
    def add(name, curve_dir, src640, ls, col, desc, param):
        srcs.append({"name": name, "curve": curve_dir, "six40": src640,
                     "ls": ls, "col": col, "desc": desc, "param": param})
    if group == "phase1b":
        add("SH@A* (exp · 实线)", root / "阶段1b长训_SH头", root / "阶段1b长训_SH头",
            "o-", "#1f77b4",
            "颜色表达 = SH 原版不变；唯一改动 = LR 协议换成 A*（对原引擎固定 LR 的单变量）",
            "train_rrf_sh.py；SH 系数直接优化；颜色 LR = exp 8e-3 → 8e-5，lr_max_steps 40000")
        add("MLP@A* (exp · 实线)", root / "阶段1b_MLP头", root / "阶段1b_MLP头",
            "s-", "#ff7f0e",
            "颜色表达换成逐点 embedding + 共享 MLP 头（vs SH 原版的结构改动），且 LR 用 A*",
            "embed 32 + MLP(h128,d2)；embed_lr 固定 2.5e-3(E0)；mlp LR = exp 8e-3 → 8e-5，lr_max_steps 40000；无 --pe_* 开关")
        add("SH 原版 (固定 · 虚线)", rrf / "run_rrf40k_sh", None,
            "o--", "#1f77b4",
            "历史基准，未改动引擎（SH + 引擎固定 LR）",
            "SH 原版；feature_lr 固定默认，无 LR 调度；run_rrf40k_sh")
        add("MLP 旧 (固定 · 虚线)", rrf / "run_rrf40k_mlp", None,
            "s--", "#ff7f0e",
            "历史基准（MLP 头 + 引擎固定 LR）",
            "embed 32 + MLP(h128,d2)；embed_lr / mlp_lr 均固定默认，无 LR 调度；run_rrf40k_mlp")
    elif group == "phase2":
        add("SH@A* (实线)", root / "阶段1b长训_SH头", root / "阶段1b长训_SH头",
            "o-", "#1f77b4",
            "基准1：SH 原版 + A* LR（Phase1b 产物，复用不重跑）",
            "同 phase1b SH@A*：train_rrf_sh.py，exp 8e-3 → 8e-5，lr_max_steps 40000")
        add("MLP@A* 无PE基线 (实线)", root / "阶段1b_MLP头", root / "阶段1b_MLP头",
            "s-", "#ff7f0e",
            "基准2：MLP 头 + A* LR 但无任何 PE（Phase1b 产物；= 各 PE 臂的对照）",
            "embed 32 + MLP(h128,d2)；A* LR；无 --pe_*；阶段1b_MLP头")
        add("SH 原版固定 (虚线)", rrf / "run_rrf40k_sh", None,
            "o--", "#1f77b4",
            "基准3：SH 原版 + 固定 LR（历史基准，复用不重跑）",
            "feature_lr 固定默认，无 LR 调度；run_rrf40k_sh（无 640 → 末点取 48 并带 *）")
        for enc, lbl, mk, col in ENC_INFO:
            add(f"{lbl}", root / f"阶段2_位置编码_{enc}", root / f"阶段2_位置编码_{enc}",
                mk + "-", col,
                f"在 MLP@A* 无PE 基线之上只加 PE（单变量 = 该臂 PE 开关）",
                PE_DESC[enc] + "；其余同 阶段1b_MLP头")
    else:  # longall / mvdrlong：长训到 RRF40000 对照（同一组源，方法名由 --methods 决定）
        add("SH 原版固定 (虚线)", rrf / "run_rrf40k_sh", None,
            "o--", "#1f77b4",
            "历史基准：SH 原版 + 引擎固定 LR（已有到 abs70000 曲线，复用不重跑）",
            "feature_lr 固定默认；run_rrf40k_sh（无 640 → 末点取 48 并带 *）")
        add("SH@A* 长训 (实线)", root / "阶段1b长训_SH头", root / "阶段1b长训_SH头",
            "o-", "#1f77b4",
            "SH 原版 + A* exp LR，延长训练到 RRF 40000（本组新跑）",
            "train_rrf_sh.py；--iterations 70000 --lr_schedule exp --color_lr_init 8e-3 "
            "--color_lr_final 8e-5 --lr_max_steps 40000")
        add("V14 PE(xyz5+dir3+dist) 长训 (实线)", root / "阶段2长训_位置编码_X5D3R1",
            root / "阶段2长训_位置编码_X5D3R1", "s-", "#2ca02c",
            "MLP 头 + PE(xyzL5+dirL3) + 距离条件，延长训练到 RRF 40000（本组新跑）",
            "train_rrf_mlp.py；--iterations 70000 --lr_schedule exp --mlp_lr_init 8e-3 "
            "--mlp_lr_final 8e-5 --lr_max_steps 40000 --pe_xyz_L 5 --pe_dir_L 3 --use_dist")
    return srcs


def _src_has_data(s, methods):
    """该源在任意一个方法文件夹下已有数据（curve.json 或 results.json）。"""
    for mm in methods:
        if s["curve"] and (s["curve"] / mm / "curve.json").is_file():
            return True
        if s["six40"] and (s["six40"] / mm / "results.json").is_file():
            return True
    return False


def write_readme(top, group, srcs, fin, base, methods):
    """组目录（三指标/）写一份 说明.md：改了什么 + 每源具体参数；对各方法文件夹通用。"""
    info = GROUP_README[group]
    covers = " / ".join(methods.split())
    L = []
    L.append(f"# {info['标题']} —— 三指标（PSNR / SSIM / LPIPS）说明")
    L.append("")
    L.append(f"> 本目录 = 三指标/，其下每个方法文件夹（{covers}）是同一组实验在一个频谱场景的结果："
             f"**PSNR曲线.png / SSIM曲线.png / LPIPS曲线.png**（48 子集逐档多源叠加）+ **末点三指标.txt**"
             f"（640 全量判优，`*`= 该源无 640 取 48 末点）+ **数据.json**。本说明对各方法文件夹通用。")
    L.append(f"> 线约定：{info['线约定']}")
    L.append(f"> 画线口径 = 48 子集；判优口径 = 640 全量。横轴 = RRF 本地千步 = (绝对迭代 − {base})/1000，"
             f"绝对迭代到 {fin}（RRF 本地 {fin - base} 步）")
    L.append("")
    L.append("## 改了什么（背景）")
    L.append(info["背景"])
    L.append("")
    L.append("## 图中各源：改了什么 + 具体参数")
    L.append("")
    L.append("| 图例 | 改了什么 | 具体参数 |")
    L.append("|---|---|---|")
    for s in srcs:
        warn = "" if _src_has_data(s, methods.split()) \
            else "　（⚠ 该源数据未就绪，图中暂未入线）"
        L.append(f"| {s['name']} | {s['desc']} | {s['param']}{warn} |")
    L.append("")
    L.append("## 全源共同参数")
    L.append("")
    for c in info["共同参数"]:
        L.append(f"- {c}")
    L.append("")
    (top / "说明.md").write_text("\n".join(L), encoding="utf-8")


def draw(ax, srcs_data, metric, base):
    for s in srcs_data:
        pts = s["pts"]
        if not pts:
            continue
        xs = [(it - base) / 1000.0 for it in sorted(pts)]
        ys = [pts[it][metric] for it in sorted(pts)]
        if all(y is not None for y in ys):
            ax.plot(xs, ys, s["ls"], color=s["col"], lw=2.0, ms=6,
                    label=s["name"])
    ax.set_xlabel(f"RRF 训练步数（千） [绝对迭代 = RRF + {base}]")
    ax.grid(alpha=.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    choices=["phase1b", "phase2", "longall", "mvdrlong"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--rrf40k", required=True)
    ap.add_argument("--methods", default="MPC")
    ap.add_argument("--iters", default="32000 35000 38000 40000")
    ap.add_argument("--base", type=int, default=30000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-font", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not args.no_font:
        try:
            import _cnfont
            _cnfont.configure(plt)
        except Exception:
            pass
    plt.rcParams["figure.dpi"] = 140

    root = Path(args.root)
    rrf = Path(args.rrf40k)
    top = Path(args.out) if args.out else root / GROUP_OUT[args.group]
    iters = [int(s) for s in args.iters.split()]
    fin = iters[-1]

    for m in args.methods.split():
        mdir = top / m
        mdir.mkdir(parents=True, exist_ok=True)
        srcs = source_dirs(root, rrf, args.group)
        data = {"group": args.group, "方法": m, "base": args.base,
                "源": []}
        for s in srcs:
            s["pts"] = (read_curve(s["curve"] / m / "curve.json", iters)
                        if s["curve"] else {})
            v640 = read_640(s["six40"], m, fin) if s["six40"] else None
            s["v640"] = v640
            data["源"].append({"名称": s["name"], "曲线档": {str(k): v for k, v in
                                 s["pts"].items()},
                               "640末点": v640})
        # —— 三张单指标图（每方法文件夹）——
        for key, zh, ylab, _dir in METRICS:
            fig, ax = plt.subplots(figsize=(8.6, 5.4))
            draw(ax, srcs, key, args.base)
            ax.set_ylabel(ylab)
            ax.set_title(f"[{m}] {zh} 收敛（{args.group}）", fontsize=12)
            if ax.get_legend_handles_labels()[0]:
                leg = ax.legend(fontsize=8, loc="best", framealpha=.9,
                                handlelength=2.8, handleheight=1.4)
                for ln in leg.get_lines():
                    ln.set_linewidth(2.6)
            fig.tight_layout()
            fig.savefig(mdir / f"{zh}曲线.png", bbox_inches="tight")
            plt.close(fig)
        # —— 末点三指标汇总（640 优先，缺则 48 曲线末点）——
        lines = [f"[{m}] 末点 abs{fin} 三指标（640 全量，* = 48 子集末点）"]
        for s in srcs:
            v = s["v640"] or (s["pts"].get(fin) if s["pts"] else None)
            if not v:
                continue
            star = "" if s["v640"] else " *"
            lines.append(
                f"  {s['name']:26s} PSNR {v.get('PSNR'):.3f}  "
                f"SSIM {v.get('SSIM'):.4f}  LPIPS {v.get('LPIPS'):.4f}{star}"
                if v.get("PSNR") is not None else f"  {s['name']}: n/a")
        (mdir / "末点三指标.txt").write_text("\n".join(lines),
                                            encoding="utf-8")
        json.dump(data, open(mdir / "数据.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("== %s/%s: 已生成 %s/{PSNR,SSIM,LPIPS}曲线.png + 末点三指标.txt + 数据.json" % (args.group, m, mdir))
        for ln in lines:
            print(ln)
    write_readme(top, args.group, source_dirs(root, rrf, args.group),
                 fin, args.base, args.methods)
    print("说明.md 已写至: %s" % (top / "说明.md"))
    print(f"输出根: {top}")


if __name__ == "__main__":
    main()
