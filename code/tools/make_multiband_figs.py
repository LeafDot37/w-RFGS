# -*- coding: utf-8 -*-
"""
make_multiband_figs.py —— 形态 A×（多频段标量分支）三件套出图（折线 / 柱状 / 原图对照）
--------------------------------------------------------------------------------
回答：在标量域把单带换成 **两个带 + 乘法调制**（ū_0·(1+λ(ū_1−init_u))，λ=0 精确退化成 A'）
      之后，A' 留下的"CBF 渲染通道 std 37.05 ≪ GT 51.08（过平滑）"这个缺口有没有补上？
      以及 MVDR 的大胜（+5.11 dB）有没有被这次改动破坏？

臂（全部同协议：起点 chkpnt30000，RRF 重训到 abs70000 = 本地 4 万步，A* exp LR 8e-3→8e-5，
     PE 全开 --pe_xyz_L 5 --pe_dir_L 3 --use_dist，冻结几何，评测 640 张 held-out）：
    A×   run_20260912_多频段标量分支         ← 本步（n_bands=2, band_mix=mult）
    A'   run_20260911_形态A标量头查表         ← 上一步（n_bands=1）
    V14  run_mlp_v2_.../阶段2长训_位置编码_X5D3R1
    SH@A* run_mlp_v2_.../阶段1b长训_SH头

三张图（数字同时落 数据.json；组目录各一份 说明.md）：
    阶段2_多频段_三指标/<M>/{PSNR,SSIM,LPIPS}曲线.png    折线（48 子集阶梯；A'/A×/对照臂）
    阶段2_多频段_柱状图/<M>/三指标_多频段对比_<M>.png      柱状（4 臂，均 abs70000 / 640 全量）
    阶段2_多频段_三指标/<M>/多频段对照_<M>.png            原图对照（GT/对照臂/A'/A× × 4 帧）

**对照臂口径**：折线图与原图对照图固定用 **原版 SH 头（SH@A* = 原版逐点 SH 系数 + A* LR）**，
**不随方法漂移**（`--baseline orig`，默认）。理由：本任务的对照应是「同一个任务、原版方法」；
逐方法去挑「更弱的那个」会让对照臂随方法变，跨方法不可读。
旧的「逐方法在 V14 / SH@A* 里取 LPIPS 更低者」保留为 `--baseline best`。
本任务的**主判据是 LPIPS 而非 PSNR**（GT 每像素只有 1 个自由度，PSNR 会被大面积平坦区拉偏）。
按 LPIPS 算的客观最强基线：MVDR → SH@A*（0.3648），**CBF → V14（0.3902，比 SH@A* 的 0.4235 低）**。
⇒ CBF 上「固定用原版」会让对照臂不是客观最强的那条，故对照图注**明确写出**实际最强的臂与它的
LPIPS（`lpips_best()`），并在 数据.json 的 `baseline_pick.note` 里留痕 —— **不隐藏、不挑软柿子**。
柱状图四臂全在，不存在这个问题。

**中文字体**：服务器无 CJK 字体 ⇒ 本脚本**在本地跑**（数据由 pull_multiband.sh 拉回暂存树）。
用法：
    python code/tools/make_multiband_figs.py --stage results/_stage_multiband
    python code/tools/make_multiband_figs.py --stage results/_stage_multiband --baseline best
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cnfont  # noqa: E402
_cnfont.configure(plt)

import torchvision.transforms.functional as tf  # noqa: E402
from splat.utils.image_utils import psnr  # noqa: E402

BASE_ITER = 30000          # 阶段一起点；RRF 本地步 = abs − BASE_ITER
FINAL = 70000
METRICS = [("PSNR", "PSNR (dB)", "↑ 好", 1), ("SSIM", "SSIM", "↑ 好", 1),
           ("LPIPS", "LPIPS (VGG)", "↓ 好", -1)]
ARMS = {  # 键 -> (图例名, 相对 run 目录, 颜色)
    "ax":  ("A× 多频段标量分支（本节）", "run_20260912_多频段标量分支", "#d62728"),
    "ap":  ("A' 标量头 + jet 查表",     "run_20260911_形态A标量头查表", "#ff7f0e"),
    "v14": ("V14（MLP+PE+dist 长训）",   "run_mlp_v2_阶段1b阶段2与长训/阶段2长训_位置编码_X5D3R1", "#2ca02c"),
    "shA": ("SH@A*（原版 SH 长训）",     "run_mlp_v2_阶段1b阶段2与长训/阶段1b长训_SH头", "#1f77b4"),
}
DESC = {
    "ax": "标量头输出 2 个带，标量域乘法合成 u = ū_0·(1+λ(ū_1−init_u))；λ=0 时精确退化成 A'",
    "ap": "标量头输出 1 个标量，混合后过固定 jet 查表（先混合再查表）",
    "v14": "MLP 颜色头（embed 32 + h128/d2）+ PE(xyzL5) + PE(dirL3) + 距离条件，输出 RGB",
    "shA": "原版逐点 SH 系数 + A* exp LR（颜色在 RGB 域线性混合）",
}


def load_json(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.is_file() else {}


def curve_of(stage: Path, rel: str, M: str):
    """48 子集曲线 {abs_iter: {PSNR,SSIM,LPIPS}}。"""
    d = load_json(stage / rel / M / "curve.json")
    out = {}
    for k, v in d.items():
        if k.startswith("ours_"):
            try:
                it = int(k.split("_")[1])
            except ValueError:
                continue
            out[it] = {m: v.get(m) for m, *_ in METRICS}
    return out


def full_of(stage: Path, rel: str, M: str, it=FINAL):
    """640 全量三指标（results.json）。"""
    v = load_json(stage / rel / M / "results.json").get(f"ours_{it}")
    return {m: v.get(m) for m, *_ in METRICS} if v else None


def std_probe_of(stage: Path, rel: str, M: str, it=FINAL):
    return load_json(stage / rel / M / f"std_probe_{it}.json") or None


def lpips_best(stage: Path, M: str):
    """在 v14/shA 里取 LPIPS 更低者（= 客观最强基线）。只用于**如实标注**，不决定对照臂。"""
    cands = {k: full_of(stage, ARMS[k][1], M) for k in ("v14", "shA")}
    cands = {k: v for k, v in cands.items() if v and v.get("LPIPS") is not None}
    return min(cands, key=lambda k: cands[k]["LPIPS"]) if cands else None


def pick_baseline(stage: Path, M: str, rule: str = "orig"):
    """对照臂怎么选。

    - `orig`（默认）：**固定用原版 SH 头**（`shA` = 原版逐点 SH 系数 + A* LR）。
      理由：本任务的对照应是「同一个任务、原版方法」，而不是「顺手挑一个更强的臂」——
      后者会让对照臂随方法漂移，跨方法不可读。缺 shA 数据才回落到 v14。
    - `best`：旧的自动判选（在 V14 与 SH@A* 中取 LPIPS 更低者）。

    注意：`orig` 下若 LPIPS 更低的其实是 V14，**必须在图注里写明**（见 `lpips_best`
    与 main 里的 `baseline_pick`），不能默默拿弱的那条当对照。
    """
    if rule == "best":
        k = lpips_best(stage, M)
        if k is None:
            return None
        return k
    if rule != "orig":
        raise ValueError(f"未知 baseline rule: {rule!r}（应为 'orig' 或 'best'）")
    for k in ("shA", "v14"):                      # 原版优先，缺数据回落
        if full_of(stage, ARMS[k][1], M):
            return k
    return None


def base_label(base_key: str, rule: str) -> str:
    return f"原版对照（{ARMS[base_key][0]}）" if rule == "orig" else f"最强基线（{ARMS[base_key][0]}）"


def _img(p: Path):
    return np.asarray(Image.open(p).convert("RGB")) if p.is_file() else None


def _psnr_one(d: Path, idx: int, gt_dir: Path = None):
    """单帧 PSNR。GT 必须**显式传入**：GT 与臂无关，而下载脚本只在 A× 臂上拉了 gt/
    （其余臂本地只有 renders/），若按 "各自臂目录下找 gt" 的写法，A'/基线行的每格
    会静默全变成 "PSNR —"。"""
    gt_dir = gt_dir if gt_dir is not None else d
    rp, gp = d / "renders" / f"{idx:05d}.png", gt_dir / "gt" / f"{idx:05d}.png"
    if not (rp.is_file() and gp.is_file()):
        return None
    r = tf.to_tensor(Image.open(rp)).unsqueeze(0)[:, :3]
    g = tf.to_tensor(Image.open(gp)).unsqueeze(0)[:, :3]
    return psnr(r, g).item()


# ── 图 1：折线（三指标曲线）────────────────────────────────────────────────────
def fig_curves(stage: Path, M: str, base_key: str, out_dir: Path, payload: dict):
    series = {}
    for k in ("ap", "ax", base_key):
        c = curve_of(stage, ARMS[k][1], M)
        if c:
            series[k] = c
    if len(series) < 2:
        print(f"[skip] 曲线 {M}: 数据不足 {list(series)}")
        return
    xs = {k: sorted(c) for k, c in series.items()}
    for m, ylab, arrow, _dir in METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=145)
        for k in ("ap", "ax", base_key):
            if k not in series:
                continue
            label, _, col = ARMS[k]
            X = [i - BASE_ITER for i in xs[k]]
            Y = [series[k][i][m] for i in xs[k]]
            ax.plot([x / 1000.0 for x in X], Y, "o-" if k == "ax" else "s--",
                    color=col, lw=2.0 if k == "ax" else 1.5,
                    ms=5.5 if k == "ax" else 4.0, label=label, zorder=3 if k == "ax" else 2)
        ax.set_xlabel("RRF 训练步数（千步；本地 0–40，= 绝对 %d–%d）" % (BASE_ITER, FINAL),
                      fontsize=9.5)
        ax.set_ylabel(f"{ylab}  {arrow}", fontsize=9.5)
        ax.grid(alpha=0.28, ls=":")
        ax.legend(fontsize=8.5, loc="best")
        if m == "LPIPS":   # 主判据，标出来
            ax.set_title(f"{M} —— {m}（**主判据**：越低越好）  48 子集抽样曲线", fontsize=11)
        else:
            ax.set_title(f"{M} —— {m} 曲线（48 子集抽样）", fontsize=11)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{m}曲线.png"); plt.close(fig)
        payload.setdefault("curve", {})[m] = {
            ARMS[k][0]: {str(i - BASE_ITER): series[k][i][m] for i in xs[k]}
            for k in series}
    print(f"[ok] 曲线 {M} -> {out_dir}")


# ── 图 2：柱状（4 臂末档 640 全量）─────────────────────────────────────────────
def fig_bars(stage: Path, M: str, base_key: str, out_dir: Path, payload: dict):
    order = ["v14", "shA", "ap", "ax"]
    vals = {k: full_of(stage, ARMS[k][1], M) for k in order}
    have = [k for k in order if vals[k]]
    if len(have) < 2:
        print(f"[skip] 柱状 {M}: 数据不足 {have}")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=145)
    for ax, (m, ylab, arrow, direction) in zip(axes, METRICS):
        ks = [k for k in order if vals[k] and vals[k][m] is not None]
        Y = [vals[k][m] for k in ks]
        cols = ["#d62728" if k == "ax" else ("#ff7f0e" if k == "ap" else "#8c8c8c") for k in ks]
        bars = ax.bar(range(len(ks)), Y, color=cols, width=0.66, zorder=3)
        for b, k in zip(bars, ks):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{vals[k][m]:.4f}" if m != "PSNR" else f"{vals[k][m]:.3f}",
                    ha="center", va="bottom", fontsize=8.6)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([ARMS[k][0].split("（")[0] for k in ks], fontsize=8.6,
                           rotation=18, ha="right")
        lo = min(Y); hi = max(Y)
        pad = max(1e-6, (hi - lo) * 0.22)
        ax.set_ylim(lo - pad, hi + pad * 1.5)
        ax.set_ylabel(f"{ylab}  {arrow}", fontsize=9.5)
        ax.grid(axis="y", alpha=0.28, ls=":")
        ax.set_title(("★ " if m == "LPIPS" else "") + f"{m}" +
                     ("（主判据）" if m == "LPIPS" else ""), fontsize=10.5)
    fig.suptitle(f"{M} —— 形态 A× vs A' vs 两条既有基线（同协议 abs{FINAL}，640 全量 held-out）",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012,
             "同协议：起点 chkpnt30000 · RRF 4 万步（A* exp LR 8e-3→8e-5）· PE 全开 · 冻结几何。"
             "灰 = 既有基线，橙 = A'，红 = A×（本节）。" + DESC["ax"],
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0.008, 0.045, 1, 0.945))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"三指标_多频段对比_{M}.png"
    fig.savefig(p); plt.close(fig)
    payload["bars_640"] = {ARMS[k][0]: vals[k] for k in order if vals[k]}
    print(f"[ok] 柱状 {M} -> {p}  四臂全在图上（对照臂 = {ARMS[base_key][0]}）")


# ── 图 3：原图对照（GT / 最强基线 / A' / A× × 4 帧）───────────────────────────
def fig_qualitative(stage: Path, M: str, base_key: str, it: int, frames, out_dir: Path,
                    payload: dict, rule: str = "orig"):
    rows = [("gt", "GT（真值）", None), (base_key, base_label(base_key, rule), ARMS[base_key][1]),
            ("ap", ARMS["ap"][0], ARMS["ap"][1]), ("ax", ARMS["ax"][0], ARMS["ax"][1])]
    dirs = {}
    for k, _lbl, rel in rows:
        if rel:
            dirs[k] = stage / rel / M / "test" / f"ours_{it}"
    axd = dirs.get("ax")
    gtd = next((dirs[k] for k in ("ax", "ap", base_key) if k in dirs and (dirs[k] / "gt").is_dir()), None)
    if axd is None or gtd is None or not (axd / "renders").is_dir():
        print(f"[skip] 定性 {M}: 渲染缺失（需要 {axd}）")
        return
    means = {k: (full_of(stage, ARMS[k][1], M, it) or {}).get("PSNR") for k in ("ap", "ax", base_key)}

    n = len(frames)
    fig, axes = plt.subplots(len(rows), n, figsize=(3.15 * n, 2.55 * len(rows)), dpi=140)
    axes = np.atleast_2d(axes)
    if axes.shape != (len(rows), n):
        axes = axes.reshape(len(rows), n)
    pf = {}
    for r, (key, label, _rel) in enumerate(rows):
        d = gtd if key == "gt" else dirs[key]
        mean = means.get(key) if key != "gt" else None
        row_label = label + (f"\n640 均值 {mean:.3f} dB" if mean is not None else "")
        for c, idx in enumerate(frames):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.8)
            img = _img(d / ("gt" if key == "gt" else "renders") / f"{idx:05d}.png")
            if img is None:
                ax.text(0.5, 0.5, "缺图", ha="center", va="center", fontsize=9, color="#bbbbbb")
                ax.set_title(f"#{idx:05d}", fontsize=8, color="#888888")
                continue
            ax.imshow(img)
            if key == "gt":
                ax.set_title(f"帧 #{idx:05d}", fontsize=9, pad=4)
            else:
                v = _psnr_one(d, idx, gtd)
                pf.setdefault(key, {})[f"{idx:05d}"] = v
                ax.set_title(f"PSNR {v:.2f} dB" if v is not None else "PSNR —",
                             fontsize=9, pad=4)
        axes[r][0].set_ylabel(row_label, fontsize=10, labelpad=8, va="center")

    a, b = means.get("ax"), means.get(base_key)
    d_s = f"   ΔPSNR(A× − 对照) = {a - b:+.3f} dB" if (a and b) else ""
    ap_m = means.get("ap")
    d_ap = f"   ΔPSNR(A× − A') = {a - ap_m:+.3f} dB" if (a and ap_m) else ""
    fig.suptitle(f"{M} 多频段对照（A× / A' / {base_label(base_key, rule)}，abs{it}）{d_s}{d_ap}",
                 fontsize=12, y=0.985)
    # 如实标注：选中的对照臂若**不是** LPIPS 更低的那条，明确写出来，不默默挑弱的当对照
    best = lpips_best(stage, M)
    if rule == "orig" and best is not None and best != base_key:
        bl, br = lpips_best(stage, M), full_of(stage, ARMS[base_key][1], M) or {}
        bb = full_of(stage, ARMS[bl][1], M) or {}
        honest = (f"⚠ 按 LPIPS（主判据）本方法的客观最强基线其实是 {ARMS[bl][0]}"
                  f"（LPIPS {bb.get('LPIPS', float('nan')):.4f}），"
                  f"高于本图对照臂的 {br.get('LPIPS', float('nan')):.4f}；"
                  "本图按既定口径固定用**原版**作对照，不随方法漂移。")
    else:
        honest = ""
    fig.text(0.5, 0.012,
             "每格 PSNR = 该帧 render 与 gt（同名配对，同 metrics.py 实现）。"
             "A× 与 A' 的差别只有标量头带数（1 → 2）与标量域乘法合成；λ=0 时 A× 精确退化成 A'。"
             + honest,
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0.012, 0.035, 1, 0.955))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"多频段对照_{M}.png"
    fig.savefig(p); plt.close(fig)
    payload["qualitative"] = {"iteration": it, "frames": list(frames),
                              "row_arms": {k: ARMS[k][0] for k, _l, _r in rows if k != "gt"},
                              "base_key": base_key, "mean_640_PSNR": means,
                              "per_frame_PSNR": pf}
    print(f"[ok] 定性 {M} -> {p}   640 均值 " + "  ".join(
        f"{k}: {means[k]:.3f}" if means.get(k) else f"{k}: —" for k in ("ax", "ap", base_key)))


def write_readme(out_dir: Path, text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "说明.md").write_text(text, encoding="utf-8")


README_CURVES = """# 阶段2 多频段标量分支 —— 三指标收敛曲线（组目录）

每方法一个文件夹：`<方法>/{PSNR,SSIM,LPIPS}曲线.png`。

- **线段**：`A'`（形态 A'，标量头 K=1）、`A×`（本节，K=2 乘法调制）、
  **`原版对照（SH@A*）`** —— 对照臂**固定用原版 SH 头**（原版逐点 SH 系数 + A* LR），
  **不随方法漂移**。理由：本任务的对照应是「同一个任务、原版方法」；若逐方法去挑「更弱的那个」，
  对照臂会随方法变，跨方法不可读。
- ⚠️ 若按 LPIPS（主判据）算，某方法的**客观最强基线**其实是 V14，图注会**明确写出**
  （见 `多频段对照_<M>.png` 底部的提示行）——固定用原版作对照是**口径选择**，不是挑软柿子。
  柱状图则四臂全在，不存在这个问题。
- **x 轴**：RRF 本地训练步数（千步），0–40 千 = 绝对 30000–70000。
- **数据口径**：绝对 32000/35000/38000/40000/45000/50000/55000/60000/65000 各 **48 子集**抽帧
  ＋ 末档 70000 **640 全量**，均取 `curve.json` 的 `ours_<i>`。48 与 640 之间有抽样噪声
  （实测 ±0.5 dB 内），故**判优以柱状图的 640 为准**，本图只看趋势与是否压平。
- **三指标方向**：PSNR/SSIM ↑ 好，LPIPS ↓ 好（主判据）。

## 单变量说明

`A×` 与 `A'` 的训练配置**逐字相同**（起点、冻结协议、A* exp LR、PE 三开关、迭代阶梯），
唯一差异 = 标量头输出带数 1 → 2，以及标量域合成方式：

    A' : u = ū_0
    A× : u = ū_0 · (1 + λ·(ū_1 − init_u))     λ = sigmoid(lam_raw) ∈ (0,1)，初值 0.5

**λ=0 时 A× 精确退化成 A'** ⇒ A× 严格包含 A'，对照是**嵌套**的，不是并列的两个假设。

## 结果（ours_70000，全 640 held-out）

| 方法 | 臂 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| MVDR | **A×** | **21.8018** | **0.8473** | **0.2537** |
| MVDR | A' | 21.7902 | 0.8469 | 0.2540 |
| MVDR | SH@A\\*（原版对照） | 16.6794 | 0.7543 | 0.3648 |
| CBF | **A×** | **16.5646** | **0.7594** | **0.3779** |
| CBF | A' | 16.2004 | 0.7575 | 0.3811 |
| CBF | SH@A\\*（原版对照） | 14.3844 | 0.7404 | 0.4235 |

**A× − A'（单变量、嵌套对照）**：MVDR **+0.0116 / +0.00043 / −0.00029**（噪声级持平）；
CBF **+0.3642 / +0.00189 / −0.00325**（小而一致的正向）。六个读数全部朝有利方向，
离板 `>5` 占比两方法均 **0.000%**（不变量保持）。

**但「CBF 过平滑源于标量头容量」这条先验推断被证伪**：CBF 渲染通道 std 只从 37.048 挪到
**40.651**（GT 51.077），**缺口只补上 25.7%**；MVDR 反而略降（52.959 → 52.680）。

**λ 轨迹**（初值 0.5）：MVDR 单调降到 0.3821；CBF **先升后降**（0.7035 → **0.7998**@abs40000
→ 0.3824）。即在症状所在的方法上优化器确实**主动增强**了调制（与设计意图同向），
但两条线终值都落在 λ ≈ 0.382 ⇒ **λ 在小 LR 末段弱辨识**。

完整判读、未验项与边界见 `docs/RRF-MLP颜色头_改动说明.md` §13 与
`docs/RF前向模型规格.md` §7.2 A× 段。
"""

README_BARS = """# 阶段2 多频段标量分支 —— 末档柱状图（组目录）

每方法一张 `三指标_多频段对比_<方法>.png`（1×3 面板：PSNR / SSIM / LPIPS）。

- **四臂同协议**，全部取 **abs70000 的 640 全量**（`results.json` 的 `ours_70000`）：
  `V14`（MLP+PE+dist 长训）/ `SH@A*`（原版 SH 长训）/ `A'`（标量头 K=1）/ `A×`（本节 K=2）。
- **主判据 = LPIPS**（见 `docs/RF前向模型规格.md`：GT 每像素只有 1 个自由度，
  PSNR 会被大面积平坦区拉偏）。图中以 ★ 标出。
- 灰 = 既有基线，橙 = A'，红 = A×。
- 每格数字即 `数据.json` 里的原始值，可直接引用。

**对照口径**：柱状图**四臂全画**（V14 / SH@A* / A' / A×），不需要判选，不存在挑对照的问题。

按 LPIPS（主判据）逐方法算，客观最强基线是：**MVDR → SH@A\***（16.679 / 0.754 / 0.365）、
**CBF → V14**（15.981 / 0.756 / 0.390）。折线图与原图对照图的**对照臂固定用原版 SH@A\***，
理由见 `阶段2_多频段_三指标/说明.md`。CBF 上该选择会让对照臂不是客观最强的那条，
故对照图注会明确标注实际最强的臂与它的 LPIPS，**不隐藏**。

## 结果一句话（abs70000，全 640）

| 方法 | 原版对照 SH@A\\* | A' | **A×** |
|---|---|---|---|
| MVDR PSNR / SSIM / LPIPS | 16.679 / 0.754 / 0.365 | 21.790 / 0.847 / 0.254 | **21.802 / 0.847 / 0.2537** |
| CBF PSNR / SSIM / LPIPS | 14.384 / 0.740 / 0.424 | 16.200 / 0.758 / 0.381 | **16.565 / 0.759 / 0.3779** |

A× 在两条方法上**六个读数全部优于 A'**，但幅度分两档：MVDR 是噪声级持平（+0.012 dB），
CBF 是小而一致的正向（**+0.364 dB / −0.0033 LPIPS**）。
关键否定结论：CBF 的渲染通道 std 缺口只补上 **25.7%**（37.048 → 40.651，GT 51.077）
⇒「瓶颈是标量头容量」**被证伪**。详见 `docs/RRF-MLP颜色头_改动说明.md` §13.7。

（CBF 上 V14 的 LPIPS 0.3902 **优于** SH@A\* 的 0.4235 —— 四臂全在本图里，可直接读。）
"""


def main():
    ap = argparse.ArgumentParser(description="形态 A× 三件套出图（本地跑，需 CJK 字体）")
    ap.add_argument("--stage", default="results/_stage_multiband",
                    help="本地暂存根（pull_multiband.sh 拉的，保持服务器相对结构）")
    ap.add_argument("--out", default=None,
                    help="输出根（默认 <stage>/run_20260912_多频段标量分支）")
    ap.add_argument("--methods", default="MVDR CBF")
    ap.add_argument("--iteration", type=int, default=FINAL)
    ap.add_argument("--frames", default="0 130 260 390")
    ap.add_argument("--baseline", default="orig", choices=["orig", "best"],
                    help="折线图/对照图的对照臂：orig=固定用原版 SH 头（默认）；"
                         "best=旧的逐方法取 LPIPS 更低者")
    args = ap.parse_args()

    stage = Path(args.stage)
    if not stage.is_dir():
        raise SystemExit(f"暂存根不存在: {stage}（先跑 code/tools/pull_multiband.sh）")
    out_root = Path(args.out) if args.out else stage / ARMS["ax"][1]
    frames = [int(x) for x in args.frames.split()]
    curves_dir = out_root / "阶段2_多频段_三指标"
    bars_dir = out_root / "阶段2_多频段_柱状图"

    for M in args.methods.split():
        print(f"===== {M} =====")
        bk = pick_baseline(stage, M, args.baseline)
        if bk is None:
            print(f"[skip] {M}: V14/SH@A* 的 results.json 都缺，无法确定对照臂")
            continue
        # 如实记录：选中的对照臂 vs 客观最强基线（LPIPS）
        _best = lpips_best(stage, M)
        _bn, _bb = full_of(stage, ARMS[bk][1], M) or {}, full_of(stage, ARMS[_best][1], M) or {}
        if _best is None:
            _note = "本方法只有一条基线臂有数据"
        elif _best == bk:
            _note = f"对照臂即客观最强基线（LPIPS {_bb.get('LPIPS', float('nan')):.4f}）"
        else:
            _note = (f"⚠ 对照臂不是客观最强基线：按 LPIPS 最强的是 {ARMS[_best][0]}"
                     f"（{_bb.get('LPIPS', float('nan')):.4f}），本条为 "
                     f"{ARMS[bk][0]}（{_bn.get('LPIPS', float('nan')):.4f}）"
                     "——固定用原版作对照是口径选择，已在图注写明")
        payload = {"method": M, "iteration": args.iteration, "base_iter": BASE_ITER,
                   "baseline_pick": {
                       "key": bk, "name": ARMS[bk][0], "rule": args.baseline,
                       "rule_note": ("固定用原版 SH 头（不随方法漂移）" if args.baseline == "orig"
                                     else "逐方法取 LPIPS 更低者"),
                       "lpips_best_key": _best, "note": _note},
                   "arms": {k: {"name": ARMS[k][0], "dir": ARMS[k][1], "spec": DESC[k]}
                            for k in ARMS},
                   "full_640": {k: full_of(stage, ARMS[k][1], M, args.iteration)
                                for k in ("v14", "shA", "ap", "ax")},
                   "std_probe": {k: std_probe_of(stage, ARMS[k][1], M, args.iteration)
                                 for k in ("v14", "shA", "ap", "ax")}}
        fig_curves(stage, M, bk, curves_dir / M, payload)
        fig_bars(stage, M, bk, bars_dir / M, payload)
        fig_qualitative(stage, M, bk, args.iteration, frames, curves_dir / M, payload,
                        args.baseline)
        (curves_dir / M).mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(curves_dir / M / "数据.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  -> {curves_dir / M / '数据.json'}")

    write_readme(curves_dir, README_CURVES)
    write_readme(bars_dir, README_BARS)
    print(f"\n输出根: {out_root}")


if __name__ == "__main__":
    main()
