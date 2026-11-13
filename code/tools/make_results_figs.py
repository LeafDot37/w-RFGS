# -*- coding: utf-8 -*-
"""
方案一 · 最终结果对比图 + 评估数据 JSON 生成脚本
运行（Windows）：python make_results_figs.py
输出：
  output/rf-3dgs_1/迭代扫描/对比图_迭代扫描.png + 评估结果.json
  output/rf-3dgs_2/三方对比/三方对比.png          + 对比数据.json
  output/mvsplat/覆盖率.png                       + 验证数据.json
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
avail = {f.name for f in font_manager.fontManager.ttflist}
font = next((c for c in candidates if c in avail), None)
if font:
    plt.rcParams["font.sans-serif"] = [font]
    print("使用字体:", font)
else:
    print("警告: 未找到中文字体, 中文将显示为方块")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

ROOT = os.path.dirname(os.path.abspath(__file__))
O1 = os.path.join(ROOT, "output", "rf-3dgs_1", "迭代扫描")
O2 = os.path.join(ROOT, "output", "rf-3dgs_2", "三方对比")
OM = os.path.join(ROOT, "output", "mvsplat")
os.makedirs(O1, exist_ok=True)
os.makedirs(O2, exist_ok=True)
os.makedirs(OM, exist_ok=True)

# ============================================================
# 数据（均来自评测日志 / 工作记录文档，已核对）
# ============================================================
# 阶段一：迭代扫描（161 held-out 视角, 800×450）
iters    = [500, 1000, 2000, 3000, 4000]
gauss    = [1.41, 1.33, 1.35, 1.47, 1.25]     # 百万高斯
psnr     = [22.08, 25.35, 28.57, 29.98, 32.07]
ssim     = [0.833, 0.875, 0.912, 0.930, 0.945]
lpips    = [0.367, 0.309, 0.241, 0.204, 0.171]
orig7k   = (29.18, 0.934, 0.176)   # 原 7000 步
orig30k  = (29.04, 0.940, 0.127)   # 原 30000 步

# 阶段二：三方对比（640 测试频谱）
methods  = ["MPC", "CBF", "Delay", "MVDR"]
# orig: 原版(30000步+RRF1万)   t40000: 方案一-40000总(4000+RRF3.6万)   r10000: 方案一-10000RRF(4000+RRF1万)
threeway = {
    "MPC":   {"orig": (15.51, 0.628, 0.385), "t40000": (17.06, 0.686, 0.331), "r10000": (15.50, 0.639, 0.381)},
    "CBF":   {"orig": (14.13, 0.723, 0.446), "t40000": (14.98, 0.759, 0.393), "r10000": (14.35, 0.736, 0.427)},
    "Delay": {"orig": (17.73, 0.691, 0.380), "t40000": (19.00, 0.738, 0.319), "r10000": (17.76, 0.698, 0.376)},
    "MVDR":  {"orig": (16.02, 0.731, 0.391), "t40000": (19.34, 0.795, 0.289), "r10000": (16.08, 0.736, 0.379)},
}

# MVSplat init 验证（K=16/V=4/G=2, 141万高斯）
coverage = {"cams": ["cam 0", "cam 200", "cam 400", "cam 600"],
            "cov": [10.3, 15.4, 16.9, 26.0], "prec": 1.0}

# ============================================================
# 图 1：迭代扫描（PSNR / SSIM / LPIPS）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

def ref_lines(ax, val7k, val30k, label7k, label30k, color7k, color30k):
    ax.axhline(val7k, ls="--", color=color7k, lw=1.2)
    ax.axhline(val30k, ls=":", color=color30k, lw=1.2)
    ax.text(510, val7k, f"原7000步 {label7k}", color=color7k, fontsize=8, va="bottom")
    ax.text(510, val30k, f"原30000步 {label30k}", color=color30k, fontsize=8, va="bottom")

c1, c2 = "#d62728", "#ff7f0e"
ax = axes[0]; ax.plot(iters, psnr, "o-", color="#1f77b4", lw=2, ms=6)
ref_lines(ax, orig7k[0], orig30k[0], f"{orig7k[0]:.2f} dB", f"{orig30k[0]:.2f} dB", c1, c2)
for x, y in zip(iters, psnr):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_title("PSNR (dB)↑", fontsize=11); ax.set_xlabel("短训迭代数"); ax.set_xticks(iters); ax.grid(alpha=.3)

ax = axes[1]; ax.plot(iters, ssim, "o-", color="#2ca02c", lw=2, ms=6)
ref_lines(ax, orig7k[1], orig30k[1], f"{orig7k[1]:.3f}", f"{orig30k[1]:.3f}", c1, c2)
for x, y in zip(iters, ssim):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_title("SSIM ↑", fontsize=11); ax.set_xlabel("短训迭代数"); ax.set_xticks(iters); ax.grid(alpha=.3)

ax = axes[2]; ax.plot(iters, lpips, "o-", color="#9467bd", lw=2, ms=6)
ref_lines(ax, orig7k[2], orig30k[2], f"{orig7k[2]:.3f}", f"{orig30k[2]:.3f}", c1, c2)
for x, y in zip(iters, lpips):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
ax.set_title("LPIPS ↓", fontsize=11); ax.set_xlabel("短训迭代数"); ax.set_xticks(iters); ax.grid(alpha=.3)

fig.suptitle("RF-3DGS 阶段一：MVSplat init + 短训迭代数扫描（vs 原版 7000/30000 步完整训练）", fontsize=12, y=1.02)
fig.tight_layout()
p1 = os.path.join(O1, "对比图_迭代扫描.png")
fig.savefig(p1, bbox_inches="tight"); print("已生成:", p1); plt.close(fig)

# ============================================================
# 图 2：三方对比（PSNR / SSIM / LPIPS）
# ============================================================
# 下限取"略低于各指标最小值"，保证所有柱子可见（PSNR 全组最小 14.13，SSIM 0.628，LPIPS 0.289）
metric_cfg = [
    ("PSNR (dB)", 0, 13.5, "↑"),
    ("SSIM", 1, 0.60, "↑"),
    ("LPIPS", 2, 0.25, "↓"),
]
bar_colors = {"orig": "#b0b0b0", "t40000": "#1f77b4", "r10000": "#2ca02c"}
bar_labels = {"orig": "原版", "t40000": "方案一-40000总", "r10000": "方案一-10000RRF"}
bar_order  = ["orig", "t40000", "r10000"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
import numpy as np
x = np.arange(len(methods)); w = 0.26
for ax, (title, idx, ymin, arrow) in zip(axes, metric_cfg):
    for j, key in enumerate(bar_order):
        vals = [threeway[m][key][idx] for m in methods]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=bar_labels[key], color=bar_colors[key], edgecolor="k", lw=.4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}" if idx else f"{v:.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(bottom=ymin); ax.set_title(f"{title} {arrow}", fontsize=11); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=.9)
fig.suptitle("RF-3DGS 阶段二：RRF 频谱重建三方对比（640 测试频谱）", fontsize=12, y=1.02)
fig.tight_layout()
p2 = os.path.join(O2, "三方对比.png")
fig.savefig(p2, bbox_inches="tight"); print("已生成:", p2); plt.close(fig)

# ============================================================
# 图 3：MVSplat init 画面覆盖率
# ============================================================
fig, ax = plt.subplots(figsize=(6.2, 3.8))
bars = ax.bar(coverage["cams"], coverage["cov"], color="#4c72b0", edgecolor="k", lw=.5)
for b, v in zip(bars, coverage["cov"]):
    ax.annotate(f"{v}%", (b.get_x() + b.get_width()/2, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 32); ax.set_ylabel("画面覆盖率 (%)")
ax.set_title("MVSplat init 点云投影覆盖率（K=16/V=4/G=2, 141万高斯）", fontsize=11)
ax.grid(axis="y", alpha=.3)
ax.text(0.02, 0.96, f"prec = {coverage['prec']:.0f}（全部投影点均在画面内）\n颜色相关性 cam600 0.43–0.46，其余 0.13–0.31，MAE ~0.13",
        transform=ax.transAxes, fontsize=8, va="top", bbox=dict(fc="white", ec="gray", alpha=.85))
fig.tight_layout()
p3 = os.path.join(OM, "覆盖率.png")
fig.savefig(p3, bbox_inches="tight"); print("已生成:", p3); plt.close(fig)

# ============================================================
# 评估数据 JSON（配套文档引用）
# ============================================================
json1 = {
    "协议": "161 held-out 测试视角, 800×450, metrics=PSNR/SSIM/LPIPS(VGG)",
    "MVSplat_init": "K=16/V=4/G=2, 1.41M 高斯",
    "迭代扫描": [
        {"迭代": i, "高斯数(百万)": g, "PSNR": p, "SSIM": s, "LPIPS": l}
        for i, g, p, s, l in zip(iters, gauss, psnr, ssim, lpips)
    ],
    "原版参考": {"7000步": list(orig7k), "30000步": list(orig30k)},
}
with open(os.path.join(O1, "评估结果.json"), "w", encoding="utf-8") as f:
    json.dump(json1, f, ensure_ascii=False, indent=2); print("已生成: 评估结果.json")

json2 = {
    "协议": "640 测试频谱, 同协议同指标; 列=PSNR/SSIM/LPIPS",
    "对比": {
        m: {"原版(30000+RRF1万)": list(threeway[m]["orig"]),
            "方案一-40000总(4000+RRF3.6万)": list(threeway[m]["t40000"]),
            "方案一-10000RRF(4000+RRF1万)": list(threeway[m]["r10000"])}
        for m in methods
    },
}
with open(os.path.join(O2, "对比数据.json"), "w", encoding="utf-8") as f:
    json.dump(json2, f, ensure_ascii=False, indent=2); print("已生成: 对比数据.json")

json3 = {
    "配置": "K=16/V=4/G=2, RF_MAX_GAUSSIANS=1500000, 32 前向 ~30min",
    "画面覆盖率": dict(zip(coverage["cams"], coverage["cov"])),
    "precision": coverage["prec"],
    "颜色相关性": "cam600 0.43-0.46, 其余 0.13-0.31",
    "MAE": 0.13, "深度p50": 6.66, "深度过滤保留": "87.7%",
    "G=2收益": "cam200 覆盖 0.6% → 15.4%（填朝向空洞）",
}
with open(os.path.join(OM, "验证数据.json"), "w", encoding="utf-8") as f:
    json.dump(json3, f, ensure_ascii=False, indent=2); print("已生成: 验证数据.json")

print("ALL DONE")
