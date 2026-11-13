# -*- coding: utf-8 -*-
"""
make_params_curve.py —— 参数量（高斯球数）— 质量（PSNR/SSIM/LPIPS）曲线评测
--------------------------------------------------------------------------------
把「我的方法（MVSplat init + 3DGS 短训迭代扫描）」与「原版方法（blender_visual_trained
完整训练）」画在同一张以 高斯球数量（≈参数量）为横轴、以 PSNR/SSIM/LPIPS 为纵轴的图上，
评估：同等参数量预算下，我的方法相对原版处于什么质量水平。

评测方法与 baseline 完全一致（见 results/run_20260905_190255_baseline 与 工作记录 §5）：
  - 测试视角：blender_visual_dataset 的 transforms_test.json —— 161 个 held-out（每第 5 帧）
  - 渲染/指标命令：render.py --eval --skip_train --resolution 2（800×450）+ metrics.py
  - 指标实现：splat.utils.loss_utils.ssim / splat.utils.image_utils.psnr /
             splat.lpipsPyTorch.lpips(VGG)（与 code/metrics.py 同一套）
指标优先复用存档里 metrics.py 生成的 results.json；缺失时按同一套函数从已有 renders+gt 重算
（不再重新渲染），并把来源写进输出 json。

用法（WSL，mvsplat env；在 code/ 目录下执行以便 import splat.*）：
    cd /path/to/w-RFGS/code
    <config.yaml paths.python> tools/make_params_curve.py \
        --run   /path/to/w-RFGS/results/run_20260905_190255_baseline \
        --orig  /path/to/w-RFGS/datasets/RF-3DGS_dataset/blender_visual_trained
    # --iters 缺省自动发现（point_cloud/iteration_* 或 迭代扫描/*）；也可显式限定，如
    #   --iters "500 1000 2000 3000 4000"

输出（默认写进 run 的 三方对比 目录）：
    参数量曲线_我的方法vs原版.png        3 面板曲线（PSNR / SSIM / LPIPS，x=高斯球数/万）
    参数量曲线_我的方法vs原版.json       每点 {方法, 迭代, 高斯球数, PSNR, SSIM, LPIPS, 指标来源}
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 让 code/ 成为可 import 根（splat.* 等来自合并后的单根）
_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

# ---------------------------------------------------------------------------
# 底层工具：ply 高斯计数 / 文件存在
# ---------------------------------------------------------------------------
def ply_gauss_count(path):
    """只读 ply ASCII 头里的 'element vertex N'，不加载整点云。"""
    with open(path, "rb") as f:
        head = f.read(4096).decode("latin-1", errors="replace")
    for line in head.splitlines():
        t = line.split()
        if len(t) == 3 and t[0] == "element" and t[1] == "vertex":
            return int(t[2])
    return 0


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ---------------------------------------------------------------------------
# 候选位置发现（覆盖 baseline 中文存档布局 + 标准 3DGS 布局）
# ---------------------------------------------------------------------------
def cloud_candidates(model, itr):
    return [
        model / "迭代扫描" / str(itr) / "point_cloud.ply",          # baseline 存档
        model / "point_cloud" / f"iteration_{itr}" / "point_cloud.ply",  # 标准 3DGS
    ]


def discover_iters(model):
    """自动发现我方档位：标准 point_cloud/iteration_*（优先），或 baseline 迭代扫描/<i>。

    只保留有 point_cloud.ply 的档，按迭代升序。
    """
    found = set()
    pc_dir = Path(model) / "point_cloud"
    if pc_dir.is_dir():
        for d in pc_dir.glob("iteration_*"):
            m = re.fullmatch(r"iteration_(\d+)", d.name)
            if m and (d / "point_cloud.ply").is_file():
                found.add(int(m.group(1)))
    scan_dir = Path(model) / "迭代扫描"
    if scan_dir.is_dir():
        for d in scan_dir.iterdir():
            if d.is_dir() and d.name.isdigit() and (d / "point_cloud.ply").is_file():
                found.add(int(d.name))
    return sorted(found)


def renders_candidates(model, itr):
    base = [
        model / "测试渲染" / f"ours_{itr}",                          # baseline 存档
        model / "迭代扫描" / str(itr) / "test" / f"ours_{itr}",       # baseline 存档(500)
        model / "test" / f"ours_{itr}",                              # 标准 3DGS
    ]
    out = []
    for d in base:
        if (d / "renders").is_dir() and (d / "gt").is_dir():
            out.append(d)
    return out


def results_json_candidates(model, itr):
    return [
        model / "迭代扫描" / str(itr) / "results.json",              # baseline 存档
        model / "results.json",                                      # 标准 3DGS 聚合
    ]


def load_metric_from_results(candidates, itr):
    """results.json 里若有 ours_<itr> 键，返回 (PSNR, SSIM, LPIPS)。"""
    for p in candidates:
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            key = f"ours_{itr}"
            if isinstance(d, dict) and key in d:
                v = d[key]
                if all(k in v for k in ("PSNR", "SSIM", "LPIPS")):
                    return p, (float(v["PSNR"]), float(v["SSIM"]), float(v["LPIPS"]))
    return None, None

# ---------------------------------------------------------------------------
# 按 code/metrics.py 同一套实现重算指标（不再渲染，只读已有 renders+gt）
# ---------------------------------------------------------------------------
def measure_metrics(renders_dir, gt_dir):
    import torch
    from PIL import Image
    import torchvision.transforms.functional as tf
    from splat.utils.loss_utils import ssim
    from splat.utils.image_utils import psnr
    from splat.lpipsPyTorch import lpips

    torch.cuda.set_device(torch.device("cuda:0"))
    names = sorted(os.listdir(renders_dir))
    if not names:
        raise RuntimeError(f"renders 为空: {renders_dir}")
    ss, ps, lps = [], [], []
    for fn in names:
        r = tf.to_tensor(Image.open(renders_dir / fn)).unsqueeze(0)[:, :3, :, :].cuda()
        g = tf.to_tensor(Image.open(gt_dir / fn)).unsqueeze(0)[:, :3, :, :].cuda()
        ss.append(ssim(r, g).item())
        ps.append(psnr(r, g).item())
        lps.append(lpips(r, g, net_type="vgg").item())
    return sum(ps) / len(ps), sum(ss) / len(ss), sum(lps) / len(lps)

# ---------------------------------------------------------------------------
# 收集一条方法-迭代 的数据点
# ---------------------------------------------------------------------------
def collect_point(tag, model, itr):
    """返回 row dict：方法/迭代/高斯球数/PSNR/SSIM/LPIPS/指标来源。找不到数据抛错。"""
    cloud = first_existing([str(c) for c in cloud_candidates(Path(model), itr)])
    if cloud is None:
        raise FileNotFoundError(f"迭代 {itr}: 找不到 point_cloud（{model}）")
    gauss = ply_gauss_count(cloud)
    if gauss <= 0:
        raise ValueError(f"迭代 {itr}: 高斯数解析失败（{cloud}）")

    jsrc, metric = load_metric_from_results(
        [str(c) for c in results_json_candidates(Path(model), itr)], itr)
    source = "results.json"
    if metric is None:
        rd = renders_candidates(Path(model), itr)
        if not rd:
            raise FileNotFoundError(
                f"迭代 {itr}: 既无 results.json[{tag}]，也找不到 renders+gt → 需先按 baseline 方法渲染")
        metric = measure_metrics(rd[0] / "renders", rd[0] / "gt")
        source = f"重算自 renders（{rd[0].name}）"
    return {
        "方法": tag, "迭代": itr, "高斯球数": gauss,
        "PSNR": round(metric[0], 4), "SSIM": round(metric[1], 4),
        "LPIPS": round(metric[2], 4), "指标来源": source,
    }

# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
def make_figure(mine_rows, orig_rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _cnfont
    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140

    def x(r):  # 高斯球数 → 万
        return r["高斯球数"] / 1e4

    m = {"PSNR": (0, "PSNR (dB) ↑"), "SSIM": (1, "SSIM ↑"), "LPIPS": (2, "LPIPS ↓")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for (key, (idx, title)) in m.items():
        ax = axes[idx]
        # 我的方法（短训迭代扫描）
        xs = [x(r) for r in mine_rows]; ys = [r[key] for r in mine_rows]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=6, label="我的方法（MVSplat init + 短训）")
        for r in mine_rows:
            ax.annotate(f"{r['迭代']}步", (x(r), r[key]), textcoords="offset points",
                        xytext=(0, 7 if key != "LPIPS" else -15), ha="center", fontsize=8, color="#1f77b4")
        # 原版方法（完整训练存档：7000 / 30000 步）
        xs = [x(r) for r in orig_rows]; ys = [r[key] for r in orig_rows]
        ax.plot(xs, ys, "s--", color="#d62728", lw=2, ms=7, label="原版（blender_visual_trained）")
        for r in orig_rows:
            ax.annotate(f"原{r['迭代']}步", (x(r), r[key]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#d62728")
        ax.set_xlabel("高斯球数（万）≈ 参数量")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="best", framealpha=.9)
    fig.suptitle("参数量-质量曲线：我的方法（MVSplat init + 3DGS 短训）vs 原版（161 held-out 视角, 800×450, "
                 "评测同 baseline）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="参数量(高斯球数)-质量曲线评测（我的方法 vs 原版，方法同 baseline）")
    ap.add_argument("--run", required=True, help="端到端 run 目录（含 rf-3dgs_1/ 阶段一 产物）")
    ap.add_argument("--iters", default=None,
                    help="我的方法要画点的短训迭代（空格分隔；缺省自动发现 "
                         "point_cloud/iteration_* 或 迭代扫描/*）")
    ap.add_argument("--orig", required=True, help="原版模型目录（blender_visual_trained）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/rf-3dgs_2/三方对比）")
    args = ap.parse_args()

    run = Path(args.run)
    mine_model = run / "rf-3dgs_1"
    orig_model = Path(args.orig)
    out = Path(args.out) if args.out else run / "rf-3dgs_2" / "三方对比"
    out.mkdir(parents=True, exist_ok=True)

    if args.iters:
        mine_iters = [int(s) for s in args.iters.split()]
    else:
        mine_iters = discover_iters(mine_model)
        if not mine_iters:
            raise RuntimeError(
                f"{mine_model} 无可用档位：标准布局 point_cloud/iteration_* 与 baseline 迭代扫描/ 都为空。"
                f"先跑 evaluate_run.sh（render+metrics 后各档才有云+指标），或显式给 --iters。")
        print(f"  自动发现我方档位: {' '.join(map(str, mine_iters))}")

    mine_rows = []
    for itr in mine_iters:
        r = collect_point("我的方法", mine_model, itr)
        mine_rows.append(r)
        print(f"  我的方法 iter {itr:>5}: {r['高斯球数']:>9,} 高斯 | PSNR {r['PSNR']:.3f} | "
              f"SSIM {r['SSIM']:.3f} | LPIPS {r['LPIPS']:.3f} | {r['指标来源']}")
    if not mine_rows:
        raise RuntimeError(f"{mine_model} 无可用评测点（先跑 evaluate_run.sh 完成多档渲染/评测）")

    # 原版：自动发现 point_cloud/iteration_*，取 results.json 里已评测过的
    orig_rows = []
    if orig_model.joinpath("point_cloud").is_dir():
        for pc in sorted(orig_model.glob("point_cloud/iteration_*/point_cloud.ply")):
            itr = int(pc.parent.name.split("_")[1])
            r = collect_point("原版", orig_model, itr)
            orig_rows.append(r)
            print(f"  原版     iter {itr:>5}: {r['高斯球数']:>9,} 高斯 | PSNR {r['PSNR']:.3f} | "
                  f"SSIM {r['SSIM']:.3f} | LPIPS {r['LPIPS']:.3f} | {r['指标来源']}")
    if not orig_rows:
        raise RuntimeError(f"原版 {orig_model} 无可用评测点")

    # 图 + json
    png = out / "参数量曲线_我的方法vs原版.png"
    make_figure(mine_rows, orig_rows, png)
    payload = {
        "协议": "blender_visual_dataset transforms_test.json（161 held-out，每第5帧）@800×450；"
                "PSNR/SSIM/LPIPS(VGG) 与 code/metrics.py 同实现（即 baseline 评测法）",
        "说明": "横轴=高斯球数(≈参数量)；我的方法各点=同一短训进程的存档（MVSplat init 起训）；"
                "原版各点=blender_visual_trained 训练存档。",
        "我的方法": mine_rows,
        "原版": orig_rows,
    }
    with open(out / "参数量曲线_我的方法vs原版.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 简短评估
    print(f"\n已生成: {png}")
    print(f"        {out / '参数量曲线_我的方法vs原版.json'}")
    best = max(mine_rows, key=lambda r: r["PSNR"])
    print(f"\n[评估] 我的方法最优点：{best['迭代']}步 / {best['高斯球数']/1e4:.0f}万高斯 / "
          f"PSNR {best['PSNR']:.2f} dB（LPIPS {best['LPIPS']:.3f}）")
    for o in orig_rows:
        print(f"[评估] 原版 {o['迭代']}步：{o['高斯球数']/1e4:.0f}万高斯 / "
              f"PSNR {o['PSNR']:.2f} dB（LPIPS {o['LPIPS']:.3f}）")

if __name__ == "__main__":
    main()
