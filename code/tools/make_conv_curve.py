# -*- coding: utf-8 -*-
"""
make_conv_curve.py —— RRF 收敛对比曲线（我方 init vs 原版 init，单变量）
--------------------------------------------------------------------------------
同一频谱数据（3dgs_<M>_100）上，两侧从各自"视觉 chkpnt30000"起做 RRF（train.py，
RF reset + 冻结几何 + densify=0）到相同绝对末迭代，沿途每档存云并评测
（render + metrics.py，640 held-out 频谱视图）。本工具把两侧结果按
x = RRF 训练步数（绝对迭代 − 视觉基准步）画出 PSNR/SSIM/LPIPS 三条收敛曲线：
  - 我的方法（MVSplat 稠密 init 视觉 seed）→ RUN/ours_<M>
  - 原版方法（colmap init 视觉 seed）      → RUN/orig_<M>
因两侧 RRF 步数完全一致，指标差异即唯一变量（init/视觉方法）是否带来优越性。

指标与 code/metrics.py 同一套（results.json 的 ours_<i> 键，PSNR/SSIM/LPIPS(VGG)）。

用法（服务器 code/ 目录，gs3d env）：
    <PY> tools/make_conv_curve.py --run /data/yuexf/3dgs/results/run_20260907_conv_MPC_收敛验证 \
                                  --methods MPC --base 30000
    # --iters 缺省自动发现 point_cloud/iteration_*；可显式如 "40000 50000 60000"

输出（默认 <run>/收敛曲线/）：
    收敛曲线_<M>.png   3 面板（PSNR / SSIM / LPIPS，x=RRF 步数），我方(o-蓝) vs 原版(s--红)
    收敛曲线_<M>.json  两侧每档 {方法, 绝对迭代, RRF步数, 高斯球数, PSNR, SSIM, LPIPS}
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

_METRICS_KEYS = ("PSNR", "SSIM", "LPIPS")


def ply_gauss_count(path):
    with open(path, "rb") as f:
        head = f.read(4096).decode("latin-1", errors="replace")
    for line in head.splitlines():
        t = line.split()
        if len(t) == 3 and t[0] == "element" and t[1] == "vertex":
            return int(t[2])
    return 0


def discover_iters(model_dir):
    """point_cloud/iteration_* 有 point_cloud.ply 的档位，升序。"""
    pc = Path(model_dir) / "point_cloud"
    out = set()
    if pc.is_dir():
        for d in pc.glob("iteration_*"):
            m = re.fullmatch(r"iteration_(\d+)", d.name)
            if m and (d / "point_cloud.ply").is_file():
                out.add(int(m.group(1)))
    return sorted(out)


def collect_rows(side, model_dir, iters):
    """读 results.json 的 ours_<i> 指标 + 每档高斯数。返回 rows 列表（按迭代升序）。"""
    rj = Path(model_dir) / "results.json"
    if not rj.is_file():
        raise FileNotFoundError(f"{side} 无 results.json: {rj}（先跑完 render + metrics.py）")
    data = json.load(open(rj, encoding="utf-8"))
    rows = []
    for itr in iters:
        key = f"ours_{itr}"
        if key not in data or not all(k in data[key] for k in _METRICS_KEYS):
            raise RuntimeError(f"{side} results.json 缺 {key} 完整指标（{rj}）")
        cloud = Path(model_dir) / "point_cloud" / f"iteration_{itr}" / "point_cloud.ply"
        gauss = ply_gauss_count(cloud) if cloud.is_file() else None
        v = data[key]
        rows.append({
            "方法": side, "绝对迭代": itr,
            "RRF步数": itr,            # 占位，main 里按 base 换算成真 RRF 步
            "高斯球数": gauss,
            "PSNR": float(v["PSNR"]), "SSIM": float(v["SSIM"]), "LPIPS": float(v["LPIPS"]),
        })
    return rows


def make_figure(side_rows, out_png, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _cnfont
    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140

    def x(r):
        return (r["绝对迭代"] - base) / 1000.0   # RRF 步数(千)

    panels = [("PSNR", "PSNR (dB) ↑"), ("SSIM", "SSIM ↑"), ("LPIPS", "LPIPS ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {"我的方法": "#1f77b4", "原版": "#d62728"}
    styles = {"我的方法": ("o-", "我的方法（MVSplat init 视觉）"),
              "原版": ("s--", "原版（colmap init 视觉）")}
    for (key, title), ax in zip(panels, axes):
        for row in side_rows:
            style, lab = styles[row["方法"]]
            xs = [x(r) for r in row["rows"]]
            ys = [r[key] for r in row["rows"]]
            ax.plot(xs, ys, style, color=colors[row["方法"]], lw=2, ms=7, label=lab)
            # 末点标注高斯球数（万），直观展示容量差异
            last = row["rows"][-1]
            if last["高斯球数"]:
                ax.annotate(f"{last['高斯球数']/1e4:.0f}万高斯", (xs[-1], ys[-1]),
                            textcoords="offset points", xytext=(4, -14 if key != "LPIPS" else 8),
                            fontsize=8, color=colors[row["方法"]])
        ax.set_xlabel(f"RRF 训练步数（千）  [绝对 = RRF + {base:,}]")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="best", framealpha=.9)
    fig.suptitle("RRF 收敛对比（单变量：仅 init/视觉 seed 不同；同频谱数据同 RRF 步数；"
                 "640 held-out, 同 metrics.py）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="RRF 收敛对比曲线（我方 init vs 原版 init，单变量）")
    ap.add_argument("--run", required=True, help="实验 run 目录（含 ours_<M> 与 orig_<M>）")
    ap.add_argument("--methods", default="MPC", help="方法名（空格分隔），默认 MPC")
    ap.add_argument("--base", type=int, default=30000, help="视觉基准步数（RRF=0 处绝对迭代）")
    ap.add_argument("--iters", default=None, help="要取的绝对存档迭代（空格分隔；缺省自动发现 point_cloud/iteration_*）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/收敛曲线）")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "收敛曲线"
    out.mkdir(parents=True, exist_ok=True)

    for m in args.methods.split():
        ours_dir = run / f"ours_{m}"
        orig_dir = run / f"orig_{m}"
        for d in (ours_dir, orig_dir):
            if not d.is_dir():
                raise RuntimeError(f"缺模型目录: {d}")

        iters = [int(s) for s in args.iters.split()] if args.iters else None
        sides = []
        for side, d in (("我的方法", ours_dir), ("原版", orig_dir)):
            its = iters if iters else discover_iters(d)
            if not its:
                raise RuntimeError(f"{d} 无可评测档位（point_cloud/iteration_* 空，或显式给 --iters）")
            rows = collect_rows(side, d, its)
            for r in rows:
                r["RRF步数"] = r["绝对迭代"] - args.base
                r["方法"] = side
            sides.append({"方法": side, "rows": rows})
            print(f"  [{m}] {side}: " + " | ".join(
                f"RRF{r['RRF步数']//1000}k(abs{r['绝对迭代']}): PSNR {r['PSNR']:.3f} "
                f"SSIM {r['SSIM']:.4f} LPIPS {r['LPIPS']:.4f}"
                + (f" {r['高斯球数']/1e4:.0f}万" if r["高斯球数"] else "") for r in rows))

        png = out / f"收敛曲线_{m}.png"
        make_figure(sides, png, args.base)
        payload = {
            "协议": "RF-3DGS 频谱 3dgs_{M}_100，640 held-out；两侧同 train.py RRF（RF reset+冻结几何"
                    f"+densify=0）同 RRF 总步数；指标=metrics.py（PSNR/SSIM/LPIPS-VGG）。"
                    f"唯一变量：视觉 seed（我方 MVSplat 稠密 init vs 原版 colmap init）。",
            "视觉基准步": args.base,
            "数据": {s["方法"]: s["rows"] for s in sides},
        }
        jf = out / f"收敛曲线_{m}.json"
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  已生成: {png}\n          {jf}")

        # 末段（等 RRF 步数）优越性判定
        ours, orig = sides
        lo, oo = ours["rows"][-1], orig["rows"][-1]
        if lo["RRF步数"] == oo["RRF步数"]:
            dP, dS, dL = (lo["PSNR"]-oo["PSNR"], lo["SSIM"]-oo["SSIM"], lo["LPIPS"]-oo["LPIPS"])
            print(f"\n[结论-{m}@RRF{lo['RRF步数']}步] 我方 − 原版 = "
                  f"ΔPSNR {dP:+.3f}  ΔSSIM {dS:+.4f}  ΔLPIPS {dL:+.4f}  → "
                  f"{'我方占优' if dP > 0 and dL < 0 else '原版占优或接近'}")


if __name__ == "__main__":
    main()
