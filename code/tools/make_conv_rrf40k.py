# -*- coding: utf-8 -*-
"""
make_conv_rrf40k.py —— 阶段二 RRF 拉长 4 万步：每方法 MLP-vs-原版SH 对比收敛曲线（本地出图）
--------------------------------------------------------------------------------
双引擎（原版 SH 逐点色头 / 我方 embedding+共享 MLP 色头）各自从同一视觉起点
orig_seed_chkpnt30000（=RF-3DGS 原版视觉 seed）整段重训 RRF 到绝对 70000（=RRF 4万步），
每方法沿途同一 10 档存档（abs 32000..70000 → RRF 2k/5k/8k/10k/15k/20k/25k/30k/35k/40k）。
服务器侧每档 render 后由 tools/metrics_fast.py 对**确定性抽帧 subset=48** 计算
PSNR/SSIM/LPIPS → <M>/curve.json（ours_<i> 键）。本工具按每方法把两引擎曲线画到一张图：
  - 原版 SH  = results/run_rrf40k_双引擎4万步/sh/<M>/curve.json
  - 我方 MLP = results/run_rrf40k_双引擎4万步/mlp/<M>/curve.json
x = RRF 训练步数（绝对迭代 − 视觉基准 30000）/1000（千步）。同方法两曲线共享同一 x 与
同 10 档、同一抽帧口径 → 唯一变量即颜色表达形式；是否随时间拉长继续上升可直接读图。

指标口径：metrics_fast.py 复用 code/metrics.py 的 PSNR/SSIM/LPIPS-VGG 实现，仅视图数
subset=48（确定性均匀抽帧，同档同样本）。末档另有全量 640 的 results.json 供柱状图/文档锚点。

用法（WSL mvsplat env；本地仓库 /path/to/w-RFGS 视图，cwd=code/）：
    <PY> tools/make_conv_rrf40k.py --run /path/to/w-RFGS/results/run_rrf40k_双引擎4万步 \
                                   --base 30000 --out /path/to/w-RFGS/results/run_rrf40k_双引擎4万步/图

输出（默认 <run>/图）：
    收敛曲线_阶段二_RRF40k_<M>.png   每方法 3 面板（PSNR/SSIM/LPIPS，x=RRF千步），
                                     MLP(o-蓝) vs 原版SH(s--红) 同图
    收敛曲线_阶段二_RRF40k_<M>.json  数值 payload（可入文档）
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
_SIDES = [("mlp", "MLP", "embedding+MLP 颜色头", "#1f77b4", "o-"),
          ("sh", "SH", "原版 逐点 SH 颜色头", "#d62728", "s--")]


def load_curve(run_dir, eng, method):
    """<run>/<eng>/<method>/curve.json → {abs_iter: metrics} 升序。"""
    cj = Path(run_dir) / eng / method / "curve.json"
    if not cj.is_file():
        raise FileNotFoundError(f"缺曲线数据: {cj}")
    d = json.load(open(cj, encoding="utf-8"))
    rows = {}
    for k, v in d.items():
        if k.startswith("ours_"):
            rows[int(k.split("_")[1])] = v
    return dict(sorted(rows.items()))


def make_figure(method, sh_rows, mlp_rows, out_png, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _cnfont
    _cnfont.configure(plt)
    plt.rcParams["figure.dpi"] = 140

    def x(it):
        return (it - base) / 1000.0   # RRF 千步

    panels = [("PSNR", "PSNR (dB) ↑"), ("SSIM", "SSIM ↑"), ("LPIPS", "LPIPS ↓")]
    fmt = {"PSNR": ".2f", "SSIM": ".3f", "LPIPS": ".3f"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    for (key, title), ax in zip(panels, axes):
        for rows, _tag, lab, color, sty in ((mlp_rows, "mlp", _SIDES[0][2], _SIDES[0][3], _SIDES[0][4]),
                                            (sh_rows, "sh", _SIDES[1][2], _SIDES[1][3], _SIDES[1][4])):
            its = list(rows)
            ax.plot([x(i) for i in its], [rows[i][key] for i in its], sty,
                    color=color, lw=2, ms=6, label=lab)
            last_it = its[-1]
            ax.annotate(f"{rows[last_it][key]:{fmt[key]}}", (x(last_it), rows[last_it][key]),
                        textcoords="offset points", xytext=(5, 0 if key != "LPIPS" else -4),
                        fontsize=8, color=color, va="center")
        ax.set_xlabel("RRF 训练步数（千）   [绝对 = RRF + %d]" % base)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="best", framealpha=.9)
    fig.suptitle(f"阶段二 RRF 拉长 4 万步对比（{method}；同起点 orig_seed 视觉 {base}，"
                 f"双引擎各整段重训 RRF→4万，同 10 档；抽帧 48/档，metrics_fast.py）",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="阶段二 RRF40k 每方法收敛对比（MLP vs 原版SH，本地出图）")
    ap.add_argument("--run", required=True, help="本地结果目录（含 sh|<eng>/<M>/curve.json）")
    ap.add_argument("--base", type=int, default=30000, help="视觉基准绝对迭代（RRF=0 处）")
    ap.add_argument("--methods", default=" ".join(_METHODS), help="方法名（空格分隔）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/图）")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "图"
    out.mkdir(parents=True, exist_ok=True)

    payload = {"协议":
               f"阶段二 RRF 双引擎各从 orig_seed 视觉 {args.base} 整段重训到绝对 70000"
               "（=RRF 4万步），10 档存档（32000..70000）。指标=metrics_fast.py（抽帧 48/档，"
               "确定性子集，同档同样本；复用 code/metrics.py 实现）。唯一变量：颜色表达形式"
               "（原版逐点SH vs 我方 embedding+共享MLP）。数据键 x = RRF 千步。",
               "数据": {}}

    for m in args.methods.split():
        mlp = load_curve(run, "mlp", m)
        sh = load_curve(run, "sh", m)
        both = set(mlp) & set(sh)
        if len(both) != len(mlp) or len(both) != len(sh):
            raise RuntimeError(f"{m}: MLP/SH 档位不一致 mlp={sorted(mlp)} sh={sorted(sh)}")
        its = sorted(both)

        png = out / f"收敛曲线_阶段二_RRF40k_{m}.png"
        make_figure(m, sh, mlp, png, args.base)

        def block(rows):
            return [{"绝对迭代": i, "RRF步数": i - args.base, "PSNR": rows[i]["PSNR"],
                     "SSIM": rows[i]["SSIM"], "LPIPS": rows[i]["LPIPS"]} for i in its]
        payload["数据"][m] = {"MLP": block(mlp), "原版SH": block(sh)}
        # 末档（RRF 4万）差
        dP = mlp[its[-1]]["PSNR"] - sh[its[-1]]["PSNR"]
        dS = mlp[its[-1]]["SSIM"] - sh[its[-1]]["SSIM"]
        dL = mlp[its[-1]]["LPIPS"] - sh[its[-1]]["LPIPS"]
        print(f"[{m}] 档位 {len(its)} 个（RRF {its[0]-args.base}..{its[-1]-args.base} 步）")
        for side, rows in (("MLP", mlp), ("原版SH", sh)):
            print("  " + " | ".join(f"RRF{(i-args.base)//1000}k:{rows[i]['PSNR']:.3f}"
                                    for i in its))
        print(f"  末档(抽帧48) Δ(MLP−原版SH) = ΔPSNR {dP:+.3f}  ΔSSIM {dS:+.4f}  ΔLPIPS {dL:+.4f}")
        print(f"  已生成: {png}")

    jf = out / f"收敛曲线_阶段二_RRF40k.json"
    json.dump(payload, open(jf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  已生成: {jf}")


if __name__ == "__main__":
    main()
