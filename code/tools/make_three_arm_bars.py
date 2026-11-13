# -*- coding: utf-8 -*-
"""
make_three_arm_bars.py —— 两组预算三臂对比柱状图（每方法一张，1×3 面板）
--------------------------------------------------------------------------------
回答：在**同一组**里（同预算），三臂的 640 全量三指标各是多少？

    ① A1 原版物理 + 原版RRF   物理 = 上游 orig_seed_chkpnt30000；颜色 = train.py 原版逐点 SH
    ② A2 原版物理 + 新RRF     物理同上；颜色 = MPC/Delay 用 MLP 头、MVDR/CBF 用 A× 标量头
    ③ A3 新物理   + 新RRF     物理 = MVSplat init + 短训；颜色同 ②

数据源**一律是拉回本地的 `清单/*.json`**（驱动在 RENDER 段逐节点写），不在这里硬编码
复用表 —— 这样「哪一臂其实是只读复用来的」是数据说了算，不是我说的算。每根柱子的高度
直接取自该臂的 `results.json[ours_<绝对迭代>]`，绝对迭代也来自清单（组1=60000，组2=50000），
不靠目录名猜。

同图画出 `config.yaml` 的 `stage2_orig`（原版 RF-3DGS 的 40000 刻 = 原版视觉 30000 + RRF 1万）
作**虚线基准**：它既不是 A1（A1 的 RRF 跑到本地 2~3 万步，训练量不同），也不是任何一臂，
只是一个历史锚点 —— 图注里写明，别读成「第四条臂」。

用法（本地仓库根；WSL mvsplat env 或服务器 gs3d env 均可）：
    python code/tools/make_three_arm_bars.py \
        --run 'results/run_20260912_两组预算_三臂对比' --group 组1_物理30000_RRF30000
    ... --methods "MPC CBF Delay MVDR" --out <组目录>/柱状图

输出（默认 <run>/<组>/柱状图/）：
    <M>/三指标_三臂对比_<M>.png
    <M>/数据.json      该方法的数值 payload（含每臂参数量的来源）
"""
import argparse
import json
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

METRICS = [("PSNR", "PSNR (dB)  ↑", 1), ("SSIM", "SSIM  ↑", 1),
           ("LPIPS", "LPIPS (VGG)  ↓", -1)]
FMT = {"PSNR": ".3f", "SSIM": ".4f", "LPIPS": ".4f"}

# 三臂：臂键 → (短标签, 图例名, 颜色, 目录名, 改了什么)
ARMS = [
    ("A1", "A1", "原版物理+原版RRF", "#1f77b4", "A1_原版物理_原版RRF",
     "物理 = 上游 orig_seed_chkpnt30000；颜色 = train.py 原版逐点 SH（引擎 SH）"),
    ("A2", "A2", "原版物理+新RRF", "#ff7f0e", "A2_原版物理_新RRF",
     "物理同上；颜色 = MPC/Delay 用 MLP 头（V14）、MVDR/CBF 用 A× 标量头"),
    ("A3", "A3", "新物理+新RRF", "#2ca02c", "A3_新物理_新RRF",
     "物理 = MVSplat init + 短训；颜色与 A2 逐字同一份 A* 协议"),
]
GROUP_NAME = {1: "组1_物理30000_RRF30000", 2: "组2_物理10000_RRF20000"}
# 组定义（与驱动 group_def 一致；图内要用到「物理/ RRF 各多少步」）
GROUP_INFO = {
    1: dict(phys=30000, rrf=30000, base="原版 orig_seed（上游）", abs_it=60000),
    2: dict(phys=10000, rrf=20000, base="组1 同源 orig_seed（上游）", abs_it=50000),
}


def load_manifest(run: Path):
    """清单/*.json → {(组号:int, 臂:str, 方法:str): rec}"""
    out = {}
    d = run / "清单"
    for f in sorted(d.glob("*.json")):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] 清单读不动 {f}: {e}")
            continue
        out[(int(r["组"]), r["臂"], r["方法"])] = r
    return out


def read_ref(cfg: Path):
    """config.yaml 的 eval.reference.stage2_orig → {M: {PSNR,SSIM,LPIPS}}"""
    try:
        import yaml
        d = yaml.safe_load(open(cfg, encoding="utf-8")) or {}
        return ((d.get("eval") or {}).get("reference") or {}).get("stage2_orig") or {}
    except Exception as e:
        print(f"  [warn] 读不到参考值 {cfg}: {e}")
        return {}


def collect(mani, run, gno, method, ref):
    """→ 三臂的 {key: {"val": {...}, "rec": rec}}，并对每臂现读 results.json。"""
    out, notes = {}, []
    for key, _s, name, _c, dirname, _d in ARMS:
        rec = mani.get((gno, key, method))
        if rec is None:
            notes.append(f"{key} {method}: 清单里没有这个节点（驱动没跑到？）")
            out[key] = None
            continue
        rj = run / rec["本树相对路径"] / "results.json"
        ab = int(rec["绝对迭代"])
        try:
            v = json.load(open(rj, encoding="utf-8"))[f"ours_{ab}"]
        except Exception as e:
            notes.append(f"{key} {method}: 读不到 {rj}[ours_{ab}] ({e})")
            out[key] = None
            continue
        out[key] = {"val": {k: v.get(k) for k, _t, _f in METRICS}, "rec": rec,
                    "n_views": v.get("n_views"), "abs_it": ab}
    return out, notes


def draw(method, gno, data, ref_m, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import _cnfont
        _cnfont.configure(plt)
    except Exception:
        pass
    plt.rcParams["figure.dpi"] = 140

    gi = GROUP_INFO[gno]
    w = 0.62
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2))
    for ax, (key, title, sign) in zip(axes, METRICS):
        vals = [(data[k]["val"][key] if data.get(k) else None) for k, _s, _n, _c, _d, _x in ARMS]
        allv = [v for v in vals if v is not None]
        rv = (ref_m or {}).get(key)
        hi = max(allv + ([rv] if rv is not None else [])) if (allv or rv) else 1.0
        ax.set_ylim(0, hi * 1.34)

        for i, (k, _s, _n, col, _d, _x) in enumerate(ARMS):
            v = vals[i]
            if v is None:
                ax.bar(i, hi * 0.52, w, facecolor="#f5f5f5", edgecolor="#bbbbbb",
                       linestyle="--", linewidth=1.0)
                ax.annotate("缺数据", (i, hi * 0.26), ha="center", va="center",
                            fontsize=7.5, color="#888888")
                continue
            b = ax.bar(i, v, w, color=col)
            ax.annotate(f"{v:{FMT[key]}}",
                        (b[0].get_x() + b[0].get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=8.2)

        # 原版参考线：历史锚点，不是第四条臂
        if rv is not None:
            ax.axhline(rv, color="#555555", linestyle="--", linewidth=1.1, alpha=.85)
            ax.annotate(f"原版 RF-3DGS 参考 {rv:{FMT[key]}}", (2.52, rv),
                        textcoords="offset points", xytext=(0, 3),
                        ha="right", va="bottom", fontsize=7.2, color="#444444")

        # A3 − A1 的差（本图最该看的那个数：换物理+换头，一起动）
        if vals[0] is not None and vals[2] is not None:
            dv = vals[2] - vals[0]
            good = (dv > 0) if sign > 0 else (dv < 0)
            ax.annotate(f"Δ(A3−A1) {dv:+.3f}", (1, hi * 1.22), ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color="#1a9850" if good else "#c0392b")
        if vals[1] is not None and vals[0] is not None:
            dv = vals[1] - vals[0]
            good = (dv > 0) if sign > 0 else (dv < 0)
            ax.annotate(f"Δ(A2−A1) {dv:+.3f}", (1, hi * 1.32), ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color="#1a9850" if good else "#c0392b")

        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([s for _k, s, _n, _c, _d, _x in ARMS], fontsize=9.5)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", alpha=.3)
        ax.margins(x=.10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _k, _s, _n, c, _d, _x in ARMS]
    fig.legend(handles, [f"{_k} {_n}" for _k, _s, _n, _c, _d, _x in ARMS],
               loc="lower center", ncol=3, fontsize=9.2, framealpha=.9,
               bbox_to_anchor=(.5, .045))
    fig.text(.5, .012,
             f"同一组内读差值：物理预算 {'相同' if gno == 1 else 'A3 只有 A1/A2 的 1/3'}、"
             f"RRF 预算相同（本地 {gi['rrf']} 步）。"
             "虚线 = 原版 RF-3DGS 历史锚点（视觉30000+RRF1万，不是第四条臂，训练量不同）。"
             "每根柱 = 640 全量 test 渲染的三指标均值。",
             ha="center", fontsize=7.8, color="#555555")
    fig.suptitle(f"[{method}] {GROUP_NAME[gno]}：三臂三指标对比"
                 f"（物理 {gi['phys']} 步 / RRF 本地 {gi['rrf']} 步，绝对末档 abs{gi['abs_it']}）",
                 fontsize=12.5, y=1.0)
    fig.subplots_adjust(bottom=.26, top=.87)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="两组预算三臂对比柱状图（1×3 面板/方法）")
    ap.add_argument("--run", required=True, help="本地 run 根（含 清单/ 与 组*/）")
    ap.add_argument("--group", default="", help="组目录名；空则 --group-no 决定")
    ap.add_argument("--group-no", type=int, default=1, choices=[1, 2])
    ap.add_argument("--methods", default="MPC CBF Delay MVDR")
    ap.add_argument("--out", default="", help="输出目录（默认 <run>/<组>/柱状图）")
    ap.add_argument("--config", default="", help="config.yaml（取 stage2_orig 参考值）")
    args = ap.parse_args()

    run = Path(args.run)
    gno = args.group_no
    if args.group:
        for k, v in GROUP_NAME.items():
            if v == args.group:
                gno = k
        else:
            if args.group not in GROUP_NAME.values():
                raise SystemExit(f"--group 不认: {args.group}（可选 {list(GROUP_NAME.values())}）")
    out = Path(args.out) if args.out else run / GROUP_NAME[gno] / "柱状图"
    cfg = Path(args.config) if args.config else Path(_CODE).parent / "config.yaml"

    mani = load_manifest(run)
    if not mani:
        raise SystemExit(f"清单为空：{run}/清单/*.json 一个都没有（先跑驱动 RENDER 段再 pull）")
    ref = read_ref(cfg)
    print(f"清单 {len(mani)} 节点；参考值 {len(ref)} 个方法（{cfg}）")

    for m in args.methods.split():
        data, notes = collect(mani, run, gno, m, ref)
        for nt in notes:
            print(f"  [note] {nt}")
        if not any(data.get(k) for k, *_ in ARMS):
            print(f"[skip] {m}: 三臂一个数据都没有")
            continue
        png = out / m / f"三指标_三臂对比_{m}.png"
        draw(m, gno, data, ref.get(m), png)
        payload = {
            "组": GROUP_NAME[gno], "组定义": GROUP_INFO[gno], "方法": m,
            "指标": [k for k, _t, _f in METRICS],
            "臂": {k: (None if data.get(k) is None else {
                "图例": n, "说明": d, "绝对迭代": data[k]["abs_it"],
                "指标": data[k]["val"], "n_views": data[k]["n_views"],
                "是否复用历史产物": data[k]["rec"].get("是否复用"),
                "源模型目录": data[k]["rec"].get("源模型目录"),
                "高斯数": data[k]["rec"].get("高斯数"),
                "头部参数": data[k]["rec"].get("头部参数"),
                "头部参数说明": data[k]["rec"].get("头部参数说明"),
            }) for k, _s, n, _c, _d, d in ARMS},
            "原版参考_stage2_orig": ref.get(m),
            "缺数据说明": notes,
            "口径": "柱高 = 该臂 results.json[ours_<绝对迭代>]（640 全量 test），"
                    "绝对迭代取自 清单/*.json；同一组内读差值",
        }
        json.dump(payload, open(png.parent / "数据.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        row = "  ".join(
            f"{k}=" + (f"{data[k]['val']['PSNR']:.3f}" if data.get(k) else "—")
            for k, *_ in ARMS)
        print(f"[ok] {png}   PSNR {row}")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
