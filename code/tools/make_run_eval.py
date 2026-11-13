#!/usr/bin/env python3
"""make_run_eval.py —— 对一次端到端 run 聚合 baseline 等价全量评测（json + 图）。

输入：results/<run> 下 rf-3dgs_1 与 rf-3dgs_2/<方法> 的 results.json
      （由 evaluate_run.sh 先逐档 render + 一次 metrics.py 生成，扁平键 ours_<i>）。
      “原版 RF-3DGS”参考列取仓库根 config.yaml 的 eval.reference（eval.enabled 可关）。
自动发现（不再依赖固定单档）：阶段一全部 ours_<i>（多档迭代扫描）；阶段二每方法
      ours_<i>（full=10000RRF 对齐档 + 40000总 档；smoke=单档）。
输出：
      <run>/rf-3dgs_1/评测/阶段一评测.json + 阶段一_指标图.png
              + 阶段一_高斯数.json + 阶段一_高斯数曲线.png   （高斯点数量随训练步数，新增）
              + 快速查看/ours_<i>_4视角对比.png     （每档仅 4 个位置 原图|渲染）
      <run>/rf-3dgs_2/评测/阶段二评测.json + 阶段二_方法对比图.png
      <run>/rf-3dgs_2/三方对比/三方对比.png + 对比数据.json

用法：python make_run_eval.py --run <run_dir>        （纯文件/matplotlib/PIL，无 GPU）
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # 仓库根（config.yaml 所在）
CFG = ROOT / "config.yaml"
METHOD_ORDER = ["MPC", "CBF", "Delay", "MVDR"]
CAMS_DEFAULT = [0, 200, 400, 600]
ORIG_LABEL = "原版(30000+RRF1万)"


# ---------------------------------------------------------------- config 参考
def load_config():
    try:
        import yaml
        with open(CFG, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    except Exception as e:                       # pragma: no cover
        print(f"[make_run_eval] 无法读取 config.yaml: {e}")
        d = {}
    ev = d.get("eval") or {}
    cov = ev.get("mvsplat_coverage") or {}
    return {
        "enabled": bool(ev.get("enabled", True)),
        "reference": ev.get("reference") or {},
        "profiles": d.get("profiles") or {},
        "data_visual": (d.get("paths") or {}).get("data_visual"),
        "cams": list(cov.get("cams", CAMS_DEFAULT)) if cov.get("enabled", True) else [],
    }


def profile_name_of(run: Path):
    p = run / "profile.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


# ---------------------------------------------------------------- 小工具
def ply_count(path: Path):
    """从 ply 头取 element vertex 数（不加载全文件）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        m = re.search(rb"element vertex (\d+)", head)
        return int(m.group(1)) if m else None
    except Exception:
        return None


_METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")


def _metric_entry(m, v):
    if isinstance(v, dict) and all(k in v for k in _METRIC_KEYS):
        return {"PSNR": float(v["PSNR"]), "SSIM": float(v["SSIM"]),
                "LPIPS": float(v["LPIPS"])}
    return None


def load_results(model_dir: Path):
    """results.json -> {int(iter): {PSNR,SSIM,LPIPS}}（只留 ours_<数字> 键）。

    兼容扁平（本项目）与嵌套（scene->method）两种形态。
    """
    rj = model_dir / "results.json"
    out = {}
    if not rj.exists():
        return out
    try:
        data = json.loads(rj.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] 解析失败 {rj}: {e}")
        return out
    items = []
    if isinstance(data, dict):
        items = list(data.items())
        # 嵌套形态：{scene: {ours_i: {...}}} 平铺到 items
        flat = []
        for k, v in data.items():
            if isinstance(v, dict) and any(str(kk).startswith("ours_") for kk in v):
                flat.extend((kk, vv) for kk, vv in v.items())
            else:
                flat.append((k, v))
        items = flat
    for k, v in items:
        if isinstance(k, str):
            mm = re.fullmatch(r"ours_(\d+)", k)
            if mm:
                ent = _metric_entry(k, v)
                if ent:
                    out[int(mm.group(1))] = ent
    return out


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  写出 {path}")


def setup_font():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import _cnfont
        _cnfont.configure(plt)
        return plt
    except Exception as e:
        print(f"  [warn] 无 matplotlib，跳过出图: {e}")
        return None


def cloud_count(model: Path, itr: int, extra_models=()):
    """跨谱系取第 itr 档高斯数：先在主模型目录，其次各续训前驱目录。"""
    for md in (model, *extra_models):
        for p in (md / "point_cloud" / f"iteration_{itr}" / "point_cloud.ply",
                  md / "迭代扫描" / str(itr) / "point_cloud.ply"):
            if p.exists():
                return ply_count(p)
    return None


def load_results_multi(model: Path, extra_models=()):
    """合并主模型与前驱模型（续训谱系）的 ours_* 指标；主模型同迭代优先。"""
    merged = {}
    for md in (*extra_models, model):
        merged.update(load_results(md))
    return merged


# ---------------------------------------------------------------- 站号 / 快速查看
def _station_of_path(p: str):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) - 1 if m else None      # 1-based 文件名 -> 0-based 环轨迹站号


def pick_test_stations(data_visual, targets):
    """test 视图(按渲染序号=json 帧序) → 站号；为每个 target 站挑最近(不重复)的 test 序号。

    返回 [(target, test_idx, test_station), ...]，失败返回 []。
    """
    if not data_visual:
        return []
    tr = Path(data_visual) / "transforms_train.json"
    ts = Path(data_visual) / "transforms_test.json"
    if not (tr.exists() and ts.exists()):
        return []
    try:
        test_frames = json.load(open(ts, encoding="utf-8")).get("frames", [])
    except Exception:
        return []
    test_st = [_station_of_path(f.get("file_path", "")) for f in test_frames]
    picks, used = [], set()
    for t in targets:
        best = None
        for j, s in enumerate(test_st):
            if s is None or j in used:
                continue
            d = abs(s - t)
            if best is None or d < best[0]:
                best = (d, j, s)
        if best is None:
            continue
        used.add(best[1])
        picks.append((t, best[1], best[2]))
    return sorted(picks, key=lambda x: x[0])


def quickview_montage(out_png: Path, model: Path, itr: int, rows,
                      left="原图 (gt)", right="生成图 (render)"):
    """每档一张 4 行 2 列蒙太奇：行=视角(标签)，列=[left | right]。

    rows: [(label, tidx), ...]；图文件 test/ours_<i>/{renders,gt}/{tidx:05d}.png。
    只需 rows 里每个文件在即可——支持本地镜像只拉了 4 视角小图的情形
    （渲染全量 161/640 也照常；某帧缺失则跳过该行，仍 ≥2 行就出图）。
    """
    from PIL import Image, ImageDraw
    import _cnfont
    fnt = _cnfont.pil_font(13)
    rd = model / "test" / f"ours_{itr}"
    renders, gt = rd / "renders", rd / "gt"
    if not (renders.is_dir() and gt.is_dir()):
        print(f"  [warn] 跳过 4 视角快速查看 ours_{itr}: 无 renders/gt")
        return False

    H_IMG = 150
    pairs = []                                        # [(gt, render, 标签)]
    for label, tidx in rows:
        fn = f"{tidx:05d}.png"
        rp, gp = renders / fn, gt / fn
        if not (rp.exists() and gp.exists()):
            print(f"  [warn] ours_{itr} 缺 {fn}，跳过该视角")
            continue

        def _thumb(p):
            im = Image.open(p).convert("RGB")
            w = max(1, int(im.width * H_IMG / im.height))
            return im.resize((w, H_IMG), Image.LANCZOS)

        pairs.append((label, _thumb(gp), _thumb(rp)))
    if len(pairs) < 2:
        print(f"  [warn] 跳过 4 视角快速查看 ours_{itr}: 可用视角 {len(pairs)} <2")
        return False

    GUTTER, PAD, HEAD, GAP = 118, 10, 26, 12
    cw = max(p[1].width for p in pairs)
    W = GUTTER + cw * 2 + GAP + PAD * 2
    H = HEAD + PAD * 2 + len(pairs) * (H_IMG + 6)
    canvas = Image.new("RGB", (W, H), (246, 246, 246))
    dr = ImageDraw.Draw(canvas)
    cx = GUTTER + PAD
    dr.text((cx, 7), left, font=fnt, fill=(20, 20, 20))
    dr.text((cx + cw + GAP, 7), right, font=fnt, fill=(20, 20, 20))
    y = HEAD + PAD
    for lab, g, rn in pairs:
        dr.text((6, y + 4), lab, font=fnt, fill=(20, 20, 20))
        canvas.paste(g, (cx, y))
        canvas.paste(rn, (cx + g.width + GAP, y))
        y += H_IMG + 6
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print(f"  写出 {out_png}  ({len(pairs)} 个位置)")
    return True


def quickview_spread(out_png: Path, model: Path, itr: int, n, tag="频谱视图"):
    """通用：在 test/ours_<i> 的渲染文件里均匀取 n 帧做 4 视角对比（stage2 用）。

    按实际存在的 PNG 文件名（0 填充序）排序后均分取样：全量渲染时取首/均/末
    ~n 帧；本地镜像只拉了部分帧时（文件数 ≤ n）则全用拉取到的文件。文件名里
    的序即原测试帧序，标签不受“本地只有几张”影响。
    """
    rd = model / "test" / f"ours_{itr}"
    renders = rd / "renders"
    if not renders.is_dir():
        return False
    names = sorted(f for f in os.listdir(renders) if f.lower().endswith(".png"))
    if len(names) < 2:
        print(f"  [warn] 跳过 {out_png.name}: 渲染文件 {len(names)} 不足")
        return False
    if len(names) <= n:
        sel = names
    else:
        sel = [names[round(i * (len(names) - 1) / (n - 1))] for i in range(n)]
    rows = [(f"{tag} {int(fn[:-4])}", int(fn[:-4])) for fn in sel]
    return quickview_montage(out_png, model, itr, rows)


# ---------------------------------------------------------------- 档位标签
def _wan(x):
    """万计数：整数/一位小数（≥10000 或整千倍数时，如 1万 / 2.8万）。"""
    x = x / 10000.0
    return ("%.0f" % x) if abs(x - round(x)) < 1e-9 else ("%.1f" % x)


def _rrf(n):
    """RRF 新增迭代显示：整千倍数用万，否则退回「N 步」（避免 200→0.0万 这类怪名）。"""
    if n % 1000 == 0:
        return f"{_wan(n)}万"
    return f"{n}步"


def profile_iters(prof_cfg: dict):
    def _int(*ks):
        for k in ks:
            if k in prof_cfg and prof_cfg[k] is not None:
                return int(prof_cfg[k])
        return None
    return _int("stage1_iters"), _int("stage2_iters"), _int("stage2_total_iters")


def label_variant(itr, itr1, itr2, total2):
    if total2 is not None and itr == total2:
        return f"方案一-{total2}总({itr1}+RRF{_rrf(total2 - itr1)})"
    if itr2 is not None and itr == itr2:
        return f"方案一-{itr2 - itr1}RRF({itr1}+RRF{_rrf(itr2 - itr1)})"
    return f"存档{itr}步"


# ---------------------------------------------------------------- 图表
def fig_stage1_scan(run_name, rows, refs):
    plt = setup_font()
    if plt is None:
        return None
    iters = [r["iteration"] for r in rows]
    m = {"PSNR": 0, "SSIM": 1, "LPIPS": 2}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, (idx) in m.items():
        ax = axes[idx]
        ys = [r[key] for r in rows]
        ax.plot(iters, ys, "o-", color="#1f77b4", lw=2, ms=5, label="我的方法（MVSplat init + 短训）")
        for r in rows:
            ax.annotate(str(r["iteration"]), (r["iteration"], r[key]),
                        textcoords="offset points", xytext=(0, 7 if key != "LPIPS" else -15),
                        ha="center", fontsize=7, color="#1f77b4")
        for lbl, v in refs.items():
            ax.axhline(v[key], color="#d62728", ls="--", lw=1.2, alpha=.8)
            ax.text(iters[-1], v[key], f" 原{lbl}步 {v[key]:.3f}", fontsize=7,
                    color="#d62728", ha="right", va="bottom")
        ax.set_xlabel("阶段一短训迭代步")
        ax.set_title(f"{key} ({'↑更好' if key != 'LPIPS' else '↓更好'})", fontsize=11)
        ax.grid(alpha=.3)
    handles = [plt.Line2D([0], [0], color="#1f77b4", marker="o", ls="-", label="我的方法")]
    for lbl, v in refs.items():
        handles.append(plt.Line2D([0], [0], color="#d62728", ls="--", label=f"原版 {lbl} 步参考"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=8, framealpha=.9)
    fig.suptitle(f"{run_name} — 阶段一迭代扫描（161 held-out @800×450, 评测同 baseline）", fontsize=12, y=1.08)
    fig.tight_layout()
    return fig


def fig_stage1_gauss(run_name, rows):
    """高斯点数量随训练步数曲线（多档存档云实测，续训谱系合并后为全程）。"""
    plt = setup_font()
    if plt is None:
        return None
    pts = [(r["iteration"], r["gaussians"]) for r in rows if r.get("gaussians")]
    if len(pts) < 2:
        print("  [warn] 高斯数-步数曲线：可用点 <2，跳过出图")
        return None
    its = [p[0] for p in pts]
    gs = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(its, gs, "o-", color="#4C72B0", lw=2, ms=5)
    for x, y in pts:
        ax.annotate(f"{_wan(y)}万", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#4C72B0")
    ax.set_xlabel("阶段一训练迭代步")
    ax.set_ylabel("高斯点数量（个）")
    ax.set_title(f"{run_name} — 高斯点数量随训练步数变化", fontsize=12)
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_stage2(methods, variants, refs, metrics=("PSNR", "SSIM", "LPIPS")):
    """methods: 有序方法列表；variants: {m: [(label, metrics_dict), ...]}；refs: {m: metrics}。"""
    plt = setup_font()
    if plt is None:
        return None
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    # 收集列标签：统一所有方法的 orig + ours 变体顺序（orig 固定第一）
    col_labels = []
    for m in methods:
        for lab, _ in variants.get(m, []):
            if lab not in col_labels:
                col_labels.append(lab)
    if any(m in refs for m in methods) and "原版" not in "".join(col_labels):
        col_labels.insert(0, ORIG_LABEL)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.4))
    x = list(range(len(methods)))
    width = 0.8 / max(1, len(col_labels))
    for ax, metric in zip(axes, metrics):
        for ci, lab in enumerate(col_labels):
            vals, col = [], colors[ci % len(colors)]
            for m in methods:
                if lab == ORIG_LABEL:
                    v = refs.get(m, {}).get(metric)
                else:
                    found = next((vv for ll, vv in variants.get(m, []) if ll == lab), None)
                    v = found.get(metric) if found else None
                vals.append(v)
            off = (ci - (len(col_labels) - 1) / 2) * width
            xs = [xx + off for xx in x]
            yys = [0 if v is None else v for v in vals]
            bars = ax.bar(xs, yys, width=width, color=col, label=lab)
            for xx, v in zip(xs, vals):
                if v is not None:
                    ax.text(xx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if metric == metrics[0]:
            ax.legend(fontsize=7, loc="upper left", framealpha=.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 阶段一
def stage1(run: Path, ref, prof_cfg, cams, extra_models=()):
    model = run / "rf-3dgs_1"
    results = load_results_multi(model, extra_models)
    if not results:
        print("[stage1] 无结果（该 run 未评测阶段B？）")
        return
    rows = []
    for itr in sorted(results):
        n = cloud_count(model, itr, extra_models)
        if n is None:
            print(f"  [warn] ours_{itr}: 缺 point_cloud，高斯数记 null")
        rows.append({"iteration": itr, "iteration_key": f"ours_{itr}",
                     "gaussians": n, **results[itr]})
    refs = {}
    if ref["enabled"]:
        for lbl, v in (ref["reference"].get("stage1_orig") or {}).items():
            refs[lbl] = dict(v)
    # 达标判断：最优 LPIPS 优于原版 30000 步
    orig30 = refs.get("30000", {}).get("LPIPS")
    best = min(rows, key=lambda r: r["LPIPS"])
    passed = None if orig30 is None else bool(best["LPIPS"] < orig30)
    for r in rows:
        print(f"[stage1] ours_{r['iteration']:>6}  高斯 {r['gaussians'] or '?':>9}  "
              f"PSNR {r['PSNR']:>7.3f}  SSIM {r['SSIM']:.4f}  LPIPS {r['LPIPS']:.4f}")
    out = {
        "view_unit": "161 held-out 视觉测试视角（transforms_test.json）",
        "profile": profile_name_of(run),
        "protocol": "评测同 baseline：render --eval --skip_train --resolution 2 + metrics.py；"
                    "PSNR/SSIM/LPIPS(VGG) 同 code/metrics.py",
        "ours": rows,
        "reference_enabled": ref["enabled"],
        "达标判断": (None if passed is None else {
            "判据": "我方最优 LPIPS 优于原版 30000 步 LPIPS",
            "原版30000步LPIPS": orig30,
            "我方最优": {"iteration": best["iteration"], "LPIPS": best["LPIPS"]},
            "达标": passed,
        }),
    }
    if ref["enabled"]:
        out["reference"] = refs
    evdir = model / "评测"
    write_json(evdir / "阶段一评测.json", out)

    fig = fig_stage1_scan(run.name, rows, refs)
    if fig is not None:
        png = evdir / "阶段一_指标图.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"  写出 {png}")

    # 高斯点数量随训练步数曲线（本次新增评测）
    pts = [{"iteration": r["iteration"], "iteration_key": r["iteration_key"],
            "gaussians": r["gaussians"]} for r in rows if r.get("gaussians")]
    write_json(evdir / "阶段一_高斯数.json",
               {"view_unit": "阶段一训练多档存档点云（point_cloud/iteration_<i> 实测）",
                "线": pts})
    gfig = fig_stage1_gauss(run.name, rows)
    if gfig is not None:
        gp = evdir / "阶段一_高斯数曲线.png"
        gfig.savefig(gp, dpi=150, bbox_inches="tight")
        print(f"  写出 {gp}")
        import matplotlib.pyplot as _gplt
        _gplt.close(gfig)

    # 快速查看：每档 4 个位置 原图|渲染（仅对实际存在 renders 的源目录；无则整体提示）
    if cams:
        picks = pick_test_stations(ref.get("data_visual"), cams)
        if not picks:
            print("  [warn] 无 4 视角站号映射（transforms 缺失？），跳过快速查看图")
        qdir = evdir / "快速查看"
        n_qv = 0
        qrows = [(f"位置{t} ≈ 测试站{s}", j) for t, j, s in picks]
        for r in rows:
            src = next((md for md in (model, *extra_models)
                        if (md / "test" / r["iteration_key"] / "renders").is_dir()), None)
            if src is None:
                continue
            if quickview_montage(qdir / f"{r['iteration_key']}_4视角对比.png", src,
                                 r["iteration"], qrows):
                n_qv += 1
        if n_qv == 0:
            print("  [warn] 阶段一快速查看：本 run 无渲染图（评测渲染未落本地），跳过全部")


# ---------------------------------------------------------------- 阶段二
def stage2(run: Path, ref, prof_cfg):
    base = run / "rf-3dgs_2"
    if not base.exists():
        print("[stage2] 无 rf-3dgs_2，跳过")
        return
    itr1, itr2, total2 = profile_iters(prof_cfg)
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name not in ("评测", "三方对比")]
    dirs.sort(key=lambda p: (METHOD_ORDER.index(p.name) if p.name in METHOD_ORDER else 99, p.name))
    variants, origs = {}, ref["reference"].get("stage2_orig") if ref["enabled"] else {}
    for d in dirs:
        results = load_results(d)
        if not results:
            print(f"[stage2] {d.name}: 无 ours_* 结果（未评测？）")
            continue
        cols = []
        for itr in sorted(results):
            lab = label_variant(itr, itr1, itr2, total2)
            cols.append((lab, {"iteration": itr, **results[itr]}))
            print(f"[stage2] {d.name:6s} ours_{itr:<6}  {lab}  "
                  f"PSNR {results[itr]['PSNR']:>7.3f}  SSIM {results[itr]['SSIM']:.4f}  "
                  f"LPIPS {results[itr]['LPIPS']:.4f}"
                  + (f"   (orig {origs[d.name]['PSNR']:.3f})" if d.name in origs else ""))
        variants[d.name] = cols
    if not variants:
        return
    evdir = base / "评测"
    ours = {m: [{"label": lab, "iteration": v["iteration"],
                 "PSNR": v["PSNR"], "SSIM": v["SSIM"], "LPIPS": v["LPIPS"]}
                for lab, v in cols] for m, cols in variants.items()}
    out = {"view_unit": "640 测试频谱 / 方法", "profile": profile_name_of(run),
           "ours": ours, "reference_enabled": ref["enabled"]}
    if ref["enabled"]:
        out["reference"] = {k: dict(v) for k, v in origs.items()}
    write_json(evdir / "阶段二评测.json", out)

    methods = [d.name for d in dirs if d.name in variants]
    refs_metrics = {m: v for m, v in origs.items()}
    fig = fig_stage2(methods, variants, refs_metrics)
    if fig is not None:
        png1 = evdir / "阶段二_方法对比图.png"
        png1.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png1, dpi=150, bbox_inches="tight")
        print(f"  写出 {png1}")
        # 三方对比（对应 baseline 产物）
        tdir = base / "三方对比"
        tdir.mkdir(parents=True, exist_ok=True)
        png2 = tdir / "三方对比.png"
        fig.savefig(png2, dpi=150, bbox_inches="tight")
        print(f"  写出 {png2}")
        import matplotlib.pyplot as _mpl
        _mpl.close(fig)
    # 对比数据.json（baseline 结构：对比:{方法:{列名:[PSNR,SSIM,LPIPS]}}）
    comp = {}
    for m in methods:
        entry = {}
        if m in refs_metrics:
            entry[ORIG_LABEL] = [refs_metrics[m]["PSNR"], refs_metrics[m]["SSIM"],
                                 refs_metrics[m]["LPIPS"]]
        for lab, v in variants.get(m, []):
            entry[lab] = [v["PSNR"], v["SSIM"], v["LPIPS"]]
        comp[m] = entry
    write_json(base / "三方对比" / "对比数据.json",
               {"协议": "640 测试频谱，同协议同指标；列=[PSNR,SSIM,LPIPS]",
                "profile": profile_name_of(run), "对比": comp})

    # 快速查看：每方法·每有渲染档 均匀 4 帧 原图|RRF重建（spectrum 无环轨迹站号，按帧序均匀取）
    qdir = evdir / "快速查看"
    n_qv = 0
    for m in methods:
        md = base / m
        for itr in sorted(variants.get(m, []), key=lambda x: x[1]["iteration"]):
            if quickview_spread(qdir / f"{m}_ours_{itr[1]['iteration']}_4视角对比.png",
                                md, itr[1]["iteration"], 4):
                n_qv += 1
    if n_qv == 0:
        print("  [warn] 阶段二快速查看：无渲染图（评测渲染未落本地），跳过全部")


def main():
    ap = argparse.ArgumentParser(description="端到端 run 评测聚合（baseline 等价 + 快速查看）")
    ap.add_argument("--run", required=True, help="run 目录（含 rf-3dgs_1 / rf-3dgs_2）")
    ap.add_argument("--iters1", type=int, default=None, help="(兼容，忽略：改为自动发现)")
    ap.add_argument("--iters2", type=int, default=None, help="(兼容，忽略：改为自动发现)")
    ap.add_argument("--lineage1", action="append", default=[],
                    help="rf-3dgs_1 续训前驱模型目录（可多次）。本 run 从这些 run 续训时，"
                         "阶段一评测/图/高斯数曲线合并前驱的全部档位（指标+云高斯数），"
                         "得到连续全程扫描。可重复传（多个前驱按序）")
    args = ap.parse_args()
    run = Path(args.run)
    ref = load_config()
    pname = profile_name_of(run) or "full"
    prof_cfg = ref["profiles"].get(pname) or {}
    extra = [Path(p) for p in args.lineage1]
    stage1(run, ref, prof_cfg, ref.get("cams", []), extra_models=extra)
    stage2(run, ref, prof_cfg)


if __name__ == "__main__":
    main()
