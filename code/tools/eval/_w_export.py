# -*- coding: utf-8 -*-
"""导出波束权 w 轴的全部测量数据，供本机出图用（服务器无中文字体，图必须回本机画）。

导出三块：

1. **逐实例三指标**：四个组合（CBF/MVDR × 波束条件臂/波束盲臂）的 `test_inst/metrics.txt`
   原样搬进 JSON，外加 w 置换那次（`test_wswap/metrics.txt`）。
2. **训练集内探针的两两 PSNR**：同位置 4 个训练实例，GT 之间与模型之间的平均 PSNR。
   这是「表示层还是泛化层」的判据，必须在服务器上算（10240 张图不搬）。
3. **定性对照的 4 张图 × 2 个算子**：取留出 w 组里 PSNR 居中的实例、误差居中的视角，
   拷出 GT / 条件臂渲染 / 条件臂换 w 后渲染 / 盲臂渲染。选法固定可复现。

用法（服务器）::

    /root/miniconda3/bin/python code/tools/eval/_w_export.py \
        --run /root/autodl-tmp/rf/results/run_20261029_服务器4090D_波束权泛化 \
        --out /root/autodl-tmp/rf/results/run_20261029_服务器4090D_波束权泛化/_小件
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import itertools
import json
import shutil
import statistics as st
from pathlib import Path

import numpy as np
from PIL import Image

COMBOS = [
    # (键, 目录, 算子, 臂的文字名)
    ("MVDR/w", "训练180k/w", "MVDR", "波束条件臂"),
    ("MVDR/blind", "训练180k/blind", "MVDR", "波束盲臂"),
    ("CBF/w", "训练CBF60k/w", "CBF", "波束条件臂"),
    ("CBF/blind", "训练CBF60k/blind", "CBF", "波束盲臂"),
]


def read_metrics(p: Path):
    """metrics.txt -> [[id, role, n, psnr, ssim, lpips], ...]（跳过表头与坏行）。"""
    rows = []
    if not p.exists():
        return rows
    for ln in p.read_text(encoding="utf-8").splitlines():
        f = ln.split()
        if len(f) != 6 or f[0] == "id":
            continue
        try:
            rows.append([int(f[0]), f[1], int(f[2]),
                         float(f[3]), float(f[4]), float(f[5])])
        except ValueError:
            continue
    return rows


def psnr01(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def load_gray(p: Path):
    """统一成 [0,1] 浮点灰度（与 `_w_swap_report` 同口径，避免 255 与 1 混用）。"""
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def probe_block(root: Path):
    """训练集 4 个实例的两两 PSNR：GT 之间 vs 模型之间。"""
    d = root / "test_trainprobe"
    insts = sorted(x for x in d.glob("inst*") if x.is_dir())
    if len(insts) != 4:
        return {"ok": False, "reason": f"探针实例 {len(insts)} 个，应为 4"}
    gts, rds = [], []
    for i in insts:
        g = sorted((i / "gt").glob("*.png"))
        r = sorted((i / "renders").glob("*.png"))
        if len(g) != len(r) or not g:
            return {"ok": False, "reason": f"{i.name} GT {len(g)} vs 渲染 {len(r)}"}
        # 帧数多时按步长抽样，控制内存；抽样只影响均值精度，不影响结论方向
        step = max(1, len(g) // 64)
        gts.append([load_gray(p) for p in g[::step]])
        rds.append([load_gray(p) for p in r[::step]])
    n = min(len(x) for x in gts)
    gp, rp = [], []
    for i, j in itertools.combinations(range(4), 2):
        gp.append(st.mean(psnr01(a, b) for a, b in zip(gts[i][:n], gts[j][:n])))
        rp.append(st.mean(psnr01(a, b) for a, b in zip(rds[i][:n], rds[j][:n])))
    return {"ok": True, "pairs_gt": gp, "pairs_model": rp,
            "mean_gt": st.mean(gp), "mean_model": st.mean(rp),
            "frames_per_inst": n,
            # 模型表达的 w 变化能量占 GT 的比例（越小越说明表达不出来）
            "energy_ratio": 10 ** (-(st.mean(rp) - st.mean(gp)) / 10.0)}


def pick_qual(root: Path, blind_root: Path):
    """选实例与视角：留出 w 组里 PSNR 居中者 + 该实例内误差居中的帧。"""
    m = read_metrics(root / "test_inst" / "metrics.txt")
    hold = [r for r in m if r[1] == "w_holdout"]
    if not hold:
        return None
    hold.sort(key=lambda r: r[3])
    rec = hold[len(hold) // 2]
    iid = f"inst{rec[0]:03d}"
    d = root / "test_inst" / iid
    g = sorted((d / "gt").glob("*.png"))
    r = sorted((d / "renders").glob("*.png"))
    errs = [float(np.mean((load_gray(a) - load_gray(b)) ** 2)) for a, b in zip(g, r)]
    k = int(np.argsort(errs)[len(errs) // 2])
    out = {"instance": rec[0], "inst_dir": iid, "frame": g[k].name,
           "as_is_psnr": rec[3]}
    for name, sub in (
        ("gt", d / "gt" / g[k].name),
        ("render", d / "renders" / r[k].name),
        ("wswap", root / "test_wswap" / iid / "renders" / g[k].name),
    ):
        if sub.exists():
            out[name] = str(sub)
    # 盲臂同一实例同一帧（用于「条件臂 vs 盲臂」并排）
    p = blind_root / "test_inst" / iid / "renders" / g[k].name
    if p.exists():
        out["blind_render"] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    (out / "qual").mkdir(parents=True, exist_ok=True)

    payload = {"combos": {}, "qual": {}}
    for key, sub, op, arm_name in COMBOS:
        root = run / sub
        blk = {
            "operator": op, "arm": arm_name, "dir": sub,
            "metrics": read_metrics(root / "test_inst" / "metrics.txt"),
            "wswap_metrics": read_metrics(root / "test_wswap" / "metrics.txt"),
            "probe": probe_block(root),
        }
        payload["combos"][key] = blk
        print(f"[{key}] 实例 {len(blk['metrics'])} 条  wswap {len(blk['wswap_metrics'])} 条  "
              f"探针 {'ok' if blk['probe'].get('ok') else blk['probe'].get('reason')}")
        if blk["probe"].get("ok"):
            print(f"        GT 之间 {blk['probe']['mean_gt']:.2f} dB   "
                  f"模型之间 {blk['probe']['mean_model']:.2f} dB   "
                  f"能量比 {blk['probe']['energy_ratio'] * 100:.1f}%")

    # 定性对照：每个算子取条件臂的同一实例/视角，盲臂配同一张
    for op, w_dir, b_dir in (("MVDR", "训练180k/w", "训练180k/blind"),
                             ("CBF", "训练CBF60k/w", "训练CBF60k/blind")):
        q = pick_qual(run / w_dir, run / b_dir)
        if not q:
            print(f"[qual {op}] 没有 w_holdout 读数，跳过")
            continue
        dst = out / "qual" / op
        dst.mkdir(parents=True, exist_ok=True)
        kept = {}
        for k in ("gt", "render", "wswap", "blind_render"):
            if k in q:
                s = Path(q[k])
                if s.exists():
                    shutil.copy2(s, dst / f"{k}.png")
                    kept[k] = f"{k}.png"
        q["copied"] = kept
        q.pop("gt", None); q.pop("render", None)
        q.pop("wswap", None); q.pop("blind_render", None)
        payload["qual"][op] = q
        print(f"[qual {op}] 实例 {q['instance']} 帧 {q['frame']}  拷出 {sorted(kept)}")

    (out / "数据.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写] {out / '数据.json'}")


if __name__ == "__main__":
    main()
