# -*- coding: utf-8 -*-
"""把 `--save_scale` 生成的逐图归一化谱图，重画成「跨图共同尺度」的版本。

输入：`--raw` 下 gen_mvdr_params 的产物（inst###/*.png + scales_inst###.jsonl）
输出：`--out` 下同布局的 PNG，亮度 = 绝对功率的全局仿射映射：

    sp ≈ pmin + u/255·(pmax − pmin)            # 由 u8 + 该图的 pmin/pmax 反演
    c  = (sp − gmin) / (gmax − gmin)           # gmin/gmax 取全部图的极值

这样同一视角、不同 w 的图**保留相对功率**，这正是逐图 min–max 删掉的东西。
量化误差 ≤ 1/255 的**全局**量程（不是逐图量程），所以弱方向的图会更暗——这是
实验要保留的信息，不是 bug。

跑法（服务器）：
    python code/tools/data/_build_scale_arm.py --raw results/_scale_arm/raw \
        --out results/_scale_arm/common
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="",
                    help="给了就按实例的 position_id **分组**定常数（推荐）：同一位置内"
                         "保留跨 w 的绝对功率差，又不被别的位置拉爆量程；不给则全局")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    rows = []
    for p in sorted(glob.glob(str(raw / "scales_inst*.jsonl"))):
        iid = int(os.path.basename(p).split("inst")[1].split(".")[0])
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows.append((iid, int(r["view"]), float(r["pmin"]), float(r["pmax"])))
    if not rows:
        raise SystemExit(f"[FAIL] {raw} 下没有 scales_inst*.jsonl")

    pos_of = {}
    if args.meta:
        m = json.load(open(args.meta, encoding="utf-8"))
        pos_of = {int(i["id"]): int(i.get("position_id", -1))
                  for i in (m.get("instances") or [])}

    def group_range(iid):
        if not pos_of:
            return (min(r[2] for r in rows), max(r[3] for r in rows))
        pid = pos_of.get(iid, -1)
        sub = [r for r in rows if pos_of.get(r[0], -2) == pid]
        return (min(r[2] for r in sub), max(r[3] for r in sub))

    ranges = {}
    for iid, *_ in rows:
        ranges.setdefault(iid, group_range(iid))
    d = [10 * np.log10(max(r[3] - r[2], 1e-12)) for r in rows]
    g_all = (min(r[2] for r in rows), max(r[3] for r in rows))
    print(f"{len(rows)} 张；全局量程 {10*np.log10(max(g_all[1]-g_all[0],1e-12)):.2f} dB；"
          f"分组常数={'按 position_id' if pos_of else '全局'}")
    print(f"逐图量程中位 {np.median(d):.2f} dB；分组后每张保留的相对量程中位 "
          f"{np.median([d[k] - 10*np.log10(max(ranges[r[0]][1]-ranges[r[0]][0],1e-12)) for k, r in enumerate(rows)]):+.2f} dB")

    for iid, view, pmin, pmax in rows:
        gmin, gmax = ranges[iid]
        span = max(gmax - gmin, 1e-12)
        src = raw / f"inst{iid:03d}" / f"{view:05d}.png"
        if not src.exists():
            print(f"[warn] 缺 {src}")
            continue
        u = np.asarray(Image.open(src).convert("L"), np.float64) / 255.0
        sp = pmin + u * (pmax - pmin)
        c = np.clip((sp - gmin) / span, 0.0, 1.0)
        dst = out / f"inst{iid:03d}" / f"{view:05d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(c * 255).astype(np.uint8)).save(dst)
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
