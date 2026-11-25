# -*- coding: utf-8 -*-
"""为「模型版 EBA」造伪实例表：同几何 × 58 个码本方向，各占一个实例号。

为什么需要：渲染器按实例的 `w_label` 取 w，没有命令行直接改 w 的开关。于是把
(base 实例, 标签) 展开成一个个伪实例（id 从 1000 起），几何/朝向/频率全部复制
base，只换 `w_label`。渲染完用映射表把 pseudo id 折回 (base, label)。

同时把 A0 扫描已经生成的真值灰度图复制成 `<out_gt>/inst<pseudo>/<view>.png`，
这样 `--export_gt` 能用**同一张 jet LUT** 导出色图，和模型渲染逐域可比。

跑法（服务器）：
    python code/tools/data/_make_sweep_meta.py \
        --sweep results/_beam_sweep --meta datasets/RF-3DGS_mvdr_w_params/meta.json \
        --out_meta results/_scale_arm/sweep_meta.json \
        --out_gt results/_scale_arm/sweep_gt --bases 0 318
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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import w_codebook  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_gt", required=True)
    ap.add_argument("--bases", type=int, nargs="+", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.meta, encoding="utf-8"))
    by_id = {int(i["id"]): i for i in meta["instances"]}
    labels = w_codebook.train_labels() + w_codebook.holdout_labels()
    out_gt = Path(args.out_gt)
    rows, mapping = [], {}
    pid = 1000
    for b in args.bases:
        if b not in by_id:
            raise SystemExit(f"[FAIL] meta 里没有实例 {b}")
        views = sorted({int(json.loads(l)["view"])
                        for p in glob.glob(f"{args.sweep}/scales_*_inst{b:03d}.jsonl")
                        for l in open(p, encoding="utf-8") if l.strip()})
        if not views:
            raise SystemExit(f"[FAIL] 找不到实例 {b} 的 scales")
        for lab in labels:
            hit = glob.glob(f"{args.sweep}/lab??_{lab}/inst{b:03d}")
            if not hit:
                raise SystemExit(f"[FAIL] 缺扫描输出 {lab}/inst{b:03d}")
            src_dir = hit[0]
            r = dict(by_id[b])
            r.update(id=pid, role="sweep", views="test", w_label=lab)
            rows.append(r)
            mapping[str(pid)] = {"base": b, "label": lab}
            for v in views:
                src = Path(src_dir) / f"{v:05d}.png"
                dst = out_gt / f"inst{pid:03d}" / f"{v:05d}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.copyfile(src, dst)
            pid += 1
    json.dump({"instances": rows}, open(args.out_meta, "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(mapping, open(Path(args.out_meta).with_name("sweep_map.json"),
                            "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(rows)} 个伪实例（{len(args.bases)} base × {len(labels)} 标签）；"
          f"GT 复制到 {out_gt}")


if __name__ == "__main__":
    main()
