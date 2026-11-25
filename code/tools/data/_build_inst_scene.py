# -*- coding: utf-8 -*-
"""为一个单发射源实例构造原版 RF-3DGS 训练用场景目录。

- images/      指向 RF-3DGS_cbf_params/spectra/instXXX 的符号链接
- sparse/0/    cameras.txt、points3D.ply 复制；images.txt 只保留该实例存在的视角
- train/test_index.txt  按排序后 4:1 划分（留出块只有 640 个 test 视角）

用法（服务器）：
    python code/tools/data/_build_inst_scene.py --ids 144 148 152 156
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import shutil
from pathlib import Path

R = Path("/root/autodl-tmp/rf")
SPC = R / "datasets/RF-3DGS_cbf_params/spectra"
BASE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
OUT = R / "datasets/_perinst"


def build(iid: int):
    src = SPC / f"inst{iid:03d}"
    names = sorted(p.name for p in src.glob("*.png"))
    if not names:
        print(f"[warn] inst{iid} 没有图")
        return
    dst = OUT / f"inst{iid:03d}"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "sparse/0").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for n in names:
        out = dst / "images" / n
        if not out.exists():
            Image.open(src / n).convert("RGB").save(out)

    shutil.copy(BASE / "sparse/0/cameras.txt", dst / "sparse/0/cameras.txt")
    ply = BASE / "sparse/0/points3D.ply"
    if not (dst / "sparse/0/points3D.ply").exists():
        (dst / "sparse/0/points3D.ply").symlink_to(ply)

    want = set(names)
    lines = (BASE / "sparse/0/images.txt").read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[-1] in want:
            out.append(line)
            if i + 1 < len(lines):
                out.append(lines[i + 1])   # 2D 点那一行
            i += 2
        else:
            i += 1
    (dst / "sparse/0/images.txt").write_text("\n".join(out) + "\n")

    test = names[::5]
    train = [n for n in names if n not in set(test)]
    (dst / "train_index.txt").write_text("\n".join(train) + "\n")
    (dst / "test_index.txt").write_text("\n".join(test) + "\n")
    print(f"inst{iid}: images={len(names)} train={len(train)} test={len(test)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for i in args.ids:
        build(i)


if __name__ == "__main__":
    main()
