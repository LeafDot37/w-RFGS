# -*- coding: utf-8 -*-
"""给「只生成了前 16 个视角」的小数据集造一个可用的 COLMAP 划分目录。

训练器 `load_cameras` 要求 images.txt 里的**每个相机**都落在 train 或 test 索引里，
否则直接报错；而 `--train_view_limit` 取的是 test_index 的前 N 个，未必等于生成器
实际产出的那 N 个（生成器按位置分组迭代）。所以按磁盘上真实存在的图号写索引。

    sparse/           -> 原数据集 sparse/（软链）
    test_index.txt    <- 生成器实际写出的 16 个图号
    train_index.txt   <- 其余全部图号（占位，训练用 test）

跑法（服务器）：
    python code/tools/data/_make_sub_data.py \
        --raw results/_scale_arm/raw --src datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
        --out datasets/_scale_arm/data_sub
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw, src, out = Path(args.raw), Path(args.src), Path(args.out)
    inst0 = sorted(p for p in raw.glob("inst*") if p.is_dir())[0]
    test_ids = sorted(p.stem for p in inst0.glob("*.png"))
    if not test_ids:
        raise SystemExit(f"[FAIL] {inst0} 下没有 png")

    names = []
    for line in open(src / "sparse/0/images.txt", encoding="utf-8"):
        f = line.split()
        if len(f) >= 10:
            names.append(os.path.basename(f[9]).split(".")[0])
    if not names:
        raise SystemExit("[FAIL] 读不到 images.txt")
    rest = [n for n in names if n not in set(test_ids)]
    missing = [n for n in test_ids if n not in set(names)]
    if missing:
        raise SystemExit(f"[FAIL] 生成图号不在 images.txt 里：{missing[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    link = out / "sparse"
    if not link.exists():
        link.symlink_to((src / "sparse").resolve())
    (out / "test_index.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    (out / "train_index.txt").write_text("\n".join(rest) + "\n", encoding="utf-8")
    print(f"test {len(test_ids)} 个：{test_ids[:4]}...{test_ids[-2:]}；train 占位 {len(rest)} 个 -> {out}")


if __name__ == "__main__":
    main()
