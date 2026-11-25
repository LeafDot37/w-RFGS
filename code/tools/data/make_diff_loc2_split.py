# -*- coding: utf-8 -*-
"""第二套漫散射位置留出划分：测试 = 前 4 个位置（16 视角），训练 = 后 12 个。

用于复核 §diffuse 的"48 视角逐场景 3DGS 低于均值图地板"不是单次划分的偶然。
第一套划分（`_diff_rf_loc`）留出的是最后 4 个位置。
"""
from __future__ import annotations

import os
from pathlib import Path

R = Path("/root/autodl-tmp/rf")
SRC = R / "results/_diff_rf_split"
DST = R / "results/_diff_rf_loc2"
IDS = [144, 196, 232, 268, 304, 318]
SPARSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100/sparse"


def main() -> None:
    for i in IDS:
        src = SRC / f"inst{i:03d}"
        out = DST / f"inst{i:03d}"
        out.mkdir(parents=True, exist_ok=True)
        names = sorted({*open(src / "train_index.txt").read().split(),
                        *open(src / "test_index.txt").read().split()})
        if len(names) != 64:
            raise RuntimeError(f"inst{i}: {len(names)} != 64")
        quads = [names[k:k + 4] for k in range(0, 64, 4)]
        test = [n for q in quads[:4] for n in q]
        train = [n for q in quads[4:] for n in q]
        (out / "train_index.txt").write_text("\n".join(train) + "\n")
        (out / "test_index.txt").write_text("\n".join(test) + "\n")
        for name, target in (("images", src / "images"), ("sparse", SPARSE)):
            p = out / name
            if p.is_symlink() or p.exists():
                p.unlink()
            os.symlink(str(target), str(p))
        print(f"inst{i:03d}: train={len(train)} test={len(test)} "
              f"test_first={test[:4]}", flush=True)


if __name__ == "__main__":
    main()
