# -*- coding: utf-8 -*-
"""把漫散射 48/16 的**按朝向留出**划分改成**按位置留出**，与 RF-3DGS 原始口径一致。

原 `_diff_rf_split` 用 `awk NR%4==1` 取测试 ⇒ 测试集全是同一种阵列朝向（mod4=1），
等于把 90° 扇区整块拿掉；逐高斯 SH 外观模型无法外推未见朝向，结果不可比。

本脚本：64 个视角 = 16 个位置 × 4 朝向。取最后 4 个位置（16 视角）测试，
其余 12 个位置（48 视角）训练。images/sparse 用符号链接，不复制数据。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
from pathlib import Path

R = Path("/root/autodl-tmp/rf")
SRC = R / "results/_diff_rf_split"
DST = R / "results/_diff_rf_loc"
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
            raise RuntimeError(f"inst{i}: 视角数 {len(names)} != 64")
        quads = [names[k:k + 4] for k in range(0, 64, 4)]
        test = [n for q in quads[-4:] for n in q]
        train = [n for q in quads[:-4] for n in q]
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
