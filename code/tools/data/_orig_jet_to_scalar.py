# -*- coding: utf-8 -*-
"""原版 RF-3DGS 数据集适配层：jet RGB CBF 谱图 -> 标量 u（8bit 灰度）+ 单实例 meta。

原版数据（training-rf-spectrum/3dgs_CBF_100）
--------------------------------------------------------------------------------
- 3200 张 300x200 jet 谱图；单发射源 TX=[6.905,0,0.287]，60 GHz，
  TX 1x1 iso / RX 8x8 tr38901（与本仓 4x4/10x10/2-4GHz 的协议不同）；
- 800 个接收位置 x 4 个阵列朝向（教程 `for angle in [-pi/2,0,pi/2,pi]`）；
- train_index.txt 2560 / test_index.txt 640（按位置整组留出）。

本脚本产出
--------------------------------------------------------------------------------
    results/_orig_cbf/scalar/inst000/<image_name>.png   # u = jet^{-1}(rgb)
    results/_orig_cbf/meta.json                         # 单实例，freq=60e9

`jet^{-1}` 用 `splat.color_scalar.make_jet_lut(256)`（与作者 matplotlib jet 逐项
一致），最近邻查表；输出 u∈[0,1] 量化到 8bit。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
from splat.color_scalar import make_jet_lut                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100")
    ap.add_argument("--out", default="results/_orig_cbf")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    dst = out / "scalar" / "inst000"
    dst.mkdir(parents=True, exist_ok=True)

    lut = (make_jet_lut(256).clamp(0, 1).numpy() * 255.0).astype(np.float32)
    names = sorted(p.name for p in (src / "images").glob("*.png"))
    print(f"[orig] {len(names)} images -> {dst}", flush=True)
    for k, name in enumerate(names):
        rgb = np.asarray(Image.open(src / "images" / name).convert("RGB"))
        a = rgb.reshape(-1, 3).astype(np.float32)
        u = np.empty(len(a), np.float32)
        for i in range(0, len(a), 32768):
            b = a[i:i + 32768]
            u[i:i + 32768] = ((b[:, None, :] - lut[None]) ** 2).sum(-1).argmin(1) / 255.0
        Image.fromarray(np.round(u.reshape(rgb.shape[:2]) * 255.0).astype(np.uint8),
                        mode="L").save(dst / name)
        if (k + 1) % 400 == 0:
            print(f"  {k + 1}/{len(names)}", flush=True)

    meta = {
        "format": "orig_rf3dgs_v1",
        "source": "RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100",
        "protocol": {
            "spectrum": "cbf",
            "tx_array": [1, 1],
            "tx_pattern": "iso",
            "polarization": "V",
            "rx_array_M": 8,
            "rx_pattern": "tr38901",
            "frequency_hz": 6.0e10,
            "time_interval_ns": 0.5,
            "train_images": 2560,
            "test_images": 640,
            "single_tx": True,
        },
        "counts": {"instances": 1, "views": len(names)},
        "instances": [{
            "id": 0,
            "position_id": 0,
            "xyz": [6.905, 0.0, 0.287],
            "offset_yaw_rad": 0.0,
            "offset_pitch_rad": 0.0,
            "yaw_abs_rad": 0.0,
            "pitch_abs_rad": 0.0,
            "freq_hz": 6.0e10,
            "role": "train",
            "views": "train",
            "label": "orig",
            "w_label": "uniform",
        }],
        "orig_tx_index": 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print("[orig] meta written", flush=True)


if __name__ == "__main__":
    main()
