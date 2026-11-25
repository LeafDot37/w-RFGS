# -*- coding: utf-8 -*-
"""G1 第二刀：这 6.4 dB 的视角内变化里，有多少能被几何闭式解释？

模型（dB，功率）：
  (a) 逐实例常数
  (b) a + b·log10(d)，d = |C_view − xyz_tx|
  (c) (b) + c·cosθ + e·cos²θ，θ = 视线与发射机视轴夹角
拟合用实例 0/1/144，**在留出实例 317 上评估**。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from sim.spectrum_gen import read_colmap_poses          # noqa: E402

POSE_ROOT = ROOT / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
D = ROOT / "results/_scale_probe"

poses = read_colmap_poses(POSE_ROOT)
test_ids = [int(x) for x in (POSE_ROOT / "test_index.txt").read_text().split()][:64]
insts = {i["id"]: i for i in json.load(open(
    ROOT / "datasets/RF-3DGS_cbf_params/instances.json", encoding="utf-8"))["instances"]}

rows = {}
for f in sorted(glob.glob(str(D / "scales_inst*.jsonl"))):
    iid = int(Path(f).stem.split("inst")[1])
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    L = np.array([10 * math.log10(max(r["pmax"] - r["pmin"], 1e-30)) for r in recs])
    inst = insts[iid]
    xyz = np.array(inst["xyz"], float)
    yaw, pit = float(inst["yaw_abs_rad"]), float(inst["pitch_abs_rad"])
    bore = np.array([math.cos(pit) * math.cos(yaw),
                     math.cos(pit) * math.sin(yaw), math.sin(pit)])
    d = np.array([np.linalg.norm(np.array(poses[i][0], float) - xyz) for i in test_ids])
    u = np.array([(np.array(poses[i][0], float) - xyz) / di
                  for i, di in zip(test_ids, d)])
    cos = np.clip(u @ bore, -1.0, 1.0)
    rows[iid] = (L, np.log10(np.maximum(d, 1e-3)), cos)

TRAIN = [0, 1, 144]
TEST = 317


def design(iid, mode):
    L, ld, c = rows[iid]
    cols = [np.ones_like(L)]
    if mode >= 1:
        cols.append(ld)
    if mode >= 2:
        cols += [c, c ** 2]
    return np.stack(cols, 1), L


print(f"{'模型':<28}{'训练残差 std':>12}{'317 残差 std':>14}{'317 MAE':>10}{'317 P90':>9}")
for mode, name in ((0, "(a) 逐实例常数"), (1, "(b) + log10(d)"),
                   (2, "(c) + cosθ, cos²θ")):
    if mode == 0:
        mu = np.mean([rows[i][0].mean() for i in TRAIN])
        res_tr = np.concatenate([rows[i][0] - rows[i][0].mean() for i in TRAIN])
        pred = mu
    else:
        X = np.concatenate([design(i, mode)[0] for i in TRAIN])
        y = np.concatenate([design(i, mode)[1] for i in TRAIN])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        res_tr = X @ w - y
        Xt, yt = design(TEST, mode)
        pred = Xt @ w
        y = yt
    if mode != 0:
        rt = pred - y
    else:
        rt = rows[TEST][0] - pred
    print(f"{name:<28}{res_tr.std(ddof=1):>12.2f}{rt.std(ddof=1):>14.2f}"
          f"{np.abs(rt).mean():>10.2f}{np.percentile(np.abs(rt), 90):>9.2f}")
if True:
    X = np.concatenate([design(i, 2)[0] for i in TRAIN])
    y = np.concatenate([design(i, 2)[1] for i in TRAIN])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    print("\n(c) 系数 [1, log10 d, cosθ, cos²θ] =", np.round(w, 3))
