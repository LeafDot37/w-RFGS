# -*- coding: utf-8 -*-
"""G1 第三刀：尺度可学性探针（纯 CPU，不训练主模型）。

任务：由 (发射源参数, 视角几何) 预测 10log10(pmax − pmin)。
两种切分：
  A) 留一实例（= 泛化到新发射源）——论文真正关心的问题；
  B) 实例内留出 25% 视角（= 头在训练时会面对的内插问题）。
模型：岭回归（线性）+ kNN（非线性，检验「是否只是非线性不足」）。
判据：留出残差 std < 2 dB ⇒ 值得接尺度头。
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
D = ROOT / "results/_scale_super"

poses = read_colmap_poses(POSE_ROOT)
test_ids = [int(x) for x in (POSE_ROOT / "test_index.txt").read_text().split()][:64]
insts = {i["id"]: i for i in json.load(open(
    ROOT / "datasets/RF-3DGS_cbf_params/instances.json", encoding="utf-8"))["instances"]}

X, y, g = [], [], []
skipped = [0]
for f in sorted(glob.glob(str(D / "scales_inst*.jsonl"))):
    iid = int(Path(f).stem.split("inst")[1])
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if len(recs) != 64:
        continue
    ins = insts[iid]
    xyz = np.array(ins["xyz"], float)
    yaw, pit, fq = (float(ins["yaw_abs_rad"]), float(ins["pitch_abs_rad"]),
                    float(ins["freq_hz"]) / 1e9)
    bore = np.array([math.cos(pit) * math.cos(yaw), math.cos(pit) * math.sin(yaw),
                     math.sin(pit)])
    for r in recs:
        if r["pmax"] - r["pmin"] < 1e-12:      # 退化视角（整幅为常量），尺度无意义
            skipped[0] += 1
            continue
        c = np.array(poses[int(r["view"])][0], float)
        dv = c - xyz
        d = float(np.linalg.norm(dv))
        u = dv / max(d, 1e-6)
        L = 10 * math.log10(max(r["pmax"] - r["pmin"], 1e-30))
        feat = [np.log10(d), *u, float(u @ bore), *(xyz / 8.44),
                math.sin(yaw), math.cos(yaw), math.sin(pit), math.cos(pit), fq]
        X.append(feat); y.append(L); g.append(iid)

X, y, g = np.array(X), np.array(y), np.array(g)


def ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return B @ w


def knn(Xtr, ytr, Xte, k=8):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, 1)[:, :k]
    return ytr[idx].mean(1)


def report(tag, pred, truth):
    e = pred - truth
    print(f"{tag:<34}{e.std(ddof=1):>9.2f}{np.abs(e).mean():>10.2f}"
          f"{np.percentile(np.abs(e), 90):>9.2f}")


print(f"{'切分 / 模型':<34}{'残差 std':>9}{'MAE':>10}{'P90':>9}")
# A) 留一实例
for name, fn in (("岭回归", ridge), ("kNN(k=8)", knn)):
    errs = []
    for iid in np.unique(g):
        m = g == iid
        errs.append(fn(X[~m], y[~m], X[m]) - y[m])
    e = np.concatenate(errs)
    print(f"{'A 留一实例 · ' + name:<34}{e.std(ddof=1):>9.2f}"
          f"{np.abs(e).mean():>10.2f}{np.percentile(np.abs(e), 90):>9.2f}")

# B) 实例内留出 25%
rng = np.random.default_rng(0)
te = np.zeros(len(y), bool)
for iid in np.unique(g):
    idx = np.where(g == iid)[0]
    te[rng.choice(idx, max(4, len(idx) // 4), replace=False)] = True
for name, fn in (("岭回归", ridge), ("kNN(k=8)", knn)):
    e = fn(X[~te], y[~te], X[te]) - y[te]
    print(f"{'B 实例内留出 · ' + name:<34}{e.std(ddof=1):>9.2f}"
          f"{np.abs(e).mean():>10.2f}{np.percentile(np.abs(e), 90):>9.2f}")

# 基线：逐实例常数
e = np.concatenate([y[g == i] - y[g == i].mean() for i in np.unique(g)])
print(f"{'基线 逐实例常数':<34}{e.std(ddof=1):>9.2f}{np.abs(e).mean():>10.2f}"
      f"{np.percentile(np.abs(e), 90):>9.2f}")
print(f"\n样本 {len(y)} 条 / {len(np.unique(g))} 个实例；"
      f"总 std {y.std(ddof=1):.2f} dB")
