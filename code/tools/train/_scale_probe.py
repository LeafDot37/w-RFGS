# -*- coding: utf-8 -*-
"""Phase B2 探针：绝对尺度能否由 (tx, 相机, 预测峰值) 的 view-conditioned 回归修正？

训练：16 个 4×4 训练实例 × 64 视角的 (预测峰值, 真值 pmax)；
测试：inst318 的 64 视角（未见配置/未见 w）。
报告：修正前后 |α| 的 CV、Spearman(pred,true)、log-RMSE。
"""
from __future__ import annotations
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
import torch

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from sim import w_codebook                                          # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir                       # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402
from rfgs.rrf.train_rrf_angular import cam_yaw                               # noqa: E402

META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
POSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
CKPT = R / "results/_csi_cbfw/arm120k/csi_120000.pth"
SEED = R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"
TRAIN_IDS = list(range(16))
TEST_ID = 318

dev = "cuda"
meta = json.load(open(META, encoding="utf-8"))
by = {int(i["id"]): i for i in meta["instances"]}
cams, _, test = load_cameras(str(POSE))
views = test[:64]
g = GaussianModel(3)
load_visual_geometry(str(SEED), g, 3)
head, embed, op, cfg, it = load_tx_state(str(CKPT), device=dev)
if op is not None:
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                    requires_grad=False)
grid = CBFGrid()
T = int(cfg.get("csi_T", 128)); dt = float(cfg.get("csi_dt_ns", 0.5))


def true_pmax(iid):
    out = {}
    p = R / f"results/_scale_targets/scales_inst{iid:03d}.jsonl"
    if not p.is_file():
        p = R / f"results/_cbf_beam_sweep/scales_{by[iid]['w_label']}_inst{iid:03d}.jsonl"
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        out[int(r["view"])] = float(r["pmax"])
    return out


def feats(inst, cam, logpred):
    y, p = float(inst["yaw_abs_rad"]), float(inst["pitch_abs_rad"])
    c = cam.camera_center.cpu().numpy().reshape(3)
    cy = cam_yaw(cam)
    return [*inst["xyz"], math.sin(y), math.cos(y), math.sin(p), math.cos(p),
            float(inst["freq_hz"]) / 1e9,
            *c, math.sin(cy), math.cos(cy), logpred]


rows = []
for iid in TRAIN_IDS + [TEST_ID]:
    inst = by[iid]
    wv = torch.as_tensor(w_codebook.vector_for(str(inst["w_label"]), 4, 4),
                         dtype=torch.complex64, device=dev)
    tr = true_pmax(iid)
    with torch.no_grad():
        for vn in views:
            vid = int(vn)
            if vid not in tr:
                continue
            x = render_cir(head, embed, g, cams[vn], inst, T, dt, grid,
                           dtau=None, w_vec=wv, tx_nr=4, tx_nc=4)
            sp = ((x.conj().sum(1)) @ grid.a_mat.to(dev)).abs()
            pred = float((sp ** 2).max())
            rows.append(dict(iid=iid, view=vid, pred=pred, true=tr[vid],
                             f=feats(inst, cams[vn], math.log(max(pred, 1e-30)))))
    print("done", iid, flush=True)

rows = [r for r in rows if r["true"] > 0 and r["pred"] > 0]
train = [r for r in rows if r["iid"] in TRAIN_IDS]
testr = [r for r in rows if r["iid"] == TEST_ID]
X = np.array([r["f"] for r in train], float)
y = np.log(np.array([r["true"] / r["pred"] for r in train]))
print("diag X nan", int(np.isnan(X).sum()), "y nan", int(np.isnan(y).sum()),
      "y inf", int(np.isinf(y).sum()), "y min/max",
      float(np.nanmin(y)), float(np.nanmax(y)), flush=True)
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd
Xs = np.c_[np.ones(len(Xs)), Xs]
w, *_ = np.linalg.lstsq(Xs, y, rcond=None)

Xt = (np.array([r["f"] for r in testr]) - mu) / sd
Xt = np.c_[np.ones(len(Xt)), Xt]
corr = Xt @ w
pred = np.array([r["pred"] for r in testr]) * np.exp(corr)
true = np.array([r["true"] for r in testr])
alpha = np.sqrt(true / pred)
alpha0 = np.sqrt(true / np.array([r["pred"] for r in testr]))
logtrue = np.log(true); logpred = np.log(pred)
print("test n", len(testr))
print("baseline |alpha| cv", alpha0.std() / alpha0.mean())
print("corrected |alpha| cv", alpha.std() / alpha.mean())
from scipy.stats import spearmanr
print("corrected pmax spearman", float(spearmanr(logpred, logtrue).correlation))
print("log RMSE before", float(np.sqrt(np.mean((np.log(np.array([r["pred"] for r in testr])) - logtrue) ** 2))))
print("log RMSE after", float(np.sqrt(np.mean((logpred - logtrue) ** 2))))
