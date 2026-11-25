# -*- coding: utf-8 -*-
"""评绝对尺度：A2+scale 检查点在留出位置上的 pred pmax vs 真值 pmax。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

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
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw             # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
POSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "results/_csi_cbfw/armA2_scale120k/csi_120000.pth"
SEED = R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"
SWEEP = R / "results/_cbf_beam_sweep"
IDS = [318, 321, 324]
NVIEW = 64

dev = "cuda"
meta = json.load(open(META, encoding="utf-8"))
by = {int(i["id"]): i for i in meta["instances"]}
cams, _, test = load_cameras(str(POSE))
views = test[:NVIEW]
g = GaussianModel(3)
load_visual_geometry(str(SEED), g, 3)
head, embed, op, cfg, it = load_tx_state(str(CKPT), device=dev)
if op is not None:
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(),
                                    requires_grad=False)
grid = CBFGrid()
T = int(cfg.get("csi_T", 128)); dt = float(cfg.get("csi_dt_ns", 0.5))
ls = float(sys.argv[2]) if len(sys.argv) > 2 else float(cfg.get("log_scale", 0.0))
print("checkpoint", it, "log_scale", ls)

for iid in IDS:
    inst = by[iid]
    lab = str(inst["w_label"])
    if cfg.get("use_tx_w_complex"):
        _wv = w_codebook.vector_for(lab, 4, 4)
        w_feat = list(np.concatenate([_wv.real, _wv.imag]).astype(float))
    else:
        _tp = w_codebook.parse_label(lab)
        w_feat = ([0.0, 0.0, 1.0] if _tp is None
                  else [float(_tp[0]), float(_tp[1]), 0.0])
    wv = torch.as_tensor(w_codebook.uniform_weight(4, 4),
                         dtype=torch.complex64, device=dev)
    truth = {}
    p = SWEEP / f"scales_{lab}_inst{iid:03d}.jsonl"
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        truth[int(r["view"])] = float(r["pmax"])
    preds, trues = [], []
    with torch.no_grad():
        for vn in views:
            vid = int(vn)
            if vid not in truth:
                continue
            x = render_cir(head, embed, g, cams[vn], inst, T, dt, grid,
                           dtau=None, w_vec=wv, tx_nr=4, tx_nc=4, w_feat=w_feat)
            sp = spec_raw(x, grid) * math.exp(ls)
            preds.append(float(sp.max()))
            trues.append(truth[vid])
    pr = np.array(preds); tr = np.array(trues)
    ok = (pr > 0) & (tr > 0)
    pr, tr = pr[ok], tr[ok]
    alpha = np.sqrt(tr / pr)
    print(f"inst{iid} n={len(pr)} |alpha| mean {alpha.mean():.3e} "
          f"cv {alpha.std()/alpha.mean():.3f} "
          f"log RMSE {np.sqrt(np.mean((np.log(pr)-np.log(tr))**2)):.3f} "
          f"spearman {__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(pr, tr).correlation:.3f}")
