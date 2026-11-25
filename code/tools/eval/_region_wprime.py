# -*- coding: utf-8 -*-
"""从 CSI 头的场合成 H，求区域平均的连续 w′ = v1(Σ HᴴH)。

输出 32 个实数（16 Re + 16 Im），供 gen_cbf_params.py --w_vector 生成真值。
"""
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
from rfgs.csi.train_csi_cbf import CBFGrid, rot_z_t, tr38901_t               # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                     # noqa: E402
from rfgs.rrf.train_rrf_angular import cam_yaw                                 # noqa: E402
from sim import w_codebook                                          # noqa: E402
from sim.w_codebook import element_positions                        # noqa: E402

C = 299792458.0
META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
POSE = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
CKPT = R / "results/_csi_cbfw/arm120k/csi_120000.pth"
SEED = R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"
IID = int(sys.argv[1]) if len(sys.argv) > 1 else 318
NVIEW = 64
NR = NC = 4
OUT = R / f"results/_csi_cbfw/wprime_region_inst{IID}.json"

dev = "cuda"
meta = json.load(open(META, encoding="utf-8"))
inst = [i for i in meta["instances"] if int(i["id"]) == IID][0]
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
yt, zt = element_positions(NR, NC)
yt = torch.as_tensor(yt, dtype=torch.float32, device=dev)
zt = torch.as_tensor(zt, dtype=torch.float32, device=dev)


@torch.no_grad()
def render_H(cam, chunk=4096):
    xyz = g.get_xyz
    P = xyz.shape[0]
    opac = torch.sigmoid(g._opacity).squeeze(-1)
    tx = torch.as_tensor(inst["xyz"], dtype=torch.float32, device=dev)
    R_tx = torch.as_tensor(w_codebook.tx_orientation_matrix(
        float(inst["yaw_abs_rad"]), float(inst["pitch_abs_rad"])),
        dtype=torch.float32, device=dev)
    R_rx = rot_z_t(cam_yaw(cam)).to(dev)
    cc = cam.camera_center.reshape(1, 3)
    tx_yaw = float(inst.get("offset_yaw_rad", 0.0))
    tx_pitch = float(inst.get("offset_pitch_rad", 0.0))
    f_hz = float(inst["freq_hz"])
    H = torch.zeros(100, NR * NC, T, dtype=torch.complex64, device=dev)
    for s0 in range(0, P, chunk):
        sl = slice(s0, min(s0 + chunk, P))
        pos = xyz[sl]
        s = head(embed[sl], torch.zeros_like(pos), pos, None, tx=tx,
                 yaw=None, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                 tx_freq=f_hz, tx_rot=R_tx, tx_w=None)
        zc = torch.complex(s[:, 0], s[:, 1]) * opac[sl]
        d_tx = pos - tx
        r_tx = d_tx.norm(dim=1).clamp_min(1e-6)
        u_in = d_tx / r_tx.unsqueeze(1)
        d_rx = pos - cc
        r_rx = d_rx.norm(dim=1).clamp_min(1e-6)
        u_out = d_rx / r_rx.unsqueeze(1)
        tau_ns = (r_tx + r_rx) / C * 1e9
        tap = torch.floor(tau_ns / dt).long().clamp(0, T - 1)
        u_in_l = u_in @ R_tx
        th_t = torch.arccos(u_in_l[:, 2].clamp(-1.0, 1.0))
        ph_t = torch.atan2(u_in_l[:, 1], u_in_l[:, 0])
        ph = 2 * math.pi * (yt[None, :] * u_in_l[:, 1:2]
                            + zt[None, :] * u_in_l[:, 2:3])
        A_t = tr38901_t(th_t, ph_t)[:, None] * torch.exp(1j * ph)
        u_out_l = u_out @ R_rx
        v, w_ = u_out_l[:, 1], u_out_l[:, 2]
        th_r = torch.arccos(w_.clamp(-1.0, 1.0))
        ph_r = torch.atan2(v, torch.sqrt((1.0 - w_ * w_).clamp_min(0.0)) + 1e-12)
        g_rx = tr38901_t(th_r, ph_r)
        phase = 2 * math.pi * (grid.y.to(dev)[None, :] * v[:, None]
                               + grid.z.to(dev)[None, :] * w_[:, None])
        A_r = g_rx[:, None] * torch.exp(1j * phase)
        zz = zc * torch.exp(-1j * 2 * math.pi * f_hz * tau_ns * 1e-9)
        contrib = zz[:, None, None] * A_r[:, :, None] * A_t[:, None, :]
        H.index_add_(2, tap, contrib.permute(1, 2, 0))
    return H


G = torch.zeros(NR * NC, NR * NC, dtype=torch.complex64, device=dev)
per_view = []
for vn in views:
    H = render_H(cams[vn])
    Gv = torch.einsum("nkt,nlt->kl", H.conj(), H)
    G += Gv
    evals, evecs = torch.linalg.eigh(Gv)
    per_view.append(evecs[:, -1].detach().cpu().numpy())
G = G / len(views)
evals, evecs = torch.linalg.eigh(G)
w = evecs[:, -1].detach().cpu().numpy()
w = w / np.linalg.norm(w)
# 逐视角 w′ 与区域 w′ 的余弦
cos = [abs(np.vdot(w, p)) for p in per_view]
print("region w' norm", np.linalg.norm(w), "eig ratio",
      float(evals[-1].real / evals[-2].real))
print("per-view vs region cosine: mean %.3f min %.3f" % (np.mean(cos), np.min(cos)))
vec = list(w.real.astype(float)) + list(w.imag.astype(float))
json.dump({"inst": IID, "views": len(views), "w": vec,
           "cos_mean": float(np.mean(cos)), "cos_min": float(np.min(cos)),
           "eig_ratio": float(evals[-1].real / evals[-2].real)},
          open(OUT, "w", encoding="utf-8"))
print("VECTOR", ",".join("%.8f" % x for x in vec))
print("wrote", OUT)
