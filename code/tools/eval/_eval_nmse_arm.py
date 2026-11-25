# -*- coding: utf-8 -*-
"""修正版：CIR NMSE/MAE（跳过近零 GT）、分块 SE/DBF、纯渲染时延。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json, sys, time
from pathlib import Path

import numpy as np, torch

R = Path('/root/autodl-tmp/rf'); sys.path.insert(0, str(R / 'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar_tx import load_tx_state
from splat.scene import GaussianModel
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir
from rfgs.rrf.train_rrf_scalar import load_visual_geometry
from sim import w_codebook

BLOCKS = {
 'Position': [144, 152, 160, 168, 176, 184],
 'Yaw': [196, 202, 208, 214, 220, 226],
 'Pitch': [232, 238, 244, 250, 256, 262],
 'Carrier': [268, 274, 280, 286, 292, 298],
 'Joint': [304, 306, 308, 310, 312, 314],
 'Weights': [318, 319, 320, 321, 322, 323]}
DIRS = {'Position': 'pos', 'Yaw': 'yaw', 'Pitch': 'pitch', 'Carrier': 'freq', 'Joint': 'double'}
meta = json.load(open(R / 'datasets/RF-3DGS_mvdr_w_params/meta.json', encoding='utf-8'))
by = {int(i['id']): i for i in meta['instances']}
cams, _, test = load_cameras(str(R / 'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
views = test[:64]
g = GaussianModel(3); load_visual_geometry(str(R / 'datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth'), g, 3)
head, embed, op, cfg, it = load_tx_state(str(R / 'results/_csi_cbfw/arm_nmse_ft/csi_10000.pth'), device='cuda')
g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
grid = CBFGrid(); dir_out = str(cfg.get('head_type', '')).endswith('bisg')
Pt, N0, BW = 0.01, 10 ** (-17.4), 1e8


def gt_x(iid, vn):
    root = R / ('results/_cir_w' if iid >= 318 else 'results/_cir_axis')
    z = np.load(root / f'inst{iid:03d}.npz')
    a = z[f'v{vn}_a']; tau = z[f'v{vn}_tau']; val = z[f'v{vn}_valid']
    tap = np.clip(np.floor(tau[val] / 0.5).astype(np.int64), 0, 255)
    x = np.zeros((a.shape[0], 256), np.complex64); np.add.at(x, (slice(None), tap), a[:, val])
    return torch.as_tensor(x, device='cuda')


res = {}; allnm = []; allmae = []; render_t = 0.0; n = 0
with torch.no_grad():
    for block, ids in BLOCKS.items():
        nms, maes, ses = [], [], []
        for iid in ids:
            inst = by[iid]
            wv = torch.as_tensor(w_codebook.vector_for(str(inst['w_label']), 4, 4), dtype=torch.complex64, device='cuda')
            pv = json.load(open(R / f'results/_wprime_pv_bisg3sup36/inst{iid:03d}/perview.json')) if iid >= 318 else \
                 json.load(open(R / f'results/_wprime_axis/{DIRS[block]}/inst{iid:03d}/perview.json'))
            probe = np.load(R / f'results/_probe_precoder/inst{iid:03d}.npz')
            pmap = {int(v): k for k, v in enumerate(probe['names'])}
            for vn in views:
                torch.cuda.synchronize(); t0 = time.time()
                x = render_cir(head, embed, g, cams[vn], inst, 256, 0.5, grid, w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
                torch.cuda.synchronize(); render_t += time.time() - t0
                xg = gt_x(iid, vn)
                if float(xg.abs().max()) < 1e-12:
                    continue
                den = (x.conj() * x).real.sum().clamp_min(1e-12)
                alpha = (x.conj() * xg).sum() / den
                e = xg - alpha * x
                nms.append(float((e.abs() ** 2).sum() / (xg.abs() ** 2).sum()))
                maes.append(float(e.abs().mean() / xg.abs().mean()))
                r = None
                if r is not None:
                    a = probe['a'][r][:, :, probe['valid'][r]]
                    B = a.transpose(0, 2, 1).reshape(-1, 16); Rm = B.conj().T @ B
                    lam = float(np.linalg.eigvalsh(Rm)[-1])
                    if lam > 1e-20:
                        v = np.asarray(pv[str(int(vn))], float); w = v[:16] + 1j * v[16:]
                        G = float((w.conj() @ Rm @ w).real)
                        if G > 0:
                            ses.append(np.log2(1 + Pt * G / (N0 * BW)) / np.log2(1 + Pt * lam / (N0 * BW)))
                n += 1
        res[block] = dict(n=len(nms), nmse=float(np.mean(nms)), mae=float(np.mean(maes)),
                          se=float(np.mean(ses)) if ses else None, se_n=len(ses))
        allnm += nms; allmae += maes
        print(block, res[block], flush=True)
res['All'] = dict(n=len(allnm), nmse=float(np.mean(allnm)), mae=float(np.mean(allmae)),
                  se=float(np.mean([res[b]['se'] for b in BLOCKS if res[b]['se'] is not None])),
                  ms_per_view=1000 * render_t / max(n, 1), params=int(sum(p.numel() for p in head.parameters())))
print('All', res['All'])
json.dump(res, open(R / 'results/_axis_all/nmse_ft.json', 'w'), indent=1)
