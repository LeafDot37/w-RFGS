# -*- coding: utf-8 -*-
"""w′ 峰值上界比值（实例级聚合 + 分位数）。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
import sys
from pathlib import Path

import numpy as np

R = Path('/root/autodl-tmp/rf'); sys.path.insert(0, str(R / 'code'))
from rfgs.csi.train_csi_cbf import CBFGrid

P = R / 'results/_probe_precoder'
AX = {'Position': [144, 152, 160, 168, 176, 184],
      'Yaw': [196, 202, 208, 214, 220, 226],
      'Pitch': [232, 238, 244, 250, 256, 262],
      'Carrier': [268, 274, 280, 286, 292, 298],
      'Joint': [304, 306, 308, 310, 312, 314],
      'Weights': list(range(318, 354))}
DIRS = {'Position': 'pos', 'Yaw': 'yaw', 'Pitch': 'pitch', 'Carrier': 'freq', 'Joint': 'double'}
grid = CBFGrid(); A = grid.a_mat.cpu().numpy()


def wdir(iid):
    if iid >= 318:
        return R / f'results/_wprime_pv_bisg3sup36/inst{iid:03d}'
    for k, ids in AX.items():
        if iid in ids:
            return R / f'results/_wprime_axis/{DIRS[k]}/inst{iid:03d}'
    raise KeyError(iid)


out = {}; all_inst = []; all_views = []
for block, ids in AX.items():
    inst = []
    for iid in ids:
        z = np.load(P / f'inst{iid:03d}.npz')
        pv = json.load(open(wdir(iid) / 'perview.json'))
        vals = []
        for r, vid in enumerate(z['names']):
            val = z['valid'][r]
            C = z['a'][r][:, :, val].sum(axis=2)
            D = C.conj().T @ A
            up = float((np.abs(D) ** 2).sum(axis=0).max())
            if up <= 1e-18:
                continue
            v = np.asarray(pv[str(int(vid))], float)
            w = v[:16] + 1j * v[16:]
            vals.append(float((np.abs(w.conj() @ D) ** 2).max() / up))
        inst.append(float(np.mean(vals))); all_views += vals
    arr = np.array(inst)
    out[block] = dict(n_inst=len(ids), mean=float(arr.mean()), std=float(arr.std()),
                      rows=inst)
    all_inst += inst
    print(block, out[block]['mean'], out[block]['std'], flush=True)
a = np.array(all_inst); vv = np.array(all_views)
out['All'] = dict(n_inst=66, mean=float(a.mean()), std=float(a.std()),
                  p5=float(np.percentile(vv, 5)), p50=float(np.median(vv)),
                  p95=float(np.percentile(vv, 95)), vmin=float(vv.min()),
                  vmax=float(vv.max()), rows=all_inst)
print('All', out['All'])
json.dump(out, open(R / 'results/_wprime_axis/upper_ratio_peak2.json', 'w'), indent=1)
