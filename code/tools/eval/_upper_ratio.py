# -*- coding: utf-8 -*-
"""w′ 的真实上界比值：w′ᴴRw′ / λ_max(R)，R 由逐阵元 GT 路径系数构成。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
from pathlib import Path

import numpy as np

R = Path('/root/autodl-tmp/rf'); P = R / 'results/_probe_precoder'
AX = {'Position': [144, 152, 160, 168, 176, 184],
      'Yaw': [196, 202, 208, 214, 220, 226],
      'Pitch': [232, 238, 244, 250, 256, 262],
      'Carrier': [268, 274, 280, 286, 292, 298],
      'Joint': [304, 306, 308, 310, 312, 314],
      'Weights': list(range(318, 354))}
DIRS = {'Position': 'pos', 'Yaw': 'yaw', 'Pitch': 'pitch', 'Carrier': 'freq', 'Joint': 'double'}


def wdir(iid):
    if iid >= 318:
        return R / f'results/_wprime_pv_bisg3sup36/inst{iid:03d}'
    for k, ids in AX.items():
        if iid in ids:
            return R / f'results/_wprime_axis/{DIRS[k]}/inst{iid:03d}'
    raise KeyError(iid)


out = {}
allr = []
for block, ids in AX.items():
    vals = []
    for iid in ids:
        z = np.load(P / f'inst{iid:03d}.npz')
        pv = json.load(open(wdir(iid) / 'perview.json'))
        for r, vid in enumerate(z['names']):
            val = z['valid'][r]
            a = z['a'][r][:, :, val]              # (100,16,Pv)
            B = a.transpose(0, 2, 1).reshape(-1, 16)
            Rm = B.conj().T @ B
            lam = float(np.linalg.eigvalsh(Rm)[-1])
            if lam <= 1e-12:
                continue
            v = np.asarray(pv[str(int(vid))], float)
            w = v[:16] + 1j * v[16:]
            vals.append(float((w.conj() @ Rm @ w).real / lam))
    arr = np.array(vals)
    out[block] = dict(n_inst=len(ids), views=len(arr), mean=float(arr.mean()), std=float(arr.std()))
    allr += vals
    print(block, out[block], flush=True)
a = np.array(allr)
out['All'] = dict(n_inst=66, views=len(a), mean=float(a.mean()), std=float(a.std()))
print('All', out['All'])
json.dump(out, open(R / 'results/_wprime_axis/upper_ratio.json', 'w'), indent=1)
