# -*- coding: utf-8 -*-
"""把逐路径大 npz 压成逐视角 CIR（100xT complex64），供训练时快速读取。"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import sys
from pathlib import Path
import numpy as np
src = Path(sys.argv[1]); out = Path(sys.argv[2]); T = int(sys.argv[3]) if len(sys.argv)>3 else 128
dt = float(sys.argv[4]) if len(sys.argv)>4 else 0.5
for p in sorted(src.glob('inst*.npz')):
    z = np.load(p)
    views = z['views']; blob = {}
    for v in views:
        vn = f'{int(v):05d}'
        a = z[f'v{vn}_a']; tau = z[f'v{vn}_tau']; val = z[f'v{vn}_valid']
        tap = np.clip(np.floor(tau[val] / dt).astype(np.int64), 0, T-1)
        x = np.zeros((a.shape[0], T), np.complex64)
        np.add.at(x, (slice(None), tap), a[:, val])
        blob[f'v{vn}'] = x
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / p.name, **blob)
    print('[pack]', p.name, len(views), 'views')
