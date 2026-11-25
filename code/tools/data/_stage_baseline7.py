# -*- coding: utf-8 -*-
"""7 个代表实例的逐发射源基线数据：64 张 jet 图 + 其余位姿用 1×1 占位。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path('/root/autodl-tmp/rf'); sys.path.insert(0, str(R / 'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar import apply_jet, make_jet_lut
from splat.scene.colmap_loader import read_extrinsics_text

IDS = [144, 196, 232, 268, 304, 318, 317]
SRC = R / 'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'
OUT = R / 'results/_baseline7'
lut = make_jet_lut(256)
_, _, test = load_cameras(str(SRC))
views = set(test[:64])
names = [os.path.basename(e.name) for e in read_extrinsics_text(str(SRC / 'sparse/0/images.txt')).values()]
dummy = OUT / '_dummy.png'
dummy.parent.mkdir(parents=True, exist_ok=True)
if not dummy.exists():
    Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(dummy)

for iid in IDS:
    d = OUT / f'inst{iid:03d}'
    img = d / 'images'; img.mkdir(parents=True, exist_ok=True)
    for nm in names:
        q = img / nm
        if q.exists() or q.is_symlink():
            continue
        base = nm.split('.')[0]
        if base in views:
            u = np.asarray(Image.open(R / f'datasets/RF-3DGS_cbf_w_params/spectra/inst{iid:03d}/{base}.png').convert('L'), np.float32) / 255.0
            rgb = (apply_jet(torch.as_tensor(u).unsqueeze(0), lut).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(q)
        else:
            os.symlink(dummy, q)
    sp = d / 'sparse'
    if not sp.exists():
        os.symlink(SRC / 'sparse', sp)
    (d / 'train_index.txt').write_text('\n'.join(sorted(views)[:48]) + '\n')
    (d / 'test_index.txt').write_text('\n'.join(sorted(views)[48:]) + '\n')
    print(iid, 'staged', flush=True)
print('done')
