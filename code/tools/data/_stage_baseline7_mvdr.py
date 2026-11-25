# -*- coding: utf-8 -*-
"""7 个代表实例的 MVDR 版逐发射源基线数据（jet）。"""
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

IDS = [144, 196, 232, 268, 304, 318, 317]
SRC = R / 'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'
OUT = R / 'results/_baseline7_mvdr'
lut = make_jet_lut(256)
_, _, test = load_cameras(str(SRC)); views = sorted(test[:64])
for iid in IDS:
    d = OUT / f'inst{iid:03d}'; img = d / 'images'; img.mkdir(parents=True, exist_ok=True)
    for vn in views:
        u = np.asarray(Image.open(R / f'datasets/RF-3DGS_mvdr_w_params/spectra/inst{iid:03d}/{vn}.png').convert('L'), np.float32) / 255.0
        rgb = (apply_jet(torch.as_tensor(u).unsqueeze(0), lut).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(rgb).save(img / f'{vn}.png')
    sp = d / 'sparse'
    if not sp.exists():
        os.symlink(SRC / 'sparse', sp)
    (d / 'train_index.txt').write_text('\n'.join(views[:48]) + '\n')
    (d / 'test_index.txt').write_text('\n'.join(views[48:]) + '\n')
print('done')
