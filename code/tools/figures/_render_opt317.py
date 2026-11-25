# -*- coding: utf-8 -*-
"""用冻结的光学高斯渲染 inst317 选定视角的真实（光学）画面。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path('/root/autodl-tmp/rf'); sys.path.insert(0, str(R / 'code'))
from rfgs.core.angular_data import load_cameras
from splat.scene import GaussianModel
from splat.gaussian_renderer import render
from rfgs.rrf.train_rrf_scalar import load_visual_geometry

VIEWS = ['00052', '00100', '00117', '00292']
OUT = R / 'results/_opt317'; OUT.mkdir(parents=True, exist_ok=True)


class Pipe:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


g = GaussianModel(3)
load_visual_geometry(str(R / 'datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth'), g, 3)
for p in (g._xyz, g._scaling, g._rotation, g._opacity, g._features_dc, g._features_rest):
    p.requires_grad_(False)
cams, _, _ = load_cameras(str(R / 'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
bg = torch.tensor([1.0, 1.0, 1.0], device='cuda', dtype=torch.float32)
with torch.no_grad():
    for v in VIEWS:
        img = render(cams[v], g, Pipe(), bg)['render'].clamp(0, 1)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(OUT / f'v{v}.png')
        print(v, arr.shape, flush=True)
print('done')
