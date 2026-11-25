# -*- coding: utf-8 -*-
"""7 个代表实例的逐发射源 SH-RRF 基线测试指标（jet 域）。"""
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
from splat.utils.image_utils import psnr
from splat.utils.loss_utils import ssim
from splat.lpipsPyTorch.modules.lpips import LPIPS

lp = LPIPS('vgg', '0.1').cuda().eval()
for p in lp.parameters():
    p.requires_grad_(False)
vals = []
for iid in [144, 196, 232, 268, 304, 318, 317]:
    d = R / f'results/_baseline7/run_corf_inst{iid:03d}/rf_eval/test/ours_20000'
    rs = sorted((d / 'renders').glob('*.png'))
    gs = sorted((d / 'gt').glob('*.png'))
    ps, ss, ls = [], [], []
    for r, g in zip(rs, gs):
        a = torch.as_tensor(np.asarray(Image.open(r).convert('RGB'), np.float32) / 255).permute(2, 0, 1).contiguous()[None].cuda()
        b = torch.as_tensor(np.asarray(Image.open(g).convert('RGB'), np.float32) / 255).permute(2, 0, 1).contiguous()[None].cuda()
        ps.append(float(psnr(a, b).mean()))
        ss.append(float(ssim(a, b).mean()))
        ls.append(float(lp(a, b).mean()))
    vals.append((iid, np.mean(ps), np.mean(ss), np.mean(ls), len(ps)))
    print('%d n=%d PSNR %.2f SSIM %.3f LPIPS %.3f' % (iid, len(ps), np.mean(ps), np.mean(ss), np.mean(ls)), flush=True)
v = np.array([x[1:] for x in vals], float)
print('MEAN n=%d PSNR %.2f SSIM %.3f LPIPS %.3f' % (int(v[:, 3].sum()), v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()))
