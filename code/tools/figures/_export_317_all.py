# -*- coding: utf-8 -*-
"""实例 317（原作者 TX）全部 64 个测试视角的 CBF/MVDR GT 与 Bi-SG+3sup 渲染。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
R = Path('/root/autodl-tmp/rf'); sys.path.insert(0, str(R / 'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar import apply_jet, make_jet_lut
from splat.color_scalar_tx import load_tx_state
from splat.scene import GaussianModel
from splat.utils.image_utils import psnr
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw, spec_from_cir_mvdr
from rfgs.rrf.train_rrf_scalar import load_visual_geometry
from sim import w_codebook

dev = 'cuda'; IID = 317; OUT = R / 'results/_fig317_all'; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(R / 'datasets/RF-3DGS_mvdr_w_params/meta.json'))
inst = [i for i in meta['instances'] if int(i['id']) == IID][0]
cams, _, test = load_cameras(str(R / 'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
g = GaussianModel(3)
load_visual_geometry(str(R / 'datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth'), g, 3)
head, embed, op, cfg, it = load_tx_state(str(R / 'results/_csi_cbfw/arm20k_T256_bisg_3sup/csi_20000.pth'), device=dev)
g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
grid = CBFGrid(); lut = make_jet_lut(256).to(dev)
wv = torch.as_tensor(w_codebook.vector_for(str(inst['w_label']), 4, 4), dtype=torch.complex64, device=dev)
dir_out = str(cfg.get('head_type', '')).endswith('bisg')
z = np.load(R / f'results/_cir_axis/inst{IID:03d}.npz')
RF = R / 'final/前馈pipeline/运行/run_20261020_本机4060_角域前馈多发射源_起步点修复/rf3dgs_released_renders'


def jet(u):
    return (apply_jet(u.unsqueeze(0), lut).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


rows = []
with torch.no_grad():
    for vn in test[:64]:
        a = z[f'v{vn}_a']; tau = z[f'v{vn}_tau']; val = z[f'v{vn}_valid']
        tap = np.clip(np.floor(tau[val].astype(np.float64) / 0.5).astype(np.int64), 0, 255)
        xg = np.zeros((a.shape[0], 256), np.complex64); np.add.at(xg, (slice(None), tap), a[:, val])
        xg = torch.as_tensor(xg, device=dev)
        x = render_cir(head, embed, g, cams[vn], inst, 256, 0.5, grid, w_vec=wv, tx_nr=4, tx_nc=4, dir_out=dir_out)
        u = spec_raw(x, grid); u = (u - u.min()) / (u.max() - u.min() + 1e-12)
        gtu = np.asarray(Image.open(R / f'datasets/RF-3DGS_cbf_w_params/spectra/inst{IID:03d}/{vn}.png').convert('L'), np.float32) / 255
        gt = torch.as_tensor(gtu, device=dev)
        um = spec_from_cir_mvdr(x, grid); ug = spec_from_cir_mvdr(xg, grid)
        pc = float(psnr(apply_jet(u.unsqueeze(0), lut).unsqueeze(0), apply_jet(gt.unsqueeze(0), lut).unsqueeze(0)).mean())
        pm = float(psnr(apply_jet(um.unsqueeze(0), lut).unsqueeze(0), apply_jet(ug.unsqueeze(0), lut).unsqueeze(0)).mean())
        for name, arr in [('cbf_gt', jet(gt)), ('cbf_ours', jet(u)), ('mvdr_gt', jet(ug)), ('mvdr_ours', jet(um))]:
            Image.fromarray(arr).save(OUT / f'v{vn}_{name}.png')
        rows.append((vn, pc, pm, 1 if (RF / f'{vn}.png').exists() else 0))
        print(vn, f'{pc:.2f}', f'{pm:.2f}', flush=True)
with open(OUT / 'index.tsv', 'w') as f:
    f.write('view\tcbf_psnr\tmvdr_psnr\trf3dgs\n')
    for r in rows:
        f.write('%s\t%.2f\t%.2f\t%d\n' % r)
print('done', len(rows), 'rf3dgs_avail', sum(r[3] for r in rows))
