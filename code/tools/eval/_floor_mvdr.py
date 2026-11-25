# -*- coding: utf-8 -*-
"""MVDR 的常数地板：训练集 GT MVDR u 图的像素均值，再在 36x64 测试视角上评。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json, sys
from pathlib import Path
import numpy as np, torch
R=Path('/root/autodl-tmp/rf'); sys.path.insert(0,str(R/'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar import apply_jet, make_jet_lut
from splat.lpipsPyTorch.modules.lpips import LPIPS
from splat.utils.image_utils import psnr
from splat.utils.loss_utils import ssim
from rfgs.csi.train_csi_cbf import CBFGrid, spec_from_cir_mvdr
from sim.spectrum_gen import array_manifold_vector
dev='cuda'
cams,_,test=load_cameras(str(R/'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
views=test[:64]; grid=CBFGrid(); lut=make_jet_lut(256).to(dev)
a_mat=torch.as_tensor(array_manifold_vector(10,grid.theta,grid.phi).reshape(100,-1),dtype=torch.complex128,device=dev)
lp=LPIPS('vgg','0.1').to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)
T=256; dt=0.5
def bin_cir(z,vn):
    a=z[f'v{vn}_a']; tau=z[f'v{vn}_tau']; val=z[f'v{vn}_valid']
    tap=np.clip(np.floor(tau[val].astype(np.float64)/dt).astype(np.int64),0,T-1)
    x=np.zeros((a.shape[0],T),np.complex64); np.add.at(x,(slice(None),tap),a[:,val]); return x
# mean training MVDR u
acc=np.zeros(grid.hw,np.float64); n=0
for iid in range(144):
    z=np.load(R/f'results/_cir_train/inst{iid:03d}.npz')
    if iid%16==0: print('train',iid,flush=True)
    for vn in views:
        x=torch.as_tensor(bin_cir(z,vn),device=dev)
        u=spec_from_cir_mvdr(x,grid,a_mat=a_mat).cpu().numpy(); acc+=u; n+=1
mu=torch.as_tensor(acc/n,dtype=torch.float32,device=dev)
print('mean over',n,'training maps',flush=True)
vals={'psnr':[],'ssim':[],'lpips':[]}
with torch.no_grad():
    for iid in range(318,354):
        z=np.load(R/f'results/_cir_w/inst{iid:03d}.npz')
        for vn in views:
            x=torch.as_tensor(bin_cir(z,vn),device=dev)
            ug=spec_from_cir_mvdr(x,grid,a_mat=a_mat)
            pr=apply_jet(mu.unsqueeze(0),lut).unsqueeze(0); gt=apply_jet(ug.unsqueeze(0),lut).unsqueeze(0)
            vals['psnr'].append(float(psnr(pr,gt).mean())); vals['ssim'].append(float(ssim(pr,gt).mean())); vals['lpips'].append(float(lp(pr,gt).mean()))
print({k:(float(np.mean(v)),float(np.std(v))) for k,v in vals.items()},flush=True)
json.dump({k:{'mean':float(np.mean(v)),'std':float(np.std(v))} for k,v in vals.items()},
          open(R/'results/_csi_cbfw/floor_mvdr.json','w'))
