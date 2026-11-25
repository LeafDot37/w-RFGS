# -*- coding: utf-8 -*-
"""跨轴定性：每个轴一个实例、同一视角 5 的 CBF GT 与 Bi-SG+3sup 渲染。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
R=Path('/root/autodl-tmp/rf'); sys.path.insert(0,str(R/'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar import apply_jet, make_jet_lut
from splat.color_scalar_tx import load_tx_state
from splat.scene import GaussianModel
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw
from rfgs.rrf.train_rrf_scalar import load_visual_geometry
from sim import w_codebook
dev='cuda'; OUT=R/'results/_axis_qual'; OUT.mkdir(parents=True,exist_ok=True)
V='00005'; IDS=[144,196,232,268,318]
meta=json.load(open(R/'datasets/RF-3DGS_mvdr_w_params/meta.json'))
by={int(i['id']):i for i in meta['instances']}
cams,_,_=load_cameras(str(R/'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
g=GaussianModel(3); load_visual_geometry(str(R/'datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth'),g,3)
head,embed,op,cfg,it=load_tx_state(str(R/'results/_csi_cbfw/arm20k_T256_bisg_3sup/csi_20000.pth'),device=dev)
g._opacity=torch.nn.Parameter(op.reshape(-1,1).contiguous(),requires_grad=False)
grid=CBFGrid(); lut=make_jet_lut(256).to(dev)
dir_out=str(cfg.get('head_type','')).endswith('bisg')
def jet(u): return (apply_jet(u.unsqueeze(0),lut).permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
for iid in IDS:
    inst=by[iid]; wv=torch.as_tensor(w_codebook.vector_for(str(inst['w_label']),4,4),dtype=torch.complex64,device=dev)
    with torch.no_grad():
        x=render_cir(head,embed,g,cams[V],inst,256,0.5,grid,w_vec=wv,tx_nr=4,tx_nc=4,dir_out=dir_out)
        u=spec_raw(x,grid); u=(u-u.min())/(u.max()-u.min()+1e-12)
        gtu=np.asarray(Image.open(R/f'datasets/RF-3DGS_cbf_w_params/spectra/inst{iid:03d}/{V}.png').convert('L'),np.float32)/255
        Image.fromarray(jet(torch.as_tensor(gtu,device=dev))).save(OUT/f'inst{iid:03d}_gt.png')
        Image.fromarray(jet(u)).save(OUT/f'inst{iid:03d}_ours.png')
    print(iid,'done',flush=True)
print('all done')
