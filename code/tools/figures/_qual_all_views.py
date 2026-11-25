# -*- coding: utf-8 -*-
"""逐视角 CBF/MVDR 定性对比（GT vs Bi-SG+3sup），实例 321。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
R=Path('/root/autodl-tmp/rf'); sys.path.insert(0,str(R/'code'))
from rfgs.core.angular_data import load_cameras
from splat.color_scalar import apply_jet, make_jet_lut
from splat.color_scalar_tx import load_tx_state
from splat.scene import GaussianModel
from splat.utils.image_utils import psnr
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir, spec_raw, spec_from_cir_mvdr
from rfgs.rrf.train_rrf_scalar import load_visual_geometry
from sim import w_codebook
dev='cuda'; IID=321
OUT=R/f'results/_qual_all/inst{IID:03d}'; OUT.mkdir(parents=True,exist_ok=True)
meta=json.load(open(R/'datasets/RF-3DGS_mvdr_w_params/meta.json'))
inst=[i for i in meta['instances'] if int(i['id'])==IID][0]
cams,_,test=load_cameras(str(R/'datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100'))
g=GaussianModel(3); load_visual_geometry(str(R/'datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth'),g,3)
head,embed,op,cfg,it=load_tx_state(str(R/'results/_csi_cbfw/arm20k_T256_bisg_3sup/csi_20000.pth'),device=dev)
g._opacity=torch.nn.Parameter(op.reshape(-1,1).contiguous(),requires_grad=False)
grid=CBFGrid(); lut=make_jet_lut(256).to(dev)
wv=torch.as_tensor(w_codebook.vector_for(str(inst['w_label']),4,4),dtype=torch.complex64,device=dev)
dir_out=str(cfg.get('head_type','')).endswith('bisg')
def gt_cir(vn):
    z=np.load(R/f'results/_cir_w/inst{IID:03d}.npz')
    a=z[f'v{vn}_a']; tau=z[f'v{vn}_tau']; val=z[f'v{vn}_valid']
    tap=np.clip(np.floor(tau[val].astype(np.float64)/0.5).astype(np.int64),0,255)
    x=np.zeros((a.shape[0],256),np.complex64); np.add.at(x,(slice(None),tap),a[:,val])
    return torch.as_tensor(x,device=dev)
def rgb(u):
    return (apply_jet(u.unsqueeze(0),lut).permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
def panel(img, scale=2):
    return Image.fromarray(img).resize((img.shape[1]*scale,img.shape[0]*scale),Image.NEAREST)
rows=[]
with torch.no_grad():
    for vn in test[:64]:
        x=render_cir(head,embed,g,cams[vn],inst,256,0.5,grid,w_vec=wv,tx_nr=4,tx_nc=4,dir_out=dir_out)
        u=spec_raw(x,grid); u=(u-u.min())/(u.max()-u.min()+1e-12)
        gtu=np.asarray(Image.open(R/f'datasets/RF-3DGS_cbf_w_params/spectra/inst{IID:03d}/{vn}.png').convert('L'),np.float32)/255
        gt=torch.as_tensor(gtu,device=dev)
        um=spec_from_cir_mvdr(x,grid); ug=spec_from_cir_mvdr(gt_cir(vn),grid)
        gtc=rgb(gt); prc=rgb(u); gtm=rgb(ug); prm=rgb(um)
        pc=float(psnr(apply_jet(u.unsqueeze(0),lut).unsqueeze(0),apply_jet(gt.unsqueeze(0),lut).unsqueeze(0)).mean())
        pm=float(psnr(apply_jet(um.unsqueeze(0),lut).unsqueeze(0),apply_jet(ug.unsqueeze(0),lut).unsqueeze(0)).mean())
        P=panel(gtc); W,H=P.size
        canvas=Image.new('RGB',(2*W+30,2*H+90),'white')
        canvas.paste(panel(gtc),(10,40)); canvas.paste(panel(prc),(W+20,40))
        canvas.paste(panel(gtm),(10,H+70)); canvas.paste(panel(prm),(W+20,H+70))
        d=ImageDraw.Draw(canvas)
        d.text((10,10),f'inst {IID}  view {vn}',fill='black')
        d.text((10,30),'CBF GT',fill='black'); d.text((W+20,30),'CBF Bi-SG+3sup',fill='black')
        d.text((10,H+60),'MVDR GT',fill='black'); d.text((W+20,H+60),'MVDR Bi-SG+3sup',fill='black')
        canvas.save(OUT/f'view_{vn}.png')
        rows.append((vn,pc,pm))
with open(OUT/'index.tsv','w') as f:
    f.write('view\tcbf_psnr\tmvdr_psnr\n')
    for r in rows: f.write('%s\t%.2f\t%.2f\n'%r)
print('done',len(rows),OUT)
