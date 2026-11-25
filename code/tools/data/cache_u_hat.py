# -*- coding: utf-8 -*-
"""把冻结模型的渲染结果 u_hat 缓存成 uint8 灰度图，供阶段 1 的条件精修器训练。

为什么要缓存：阶段 1 的输入是「冻结检查点对该实例、该视角的渲染」。若不缓存，
训练每一步都要走一次 3DGS 前向（P≈1e6 高斯），几十万步不可能。渲染是一次性开销。

口径与 `_chk_stage0_w_mix.py` 完全一致（同一个 `compute_scalar_tx` + `render`
+ `combine_bands`），只是**不加载 GT、不算指标**，所以快得多。

输出：

    <out>/u/inst{id}/{view}.png      uint8 灰度，0–255（= round(u*255)）
    <out>/manifest.json              检查点、实例表、视角集、参数快照

用法（WSL）::

    $PY code/tools/data/cache_u_hat.py \
        -s datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
        --meta datasets/RF-3DGS_mvdr_w_params/meta.json \
        --checkpoint results/_ckpt/angular_180000.pth \
        --start_checkpoint datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
        --roles train,pos_holdout,w_holdout --views 640 --stride 1 \
        --out datasets/_uhat_mvdr_w
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from rfgs.core.angular_data import load_cameras, load_instances                  # noqa: E402
from splat.arguments import ModelParams, PipelineParams                # noqa: E402
from splat.color_scalar import combine_bands, to_override_color        # noqa: E402
from splat.color_scalar_tx import compute_scalar_tx, load_tx_state     # noqa: E402
from splat.gaussian_renderer import render                             # noqa: E402
from splat.scene import GaussianModel                                  # noqa: E402

from rfgs.rrf.train_rrf_scalar import load_visual_geometry                      # noqa: E402


def _md5_head(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--start_checkpoint", required=True)
    ap.add_argument("--roles", default="train,pos_holdout,w_holdout",
                    help="逗号分隔的实例角色；all=全部")
    ap.add_argument("--instances", default="", help="直接指定 id，逗号分隔，优先于 --roles")
    ap.add_argument("--views", type=int, default=640, help="每实例最多取多少个 test 视角")
    ap.add_argument("--stride", type=int, default=1, help="视角抽样步长")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    cams, _, test_names = load_cameras(dataset.source_path)
    names = test_names[:: max(args.stride, 1)][: args.views]
    inst = load_instances(args.meta)
    if inst is None:
        raise SystemExit("meta 里没有 instances")

    if args.instances:
        want = [int(x) for x in args.instances.split(",") if x.strip()]
        rows = [int(np.searchsorted(inst.ids, i)) for i in want]
    elif args.roles == "all":
        rows = list(range(len(inst)))
    else:
        roles = {r.strip() for r in args.roles.split(",") if r.strip()}
        rows = sorted(int(r) for r in np.where(np.isin(inst.role, list(roles)))[0])
    print(f"[cache] {len(rows)} 个实例 × {len(names)} 视角 = "
          f"{len(rows) * len(names)} 张", flush=True)

    head, embed, opacity, cfg, it = load_tx_state(args.checkpoint, device=args.device)
    gaussians = GaussianModel(dataset.sh_degree)
    load_visual_geometry(args.start_checkpoint, gaussians, dataset.sh_degree)
    if opacity is not None:
        gaussians._opacity = torch.nn.Parameter(
            opacity.reshape(-1, 1).contiguous(), requires_grad=False)
    scale = float(cfg.get("spatial_lr_scale", 0.0) or 0.0)
    if scale <= 0:
        raise SystemExit("cfg 里没有 spatial_lr_scale")
    bg = torch.zeros(3, device=args.device)

    udir = os.path.join(args.out, "u")
    os.makedirs(udir, exist_ok=True)
    manifest = dict(checkpoint=args.checkpoint, ckpt_iter=int(it),
                    ckpt_md5_head=_md5_head(args.checkpoint),
                    start_checkpoint=args.start_checkpoint, meta=args.meta,
                    views=names, stride=args.stride, n_gaussians=int(
                        gaussians.get_xyz.shape[0]),
                    spatial_lr_scale=scale, instances=[])
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for ri, row in enumerate(rows):
            iid = int(inst.ids[row])
            d = os.path.join(udir, f"inst{iid:03d}")
            os.makedirs(d, exist_ok=True)
            params = dict(tx_yaw=float(inst.yaw_off[row]),
                          tx_pitch=float(inst.pitch_off[row]),
                          tx_freq=float(inst.freq_hz[row]),
                          tx_rot=torch.as_tensor(inst.rot[row], dtype=torch.float32),
                          tx_w=[float(inst.w_theta[row]), float(inst.w_phi[row]),
                                1.0 if bool(inst.w_uniform[row]) else 0.0])
            txpos = torch.as_tensor(inst.xyz[row], device=args.device,
                                    dtype=torch.float32)
            for nm in names:
                p = os.path.join(d, f"{nm}.png")
                if os.path.exists(p) and not args.overwrite:
                    done += 1
                    continue
                cam = cams[nm]
                s = compute_scalar_tx(head, embed, gaussians.get_xyz,
                                      cam.camera_center, scale, txpos, **params)
                pkg = render(cam, gaussians, pipe, bg,
                             override_color=to_override_color(s))
                u = combine_bands(pkg["render"], head)[0]
                a = (u.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
                Image.fromarray(a).save(p)
                done += 1
            manifest["instances"].append(dict(
                id=iid, row=int(row), role=str(inst.role[row]),
                position_id=int(inst.position_id[row]),
                xyz=[float(x) for x in inst.xyz[row]],
                yaw_off=float(inst.yaw_off[row]), pitch_off=float(inst.pitch_off[row]),
                freq_hz=float(inst.freq_hz[row]),
                w_label=str(inst.w_label[row]),
                w_theta=float(inst.w_theta[row]), w_phi=float(inst.w_phi[row]),
                w_uniform=bool(inst.w_uniform[row]),
                base_instance=int(inst.base_instance[row])))
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"[cache] inst{iid:03d} ({ri+1}/{len(rows)})  {done} 张  "
                  f"{rate:.2f} 张/s  余 {(len(rows)*len(names)-done)/max(rate,1e-9)/60:.1f} min",
                  flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[cache] 完成 {done} 张，{time.time()-t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
