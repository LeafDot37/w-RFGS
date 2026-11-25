
# -*- coding: utf-8 -*-
"""波束管理指标：用预测场从 58 个码本波束里选 w，和真值最优波束比。

口径
----
- 真值协方差来自 `results/_probe_precoder/instXXX.npz`（`a` 为 [M²,16,P]，`valid` 掩码）：
      Rm_true = B^H B,  B = a[:,:,valid].transpose(0,2,1).reshape(-1,16)
- 预测协方差：对 16 个 TX 阵元各用基向量 w=e_k 渲染一次，得到 x_k [M²,T]；
      B_pred = [x_0..x_15] 展平,  Rm_pred = B_pred^H B_pred
- 码本：`sim.w_codebook.train_labels()+holdout_labels()`（37+21=58 个 w）
- 指标：top-1 / top-3 命中（相对真值最优 w），选中波束的 SE 相对最优 λ 的比值，
  以及相对增益损失 dB。扫描开销按 0 次测量（预测）vs 58 次（穷举）报告。

跑法（服务器，GPU 空闲时）：
    python code/tools/eval/eval_beam_mgmt.py \
        --checkpoint results/_csi_cbfw/arm20k_full_rfalpha_geom/csi_20000.pth \
        --ids 144 196 232 268 304 318 --views 8 --out results/_down_check/beam_mgmt.jsonl
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path("/root/autodl-tmp/rf")
sys.path.insert(0, str(R / "code"))
from rfgs.core.angular_data import load_cameras                     # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, render_cir              # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry          # noqa: E402
from splat.color_scalar_tx import load_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from sim import w_codebook                                          # noqa: E402

SRC = R / "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
META = R / "datasets/RF-3DGS_mvdr_w_params/meta.json"
PROBE = R / "results/_probe_precoder"
PT, N0, BW = 0.01, 10 ** (-17.4), 1e8


def se(g):
    return np.log2(1.0 + PT * g / (N0 * BW))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", type=int, nargs="+",
                    default=[144, 196, 232, 268, 304, 318])
    ap.add_argument("--views", type=int, default=8,
                    help="每个实例取前 N 个测试视角")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(META, encoding="utf-8"))
    by = {int(i["id"]): i for i in meta["instances"]}
    cams, _, test = load_cameras(str(SRC))
    g = GaussianModel(3)
    load_visual_geometry(
        str(R / "datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"), g, 3)
    head, embed, op, cfg, it = load_tx_state(args.checkpoint, device=dev)
    g._opacity = torch.nn.Parameter(op.reshape(-1, 1).contiguous(), requires_grad=False)
    grid = CBFGrid()
    dir_out = str(cfg.get("head_type", "")).endswith("bisg")
    labels = w_codebook.train_labels() + w_codebook.holdout_labels()
    wmat = torch.as_tensor(
        np.stack([w_codebook.vector_for(lab, 4, 4) for lab in labels], 0),
        dtype=torch.complex64, device=dev)          # (58,16)
    nW = len(labels)
    print(f"[beam] ckpt={Path(args.checkpoint).name} ids={len(args.ids)} "
          f"views={args.views} codebook={nW}", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    all_top1, all_top3, all_se, all_gap, all_gap_top3 = [], [], [], [], []
    all_se_rand, all_se_uni = [], []
    uni_idx = labels.index(w_codebook.UNIFORM)
    k_idx = torch.eye(16, dtype=torch.complex64, device=dev)
    for iid in args.ids:
        inst = by[iid]
        probe = np.load(PROBE / f"inst{iid:03d}.npz")
        pmap = {int(v): k for k, v in enumerate(probe["names"])}
        b1, b3, ss, gg, g3 = [], [], [], [], []
        for vn in test[:args.views]:
            r = pmap.get(int(vn))
            if r is None:
                continue
            with torch.no_grad():
                xs = []
                for k in range(16):
                    xs.append(render_cir(head, embed, g, cams[vn], inst, args.T,
                                         args.dt, grid, w_vec=k_idx[k],
                                         tx_nr=4, tx_nc=4, dir_out=dir_out)
                              .reshape(-1))            # 100*T
                Bp = torch.stack(xs, -1)               # (100*T,16)
                Rp = Bp.conj().T @ Bp
            a = probe["a"][r][:, :, probe["valid"][r]]
            Bt = a.transpose(0, 2, 1).reshape(-1, 16)
            Rt = Bt.conj().T @ Bt
            lam = float(np.linalg.eigvalsh(Rt)[-1])
            wt = wmat.conj()                            # for w^H R w
            Gt = torch.einsum("wi,ij,wj->w", wt, torch.as_tensor(Rt, device=dev), wmat).real
            Gp = torch.einsum("wi,ij,wj->w", wt, Rp, wmat).real
            oracle = int(Gt.argmax())
            order = torch.argsort(Gp, descending=True)
            sel = int(order[0])
            top3 = set(int(x) for x in order[:3])
            b1.append(float(sel == oracle))
            b3.append(float(oracle in top3))
            ss.append(float(se(float(Gt[sel])) / se(lam)))
            gg.append(float(10 * np.log10(max(float(Gt[sel]), 1e-30) / max(lam, 1e-30))))
            if oracle in top3:
                g3.append(gg[-1])
            all_se_rand.append(float(np.mean([se(float(g)) for g in Gt]) / se(lam)))
            all_se_uni.append(float(se(float(Gt[uni_idx])) / se(lam)))
        if b1:
            row = {"id": iid, "n": len(b1), "top1": float(np.mean(b1)),
                   "top3": float(np.mean(b3)), "se_ratio": float(np.mean(ss)),
                   "gain_gap_db": float(np.mean(gg))}
            out.write(json.dumps(row) + "\n"); out.flush()
            all_top1 += b1; all_top3 += b3; all_se += ss; all_gap += gg; all_gap_top3 += g3
            print(f"  inst{iid:03d} n={len(b1)} top1={row['top1']:.3f} "
                  f"top3={row['top3']:.3f} SE={row['se_ratio']:.3f} "
                  f"gap={row['gain_gap_db']:.2f}dB", flush=True)
    print("\n=== beam management (58-beam codebook) ===", flush=True)
    print(f"top1 {np.mean(all_top1):.3f}  top3 {np.mean(all_top3):.3f}  "
          f"SE ratio {np.mean(all_se):.3f}  gain gap {np.mean(all_gap):.2f} dB", flush=True)
    print(f"baselines: uniform SE {np.mean(all_se_uni):.3f}  "
          f"random SE {np.mean(all_se_rand):.3f}  exhaustive SE 1.000 (58 scans)", flush=True)
    out.write(json.dumps({"summary": True, "top1": float(np.mean(all_top1)),
                          "top3": float(np.mean(all_top3)),
                          "se_ratio": float(np.mean(all_se)),
                          "gain_gap_db": float(np.mean(all_gap)),
                          "se_uniform": float(np.mean(all_se_uni)),
                          "se_random": float(np.mean(all_se_rand)),
                          "se_exhaustive": 1.0}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
