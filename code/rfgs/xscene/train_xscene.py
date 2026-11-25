# -*- coding: utf-8 -*-
"""跨场景可行性实验：多布局共享一个 CSI 头，留出一个场景做测试。

两种模式
--------------------------------------------------------------------------------
`free`     每场景一套逐点 embedding + opacity（与主实验相同的参数化），头共享。
           测试场景冻结头、只拟合该场景的 embedding/opacity（few-shot）。
`material` embedding 固定为材质的 one-hot（不可学习），全模型零逐场景参数。
           测试场景**零样本**直接前馈。

为什么还要 `material` 模式
--------------------------------------------------------------------------------
`free` 的逐点 embedding 让任何头都能拟合新场景，跨场景迁移只体现在「优化更快」；
`material` 模式把「材质 + 局部入射几何 → 复反射系数」这条物理映射交给共享头，
新场景没有任何可拟合参数，才是真正的零样本检验。两臂一起报，避免只挑好看的那条。

用法：见 `_srv_xscene_train.sh`。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_mlp import set_scene_norm                          # noqa: E402
from splat.color_scalar_coh import TxCoherentColorHead            # noqa: E402
from splat.color_scalar_tx import tx_head_cfg                      # noqa: E402
from rfgs.csi.train_csi_cbf import CBFGrid, rot_z_t, spec_raw, tr38901_t     # noqa: E402
from rfgs.rrf.train_rrf_angular import cam_yaw                               # noqa: E402
from sim import w_codebook                                          # noqa: E402
from sim.w_codebook import element_positions                        # noqa: E402

C_LIGHT = 299792458.0
# 直达径复标量：沿用主模型 F20k 的标定值（同一 4×4 TX / 10×10 RX 口径）。
LOS_KAPPA = complex(1.5e6, -7e5)


class XScene:
    def __init__(self, name: str, root: Path, dev: str):
        self.name = name
        self.root = Path(root)
        z = np.load(self.root / "points.npz")
        self.xyz = torch.as_tensor(z["xyz"], dtype=torch.float32, device=dev)
        self.normal = torch.as_tensor(z["normal"], dtype=torch.float32, device=dev)
        self.mat = np.asarray(z["mat"], np.int64)
        meta = json.load(open(self.root / "instances.json", encoding="utf-8"))
        self.insts = [r for r in meta["instances"]
                      if r.get("role", "train") == "train"]
        cams, train, test = load_cameras(str(self.root))
        self.cams, self.train, self.test = cams, train, test
        self.spectra = self.root / "spectra"
        self.P = self.xyz.shape[0]
        oh = torch.zeros(self.P, 8, device=dev)
        oh[torch.arange(self.P, device=dev),
           torch.as_tensor(self.mat, dtype=torch.long, device=dev)] = 1.0
        self.mat_oh = oh


def make_embed(scene: XScene, mode: str, dev: str):
    if mode == "material":
        return torch.zeros(scene.P, 17, device=dev)
    return nn.Parameter(torch.zeros(scene.P, 17, device=dev).normal_(0, 0.02),
                        requires_grad=True)


def make_head(mode: str, hidden: int, depth: int, dev: str):
    return TxCoherentColorHead(
        embed_dim=32, hidden=hidden, depth=depth, use_dist=False, use_rel=True,
        use_tx=True, use_yaw=False, use_tx_orient=True, use_tx_local_dir=True,
        use_tx_freq=True, freq_lo_hz=2.0e9, freq_hi_hz=4.5e9,
        init_u=0.05, n_ant=16, use_af=False, use_geom=False).to(dev)


@torch.no_grad()
def _w_uniform(tx_nr, tx_nc, dev):
    return torch.as_tensor(w_codebook.uniform_weight(tx_nr, tx_nc),
                           dtype=torch.complex64, device=dev)


def _geom7(xyz, tx, dirv, normal):
    """与 `TxScalarColorHead._geom_block` 同式的 7 维几何块（复场头不支持 use_geom）。"""
    n = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    t = tx if tx.dim() == 2 else tx.unsqueeze(0)
    if t.shape[0] == 1 and n.shape[0] != 1:
        t = t.expand(n.shape[0], 3)
    utx = xyz - t
    utx = utx / utx.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    urx = dirv / dirv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = utx + urx
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat([n, (n * utx).sum(-1, keepdim=True),
                      (n * urx).sum(-1, keepdim=True),
                      (n * b).sum(-1, keepdim=True).abs(),
                      torch.zeros(n.shape[0], 1, dtype=n.dtype, device=n.device)],
                     dim=1)


def render_cir_x(head, embed, scene: XScene, cam, inst, T, dt_ns, grid,
                 tx_nr=4, tx_nc=4, opacity=None, chunk=65536):
    """与 `train_csi_cbf.render_cir` 同口径，额外传入视线方向与法向。"""
    dev = scene.xyz.device
    xyz = scene.xyz
    P = xyz.shape[0]
    op = (torch.sigmoid(opacity).reshape(-1) if opacity is not None
          else torch.ones(P, device=dev))
    tx = torch.as_tensor(inst["xyz"], dtype=torch.float32, device=dev)
    R_tx = torch.as_tensor(w_codebook.tx_orientation_matrix(
        float(inst["yaw_abs_rad"]), float(inst["pitch_abs_rad"])),
        dtype=torch.float32, device=dev)
    R_rx = rot_z_t(cam_yaw(cam)).to(dev)
    cc = cam.camera_center.reshape(1, 3)
    tx_yaw = float(inst.get("offset_yaw_rad", 0.0))
    tx_pitch = float(inst.get("offset_pitch_rad", 0.0))
    f_hz = float(inst["freq_hz"])
    yt, zt = element_positions(tx_nr, tx_nc)
    yt = torch.as_tensor(yt, dtype=torch.float32, device=dev)
    zt = torch.as_tensor(zt, dtype=torch.float32, device=dev)
    w_vec = _w_uniform(tx_nr, tx_nc, dev)
    x = torch.zeros(grid.M * grid.M, T, dtype=torch.complex64, device=dev)
    d_rx_all = xyz - cc
    dirv_all = d_rx_all / d_rx_all.norm(dim=1, keepdim=True).clamp_min(1e-9)
    for s0 in range(0, P, chunk):
        sl = slice(s0, min(s0 + chunk, P))
        pos = xyz[sl]
        geom = _geom7(pos, tx, dirv_all[sl], scene.normal[sl])
        inp = torch.cat([scene.mat_oh[sl], geom, embed[sl]], dim=1)
        s = head(inp, dirv_all[sl], pos, None, tx=tx,
                 yaw=None, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                 tx_freq=f_hz, tx_rot=R_tx, tx_w=None)
        zc = torch.complex(s[:, 0], s[:, 1]) * op[sl]
        d_tx = pos - tx
        r_tx = d_tx.norm(dim=1).clamp_min(1e-6)
        u_in = d_tx / r_tx.unsqueeze(1)
        r_rx = d_rx_all[sl].norm(dim=1).clamp_min(1e-6)
        u_out = d_rx_all[sl] / r_rx.unsqueeze(1)
        tau_ns = (r_tx + r_rx) / C_LIGHT * 1e9
        tap = torch.floor(tau_ns / dt_ns).long().clamp(0, T - 1)
        u_in_l = u_in @ R_tx
        th_t = torch.arccos(u_in_l[:, 2].clamp(-1.0, 1.0))
        ph_t = torch.atan2(u_in_l[:, 1], u_in_l[:, 0])
        ph = 2 * math.pi * (yt[None, :] * u_in_l[:, 1:2]
                            + zt[None, :] * u_in_l[:, 2:3])
        A = tr38901_t(th_t, ph_t)[:, None] * torch.exp(1j * ph)
        g_tx = (A * w_vec[None, :]).sum(1)
        u_out_l = u_out @ R_rx
        v, w = u_out_l[:, 1], u_out_l[:, 2]
        th_r = torch.arccos(w.clamp(-1.0, 1.0))
        ph_r = torch.atan2(v, torch.sqrt((1.0 - w * w).clamp_min(0.0)) + 1e-12)
        g_rx = tr38901_t(th_r, ph_r)
        phase = 2 * math.pi * (grid.y.to(dev)[None, :] * v[:, None]
                               + grid.z.to(dev)[None, :] * w[:, None])
        M = g_rx[:, None] * torch.exp(1j * phase)
        zz = zc * torch.exp(-1j * 2 * math.pi * f_hz * tau_ns * 1e-9) * g_tx
        x = x.index_add(1, tap, (zz[:, None] * M).transpose(0, 1))
    # 解析直达径：与主模型同式。合成房间的 GT 含 LOS，而点云全在表面上，
    # 纯散射模型对直达径的支撑为零（主模型加 LOS 后 CBF +2 dB 量级）。
    dvec = cc.reshape(3) - tx.reshape(3)
    d_los = dvec.norm().clamp_min(1e-6)
    u_w = dvec / d_los
    u_in_l = u_w @ R_tx
    th_t = torch.arccos(u_in_l[2].clamp(-1.0, 1.0))
    ph_t = torch.atan2(u_in_l[1], u_in_l[0])
    ph = 2 * math.pi * (yt * u_in_l[1] + zt * u_in_l[2])
    A = tr38901_t(th_t, ph_t)[None] * torch.exp(1j * ph)
    g_tx_los = (A * w_vec[None, :]).sum()
    u_out_l = (-u_w) @ R_rx
    v_l, w_l = u_out_l[1], u_out_l[2]
    th_r = torch.arccos(w_l.clamp(-1.0, 1.0))
    ph_r = torch.atan2(v_l, torch.sqrt((1.0 - w_l * w_l).clamp_min(0.0)) + 1e-12)
    g_rx_los = tr38901_t(th_r, ph_r)
    M_los = g_rx_los * torch.exp(1j * 2 * math.pi
                                 * (grid.y.to(dev) * v_l + grid.z.to(dev) * w_l))
    tau_los = d_los / C_LIGHT * 1e9
    tap_los = torch.floor(tau_los / dt_ns).long().clamp(0, T - 1)
    amp = (LOS_KAPPA / d_los
           * torch.exp(-1j * 2 * math.pi * f_hz * tau_los * 1e-9) * g_tx_los)
    x = x.index_add(1, tap_los.reshape(1),
                    (amp * M_los).to(torch.complex64).reshape(x.shape[0], 1))
    return x


def norm_spec(x, grid):
    sp = spec_raw(x, grid)
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def load_gt(scene: XScene, inst, vn: str, dev: str):
    p = scene.spectra / f"inst{int(inst['id']):03d}" / f"{vn}.png"
    a = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
    return torch.as_tensor(a, device=dev)


def build(args):
    dev = "cuda"
    names = args.scenes.split(",")
    scenes = [XScene(n, Path(args.root) / n, dev) for n in names]
    head = make_head(args.mode, args.hidden, args.depth, dev)
    all_xyz = torch.cat([s.xyz for s in scenes], 0)
    set_scene_norm(head, all_xyz)
    embeds, ops = {}, {}
    for s in scenes:
        embeds[s.name] = make_embed(s, args.mode, dev)
        if args.mode == "free":
            ops[s.name] = nn.Parameter(
                torch.full((s.P, 1), 2.2, device=dev), requires_grad=True)
        else:
            ops[s.name] = None
    return dev, scenes, head, embeds, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/rf/_external/xrooms")
    ap.add_argument("--scenes", required=True, help="逗号分隔；第一个是测试场景")
    ap.add_argument("--mode", choices=["free", "material"], default="free")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--views_per_step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_lr", type=float, default=1e-3)
    ap.add_argument("--opacity_lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--dt_ns", type=float, default=0.5)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev, scenes, head, embeds, ops = build(args)
    train_scenes = [s for s in scenes if s.name != args.scenes.split(",")[0]]
    if not train_scenes:
        raise SystemExit("没有训练场景")
    grid = CBFGrid()
    params = [{"params": list(head.parameters()), "lr": args.lr}]
    for s in train_scenes:
        if isinstance(embeds[s.name], nn.Parameter):
            params.append({"params": [embeds[s.name]], "lr": args.embed_lr})
        if ops[s.name] is not None:
            params.append({"params": [ops[s.name]], "lr": args.opacity_lr})
    opt = torch.optim.Adam(params)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.model_path, "train.log"), "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    say(f"[xs] mode={args.mode} train={[s.name for s in train_scenes]} "
        f"holdout={args.scenes.split(',')[0]} "
        f"P={[s.P for s in scenes]} iters={args.iterations}")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(args.views_per_step):
            s = train_scenes[int(rng.integers(0, len(train_scenes)))]
            inst = s.insts[int(rng.integers(0, len(s.insts)))]
            vn = s.train[int(rng.integers(0, len(s.train)))]
            x = render_cir_x(head, embeds[s.name], s, s.cams[vn], inst,
                             args.T, args.dt_ns, grid, opacity=ops[s.name])
            u = norm_spec(x, grid)
            gt = load_gt(s, inst, vn, dev)
            loss = loss + (u - gt).abs().mean()
        loss = loss / args.views_per_step
        loss.backward()
        opt.step()
        if it % args.log_every == 0:
            el = time.time() - t0
            say(f"[xs] {it}/{args.iterations} loss={float(loss):.4f} "
                f"{it/el:.2f} it/s {el/60:.1f} min")
        if args.save_every and it % args.save_every == 0:
            save(args, head, embeds, ops, it)
    save(args, head, embeds, ops, args.iterations)
    say(f"[xs] done {args.iterations}")


def save(args, head, embeds, ops, it):
    blob = {"iteration": int(it),
            "cfg": tx_head_cfg(head, {"xsec_mode": args.mode,
                                      "xsec_T": args.T,
                                      "xsec_dt_ns": args.dt_ns,
                                      "xsec_scenes": args.scenes,
                                      "xsec_hidden": args.hidden,
                                      "xsec_depth": args.depth}),
            "head": head.state_dict(),
            "embed": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                      for k, v in embeds.items()},
            "opacity": {k: (v.detach().cpu() if torch.is_tensor(v) else None)
                        for k, v in ops.items()}}
    torch.save(blob, os.path.join(args.model_path, f"xs_{it}.pth"))


if __name__ == "__main__":
    main()
