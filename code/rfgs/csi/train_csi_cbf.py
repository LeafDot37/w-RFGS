# -*- coding: utf-8 -*-
"""CSI 头 v0：逐高斯复散射 + 解析 CBF 渲染（G1 的最小可行实现）。

与标量头的区别
--------------------------------------------------------------------------------
- 头**不看视线方向**（dirv 传零、use_dist 关）：学的是与接收机无关的环境散射；
- 输出是逐高斯复散射 S_i（Re/Im）；
- 渲染不走光栅化器：每个高斯按几何时延落到 CIR 的时延格，乘以 TX 方向图、
  RX 阵列流形与 exp(-j2πfτ)，求和得 x[100, T]；
- CBF 由 x 解析计算（与 `spectrum_gen.cbf_spectrum` 同一公式），再逐图 min-max。

监督：只用频谱 L1（u 域）。绝对功率暂不监督——逐图归一化的目标里没有它；
后面用导出的 CIR 再加逐路径复监督（G1b）。

跑法（服务器）：
    python train_csi_cbf.py -s <data> --spectra_root <spc> --meta <meta> \
        --start_checkpoint <seed.pth> --model_path <out> --iterations 5000
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

_CODE = str(Path(__file__).resolve().parent)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from rfgs.core.angular_data import load_cameras                              # noqa: E402
from splat.color_mlp import set_scene_norm                          # noqa: E402
from splat.color_scalar_coh import TxCoherentColorHead              # noqa: E402
from splat.color_scalar_tx import save_tx_state                     # noqa: E402
from splat.scene import GaussianModel                               # noqa: E402
from sim import w_codebook                                          # noqa: E402
from sim.spectrum_gen import (compute_angle_matrices, steering_vector,  # noqa: E402
                              tr38901_gain)
from rfgs.rrf.train_rrf_angular import cam_yaw                               # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                   # noqa: E402

C_LIGHT = 299792458.0
RF_MARGIN = 0.2      # 透射率积分在散射体/收发端附近留的空隙（m），避免自遮挡


def tr38901_t(theta, phi):
    """torch 版 tr38901_gain（与 sim.spectrum_gen 的公式逐项一致）。"""
    t3db = math.radians(65.0)
    a_v = -torch.clamp(12.0 * ((theta - math.pi / 2) / t3db) ** 2, max=30.0)
    phi = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi
    a_h = -torch.clamp(12.0 * (phi / t3db) ** 2, max=30.0)
    a_db = -torch.clamp(-(a_v + a_h), max=30.0) + 8.0
    return torch.sqrt(10.0 ** (a_db / 10.0))


def rot_z_t(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=torch.float32)


class CBFGrid:
    def __init__(self, M=10, width=300, height=200, fov=90.0, stride=1):
        self.M = M
        self.stride = int(stride)
        th, ph = compute_angle_matrices(width, height, fov)
        if self.stride > 1:
            th, ph = th[::self.stride, ::self.stride], ph[::self.stride, ::self.stride]
        self.theta, self.phi = th, ph
        a = steering_vector(M, th, ph).reshape(M * M, -1)
        self.a_mat = torch.as_tensor(a, dtype=torch.complex64)
        y, z = w_codebook.element_positions(M, M)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.hw = (th.shape[0], th.shape[1])


def render_cir(head, embed, gaussians, cam, inst, T, dt_ns, grid, chunk=150000,
               dtau=None, w_vec=None, tx_nr=1, tx_nc=1, w_feat=None,
               dir_out=False, tx_pattern="tr38901"):
    dev = gaussians.get_xyz.device
    xyz = gaussians.get_xyz
    P = xyz.shape[0]
    op = torch.sigmoid(gaussians._opacity).squeeze(-1)
    tx = torch.as_tensor(inst["xyz"], dtype=torch.float32, device=dev)
    R_tx = torch.as_tensor(w_codebook.tx_orientation_matrix(
        float(inst["yaw_abs_rad"]), float(inst["pitch_abs_rad"])),
        dtype=torch.float32, device=dev)
    R_rx = rot_z_t(cam_yaw(cam)).to(dev)
    cc = cam.camera_center.reshape(1, 3)
    tx_yaw = float(inst.get("offset_yaw_rad", 0.0))
    tx_pitch = float(inst.get("offset_pitch_rad", 0.0))
    tx_freq = float(inst["freq_hz"])
    f_hz = float(inst["freq_hz"])
    x = torch.zeros(grid.M * grid.M, T, dtype=torch.complex64, device=dev)
    rf = getattr(head, "rf_alpha", None)
    rf_use_amp = bool(getattr(head, "rf_use_amp", False))
    dif = getattr(head, "diffuse", None)
    d_all = None
    if dif is not None:
        d_all = dif(embed, xyz, center=getattr(head, "xyz_center", None),
                    scale=float(getattr(head, "xyz_scale", 0.0) or 0.0))
    nrm = ani = None
    if getattr(head, "use_geom", False):
        from splat.color_scalar_tx import gaussian_geom
        nrm, ani = gaussian_geom(gaussians)
    dens = None
    if rf is not None:
        # 静态 RF 不透明度：α = sigmoid(logit(o) + δ(embed,xyz))，δ 零初始化。
        _c = getattr(head, "xyz_center", None)
        _s = float(getattr(head, "xyz_scale", 0.0) or 0.0)
        alpha_all = rf.alpha(embed, xyz, op, center=_c, scale=_s)
        lo = xyz.min(0).values - 0.25
        hi = xyz.max(0).values + 0.25
        rng = (hi - lo).clamp_min(1e-6)
        dims = torch.tensor([rf.gx, rf.gy, rf.gz], dtype=torch.float32, device=dev)
        ii = ((xyz - lo) / rng * (dims - 1)).long().clamp_min(0)
        ii[:, 0].clamp_(max=rf.gx - 1)
        ii[:, 1].clamp_(max=rf.gy - 1)
        ii[:, 2].clamp_(max=rf.gz - 1)
        flat = ii[:, 2] * (rf.gy * rf.gx) + ii[:, 1] * rf.gx + ii[:, 0]
        rho = torch.zeros(rf.gz * rf.gy * rf.gx, dtype=torch.float32, device=dev)
        rho = rho.index_add(0, flat, alpha_all) * nn.functional.softplus(rf.log_k)
        rho = rho.reshape(1, 1, rf.gz, rf.gy, rf.gx)

        def dens(pts):
            g = (pts - lo) / rng * 2 - 1
            g = g.reshape(1, pts.shape[0], 1, 1, 3)
            return nn.functional.grid_sample(
                rho, g, align_corners=True, mode="bilinear",
                padding_mode="border").reshape(-1)
    for s0 in range(0, P, chunk):
        sl = slice(s0, min(s0 + chunk, P))
        pos = xyz[sl]
        dirv = ((pos - cc) / (pos - cc).norm(dim=1, keepdim=True).clamp_min(1e-9)
                if dir_out else torch.zeros_like(pos))
        s = head(embed[sl], dirv, pos, None, tx=tx,
                 yaw=None, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                 tx_freq=tx_freq, tx_rot=R_tx, tx_w=w_feat,
                 normal=(nrm[sl] if nrm is not None else None),
                 aniso=(ani[sl] if ani is not None else None))
        zc = torch.complex(s[:, 0], s[:, 1]) * (
            alpha_all[sl] if rf_use_amp else op[sl])
        if d_all is not None:
            zc = zc + d_all[sl].to(torch.complex64) * op[sl]
        d_tx = pos - tx
        r_tx = d_tx.norm(dim=1).clamp_min(1e-6)
        u_in = d_tx / r_tx.unsqueeze(1)
        d_rx = pos - cc
        r_rx = d_rx.norm(dim=1).clamp_min(1e-6)
        u_out = d_rx / r_rx.unsqueeze(1)
        tau_ns = (r_tx + r_rx) / C_LIGHT * 1e9
        if dtau is not None:
            tau_ns = tau_ns + dtau[sl]
        tap = torch.floor(tau_ns / dt_ns).long().clamp(0, T - 1)
        u_in_l = u_in @ R_tx
        th_t = torch.arccos(u_in_l[:, 2].clamp(-1.0, 1.0))
        vt = u_in_l[:, 1]
        # 方向图的方位角用 atan2(y, x)；阵列相位用 (v,w)=(y,z)。两者不要混。
        ph_t = torch.atan2(u_in_l[:, 1], u_in_l[:, 0])
        if tx_nr * tx_nc > 1 and w_vec is not None:
            # 4×4：解析阵列响应 × w。阵元序与 w_codebook 同一套（G0 已验证 1×1；
            # 4×4 的 GT 由生成器 `_prep_with_w` 的 a·w 合成，这里同号）。
            from sim.w_codebook import element_positions
            yt, zt = element_positions(tx_nr, tx_nc)
            yt = torch.as_tensor(yt, dtype=torch.float32, device=dev)
            zt = torch.as_tensor(zt, dtype=torch.float32, device=dev)
            ph = 2 * math.pi * (yt[None, :] * vt[:, None]
                                + zt[None, :] * u_in_l[:, 2][:, None])
            A = tr38901_t(th_t, ph_t)[:, None] * torch.exp(1j * ph)
            g_tx = (A * w_vec[None, :]).sum(1)          # 复增益
        elif tx_pattern == "iso":
            g_tx = torch.ones_like(th_t)
        else:
            g_tx = tr38901_t(th_t, ph_t)
        u_out_l = u_out @ R_rx
        v = u_out_l[:, 1]
        w = u_out_l[:, 2]
        th_r = torch.arccos(w.clamp(-1.0, 1.0))
        ph_r = torch.atan2(v, torch.sqrt((1.0 - w * w).clamp_min(0.0)) + 1e-12)
        g_rx = tr38901_t(th_r, ph_r)
        phase = 2 * math.pi * (grid.y.to(dev)[None, :] * v[:, None]
                               + grid.z.to(dev)[None, :] * w[:, None])
        M = g_rx[:, None] * torch.exp(1j * phase)
        zz = zc * torch.exp(-1j * 2 * math.pi * f_hz * tau_ns * 1e-9) * g_tx
        contrib = zz[:, None] * M                       # [chunk, 100]
        if rf is not None:
            kk = 8
            d1 = (pos - tx).norm(dim=1).clamp_min(1e-6)
            d2 = (cc.reshape(3) - pos).norm(dim=1).clamp_min(1e-6)
            m1 = (RF_MARGIN / d1).clamp(0.0, 0.45)
            m2 = (RF_MARGIN / d2).clamp(0.0, 0.45)
            tt = torch.linspace(0, 1, kk, device=dev)[None, :]
            t1 = m1[:, None] + (1 - 2 * m1)[:, None] * tt
            t2 = m2[:, None] + (1 - 2 * m2)[:, None] * tt
            s1 = tx[None, None, :] + t1[..., None] * (pos - tx)[:, None, :]
            s2 = (pos[:, None, :] + t2[..., None]
                  * (cc.reshape(3)[None, :] - pos)[:, None, :])
            r1 = dens(s1.reshape(-1, 3)).reshape(-1, kk)
            r2 = dens(s2.reshape(-1, 3)).reshape(-1, kk)
            ds1 = d1 * (1 - 2 * m1) / kk
            ds2 = d2 * (1 - 2 * m2) / kk
            v = torch.exp(-(r1.sum(1) * ds1 + r2.sum(1) * ds2))
            contrib = contrib * v[:, None]
        x = x.index_add(1, tap, contrib.transpose(0, 1))
    if getattr(head, "use_los", False):
        # 解析直达径。TX–RX 连线上没有高斯 ⇒ 纯表面散射模型对这一类路径的支撑为零，
        # 而数据集的 GT 含 los=True（实测各留出块 33%–48% 的能量在直达径桶里）。
        # 幅度 = κ/d · TX 方向图(含 w) · RX 方向图 · exp(−j2πfτ)，κ 是唯一自由复标量。
        dvec = cc.reshape(3) - tx.reshape(3)
        d_los = dvec.norm().clamp_min(1e-6)
        u_w = dvec / d_los
        u_in_l = u_w @ R_tx
        th_t = torch.arccos(u_in_l[2].clamp(-1.0, 1.0))
        ph_t = torch.atan2(u_in_l[1], u_in_l[0])
        if tx_nr * tx_nc > 1 and w_vec is not None:
            from sim.w_codebook import element_positions
            yt, zt = element_positions(tx_nr, tx_nc)
            yt = torch.as_tensor(yt, dtype=torch.float32, device=dev)
            zt = torch.as_tensor(zt, dtype=torch.float32, device=dev)
            ph = 2 * math.pi * (yt * u_in_l[1] + zt * u_in_l[2])
            A = tr38901_t(th_t, ph_t)[None] * torch.exp(1j * ph)
            g_tx = (A * w_vec[None, :]).sum()
        elif tx_pattern == "iso":
            g_tx = torch.ones((), dtype=torch.float32, device=dev)
        else:
            g_tx = tr38901_t(th_t, ph_t)
        u_out_l = (-u_w) @ R_rx
        v_l = u_out_l[1]
        w_l = u_out_l[2]
        th_r = torch.arccos(w_l.clamp(-1.0, 1.0))
        ph_r = torch.atan2(v_l, torch.sqrt((1.0 - w_l * w_l).clamp_min(0.0)) + 1e-12)
        g_rx = tr38901_t(th_r, ph_r)
        M_los = g_rx * torch.exp(1j * 2 * math.pi
                                 * (grid.y.to(dev) * v_l + grid.z.to(dev) * w_l))
        tau_los = d_los / C_LIGHT * 1e9
        tap_los = torch.floor(tau_los / dt_ns).long().clamp(0, T - 1)
        if getattr(head, "los_log", None) is not None:
            kap = head.los_kappa0 * torch.exp(
                torch.complex(head.los_log[0], head.los_log[1]))
        else:
            kap = head.los_kappa
        amp = (kap / d_los
               * torch.exp(-1j * 2 * math.pi * f_hz * tau_los * 1e-9) * g_tx)
        if rf is not None:
            kk = 48
            p0 = tx.reshape(3)
            p1 = cc.reshape(3)
            d_l = (p1 - p0).norm().clamp_min(1e-6)
            m = (RF_MARGIN / d_l).clamp(0.0, 0.45)
            ts = m + (1 - 2 * m) * torch.linspace(0, 1, kk, device=dev)
            pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
            ds = d_l * (1 - 2 * m) / kk
            amp = amp * torch.exp(-(dens(pts).sum() * ds))
        x = x.index_add(1, tap_los.reshape(1),
                        (amp * M_los).to(torch.complex64).reshape(x.shape[0], 1))
    return x


def spec_from_cir_mvdr(x, grid, loading=0.01, db=True, a_mat=None):
    from sim.spectrum_gen import array_manifold_vector
    if a_mat is None:
        a_mat = torch.as_tensor(array_manifold_vector(
            10, grid.theta, grid.phi).reshape(100, -1),
            dtype=torch.complex128, device=x.device)
    xx = x.to(torch.complex128)
    Rm = xx @ xx.conj().transpose(0, 1)
    tr = torch.diagonal(Rm).real.sum() / 100.0
    eye = torch.eye(100, dtype=Rm.dtype, device=x.device)
    L = None
    # 零/近零 CIR（整幅 u=0 的视角）会让 R 秩 1 甚至为 0，Cholesky 直接失败；
    # 依次放大 loading 重试，仍失败就返回常数图（归一化后为 0）。
    for scale in (loading, max(loading, 0.1), 1.0):
        try:
            L = torch.linalg.cholesky(
                Rm + (scale * tr.clamp_min(1e-30) + 1e-12) * eye)
            break
        except Exception:
            L = None
    if L is None:
        return torch.zeros(grid.hw, dtype=torch.float32, device=x.device)
    v = torch.linalg.solve_triangular(L, a_mat, upper=False)
    denom = (v.real ** 2 + v.imag ** 2).sum(0)
    p = 1.0 / denom.clamp_min(1e-30)
    if db:
        p = 10.0 * torch.log10(p)
    sp = p.reshape(grid.hw).float()
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def build_gt_cir(npz_path, vn, T, dt, dev):
    """读逐视角 CIR。

    两种输入都支持：
    * `_pack_cir.py` 的产物：键 `v<view>`，已按 dt 分箱；
    * `_g1_export_csi_cbf_w.py` 的产物：键 `v<view>_{a,tau,valid}`，逐路径。
      逐路径版可以按训练端的 T 现场分箱——**这一点很关键**：旧主臂 T=128
      （64 ns）会把 15% 的平均能量、13.7% 视角的过半能量截掉，而 CBF 真值是
      按全时延算的；用逐路径输入才能在 T=256 时把晚到路径补回来。
    """
    z = np.load(npz_path)
    _hk = f"v{int(vn):05d}_h"
    if _hk in z.files:                    # 漫散射子集：聚合 H [100,T]
        x = torch.as_tensor(z[_hk], dtype=torch.complex64, device=dev)
        return x[:, :T]
    if f"v{vn}" in z.files:
        x = torch.as_tensor(z[f"v{vn}"], dtype=torch.complex64, device=dev)
        return x[:, :T]
    a = z[f"v{vn}_a"]
    tau = z[f"v{vn}_tau"]
    valid = z[f"v{vn}_valid"]
    tap = np.clip(np.floor(tau[valid].astype(np.float64) / dt).astype(np.int64),
                  0, T - 1)
    xg = np.zeros((a.shape[0], T), np.complex64)
    np.add.at(xg, (slice(None), tap), a[:, valid])
    return torch.as_tensor(xg, dtype=torch.complex64, device=dev)


def spec_raw(x, grid):
    """归一化前的 CBF 幅度图；pmax 与生成器的 scales_*.jsonl 同一量纲。"""
    x_sum = x.conj().sum(1)
    return (x_sum @ grid.a_mat.to(x.device)).abs().reshape(grid.hw)


def spec_from_cir(x, grid):
    sp = spec_raw(x, grid)
    mn, mx = sp.min(), sp.max()
    return (sp - mn) / torch.clamp(mx - mn, min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--spectra_root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--start_checkpoint", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--views_per_step", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子；TMC 的 3-seed 方差检查用")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_lr", type=float, default=1e-3)
    ap.add_argument("--opacity_lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--dt_ns", type=float, default=0.5)
    ap.add_argument("--train_roles", nargs="*", default=["train"])
    ap.add_argument("--max_instances", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--view_list", default="",
                    help="逗号分隔的视角名白名单（成对数据只用这些视角）")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--mvdr_spectra", default="",
                    help="给了就做双算子监督：同一份 CIR 同时解释 CBF 与 MVDR 图")
    ap.add_argument("--mvdr_dt_ns", type=float, default=0.1)
    ap.add_argument("--mvdr_T", type=int, default=600)
    ap.add_argument("--w_mvdr", type=float, default=0.5)
    ap.add_argument("--train_op", choices=["cbf", "mvdr"], default="cbf")
    ap.add_argument("--head", choices=["coh", "bisg"], default="coh",
                    help="coh=当前主头；bisg=方向相关 Bi-SG 散射核")
    ap.add_argument("--op_dt_ns", type=float, default=0.1)
    ap.add_argument("--op_T", type=int, default=600)
    ap.add_argument("--cir_root", default="",
                    help="给了就加复 CIR 相关损失（尺度不变），用于钉住相位/时延")
    ap.add_argument("--cir_h_root", default="",
                    help="漫散射子集：聚合 H 的 npz 根（键 v{i}_h）")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="训练实例白名单（漫散射的 24 个训练实例）")
    ap.add_argument("--w_cir", type=float, default=0.5)
    ap.add_argument("--w_img", type=float, default=1.0,
                    help="图像 L1 损失权重；0=纯 CIR 监督（CSI-only 臂）")
    ap.add_argument("--w_mvdr_img", type=float, default=0.0,
                    help="MVDR 图 L1 权重；GT 由同一份 CIR 现算（需 --cir_root）")
    ap.add_argument("--mvdr_stride", type=int, default=2,
                    help="MVDR 损失用的角域下采样倍数（训练用，评测仍全分辨率）")
    ap.add_argument("--cir_multi", type=int, default=0,
                    help="每步额外监督的 w 个数（需 --cir_multi_dir）")
    ap.add_argument("--cir_multi_dir", default="",
                    help="逐阵元响应 a16 目录（_g1_export_csi_cbf_w.py --save_a16）")
    ap.add_argument("--cir_multi_views", type=int, default=128,
                    help="多 w 监督只在这些前 N 个视角上做")
    ap.add_argument("--w_cir_multi", type=float, default=1.0,
                    help="多 w 相关损失的总权重（内部按 N 平均）")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--train_stride", type=int, default=1,
                    help="训练时角域下采样倍数（评测仍全分辨率）")
    ap.add_argument("--tx_nr", type=int, default=1)
    ap.add_argument("--tx_nc", type=int, default=1)
    ap.add_argument("--tx_pattern", choices=["tr38901", "iso"], default="tr38901",
                    help="原版 RF-3DGS 数据集的 TX 是 1x1 iso；本项目数据集是 tr38901")
    ap.add_argument("--M", type=int, default=10, help="RX UPA 行/列数（原版是 8）")
    ap.add_argument("--freq_lo", type=float, default=2.0e9,
                    help="头部频段归一化的下界（原版外部层用 55e9）")
    ap.add_argument("--freq_hi", type=float, default=4.5e9)
    ap.add_argument("--view_split", choices=["train", "test"], default="test",
                    help="训练用哪个 index 划分；本项目一律 test（作者 640 视角），"
                         "原版外部层用 train（2560 训练 / 640 测试）")
    ap.add_argument("--w_uniform", type=int, default=0,
                    help="1=阵列渲染一律用 uniform w（w-盲对照；GT 仍是各实例的真值 w）")
    ap.add_argument("--scale_root", default="",
                    help="逐实例 scales_instXXX.jsonl 目录；给了就监督绝对尺度")
    ap.add_argument("--w_scale", type=float, default=0.0,
                    help="log pmax 尺度损失权重（配合 --scale_root）")
    ap.add_argument("--w_input", choices=["none", "dir", "complex"], default="none",
                    help="dir: w 码本指向 (θ,φ) 进头；complex: 16 维复权 Re/Im 进头；渲染器都用 uniform 阵列")
    ap.add_argument("--dtau_lr", type=float, default=0.0,
                    help=">0 时启用逐高斯时延残差（ns），该值即学习率")
    ap.add_argument("--init_csi", default="", help="从 CSI checkpoint 热启动头/嵌入/不透明度")
    ap.add_argument("--cir_nmse", type=int, default=0,
                    help="1=复 CIR 损失用全抽头 NMSE（最优复标量）而非相关")
    ap.add_argument("--w_prof", type=float, default=0.0,
                    help="归一化能量-时延剖面 L1 权重（尺度不变、但对直达径占比敏感）")
    ap.add_argument("--w_tap", type=float, default=0.0,
                    help="逐时延桶角谱 L1 权重（约束桶内能量来自哪个方向）")
    ap.add_argument("--tap_w", type=int, default=30)
    ap.add_argument("--tap_h", type=int, default=20)
    ap.add_argument("--use_los", type=int, default=0,
                    help="1=加解析直达径（TX–RX 直连，幅度 κ/d×方向图×exp(−j2πfτ)）")
    ap.add_argument("--los_param", choices=["direct", "log"], default="direct",
                    help="log: κ=κ0·exp(u+iv)，相对学习率；direct: 直接复参数")
    ap.add_argument("--los_kappa_re", type=float, default=1.0)
    ap.add_argument("--los_kappa_im", type=float, default=0.0)
    ap.add_argument("--los_lr", type=float, default=2e-2,
                    help="直达径复标量 κ 的学习率（单独一组；量级与头不同）")
    ap.add_argument("--use_rf_alpha", type=int, default=0,
                    help="1=静态 RF 不透明度修正 + 体素透射率（LOS 与反射双段）")
    ap.add_argument("--rf_alpha_lr", type=float, default=1e-3)
    ap.add_argument("--rf_gx", type=int, default=64)
    ap.add_argument("--rf_gy", type=int, default=64)
    ap.add_argument("--rf_gz", type=int, default=32)
    ap.add_argument("--rf_k_init", type=float, default=-2.0,
                    help="体素密度全局尺度 softplus(log_k) 的初值（log 空间）")
    ap.add_argument("--rf_bias_init", type=float, default=0.0,
                    help="RF α 最后一层 bias 初值；0 ⇒ α 从光学 opacity 开始")
    ap.add_argument("--rf_use_amp", type=int, default=0,
                    help="1=RF α 同时作为幅度门（替代光学 opacity）；密度仍用 α")
    ap.add_argument("--freeze_opacity", type=int, default=0,
                    help="1=冻结光学 opacity（只作为 RF α 的初始化先验）")
    ap.add_argument("--use_geom", type=int, default=0,
                    help="1=把高斯法向/各向异性接进头（M4）")
    ap.add_argument("--use_diffuse", type=int, default=0,
                    help="1=加逐高斯漫散射权重 d_i（宽角度、实数功率）")
    ap.add_argument("--diffuse_lr", type=float, default=1e-2)
    ap.add_argument("--diffuse_hidden", type=int, default=64)
    ap.add_argument("--diffuse_bias_init", type=float, default=-8.0,
                    help="softplus 偏置初值：-8 时 d≈3e-4（几乎不扰动镜面），"
                         "D1 实测 5k 内偏置不动 ⇒ 表示没被激活；重跑用 -2")
    ap.add_argument("--freeze_body", type=int, default=0,
                    help="1=冻结镜面主体，只训 diffuse 头（漫散射适配）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    meta = json.load(open(args.meta, encoding="utf-8"))
    rows_all = meta["instances"]
    scales = {}
    if args.scale_root and args.w_scale > 0:
        for f in Path(args.scale_root).glob("scales_inst*.jsonl"):
            iid = int(f.stem.split("inst")[1])
            d = {}
            for ln in open(f, encoding="utf-8"):
                r = json.loads(ln)
                d[int(r["view"])] = (float(r["pmax"]), float(r["pmean"]))
            scales[iid] = d
        print(f"[scale] {len(scales)} instances, {sum(len(v) for v in scales.values())} views")
    rows = [r for r in rows_all if r.get("role", "train") in args.train_roles]
    if args.ids:
        _want = set(int(x) for x in args.ids)
        rows = [r for r in rows if int(r["id"]) in _want]
    if args.max_instances:
        rows = rows[: args.max_instances]
    if not rows:
        raise SystemExit("没有训练实例")

    cams, train_names, test_names = load_cameras(args.source_path)
    view_names = train_names if args.view_split == "train" else test_names
    if args.view_list:
        want = set(args.view_list.split(","))
        view_names = [v for v in view_names if v in want]
    if not view_names:
        raise SystemExit("view_list 过滤后没有视角")
    gaussians = GaussianModel(3)
    load_visual_geometry(args.start_checkpoint, gaussians, 3)
    gaussians._opacity.requires_grad_(not bool(args.freeze_opacity))
    P = gaussians.get_xyz.shape[0]
    embed = nn.Parameter(torch.zeros((P, 32), device=dev).normal_(0, 0.02),
                         requires_grad=True)
    dtau = (nn.Parameter(torch.zeros(P, device=dev))
            if args.dtau_lr > 0 else None)
    log_scale = (nn.Parameter(torch.zeros((), device=dev))
                 if args.scale_root and args.w_scale > 0 else None)
    _kw = dict(embed_dim=32, hidden=args.hidden, depth=args.depth,
               use_dist=False, use_rel=True, use_tx=True, use_yaw=False,
               use_tx_orient=True, use_tx_local_dir=True, use_tx_freq=True,
               freq_lo_hz=args.freq_lo, freq_hi_hz=args.freq_hi,
               init_u=0.05, n_ant=16,
               use_af=False, use_tx_w=(args.w_input == "dir"),
               use_tx_w_complex=(args.w_input == "complex"),
               use_geom=bool(args.use_geom))
    if args.head == "bisg":
        from splat.color_scalar_bisg import TxBiSGColorHead
        _kw2 = {k: v for k, v in _kw.items()
                if k not in ("use_tx_w", "use_tx_w_complex")}
        head = TxBiSGColorHead(**_kw2).to(dev)
    else:
        head = TxCoherentColorHead(**_kw).to(dev)
    if args.use_los:
        head.use_los = True
        if args.los_param == "log":
            head.los_kappa0 = complex(args.los_kappa_re, args.los_kappa_im)
            head.los_log = nn.Parameter(torch.zeros(2, device=dev))
        else:
            head.los_kappa = nn.Parameter(torch.tensor(
                complex(args.los_kappa_re, args.los_kappa_im),
                dtype=torch.complex64, device=dev))
    if args.use_rf_alpha:
        from splat.rf_alpha import RfAlphaHead
        head.rf_alpha = RfAlphaHead(embed_dim=32, hidden=64, gx=args.rf_gx,
                                    gy=args.rf_gy, gz=args.rf_gz,
                                    bias_init=args.rf_bias_init).to(dev)
        head.rf_use_amp = bool(args.rf_use_amp)
        with torch.no_grad():
            head.rf_alpha.log_k.fill_(args.rf_k_init)
    if args.use_diffuse:
        from splat.diffuse import DiffuseHead
        head.use_diffuse = True
        head.diffuse = DiffuseHead(embed_dim=32, hidden=args.diffuse_hidden,
                                   bias_init=args.diffuse_bias_init).to(dev)
    set_scene_norm(head, gaussians.get_xyz)
    if args.init_csi:
        from splat.color_scalar_tx import load_tx_state
        _h, _e, _o, _cfg, _it = load_tx_state(args.init_csi, device=dev)
        _sd = _h.state_dict()
        _allowed = set()
        if getattr(head, "use_los", False):
            if getattr(head, "los_log", None) is not None and "los_log" not in _sd:
                _allowed.add("los_log")
            if getattr(head, "los_kappa", None) is not None and "los_kappa" not in _sd:
                _allowed.add("los_kappa")
        if getattr(head, "rf_alpha", None) is not None:
            _allowed |= {k for k in head.state_dict() if k.startswith("rf_alpha.")}
        if getattr(head, "diffuse", None) is not None:
            _allowed |= {k for k in head.state_dict() if k.startswith("diffuse.")}
        if _allowed:
            _inc = head.load_state_dict(_sd, strict=False)
            assert set(_inc.missing_keys) <= _allowed, _inc
            assert not _inc.unexpected_keys, _inc
            print("[init_csi] 允许缺失键 %s，保留初值" % sorted(_allowed), flush=True)
        else:
            head.load_state_dict(_sd)
        embed.data.copy_(_e.reshape_as(embed))
        gaussians._opacity.data.copy_(_o.reshape_as(gaussians._opacity))
        print(f"[init_csi] loaded {args.init_csi} (iter {_it})", flush=True)
    grid = CBFGrid(M=args.M, stride=args.train_stride)
    a_tap = (CBFGrid(M=args.M, width=args.tap_w, height=args.tap_h,
                     fov=90.0).a_mat.to(dev) if args.w_tap > 0 else None)
    grid_mv = (CBFGrid(M=args.M, stride=args.mvdr_stride)
               if args.w_mvdr_img > 0 else None)
    a_mv = None
    if grid_mv is not None:
        from sim.spectrum_gen import array_manifold_vector
        a_mv = torch.as_tensor(
            array_manifold_vector(10, grid_mv.theta, grid_mv.phi).reshape(100, -1),
            dtype=torch.complex128, device=dev)
    if args.freeze_body:
        for n, p in head.named_parameters():
            if not n.startswith("diffuse."):
                p.requires_grad_(False)
        embed.requires_grad_(False)
        gaussians._opacity.requires_grad_(False)
    _hp = [p for n, p in head.named_parameters()
           if n not in ("los_kappa", "los_log")
           and not n.startswith("rf_alpha.") and not n.startswith("diffuse.")]
    if args.freeze_body:
        _hp = []
    _og = ([] if (args.freeze_opacity or args.freeze_body) else
           [{"params": [gaussians._opacity], "lr": args.opacity_lr}])
    _eg = [] if args.freeze_body else [{"params": [embed], "lr": args.embed_lr}]
    _lg = (([] if args.freeze_body else
            ([{"params": [head.los_log], "lr": args.los_lr}]
             if getattr(head, "los_log", None) is not None else
             [{"params": [head.los_kappa], "lr": args.los_lr}]))
           if getattr(head, "use_los", False) else [])
    _rg = ([] if args.freeze_body else
           ([{"params": list(head.rf_alpha.parameters()), "lr": args.rf_alpha_lr}]
            if getattr(head, "rf_alpha", None) is not None else []))
    _dg = ([{"params": list(head.diffuse.parameters()), "lr": args.diffuse_lr}]
           if getattr(head, "diffuse", None) is not None else [])
    opt = torch.optim.Adam(_og + _eg + ([{"params": _hp, "lr": args.lr}] if _hp else [])
      + ([{"params": [dtau], "lr": args.dtau_lr}] if dtau is not None else [])
      + _lg + _rg + _dg
      + ([{"params": [log_scale], "lr": 0.5}] if log_scale is not None else []))
    os.makedirs(args.model_path, exist_ok=True)
    log = open(os.path.join(args.model_path, "train.log"), "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"[csi] P={P} instances={len(rows)} views={len(view_names)} T={args.T}")
    t0 = time.time()
    rng = np.random.default_rng(0)
    cir_cache = {}
    last_scale = 0.0
    for it in range(1, args.iterations + 1):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(args.views_per_step):
            inst = rows[int(rng.integers(0, len(rows)))]
            vn = view_names[int(rng.integers(0, len(view_names)))]
            cam = cams[vn]
            w_vec, w_feat = None, None
            if args.tx_nr * args.tx_nc > 1:
                from sim import w_codebook as _wb
                _lab = ("uniform" if args.w_uniform
                        else str(inst.get("w_label", "uniform")))
                if args.w_input == "dir":
                    _tp = _wb.parse_label(_lab)
                    w_feat = ([0.0, 0.0, 1.0] if _tp is None
                              else [float(_tp[0]), float(_tp[1]), 0.0])
                    w_vec = torch.as_tensor(
                        _wb.uniform_weight(args.tx_nr, args.tx_nc),
                        dtype=torch.complex64, device=dev)
                elif args.w_input == "complex":
                    _wv = _wb.vector_for(_lab, args.tx_nr, args.tx_nc)
                    w_feat = list(np.concatenate([_wv.real, _wv.imag]).astype(float))
                    w_vec = torch.as_tensor(
                        _wb.uniform_weight(args.tx_nr, args.tx_nc),
                        dtype=torch.complex64, device=dev)
                else:
                    w_vec = torch.as_tensor(_wb.vector_for(_lab, args.tx_nr, args.tx_nc),
                                            dtype=torch.complex64, device=dev)
            x = render_cir(head, embed, gaussians, cam, inst, args.T,
                           args.dt_ns, grid, dtau=dtau, w_vec=w_vec,
                           tx_nr=args.tx_nr, tx_nc=args.tx_nc, w_feat=w_feat,
                           dir_out=(args.head == "bisg"),
                           tx_pattern=args.tx_pattern)
            if args.train_op == "cbf":
                xs = x * torch.exp(log_scale) if log_scale is not None else x
                sp = spec_raw(xs, grid)
                mn, mx = sp.min(), sp.max()
                u = (sp - mn) / torch.clamp(mx - mn, min=1e-12)
            else:
                xo = render_cir(head, embed, gaussians, cam, inst, args.op_T,
                                args.op_dt_ns, grid, w_vec=w_vec,
                                tx_nr=args.tx_nr, tx_nc=args.tx_nc,
                                tx_pattern=args.tx_pattern)
                u = spec_from_cir_mvdr(xo, grid)
            li = 0.0
            if args.w_img > 0:
                gt = Image.open(Path(args.spectra_root) /
                                f"inst{int(inst['id']):03d}" / f"{vn}.png")
                _gt = np.asarray(gt.convert("L"), np.float32) / 255.0
                if args.train_stride > 1:
                    _gt = _gt[::args.train_stride, ::args.train_stride]
                gt = torch.as_tensor(_gt, device=dev)
                li = args.w_img * (u - gt).abs().mean()
            last_scale = 0.0
            if log_scale is not None:
                _iid = int(inst["id"])
                _tr = scales.get(_iid, {}).get(int(vn))
                if _tr is not None:
                    _pred = sp.max().clamp_min(1e-30)
                    _sc = args.w_scale * torch.abs(
                        torch.log(_pred) - math.log(max(_tr[0], 1e-30)))
                    li = li + _sc
                    last_scale = float(_sc.detach())
            if args.w_mvdr_img > 0:
                if not (args.cir_root or args.cir_h_root):
                    raise SystemExit("--w_mvdr_img 需要 --cir_root 或 --cir_h_root")
                key = int(inst["id"])
                if key not in cir_cache:
                    if args.cir_h_root:
                        cir_cache[key] = str(Path(args.cir_h_root)
                                             / f"inst{key:03d}" / "cir.npz")
                    else:
                        cir_cache[key] = str(Path(args.cir_root)
                                             / f"inst{key:03d}.npz")
                xg = build_gt_cir(cir_cache[key], vn, args.T, args.dt_ns, dev)
                with torch.no_grad():
                    ug = spec_from_cir_mvdr(xg, grid_mv, a_mat=a_mv)
                um = spec_from_cir_mvdr(x, grid_mv, a_mat=a_mv)
                li = li + args.w_mvdr_img * (um - ug).abs().mean()
            if args.cir_root or args.cir_h_root:
                key = int(inst["id"])
                if key not in cir_cache:
                    if args.cir_h_root:
                        cir_cache[key] = str(Path(args.cir_h_root)
                                             / f"inst{key:03d}" / "cir.npz")
                    else:
                        cir_cache[key] = str(Path(args.cir_root)
                                             / f"inst{key:03d}.npz")
                xg = build_gt_cir(cir_cache[key], vn, args.T, args.dt_ns, dev)
                if args.cir_nmse:
                    den = xg.norm() ** 2
                    if float(den) > 1e-24:
                        alpha = ((x * xg.conj()).sum()
                                 / x.norm().clamp_min(1e-12) ** 2).detach()
                        nm = (alpha * x - xg).norm() ** 2 / den
                        li = li + args.w_cir * nm
                else:
                    num = (x * xg.conj()).sum().abs()
                    den = x.norm() * xg.norm()
                    li = li + args.w_cir * (1.0 - num / den.clamp_min(1e-12))
                if args.w_prof > 0 and float(xg.abs().max()) > 1e-12:
                    # 逐视角归一化的能量-时延剖面。相关损失是全局尺度不变的，
                    # 无法约束"直达径 vs 反射各占多少"；这一项正好补这个洞。
                    pe = (x.conj() * x).real.sum(0).clamp_min(1e-30)
                    pg = (xg.conj() * xg).real.sum(0).clamp_min(1e-30)
                    pe = pe / pe.sum().clamp_min(1e-30)
                    pg = pg / pg.sum().clamp_min(1e-30)
                    li = li + args.w_prof * (pe - pg).abs().sum()
                if args.w_tap > 0 and float(xg.abs().max()) > 1e-12:
                    # 逐桶角谱：CBF/MVDR 对"桶内角度"是零空间，这一项直接补上。
                    pt = (x.conj().transpose(0, 1) @ a_tap).abs() ** 2
                    pgt = (xg.conj().transpose(0, 1) @ a_tap).abs() ** 2
                    pt = pt / pt.sum(1, keepdim=True).clamp_min(1e-30)
                    pgt = pgt / pgt.sum(1, keepdim=True).clamp_min(1e-30)
                    wt = (xg.abs() ** 2).sum(0)
                    wt = wt / wt.sum().clamp_min(1e-30)
                    li = li + args.w_tap * (wt[:, None] * (pt - pgt).abs()).sum()
            if False and args.mvdr_spectra:
                xm = render_cir(head, embed, gaussians, cam, inst, args.mvdr_T,
                                args.mvdr_dt_ns, grid)
                um = spec_from_cir_mvdr(xm, grid)
                gm = Image.open(Path(args.mvdr_spectra) /
                                f"inst{int(inst['id']):03d}" / f"{vn}.png")
                gm = torch.as_tensor(np.asarray(gm.convert("L"), np.float32) / 255.0,
                                     device=dev)
                li = li + args.w_mvdr * (um - gm).abs().mean()
            loss = loss + li
        if args.cir_multi > 0:
            if not args.cir_multi_dir:
                raise SystemExit("--cir_multi 需要 --cir_multi_dir")
            iid = int(inst["id"])
            nv = min(args.cir_multi_views, len(view_names))
            cvn = view_names[int(rng.integers(0, nv))]
            z = np.load(Path(args.cir_multi_dir) / f"inst{iid:03d}.npz")
            a16 = z[f"v{cvn}_a16"]
            ctau = z[f"v{cvn}_tau"]; cval = z[f"v{cvn}_valid"]
            for _ in range(args.cir_multi):
                wr = rng.normal(size=a16.shape[1])
                wi = rng.normal(size=a16.shape[1])
                wnp = (wr + 1j * wi).astype(np.complex64)
                wnp /= max(float(np.linalg.norm(wnp)), 1e-12)
                wv = torch.as_tensor(wnp, dtype=torch.complex64, device=dev)
                xm = render_cir(head, embed, gaussians, cams[cvn], inst,
                                args.T, args.dt_ns, grid, dtau=dtau,
                                w_vec=wv, tx_nr=args.tx_nr, tx_nc=args.tx_nc,
                                dir_out=(args.head == "bisg"))
                aw = np.einsum("mnp,n->mp", a16[:, :, cval], wnp)
                tap = np.clip(np.floor(ctau[cval].astype(np.float64) /
                                       args.dt_ns).astype(np.int64),
                              0, args.T - 1)
                xg = np.zeros((aw.shape[0], args.T), np.complex64)
                np.add.at(xg, (slice(None), tap), aw)
                xg = torch.as_tensor(xg, dtype=torch.complex64, device=dev)
                num = (xm * xg.conj()).sum().abs()
                den = xm.norm() * xg.norm()
                li = li + (args.w_cir_multi / max(args.cir_multi, 1)) * (
                    1.0 - num / den.clamp_min(1e-12))
        loss = loss / args.views_per_step
        loss.backward()
        opt.step()
        if it % args.log_every == 0:
            el = time.time() - t0
            _s = (f" ls={float(log_scale):.2f} sc={last_scale:.3f}"
                  if log_scale is not None else "")
            say(f"[csi] {it}/{args.iterations} loss={float(loss):.4f}"
                + _s + f" {it/el:.2f} it/s  {el/60:.1f} min")
        if args.save_every and it % args.save_every == 0:
            save_tx_state(os.path.join(args.model_path, f"csi_{it}.pth"),
                          head, embed, it, extra={"csi_T": args.T,
                                                  "log_scale": (float(log_scale.detach())
                                                                if log_scale is not None else 0.0),
                                                  "csi_dt_ns": args.dt_ns,
                                                  "has_dtau": dtau is not None},
                          opacity=gaussians._opacity)
            if dtau is not None:
                torch.save(dtau.detach().cpu(),
                           os.path.join(args.model_path, f"dtau_{it}.pt"))
    save_tx_state(os.path.join(args.model_path,
                               f"csi_{args.iterations}.pth"),
                  head, embed, args.iterations,
                  extra={"csi_T": args.T, "csi_dt_ns": args.dt_ns,
                         "log_scale": (float(log_scale.detach())
                                       if log_scale is not None else 0.0),
                         "has_dtau": dtau is not None},
                  opacity=gaussians._opacity)
    if dtau is not None:
        torch.save(dtau.detach().cpu(),
                   os.path.join(args.model_path, f"dtau_{args.iterations}.pt"))
    say(f"[csi] done {args.iterations} steps")


if __name__ == "__main__":
    main()
