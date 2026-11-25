# -*- coding: utf-8 -*-
"""Bi-SG 角度瓣头：在相干复场上乘一个**方向相关**的散射核。

动机（GSBF, arXiv:2608.05896 的 Bi-SG kernel）
--------------------------------------------------------------------------------
已有的头都是「逐高斯一个复标量」：`coh` 的 Re/Im 由 MLP 从位置/视线/参数预测，
方向相关性只存在于共享 MLP 的输入里。GSBF 的做法是给每个高斯一个显式的双向散射核

    S_i(ŝ_in, ŝ_out) = α_i + Σ_{k=1..K} a_{k,i} · exp[ κ_{k,i} (cosθ_{k,i} − 1) ]

其中 a 是复幅度、v_k 是可学习轴向、κ_k 是锐度；cosθ 在「近共线传播」与「折返散射」
之间按入射/出射的共线度加权：

    cosθ_k = σ(γ(cosφ − η))·(ŝ_in·v_k)(ŝ_out·v_k) + (1−σ)·(e·v_k)

本仓的旧记录把「相干」与「解析 AF」两条线都判死了
（`results/_w_coh_gate640/`、`results/_w_loss640/`），但都没有**方向瓣**：它们仍是逐点
复标量。这里补上的正是那一块。

本实现的选择（与 GSBF 的差异，写清楚）
--------------------------------------------------------------------------------
1. **摊销而非逐场景优化**：轴/锐度/幅度都由共享 MLP 从逐点特征预测（外加 K 个全局
   基轴），所以换发射源仍是前馈；GSBF 是单场景拟合。
2. **乘性调制**：`z_p = z_base,p · S_p`，`z_base` 就是 `coh` 头的复场。因此
   `a_k → 0` 时 **精确退化**成相干臂——这是本臂的嵌套对照。
3. **不套 sigmoid**：与 coh 头一致，输出原始 Re/Im。

冷启动
--------------------------------------------------------------------------------
- 基座两通道 bias 取 `(a·cos0.5, a·sin0.5)`（与 coh 头一致，模长 = init_u）。
- 幅度通道权重置零、bias 取小的互异值 ⇒ 初始 S ≈ 1，两臂在起点几乎逐位相同。
- 轴通道权重为默认初始化 ⇒ 初始轴分散；κ = softplus(·) + κ0。

跑法
----
    训练：--w_head bisg --n_bands 1 --band_mix single（头内部用 2 通道/coh 合成）
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from splat.color_mlp import _dist, make_dir
from splat.color_scalar_coh import TxCoherentColorHead

HEAD_TYPE = "scalar_jet_tx_bisg"


class TxBiSGColorHead(TxCoherentColorHead):
    """→ (P,2)：相干复场 × Bi-SG 角度瓣调制。"""

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5, n_ant: int = 16, use_af: bool = False,
                 n_lobes: int = 4, bisg_eta: float = 0.5,
                 bisg_gamma: float = 8.0, bisg_kappa0: float = 4.0,
                 use_geom: bool = False):
        super().__init__(
            embed_dim=embed_dim, hidden=hidden, depth=depth,
            pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L, pe_rel_L=pe_rel_L,
            pe_tx_L=pe_tx_L, use_dist=use_dist, use_rel=use_rel, use_tx=use_tx,
            use_yaw=use_yaw, use_tx_orient=use_tx_orient,
            use_tx_local_dir=use_tx_local_dir, use_tx_freq=use_tx_freq,
            freq_lo_hz=freq_lo_hz, freq_hi_hz=freq_hi_hz,
            init_u=init_u, n_ant=n_ant, use_af=use_af, use_geom=use_geom)
        self.head_type = HEAD_TYPE
        self.n_lobes = int(n_lobes)
        self.bisg_eta = float(bisg_eta)
        self.bisg_gamma = float(bisg_gamma)
        self.bisg_kappa0 = float(bisg_kappa0)
        K = self.n_lobes
        hid = self.out.in_features
        # 输出：基座 Re/Im(2) + 复幅度 a_k(2K) + 轴偏移(3K) + 锐度 logit(K)
        self.out = nn.Linear(hid, 2 + 2 * K + 3 * K + K)
        # 全局基轴（每瓣一个方向），预测的是它的偏移
        base = torch.randn(K, 3)
        base = base / base.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.lobe_axes = nn.Parameter(base)
        a = float(init_u)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()
            self.out.bias[0] = a * math.cos(0.5)
            self.out.bias[1] = a * math.sin(0.5)
            # 幅度：小的互异初值，打破瓣间对称；权重为零 ⇒ 起点 S≈1
            for k in range(K):
                self.out.bias[2 + 2 * k] = 0.01 * math.cos(0.7 * (k + 1))
                self.out.bias[3 + 2 * k] = 0.01 * math.sin(0.7 * (k + 1))

    # ------------------------------------------------------------------ #
    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None, normal=None, aniso=None) -> torch.Tensor:
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=None, normal=normal, aniso=aniso)
        o = self.out(self.net(x))
        K = self.n_lobes
        i0 = 2
        i1 = i0 + 2 * K
        i2 = i1 + 3 * K
        zr, zi = o[:, 0], o[:, 1]
        amp = o[:, i0:i1].reshape(-1, K, 2)           # [P,K,2] (Re,Im)
        dv = o[:, i1:i2].reshape(-1, K, 3)            # 轴偏移
        kap = F.softplus(o[:, i2:i2 + K]) + self.bisg_kappa0

        # 入射（TX→点）与出射（点→相机）方向，世界系
        tin = xyz - tx.reshape(1, 3)
        u_in = tin / tin.norm(dim=1, keepdim=True).clamp_min(1e-6)
        u_out = dirv
        cphi = (u_in * u_out).sum(-1, keepdim=True)                  # [P,1]
        e = u_in - u_out
        e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-6)
        w = torch.sigmoid(self.bisg_gamma * (cphi - self.bisg_eta))  # [P,1]
        v = self.lobe_axes[None] + 0.3 * dv
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)         # [P,K,3]
        c1 = (u_in[:, None, :] * v).sum(-1)                          # [P,K]
        c2 = (u_out[:, None, :] * v).sum(-1)
        ce = (e[:, None, :] * v).sum(-1)
        cos_t = w * (c1 * c2) + (1.0 - w) * ce
        E = torch.exp(kap * (cos_t - 1.0))

        Sr = 1.0 + (amp[..., 0] * E).sum(-1)
        Si = (amp[..., 1] * E).sum(-1)
        return torch.stack([zr * Sr - zi * Si, zr * Si + zi * Sr], dim=1)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: dict) -> "TxBiSGColorHead":
        h = cls(
            embed_dim=cfg["embed_dim"], hidden=cfg["mlp_hidden"],
            depth=cfg["mlp_depth"], pe_xyz_L=cfg["pe_xyz_L"],
            pe_dir_L=cfg["pe_dir_L"], pe_rel_L=cfg.get("pe_rel_L", 0),
            pe_tx_L=cfg.get("pe_tx_L", 0), use_dist=cfg["use_dist"],
            use_rel=cfg.get("use_rel", False), use_tx=cfg.get("use_tx", False),
            use_yaw=cfg.get("use_yaw", False),
            use_tx_orient=cfg.get("use_tx_orient", False),
            use_tx_local_dir=cfg.get("use_tx_local_dir", False),
            use_tx_freq=cfg.get("use_tx_freq", False),
            freq_lo_hz=cfg.get("freq_lo_hz", 2.0e9),
            freq_hi_hz=cfg.get("freq_hi_hz", 4.5e9),
            init_u=cfg["init_u"], n_ant=cfg.get("n_ant", 16),
            use_af=cfg.get("use_af", False),
            n_lobes=cfg.get("n_lobes", 4), bisg_eta=cfg.get("bisg_eta", 0.5),
            bisg_gamma=cfg.get("bisg_gamma", 8.0),
            bisg_kappa0=cfg.get("bisg_kappa0", 4.0),
            use_geom=cfg.get("use_geom", False))
        h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
        h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
        return h


def compute_scalar_bisg(head: TxBiSGColorHead, embed: torch.Tensor,
                        xyz: torch.Tensor, cam_center: torch.Tensor,
                        spatial_lr_scale: float, tx: torch.Tensor,
                        yaw=None, tx_yaw=None, tx_pitch=None,
                        tx_freq=None, tx_rot=None, tx_w=None):
    """与 `compute_scalar_tx` 同签名，返回 (P,2)。"""
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    s = head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
             tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
             tx_rot=tx_rot, tx_w=None)
    if head.use_af:
        from splat.color_scalar_tx_cplx import w_from_params
        from splat.color_scalar_af import array_factor_complex
        wr, wi = w_from_params(tx_w, head.n_ant, xyz.device, dtype=xyz.dtype)
        ar, ai = array_factor_complex(head, xyz, tx, tx_rot, wr, wi)
        zr, zi = s[:, 0], s[:, 1]
        s = torch.stack([zr * ar + zi * ai, zi * ar - zr * ai], dim=1)
    return s


def bisg_head_cfg(head: TxBiSGColorHead, extra: Optional[dict] = None) -> dict:
    from splat.color_scalar_tx import _base_head_cfg
    cfg = _base_head_cfg(head)
    cfg.update({
        "head_type": HEAD_TYPE,
        "n_ant": int(head.n_ant), "use_af": bool(head.use_af),
        "n_lobes": int(head.n_lobes), "bisg_eta": float(head.bisg_eta),
        "bisg_gamma": float(head.bisg_gamma),
        "bisg_kappa0": float(head.bisg_kappa0),
        "band_mix": "coh",
    })
    if extra:
        cfg.update(extra)
    return cfg
