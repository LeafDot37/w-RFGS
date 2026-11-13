# -*- coding: utf-8 -*-
"""相干双通道标量头：把逐高斯复场渲成 (Re, Im) 两通道，像素上再取模。

为什么要有这个头（闸一的实测结论）
--------------------------------------------------------------------------------
`results/_w_af_gate/说明.md` 第六节把三种渲染算子放在同一批自由参数下比：

    逐点取模再叠加（历史口径）      w 盲拟合 u-RMSE 0.1224
    相干双通道、像素上再取模        w 盲拟合 u-RMSE **0.0985**，corr 0.184 → **0.286**

机理：光栅化器对颜色是**逐通道线性**叠加（`C[ch] += f[id*C+ch]·α·T`），所以把
复场的实部、虚部分别放进两个通道渲染一次，得到的混合场就是

    R(pix) = Σ_i α_i T_i · Re_i,    G(pix) = Σ_i α_i T_i · Im_i

即**跨高斯的相干求和**（权重 α_i T_i ≥ 0 是实数）。最后在像素上取
`sqrt(R²+G²)`。历史口径是逐高斯先取模 `|z_i|` 再叠加，等于在逐高斯处就丢掉
跨点相位 —— 消去项全丢，图像只能变亮不会变暗。

这个头把「相干」做成**基础保真度**的杠杆：它连 w 都不需要就能压低误差
（0.1224 → 0.0985），是在 w 轴之外唯一实测有效的提升项。`use_af=True` 时再把
解析阵列因子以**复数**乘进来（`z · conj(AF)`），w 因此同时进入幅度与相位。

输出层为什么不套 sigmoid
--------------------------------------------------------------------------------
基类 `forward` 出 `sigmoid(out(net(x)))`。本头的两个通道是 Re/Im，可以是负数，
sigmoid 会把它压成 (0,1) 并丢掉符号 —— 复场退化成第一象限的实场，等于把
相干性的一半废掉。所以 `forward` 直接返回原始 logit（(P,2)）。

冷启动：权重 0 + 偏置 `(0.878·a, 0.479·a)`（模长 = `init_u`，`a = init_u`）。
**必须两个通道都非零且不相等**：
- 若两通道都为 0，`sqrt(R²+G²)` 在 0 处对任一通道的导数都是 0 ⇒ 两通道永久冻结；
- 若两通道相等，梯度也相等 ⇒ 相对相位锁死，复场退化成带符号的实场。
0.878 / 0.479 是 cos(0.5) / sin(0.5)，模长恰好 1，乘 `init_u` 后初值幅度与历史头一致。

不做渲染后 min-max：GT 是逐图 min-max，但幅度由 `sqrt` 给出、尺度是**可观测**的
（损失对这个尺度敏感），所以不需要也不应该再归一化一次。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from splat.color_mlp import _dist, make_dir
from splat.color_scalar_tx import TxScalarColorHead

HEAD_TYPE = "scalar_jet_tx_coh"


class TxCoherentColorHead(TxScalarColorHead):
    """→ (P,2)：逐高斯复场的 (Re, Im)；`use_af` 时再乘 `conj(AF(w))`。"""

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5, n_ant: int = 16, use_af: bool = False):
        super().__init__(
            embed_dim=embed_dim, hidden=hidden, depth=depth,
            pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L, pe_rel_L=pe_rel_L,
            pe_tx_L=pe_tx_L, use_dist=use_dist, use_rel=use_rel, use_tx=use_tx,
            use_yaw=use_yaw, use_tx_orient=use_tx_orient,
            use_tx_local_dir=use_tx_local_dir, use_tx_freq=use_tx_freq,
            use_tx_w=False,                       # w 只走解析阵列因子（复数）
            freq_lo_hz=freq_lo_hz, freq_hi_hz=freq_hi_hz,
            init_u=init_u, n_bands=2, band_mix="coh")
        n_ant = int(n_ant)
        if use_af:
            from splat.color_scalar_af import element_positions
            element_positions(n_ant)              # 早失败：非方阵直接报错
        self.head_type = HEAD_TYPE
        self.n_ant = n_ant
        self.use_af = bool(use_af)

        # 基类按 n_bands>1 建了 lam_raw（给 mult 用）。本头不用它，删掉以免留下
        # 一个永远拿不到梯度的参数，写进 cfg 时也会被 band_lambda() 误报。
        if hasattr(self, "lam_raw"):
            del self.lam_raw

        # 冷启动（见模块头）：模长 = init_u，两通道不相等、都非零。
        a = float(init_u)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[0] = a * math.cos(0.5)
            self.out.bias[1] = a * math.sin(0.5)

    # ------------------------------------------------------------------ #
    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None) -> torch.Tensor:
        """→ (P,2) **原始 logit**：(Re, Im)。不出 sigmoid（见模块头）。"""
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=None)
        return self.out(self.net(x))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: dict) -> "TxCoherentColorHead":
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
            use_af=cfg.get("use_af", False))
        h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
        h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
        return h


# ---------------------------------------------------------------------- #
def compute_scalar_coh(head: TxCoherentColorHead, embed: torch.Tensor,
                       xyz: torch.Tensor, cam_center: torch.Tensor,
                       spatial_lr_scale: float, tx: torch.Tensor,
                       yaw=None, tx_yaw=None, tx_pitch=None,
                       tx_freq=None, tx_rot=None, tx_w=None):
    """与 `compute_scalar_tx` 同签名，返回 (P,2) = 逐高斯复场的 (Re, Im)。"""
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    s = head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
             tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
             tx_rot=tx_rot, tx_w=None)                        # (P,2)
    if head.use_af:
        from splat.color_scalar_tx_cplx import w_from_params   # 码本唯一定义处
        from splat.color_scalar_af import array_factor_complex
        wr, wi = w_from_params(tx_w, head.n_ant, xyz.device, dtype=xyz.dtype)
        ar, ai = array_factor_complex(head, xyz, tx, tx_rot, wr, wi)
        zr, zi = s[:, 0], s[:, 1]
        # z · conj(AF) = (zr·ar + zi·ai) + j·(zi·ar − zr·ai)
        s = torch.stack([zr * ar + zi * ai, zi * ar - zr * ai], dim=1)
    return s


def coh_head_cfg(head: TxCoherentColorHead, extra: Optional[dict] = None) -> dict:
    from splat.color_scalar_tx import _base_head_cfg
    cfg = _base_head_cfg(head)
    cfg.update({
        "head_type": HEAD_TYPE,
        "n_ant": int(head.n_ant),
        "use_af": bool(head.use_af),
        "band_mix": "coh",
        "w_form": ("render([Re(z·conj(AF)), Im(z·conj(AF))]) -> sqrt(R²+G²) -> jet"
                   if head.use_af else
                   "render([Re z, Im z]) -> sqrt(R²+G²) -> jet（w 不进图）"),
    })
    if extra:
        cfg.update(extra)
    return cfg
