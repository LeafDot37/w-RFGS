# -*- coding: utf-8 -*-
"""解析阵列因子头：网络只学「每点一个标量」，阵元轴由阵元几何**解析**给出。

为什么换成这个（闸一的结论）
--------------------------------------------------------------------------------
`color_scalar_tx_fold.py`（逐阵元场全部由网络学）在闸一上判负：
`results/_w_fold_gate/` 实测 128 维隐特征在点云上 top1 占 99.99%、逐阵元复场矩阵
C(P×N) 的 rank-1 能量 = 1.0000，于是 c_{p,n} = α_n·g_p、折叠后只剩全局乘子 S(w)，
被逐图 min-max 精确抹掉（ratio 0.001，与盲臂的 0.000 不可分）。

根因：损失只看归一化后的图，全局标量不可观测 ⇒ 阵元轴拿不到梯度；主干也只要拟合
一张标量场 ⇒ 隐层没有动力维持高秩。**只修阵元轴的容量不解决问题，因为它会再塌回去。**

本头把阵元轴从「要学的东西」换成**闭式几何**：每个阵元对某个散射点的贡献只差一个
由阵元位置与来向决定的相位，

    Z_n(p) = g_p · exp(−j·2π·(y_n·u_y + z_n·u_z))        （远场，单一来向）

其中 (u_y, u_z) 是「发射源 → 该高斯」的方向在**发射源本地系**下的 y/z 分量，
(y_n, z_n) 是阵元位置（**波长**单位 ⇒ 与频率无关，见下）。于是

    A_p(w) = | Σ_n conj(w_n) · Z_n(p) |
           = |g_p| · | Σ_n conj(w_n) · exp(−j·2π·(y_n u_y + z_n u_z)) |
           = |g_p| · |AF_p(w)|

网络只出 g_p（与历史单发射源头同宽），w 的进入方式是**结构性的**，
阵元轴退化成常数的路径不存在：AF_p 依赖逐点来向，不同 w 给不同的空间图样。

约定（都来自实测，不在这里重推）
--------------------------------------------------------------------------------
- `code/sim/w_codebook.py` 文档头：`paths.a` 的**信道**相位实测为 exp(+j2π p_n·û)，
  匹配权 w_n = exp(−j2π(...))/√N（激励 x = w，不是 conj）。
- CBF 恒等式（`_chk_w_fold.py:156` 验到 8.6e-07）：I = |Σ_n conj(w_n)·Z_n|，
  而 `Z_n = Σ_m conj(x_n[m])·a_m` —— 这里对信道又取了一次共轭，两次相消后
  逐阵元因子落到 **exp(−j2π p_n·û)**。符号不能省：用 +j 时波束会照到镜像方向
  （û → −û），且这个错误不会在损失里显形为「学不动」，只会让 AF 张冠李戴。
  自检见 `code/tools/archive/_chk_af_sign.py`（峰值方向必须与 `direction_weight` 的 (θ,φ) 一致）。
- 阵元序与位置：`w_codebook.element_positions(nr, nc)`（y 慢轴升序、z 快轴降序），
  与 Sionna `PlanarArray.positions()` 同序。

频率为什么不进 AF
--------------------------------------------------------------------------------
数据集把阵元间距按**波长**给（vertical_spacing=0.5），即「同一电阵列」口径
（见 `docs/发射源参数泛化_改动说明.md` 阶段 1 的两条口径）。位置以波长为单位时
2π·p_n·û 与频率无关 —— 换频率时波束形状不变是这条口径的直接推论。

归一口径（踩过一次坑，别再改回去）
--------------------------------------------------------------------------------
A_p = |z_p| · |AF_p(w)|，然后除以**逐点均值**（detach）锚定尺度；渲染后再做逐图
min-max，与 GT 的 `per_image_minmax` 同口径。

**不要用 sigmoid 再配渲染后 min-max。** 实测（`/tmp/afsmoke`，200 步）：
渲染后 min-max 让损失对全局尺度完全不变，输出层没有任何锚 ⇒ logit 一路往负无穷跑；
sigmoid 一旦饱和到 σ ≈ 5e-34，其导数也是 5e-34，梯度再也回不来，A ≡ 0、渲染图恒零、
ratio 精确 0.000。检查点里 `out.bias = −3.000` 看着正常，但 `|W|₁ = 6.42` 配上
爆炸的隐特征，实际 logit 是 −70 量级 —— **只看偏置会误判成「头没塌」**。

`|z|` 的导数是 sign(z)，处处不退化；逐点均值归一化把尺度钉死（损失对它不变，
所以它本来也不会漂），离群点也不会像 min-max 那样吃掉量程。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from splat.color_mlp import _dist, make_dir
from splat.color_scalar_tx import TxScalarColorHead

HEAD_TYPE = "scalar_jet_tx_af"

_EL_CACHE: dict = {}


def element_positions(n_ant: int = 16):
    """阵元 (y, z) 位置（**波长**单位），返回 numpy (N,) 各一。"""
    from sim.w_codebook import element_positions as _pos
    nc = int(round(math.sqrt(int(n_ant))))
    if nc * nc != int(n_ant):
        raise ValueError(f"n_ant={n_ant} 不是方阵，本头只支持方阵发射阵")
    return _pos(nc, nc)


def _element_pos_t(head: "TxArrayFactorColorHead", device, dtype):
    key = (head.n_ant, str(device), str(dtype))
    hit = _EL_CACHE.get(key)
    if hit is None:
        y, z = element_positions(head.n_ant)
        hit = (torch.as_tensor(y, dtype=dtype, device=device),
               torch.as_tensor(z, dtype=dtype, device=device))
        _EL_CACHE[key] = hit
    return hit


class TxArrayFactorColorHead(TxScalarColorHead):
    """→ (P,1)：σ(MLP(features)) · |AF(tx_w, 逐点来向)|。"""

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5, n_ant: int = 16,
                 norm: str = "mean", post_norm: str = "minmax",
                 eps_norm: float = 1e-8, use_geom: bool = False):
        super().__init__(
            embed_dim=embed_dim, hidden=hidden, depth=depth,
            pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L, pe_rel_L=pe_rel_L,
            pe_tx_L=pe_tx_L, use_dist=use_dist, use_rel=use_rel, use_tx=use_tx,
            use_yaw=use_yaw, use_tx_orient=use_tx_orient,
            use_tx_local_dir=use_tx_local_dir, use_tx_freq=use_tx_freq,
            use_tx_w=False,                       # w 只走解析阵列因子
            freq_lo_hz=freq_lo_hz, freq_hi_hz=freq_hi_hz,
            init_u=init_u, n_bands=1, band_mix="single")
        n_ant = int(n_ant)
        element_positions(n_ant)                  # 早失败：非方阵直接报错
        if norm not in ("mean", "minmax", "none"):
            raise ValueError(f"norm 须 ∈ {{'mean','minmax','none'}}，收到 {norm!r}")
        if post_norm not in ("minmax", "none"):
            raise ValueError(f"post_norm 须 ∈ {{'minmax','none'}}，收到 {post_norm!r}")
        if use_geom:
            raise ValueError("该头不支持 use_geom（几何块只接在基础/复值二次型头上）")
        self.head_type = HEAD_TYPE
        self.n_ant = n_ant
        self.norm = norm                 # 逐点尺度锚定：mean（默认）/ minmax / none
        self.post_norm = post_norm       # 渲染后（可见像素）min-max；默认做
        self.eps_norm = float(eps_norm)

    # ------------------------------------------------------------------ #
    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None) -> torch.Tensor:
        """→ (P,1) **原始 logit**（不出 sigmoid，见模块头「归一口径」）。

        基类的 sigmoid 在这里是毒药：渲染后 min-max 让全局尺度不可观测，
        logit 无锚会走向负无穷，饱和后梯度恒零。幅度由 `|z|` 给出。
        """
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=None)
        return self.out(self.net(x))

    # ------------------------------------------------------------------ #
    def post_render(self, img: torch.Tensor) -> torch.Tensor:
        """渲染后逐图 min-max —— 与 GT 的 `normalize: per_image_minmax` 同口径。"""
        if self.post_norm != "minmax":
            return img
        mn = img.min().detach()
        mx = img.max().detach()
        rng = mx - mn
        if float(rng) < 1e-8:
            return img
        return (img - mn) / rng

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: dict) -> "TxArrayFactorColorHead":
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
            norm=cfg.get("norm", "mean"),
            post_norm=cfg.get("post_norm", "minmax"),
            eps_norm=cfg.get("eps_norm", 1e-8))
        h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
        h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
        return h


# ---------------------------------------------------------------------- #
def array_factor_complex(head: TxArrayFactorColorHead, xyz: torch.Tensor,
                         tx: torch.Tensor, tx_rot, w_re: torch.Tensor,
                         w_im: torch.Tensor):
    """返回 (re, im)：`Σ_n conj(w_n)·exp(−j·2π(y_n u_y + z_n u_z))`。

    分出实虚部是为了「先渲染再取模」这条路径：把 Re/Im 分别渲成两个通道，
    光栅化的线性叠加就等于**相干求和**，最后在像素上取 `sqrt(R²+G²)`。
    `array_factor` 只取模，等于在**逐点**就丢掉相位（跨高斯的相干性随之丢失）。
    """
    if tx_rot is None:
        raise ValueError("解析阵列因子头必须给 tx_rot")
    if tx.dim() == 1:
        tx = tx.unsqueeze(0)
    if tx.shape[0] == 1 and xyz.shape[0] != 1:
        tx = tx.expand(xyz.shape[0], 3)
    dlt = xyz - tx
    u = dlt / dlt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    R = torch.as_tensor(tx_rot, dtype=xyz.dtype, device=xyz.device)
    if R.shape != (3, 3):
        raise ValueError(f"tx_rot 须为 (3,3)，收到 {tuple(R.shape)}")
    uloc = u @ R                                   # (P,3)，列 = (x, y, z)

    ey, ez = _element_pos_t(head, xyz.device, xyz.dtype)      # (N,)
    ph = -2.0 * math.pi * (uloc[:, 1:2] * ey + uloc[:, 2:3] * ez)   # (P,N)，负号见模块头
    co, si = torch.cos(ph), torch.sin(ph)
    re = (w_re * co + w_im * si).sum(-1)            # Re Σ conj(w)·e^{-jφ}
    im = (w_re * si - w_im * co).sum(-1)            # Im 同上
    return re, im


def array_factor(head: TxArrayFactorColorHead, xyz: torch.Tensor,
                 tx: torch.Tensor, tx_rot, w_re: torch.Tensor,
                 w_im: torch.Tensor) -> torch.Tensor:
    """返回 (P,) ≥ 0：`|Σ_n conj(w_n)·exp(−j·2π(y_n u_y + z_n u_z))|`。

    (u_y, u_z) 是「发射源 → 高斯」方向在发射源本地系下的 y/z 分量，
    与 `_param_block` 的 `use_tx_local_dir` 逐位同源（`u @ R`）。
    """
    re, im = array_factor_complex(head, xyz, tx, tx_rot, w_re, w_im)
    return torch.sqrt(re.square() + im.square() + 1e-12)


def compute_scalar_af(head: TxArrayFactorColorHead, embed: torch.Tensor,
                      xyz: torch.Tensor, cam_center: torch.Tensor,
                      spatial_lr_scale: float, tx: torch.Tensor,
                      yaw=None, tx_yaw=None, tx_pitch=None,
                      tx_freq=None, tx_rot=None, tx_w=None):
    """与 `compute_scalar_tx` 同签名、同输入，返回 (P,1)。"""
    from splat.color_scalar_tx_cplx import w_from_params   # 码本唯一定义处
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    s = head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
             tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
             tx_rot=tx_rot, tx_w=None)                       # (P,1) ∈ (0,1)
    wr, wi = w_from_params(tx_w, head.n_ant, xyz.device, dtype=xyz.dtype)
    a = s.reshape(-1).abs() * array_factor(head, xyz, tx, tx_rot, wr, wi)
    if head.norm == "mean":
        a = a / a.detach().mean().clamp_min(head.eps_norm)
    elif head.norm == "minmax":
        mn = a.min().detach()
        mx = a.max().detach()
        a = (a - mn) / (mx - mn).clamp_min(head.eps_norm)
    return a.reshape(-1, 1)


def af_head_cfg(head: TxArrayFactorColorHead, extra: Optional[dict] = None) -> dict:
    from splat.color_scalar_tx import _base_head_cfg
    cfg = _base_head_cfg(head)
    cfg.update({
        "head_type": HEAD_TYPE,
        "n_ant": int(head.n_ant),
        "norm": head.norm, "post_norm": head.post_norm,
        "eps_norm": float(head.eps_norm),
        "w_form": "postminmax(render(|z|/mean|z| * |sum_n conj(w_n) e^{-j2pi(y_n u_y + z_n u_z)}|))",
    })
    if extra:
        cfg.update(extra)
    return cfg
