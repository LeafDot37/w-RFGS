# -*- coding: utf-8 -*-
"""复值/二次型波束头：让**波束权 w 以闭式形式**进入渲染，而不是当 5 维实数条件喂 MLP。

为什么必须换掉「5 维条件」这条路
--------------------------------------------------------------------------------
`color_scalar_tx.py` 的波束块把 w 压成 `[sinθ_w, cosθ_w, sinφ_w, cosφ_w, uniform]`。
2026-10-29/30 的 180k 两臂实验判死在**表示层**（见
`results/run_20261029_服务器4090D_波束权泛化/说明.md` §0.3、§0.4）：

- 留出 w 上条件臂 − 盲臂 = +0.0012 dB（噪声地板 0.365）；
- 训练集内同位置 4 组 w：GT 之间 13.30 dB，模型输出之间 20.32 dB
  ⇒ 只表达了 **19.9%** 的 w 能量，**连见过的 w 都画不出来**；
- 把 w 换成同位置训练域的另一组，渲染图只动 **0.09 dB**（GT 动 18.80 dB）。

也就是说 MLP 把 w 这几列学成了死列。本文件给的替代表示是：**w 不在输入里，而在输出之后**
——逐高斯输出一组复系数 c，用码本里的 w 做一次收缩，再取模方。

表示（本文件的核心）
--------------------------------------------------------------------------------
    q_k(p) = |wᴴ · c_k(p)|²,   c_k(p) ∈ ℂ^{N_ant}，N_ant = 16（4×4 阵）
    u(p)   = σ( a(p) + Σ_{k<n_pos} q_k(p) − Σ_{k≥n_pos} q_k(p) )

三点说明：

1. **Π 项（n_pos）是 CBF 的闭式结构**。CBF 图 `p = |Σ_n conj(w_n)·S_n|²`，若把 `S_n`
   换成逐高斯的复响应 `c_n`，就正好是 `q_1`。所以 `n_neg=0` 这一档等价于「复值场」，
   是说明.md §0.5 里点名「没测过」的第一条。
2. **减号项（n_neg）是 MVDR 的拓扑**。MVDR `/正则化 Capon` 的亮像素是**零陷**：
   `P_v ≈ ε/(|a_v|²·sin²∠(a_v, u))`，亮区是「w 与该像素方向正交」的那张超平面。
   `−|wᴴc|²` 在 `c` 等于该像素的匹配向量时，恰好在零陷超平面上取 0、其余方向取负
   ⇒ 加号取一次「波束在上」，减号取一次「零陷在上」，两种拓扑都能表达。
   （2026-10-30 的 `_chk_w_factor_image.py` 已证「逐像素波束增益乘子」不成立：corr≈0.008，
   所以 CBF 那套纯乘法因子对 MVDR 是错的，减号项不是可选项而是必需项。）
3. **正负项合并就是一个一般的（不定）Hermite 二次型** `wᴴ C w`，`C = Σ_k s_k c_k c_kᴴ`，
   秩 ≤ n_pos+n_neg、符号由 s_k 给。对「留出 w 是方向内插」这条泛化轴，二次型在 w 方向
   上光滑 ⇒ 内插天然成立，这正是需要的归纳偏置。

初始化（`|C|=0` 是鞍点）
--------------------------------------------------------------------------------
`d|wᴴc|²/dc ∝ wᴴc·w`，在 c=0 处为零 ⇒ 末层若按父类那样零初始化权重，w 这条通路
一开始就死。故末层权重与 c 部分用小方差**非零**初始化（见 `_init_out`），
`a` 的偏置仍取 `logit(init_u)`，保证第 0 步图像接近常量。

与父类的关系
--------------------------------------------------------------------------------
输入特征矩阵**逐位复用** `TxScalarColorHead._features()`（几何/视线/发射源/参数编码
完全一致），差别只在末层宽度与后处理。这样「复值头 vs 5 维条件头」是同输入、
同步数、只差输出侧的干净单变量对照。`use_tx_w` 在本类里强制为 False。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from splat.color_mlp import _dist, make_dir
from splat.color_scalar_tx import TxScalarColorHead

HEAD_TYPE = "scalar_jet_tx_cplx"


def _logit(u: float) -> float:
    u = min(max(float(u), 1e-4), 1 - 1e-4)
    return math.log(u / (1.0 - u))


class TxComplexColorHead(TxScalarColorHead):
    """逐高斯输出 `n_pos + n_neg` 组复系数 + 一个静态偏置 `a`。

    `forward` 返回**未收缩**的原始系数 `(P, 1 + 2·N_ant·n_k)`，收缩在
    `compute_scalar_cplx()` 里做（w 是逐实例的，不能进模型参数）。
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5,
                 n_ant: int = 16, n_pos: int = 1, n_neg: int = 1,
                 c_init: float = 0.3, a_init: float = 0.01):
        super().__init__(
            embed_dim=embed_dim, hidden=hidden, depth=depth,
            pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L, pe_rel_L=pe_rel_L,
            pe_tx_L=pe_tx_L, use_dist=use_dist, use_rel=use_rel, use_tx=use_tx,
            use_yaw=use_yaw, use_tx_orient=use_tx_orient,
            use_tx_local_dir=use_tx_local_dir, use_tx_freq=use_tx_freq,
            use_tx_w=False,                       # w 只走收缩，不进 MLP
            freq_lo_hz=freq_lo_hz, freq_hi_hz=freq_hi_hz,
            init_u=init_u, n_bands=1, band_mix="single")
        n_ant, n_pos, n_neg = int(n_ant), int(n_pos), int(n_neg)
        if n_ant < 1:
            raise ValueError(f"n_ant 须 >= 1，收到 {n_ant}")
        if n_pos + n_neg < 1:
            raise ValueError("n_pos + n_neg 须 >= 1（否则 w 完全不进图）")
        if c_init <= 0:
            raise ValueError(f"c_init 须 > 0（|wᴴc|² 在 c=0 处梯度为零）")
        self.head_type = HEAD_TYPE
        self.n_ant = n_ant
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_k = n_pos + n_neg
        self.c_init = float(c_init)
        self.a_init = float(a_init)

        # 末层：1 个静态偏置 + n_k 组 (Re, Im) 各 N_ant 维
        self.out = nn.Linear(hidden, 1 + 2 * self.n_ant * self.n_k)
        self._init_out()

    def _init_out(self):
        nn.init.normal_(self.out.weight, mean=0.0, std=self.a_init)
        b = self.out.bias.detach()
        b[0] = _logit(self.init_u)
        b[1:] = torch.randn_like(b[1:]) * self.c_init

    # ------------------------------------------------------------------ #
    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None) -> torch.Tensor:
        """→ (P, 1 + 2·N_ant·n_k) 原始系数。**不出 sigmoid、不收缩**。"""
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=None)
        return self.out(self.net(x))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: dict) -> "TxComplexColorHead":
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
            n_pos=cfg.get("n_pos", 1), n_neg=cfg.get("n_neg", 1),
            c_init=cfg.get("c_init", 0.3), a_init=cfg.get("a_init", 0.01))
        h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
        h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
        return h


# ---------------------------------------------------------------------- #
# 波束权：码本是唯一定义点，这里只做 (θ,φ,uni) → (Re, Im) 的缓存查表
# ---------------------------------------------------------------------- #
_W_CACHE: dict = {}


def w_from_params(tx_w, n_ant: int, device, dtype=torch.float32):
    """`tx_w = [θ_deg, φ_deg, is_uniform]` → (w_re, w_im)，各 (N_ant,)。

    直接调 `sim.w_codebook`（符号与阵元序在那里实测核对过），不在这里重推公式。
    """
    if tx_w is None:
        raise ValueError("复值头必须给 tx_w = [θ_deg, φ_deg, is_uniform]")
    if torch.is_tensor(tx_w):
        tw = tx_w.detach().reshape(-1).cpu().tolist()
    else:
        tw = list(tx_w)
    if len(tw) < 2:
        raise ValueError(f"tx_w 至少要有 [θ_deg, φ_deg]，收到 {tw}")
    th, ph = round(float(tw[0]), 6), round(float(tw[1]), 6)
    uni = bool(int(round(float(tw[2])))) if len(tw) > 2 else False
    key = (th, ph, uni, int(n_ant))
    hit = _W_CACHE.get(key)
    if hit is None:
        from sim import w_codebook as wb
        nc = int(round(math.sqrt(n_ant)))
        if nc * nc != int(n_ant):
            raise ValueError(f"n_ant={n_ant} 不是方阵，码本只定义方阵")
        w = wb.uniform_weight(nc, nc) if uni else wb.direction_weight(th, ph, nc, nc)
        hit = (torch.as_tensor(w.real.copy(), dtype=dtype),
               torch.as_tensor(w.imag.copy(), dtype=dtype))
        _W_CACHE[key] = hit
    wr, wi = hit
    return wr.to(device), wi.to(device)


def contract(head: TxComplexColorHead, raw: torch.Tensor,
             w_re: torch.Tensor, w_im: torch.Tensor) -> torch.Tensor:
    """原始系数 (P, 1+2·N·K) × w → 标量 u (P,1)。

        q_k = |Σ_n conj(w_n)·c_kn|²
        u   = σ( a + Σ_{k<n_pos} q_k − Σ_{k≥n_pos} q_k )

    全程实数运算（复数张量在 P=1e6 上分配开销大）：复用同一份 (P,K,N) 视图，
    只读不写。
    """
    p = raw.shape[0]
    a = raw[:, 0:1]
    z = raw[:, 1:].view(p, head.n_k, 2, head.n_ant)
    cr = z[:, :, 0, :]                       # (P,K,N) 实部
    ci = z[:, :, 1, :]                       # (P,K,N) 虚部
    pre = (cr * w_re + ci * w_im).sum(-1)    # Re wᴴc
    pim = (ci * w_re - cr * w_im).sum(-1)    # Im wᴴc
    q = pre.square() + pim.square()          # (P,K)
    pos = q[:, :head.n_pos].sum(1, keepdim=True)
    neg = q[:, head.n_pos:].sum(1, keepdim=True)
    return torch.sigmoid(a + pos - neg)


def compute_scalar_cplx(head: TxComplexColorHead, embed: torch.Tensor,
                        xyz: torch.Tensor, cam_center: torch.Tensor,
                        spatial_lr_scale: float, tx: torch.Tensor,
                        yaw=None, tx_yaw=None, tx_pitch=None,
                        tx_freq=None, tx_rot=None, tx_w=None):
    """与 `compute_scalar_tx` 同签名、同输入，返回 (P,1)。"""
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    raw = head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
               tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq, tx_rot=tx_rot)
    wr, wi = w_from_params(tx_w, head.n_ant, raw.device, dtype=raw.dtype)
    return contract(head, raw, wr, wi)


def cplx_head_cfg(head: TxComplexColorHead, extra: Optional[dict] = None) -> dict:
    from splat.color_scalar_tx import _base_head_cfg
    cfg = _base_head_cfg(head)
    cfg.update({
        "head_type": HEAD_TYPE,
        "n_ant": int(head.n_ant), "n_pos": int(head.n_pos),
        "n_neg": int(head.n_neg), "n_k": int(head.n_k),
        "c_init": float(head.c_init), "a_init": float(head.a_init),
        "w_form": ("sigma(a + sum_k<n_pos |w^H c_k|^2 - sum_k>=n_pos |w^H c_k|^2)"),
    })
    if extra:
        cfg.update(extra)
    return cfg
