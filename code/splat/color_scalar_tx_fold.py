# -*- coding: utf-8 -*-
"""相干折叠头：逐高斯输出 **N 个阵元的复场**，w 在输出侧做相干叠加再取模。

与「5 维条件头」和「二次型头」的区别
--------------------------------------------------------------------------------
`color_scalar_tx.py` 把 w 压成 5 维实数条件喂 MLP —— 已判死在表示层
（跨配置 corr≈0，给对 w 与给错 w 只差 0.01 dB）。

`color_scalar_tx_cplx.py` 把 w 放到输出侧，但收缩式是 **|wᴴc|²**（二次型），
那是 **MVDR** 的拓扑（功率谱）。本数据集是 **CBF**，生成器存下来的是
**幅度谱**（`cbf_spectrum` 返回 np.abs，没有平方），所以对 CBF 正确的折叠是

    A_p = | Σ_n conj(w_n) · c_{p,n} |          （取模，**不平方**）
    u   = (render(A) − min) / (max − min)       （渲染后、可见像素上的 min-max）

这个形式不是近似：`code/tools/_chk_w_fold.py` 已在真值上验成恒等式
（最大相对误差 8.6e-07）。所以 w 不再是网络要学的函数，而是一个**解析算子**；
网络只负责与 w 无关的逐阵元场 c_n。

归一化要**两级**：先逐点、再逐图
--------------------------------------------------------------------------------
只做一级都不行，两条臂的冒烟读数把这件事测了出来（同一份数据、100 步）：

| 归一化 | u-RMSE | ratio |
|---|---|---|
| 只在头里（全点云 min-max） | 0.1433 | 0.283 |
| 只在渲染后（可见像素 min-max） | 0.4603 | 0.007 |

- **只做渲染后**：光栅化前每个点可以是任意大的幅度，单个离群高斯就能把整幅图的
  `max−min` 撑爆，归一化后其余全被压到 0 附近，梯度随之消失 —— 训练几乎不动。
- **只做头里**：min/max 是**全部 P 个高斯**上的（P≈1e6，绝大多数在视锥外），
  可见区的动态范围被按全点云极值压扁，`ratio = RMS(Δu)/RMS(ΔGT)` 跟着偏小。

两级串起来正好各修一半：逐点 min-max 把每个点**钳到 [0,1]**（离群点无处放大），
渲染后再按可见像素 min-max 把整幅图拉回 [0,1]（与 GT 的构造口径一致）。
注意 GT 本来就是**逐图**归一化的，所以「每个 w 各用各的常数」是对齐的，不是失真。

两级的 min/max 都 **detach**：只当归一化常数，不参与反传。
整幅图恒定（max−min < 1e-8）时原样返回，避免 0/0。

初始化
--------------------------------------------------------------------------------
|c|=0 时 A≡0，|·| 在 0 处次梯度为 0 ⇒ w 通路一开始就死（min-max 也会退化成
常量图、梯度同样为零）。故末层偏置按 `c_init` 给非零随机初值（与二次型头同法），
权重仍取小方差。代价：第 0 步不再是常量 u=init_u，而是「随机场归一化后的图」。
这是本头与历史臂唯一的口径差异，`init_u` 在这里只用于记录，不参与输出。

`P` 可达 1e6，全部运算是逐点 + 两个全局 reduction，没有近邻 gather。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from splat.color_mlp import _dist, make_dir
from splat.color_scalar_tx import TxScalarColorHead

HEAD_TYPE = "scalar_jet_tx_fold"


class TxFoldColorHead(TxScalarColorHead):
    """逐高斯输出 N 个复阵元场；`forward` 返回**未折叠**的原始系数 (P, 2N)。

    折叠在 `compute_scalar_fold()` 里做：w 是逐实例的，不能进模型参数。
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5, n_ant: int = 16,
                 c_init: float = 0.3, a_init: float = 0.01,
                 norm: str = "minmax", post_norm: str = "none",
                 eps_norm: float = 1e-8):
        super().__init__(
            embed_dim=embed_dim, hidden=hidden, depth=depth,
            pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L, pe_rel_L=pe_rel_L,
            pe_tx_L=pe_tx_L, use_dist=use_dist, use_rel=use_rel, use_tx=use_tx,
            use_yaw=use_yaw, use_tx_orient=use_tx_orient,
            use_tx_local_dir=use_tx_local_dir, use_tx_freq=use_tx_freq,
            use_tx_w=False,                       # w 只走折叠，不进 MLP
            freq_lo_hz=freq_lo_hz, freq_hi_hz=freq_hi_hz,
            init_u=init_u, n_bands=1, band_mix="single")
        n_ant = int(n_ant)
        if n_ant < 1:
            raise ValueError(f"n_ant 须 >= 1，收到 {n_ant}")
        if c_init <= 0:
            raise ValueError("c_init 须 > 0（|·| 在 0 处次梯度为零，w 通路会死）")
        for _nm, _v in (("norm", norm), ("post_norm", post_norm)):
            if _v not in ("minmax", "none"):
                raise ValueError(f"{_nm} 须 ∈ {{'minmax','none'}}，收到 {_v!r}")
        self.head_type = HEAD_TYPE
        self.n_ant = n_ant
        self.c_init = float(c_init)
        self.a_init = float(a_init)
        self.norm = norm                 # 逐点（全点云）min-max
        self.post_norm = post_norm       # 渲染后（可见像素）min-max
        self.eps_norm = float(eps_norm)

        # 末层：N 组复数（实部/虚部交错，视图按 (P,N,2) 读）
        self.out = nn.Linear(hidden, 2 * self.n_ant)
        self._init_out()

    # ------------------------------------------------------------------ #
    def post_render(self, img: torch.Tensor) -> torch.Tensor:
        """渲染后的逐图 min-max（见模块头「归一化必须放在渲染之后」）。

        `combine_bands()` 会在返回前调用本方法（只对有 `post_render` 的头生效），
        所以训练、渲染、评测三条路都自动同口径，不必各自改。
        """
        if self.post_norm != "minmax":
            return img
        mn = img.min().detach()
        mx = img.max().detach()
        rng = mx - mn
        if float(rng) < 1e-8:
            return img
        return (img - mn) / rng

    def _init_out(self):
        nn.init.normal_(self.out.weight, mean=0.0, std=self.a_init)
        with torch.no_grad():
            self.out.bias.normal_(mean=0.0, std=self.c_init)

    # ------------------------------------------------------------------ #
    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None) -> torch.Tensor:
        """→ (P, 2·N_ant) 原始系数。**不折叠、不出 sigmoid、不归一化**。"""
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=None)
        return self.out(self.net(x))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: dict) -> "TxFoldColorHead":
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
            c_init=cfg.get("c_init", 0.3), a_init=cfg.get("a_init", 0.01),
            norm=cfg.get("norm", "minmax"),
            post_norm=cfg.get("post_norm", "none"),
            eps_norm=cfg.get("eps_norm", 1e-8))
        h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
        h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
        return h


# ---------------------------------------------------------------------- #
def fold(head: TxFoldColorHead, raw: torch.Tensor,
         w_re: torch.Tensor, w_im: torch.Tensor) -> torch.Tensor:
    """原始系数 (P,2N) × w → 逐点**原始幅度** A (P,1) ≥ 0。

        pre = Re(Σ_n conj(w_n)c_n) = Σ_n (w_r·c_r + w_i·c_i)
        pim = Im(Σ_n conj(w_n)c_n) = Σ_n (w_i·c_r − w_r·c_i)
        A   = sqrt(pre² + pim²)

    第一级归一化（`norm='minmax'`）：在**全部 P 个点**上做 min-max，把每个点钳到
    [0,1]。目的是不许单个离群高斯把渲染图的量程撑爆（见模块头「两级归一化」）。
    第二级（可见像素）在 `post_render()` 里做。
    """
    p = raw.shape[0]
    z = raw.view(p, head.n_ant, 2)
    cr, ci = z[:, :, 0], z[:, :, 1]
    pre = (cr * w_re + ci * w_im).sum(-1)          # (P,)
    pim = (ci * w_re - cr * w_im).sum(-1)
    a = torch.sqrt(pre.square() + pim.square() + 1e-12)
    if head.norm == "minmax":
        mn = a.min().detach()
        mx = a.max().detach()
        a = (a - mn) / (mx - mn).clamp_min(head.eps_norm)
    return a.reshape(-1, 1)


def compute_scalar_fold(head: TxFoldColorHead, embed: torch.Tensor,
                        xyz: torch.Tensor, cam_center: torch.Tensor,
                        spatial_lr_scale: float, tx: torch.Tensor,
                        yaw=None, tx_yaw=None, tx_pitch=None,
                        tx_freq=None, tx_rot=None, tx_w=None):
    """与 `compute_scalar_tx` 同签名、同输入，返回 (P,1) ∈ [0,1]。"""
    from splat.color_scalar_tx_cplx import w_from_params   # 码本唯一定义处
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    raw = head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
               tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq, tx_rot=tx_rot)
    wr, wi = w_from_params(tx_w, head.n_ant, raw.device, dtype=raw.dtype)
    return fold(head, raw, wr, wi)


def fold_head_cfg(head: TxFoldColorHead, extra: Optional[dict] = None) -> dict:
    from splat.color_scalar_tx import _base_head_cfg
    cfg = _base_head_cfg(head)
    cfg.update({
        "head_type": HEAD_TYPE,
        "n_ant": int(head.n_ant), "c_init": float(head.c_init),
        "a_init": float(head.a_init), "norm": head.norm,
        "post_norm": head.post_norm, "eps_norm": float(head.eps_norm),
        "w_form": "postminmax(render(minmax_points(|sum_n conj(w_n) c_n|)))",
    })
    if extra:
        cfg.update(extra)
    return cfg


_ = math  # 保持与同目录其它头一致的导入形状
