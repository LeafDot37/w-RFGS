# -*- coding: utf-8 -*-
"""前馈「光学点云 → 射频点云」头：**逐点读数，一次前向，零逐场景优化**。

和 `corf/coupled_head.py` 的关系
--------------------------------------------------------------------------------
`CoupledHead` 吃的是「位置 + 12 通道投影统计 + 射频近邻证据」，几何来自
MVSplat 前馈点云（未优化过）。本文件换成**已经被光学 3DGS 优化过的点云**：
每个点自带尺度、旋转、不透明度、3 阶球谐（48 系数），这些是「光学外观证据」。

    输入  xyz [P,3] + app [P,56]（scale·rot·opacity·SH，全部**激活前**参数）
    输出  射频不透明度 [P,1]（logit，交给渲染器的 sigmoid）
          + 射频球谐 [P,16,3]（`sh_bound·tanh(·)`，**有界**，见 `sh_bound` 的注释）

为什么值得换
--------------------------------------------------------------------------------
- 光学的上限由**几何**定，不由老师定：`L_opt` 本来就是对着真值图像算的。
- 优化过的点云把几何（位置/尺度/朝向/不透明度）与光学外观一起定下来了，
  射频头只需要学「这些光学属性 → 该点的频谱辐射」这一层跨模态映射。
- 输入是逐点的、稠密的、每个点都有的；不像 12 通道投影统计要在 809 张图里聚合，
  也不像 RF 近邻证据需要把测量位置也算进来。

两条臂
--------------------------------------------------------------------------------
    geo_app   位置 + 全部 56 维光学属性（主臂）
    geo_xyz   只用位置（阴性对照：光学外观到底贡献多少）

⚠ 本头的输入**不包含任何射频侧信息** ⇒ 推理时只需要一份光学 3DGS 重建，
  射频数据只用于训练。这正是「射频数据稀缺、光学数据充足」的落地方式。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from corf.rf_head import PositionalEncoding, RF_SH_C, RF_SH_COEFFS
from splat.color_mlp import fourier_feat

# 逐点光学属性的分段定义（build_app 的拼接顺序，改这里要同步改 APP_DIM）
APP_SEGS = (("scale", 3), ("rot", 4), ("opacity", 1), ("sh_dc", 3), ("sh_rest", 45))
APP_DIM = sum(n for _, n in APP_SEGS)          # 56


def build_app(gm) -> torch.Tensor:
    """从 `GaussianModel` 取逐点光学属性，拼成 [P, 56]（**激活前**参数）。

    - scale 是 log 尺度、opacity 是 logit、SH 是原始系数 —— 一律不做激活：
      激活函数是单调的，网络自己能学，做了反而压缩动态范围。
    - rot 做单位化：ply 里存的是未归一化四元数（实测模长 0.50–3.36），
      不归一化的话同一个旋转会对应不同输入，白送一份噪声。
    """
    p = gm.get_xyz.shape[0]
    scale = gm._scaling.detach().float().reshape(p, 3)
    rot = gm._rotation.detach().float().reshape(p, 4)
    rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    op = gm._opacity.detach().float().reshape(p, 1)
    dc = gm._features_dc.detach().float().reshape(p, 3)
    rest = gm._features_rest.detach().float().reshape(p, -1)
    app = torch.cat([scale, rot, op, dc, rest], dim=-1)
    if app.shape[1] != APP_DIM:
        raise RuntimeError(f"逐点属性维度 {app.shape[1]} != 约定的 {APP_DIM}")
    if not torch.isfinite(app).all():
        raise RuntimeError("逐点属性里有 NaN/Inf —— 点云本身不干净，别拿来训")
    return app.contiguous()


def app_stats(app: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """逐维均值/标准差（CPU float32）。训练与渲染两端必须用同一份统计量：
    训练时写进检查点 config，渲染时从 config 读，**不重新估**。"""
    a = app.detach().float()
    return a.mean(dim=0).cpu(), a.std(dim=0).cpu().clamp_min(1e-6)


class GeoToRfHead(nn.Module):
    def __init__(self, app_dim: int = 0, hidden: int = 64, n_layers: int = 3,
                 pe_bands: int = 6, chunk: int = 250_000,
                 sh_coeffs: int = RF_SH_COEFFS, sh_c: int = RF_SH_C,
                 pe_mean: torch.Tensor | None = None,
                 pe_std: torch.Tensor | None = None,
                 app_mean: torch.Tensor | None = None,
                 app_std: torch.Tensor | None = None,
                 opacity_init: float = 0.1, app_clip: float = 5.0,
                 sh_bound: float = 2.0,
                 out_kind: str = "sh", n_bands: int = 1, init_u: float = 0.5,
                 use_dir: bool = False, pe_dir_bands: int = 0,
                 use_dist: bool = False,
                 use_rel: bool = False, pe_rel_bands: int = 0):
        super().__init__()
        # 输出函数类：
        #   "sh"     逐点 16×3 球谐（原臂，输出空间大，实测会渲出调色板外的颜色）
        #   "scalar" 逐点 K 个标量 ∈ (0,1)；光栅化在**标量域**混合，查 jet 表得 RGB
        #            （与 splat/color_scalar.py 的形态 A'/A× 同一套机制）。
        #            依据：本数据集 GT 像素 100% 落在 jet 调色板上（tools/check_jet_palette.py
        #            实测中位距离 0.0022，而 SH 渲染只有 14–23% 落在 2/255 内）。
        if out_kind not in ("sh", "scalar"):
            raise ValueError(f"out_kind 须 ∈ {{'sh','scalar'}}，收到 {out_kind!r}")
        if int(n_bands) not in (1, 2, 3):
            # 光栅化器硬要求 NUM_CHANNELS == 3，多带只能借三个通道
            raise ValueError(f"n_bands 须 ∈ {{1,2,3}}，收到 {n_bands}")
        # 视线条件：逐场景标量头（run_20260912_多频段标量分支）的实测依据 ——
        # 同一逐点参数在不同视线下的标量 std = 0.061，占空间场 std(0.095) 的 64%
        # （tools/diag_scalar_smoothness.py）。只吃位置的标量头无法表示这 64%
        # ⇒ 训练视角都拟合不动（实测 11.7 dB，训练/测试几乎相同）。
        self.use_dir = bool(use_dir)
        self.pe_dir_bands = int(pe_dir_bands)
        self.use_dist = bool(use_dist)
        if self.use_dist and not self.use_dir:
            raise ValueError("use_dist=True 需要 use_dir=True（距离与方向成对使用）")
        # 发射源相对几何：显式把 Δ=xyz−tx、距离、单位方向喂进去。
        # 理由是空间场的主变化轴就是「点到发射源的距离」；让 MLP 自己从
        # γ(xyz) 与 γ(tx) 两组展开里做减法，在低样本数下是浪费容量。
        # 该假设本身要由 `空间场-相对特征` 消融臂验证，未验证前不当结论用。
        self.use_rel = bool(use_rel)
        self.pe_rel_bands = int(pe_rel_bands)
        self.out_kind = out_kind
        self.n_bands = int(n_bands)
        self.init_u = float(init_u)
        # 供 splat/color_scalar.combine_bands() 复用：K=1 即 A'，K=2 走乘性调制 A×
        self.band_mix = "mult" if self.n_bands > 1 else "single"
        self.app_dim = int(app_dim)
        # 球谐输出**有界**：sh = sh_bound·tanh(raw)。
        # 无界版本实测会发散 —— 训练的档 |SH| 均值从 ~0.1 一路涨到 82.8（老师是
        # 0.028），渲染被推爆、PSNR 掉到 8 dB；而损失里没有任何一项惩罚幅值。
        # 上界 2.0 的依据：光栅化算的是 0.5 + Σ C_k·sh_k，C_0=0.282，
        # 单点要渲满 [0,1] 只需 |sh|≈1.8；老师的 |SH| 上界 1.02、
        # 99% 在 0.09 以内 ⇒ 2.0 对可达解不构成限制。
        self.sh_bound = float(sh_bound)
        self.use_app = self.app_dim > 0
        self.sh_coeffs = int(sh_coeffs)
        self.sh_c = int(sh_c)
        self.chunk = int(chunk)
        self.app_clip = float(app_clip)

        self.pe = PositionalEncoding(pe_bands) if pe_bands > 0 else None
        pos_dim = self.pe.out_dim if self.pe is not None else 3
        if self.pe is not None:
            if pe_mean is None or pe_std is None:
                raise ValueError("pe_bands > 0 时须显式传 pe_mean / pe_std")
            self.register_buffer("_pe_mean", torch.as_tensor(pe_mean).float().reshape(3).clone())
            self.register_buffer("_pe_std",
                                 torch.as_tensor(pe_std).float().reshape(3).clamp_min(1e-6).clone())
        if self.use_app:
            if app_mean is None or app_std is None:
                raise ValueError("app_dim > 0 时须显式传 app_mean / app_std")
            self.register_buffer("_app_mean",
                                 torch.as_tensor(app_mean).float().reshape(-1).clone())
            self.register_buffer("_app_std",
                                 torch.as_tensor(app_std).float().reshape(-1).clamp_min(1e-6).clone())
            if self._app_mean.numel() != self.app_dim:
                raise ValueError(f"app_mean 维度 {self._app_mean.numel()} != app_dim {self.app_dim}")

        c_in = pos_dim + (self.app_dim if self.use_app else 0)
        if self.use_dir:
            c_in += 3 + (6 * self.pe_dir_bands if self.pe_dir_bands > 0 else 0)
            c_in += 1 if self.use_dist else 0
        if self.use_rel:
            c_in += 3 + 1 + 3 + (6 * self.pe_rel_bands if self.pe_rel_bands > 0 else 0)
        dims = [c_in] + [hidden] * (n_layers - 1)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)

        if self.out_kind == "sh":
            self.head = nn.Linear(dims[-1], self.sh_coeffs * self.sh_c + 1)
        else:
            self.head = nn.Linear(dims[-1], self.n_bands + 1)
        # 与 `rf_head.py` / `coupled_head.py` 同一约定：外观零初始化（= 中灰），
        # 不透明度偏置设到 0.1。权重全 0 起手但输入非零 ⇒ 第一步就有梯度。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = float(torch.log(torch.tensor(opacity_init / (1 - opacity_init))))
            if self.out_kind == "scalar":
                # 标量初值 = init_u（缺省 0.5 ⇒ jet(0.5) 亮绿，在调色板上）。
                # 注意这与 SH 头的「零 = 中灰」不同：灰不在 jet 曲线上。
                u0 = min(max(self.init_u, 1e-4), 1 - 1e-4)
                self.head.bias[1:] = float(torch.log(torch.tensor(u0 / (1.0 - u0))))
        # 形态 A×（两带乘性调制）的 λ；只有 scalar 且 n_bands>1 才存在
        if self.out_kind == "scalar" and self.n_bands > 1:
            self.lam_raw = nn.Parameter(torch.zeros(1))   # λ = sigmoid(0) = 0.5

    def _build(self, xyz: torch.Tensor, app: torch.Tensor | None,
               dirv: torch.Tensor | None = None,
               dist: torch.Tensor | None = None,
               txpos: torch.Tensor | None = None) -> torch.Tensor:
        """拼一**段**点的输入向量。归一化放在这里（不是在外面做一次），
        于是 [P, 95] 的整块只有 chunk 那么大 —— 1.44e6 点全拼要 545 MB。

        视线块**追加在最后**（与逐场景 ScalarColorHead 的布局一致：
        cat[γ(xyz), app?, dir, γ(dir)?, r?]）；顺序只在追加处，旧配置
        （use_dir=False）的输入宽度与逐位数值都不变。
        """
        parts = []
        if self.pe is not None:
            parts.append(self.pe((xyz - self._pe_mean) / self._pe_std))
        else:
            parts.append(xyz)
        if self.use_app:
            if app is None:
                raise ValueError("本头用光学属性但没给 app")
            a = (app - self._app_mean) / self._app_std
            # 重尾维度（如死点的 opacity logit）会拖坏第一步的梯度，钳到 ±5σ
            parts.append(a.clamp(-self.app_clip, self.app_clip))
        if self.use_dir:
            if dirv is None:
                raise ValueError("use_dir=True 但没给 dirv（单位视线方向 [P,3]）")
            parts.append(dirv)
            if self.pe_dir_bands > 0:
                parts.append(fourier_feat(dirv, self.pe_dir_bands))
            if self.use_dist:
                if dist is None:
                    raise ValueError("use_dist=True 但没给 dist（[P,1] 场景尺度距离）")
                parts.append(dist)
        if self.use_rel:
            if txpos is None:
                raise ValueError("use_rel=True 但没给 txpos（[P,3] 发射源位置）")
            if txpos.shape != xyz.shape:
                # 允许 [3] 自动广播，但不做静默 reshape：形状错通常是坐标系搞混了
                txpos = txpos.expand_as(xyz)
            d = xyz - txpos
            r = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            parts.append(d)
            parts.append(r)
            parts.append(d / r)
            if self.pe_rel_bands > 0:
                parts.append(fourier_feat(d, self.pe_rel_bands))
        return torch.cat(parts, dim=-1)

    def forward(self, xyz: torch.Tensor, app: torch.Tensor | None = None,
                dirv: torch.Tensor | None = None, dist: torch.Tensor | None = None,
                txpos: torch.Tensor | None = None):
        """返回 (op, attr)。

        out_kind="sh"     → attr = 球谐 [P,16,3]，激活前（= sh_bound·tanh）
        out_kind="scalar" → attr = 标量 [P,K] ∈ (0,1)，供标量域混合后查 jet 表

        use_dir=True 时 `dirv`（[P,3] 单位向量）必传，`dist`（[P,1]）在 use_dist 时必传；
        use_rel=True 时 `txpos`（[P,3] 或可广播的 [3]）必传。
        """
        p = xyz.shape[0]
        if txpos is not None and txpos.shape != xyz.shape:
            # 先广播再分段，否则 [3] 在 chunk 路径里会被切错
            txpos = txpos.expand_as(xyz).contiguous()
        out = None
        if p <= self.chunk:
            out = self.head(self.trunk(self._build(xyz, app, dirv, dist, txpos)))
        else:
            chunks = []
            for s in range(0, p, self.chunk):
                e = s + self.chunk
                chunks.append(self.head(self.trunk(self._build(
                    xyz[s:e], None if app is None else app[s:e],
                    None if dirv is None else dirv[s:e],
                    None if dist is None else dist[s:e],
                    None if txpos is None else txpos[s:e]))))
            out = torch.cat(chunks, dim=0)

        op = out[:, 0:1]                                   # 不透明度：交给 sigmoid
        if self.out_kind == "scalar":
            return op, torch.sigmoid(out[:, 1:])           # [P,K] ∈ (0,1)
        sh = self.sh_bound * torch.tanh(out[:, 1:])        # 球谐：有界
        return op, sh.reshape(p, self.sh_coeffs, self.sh_c)

    def extra_repr(self) -> str:
        nl = sum(1 for m in self.trunk if isinstance(m, nn.Linear))
        extra = (f"out_kind={self.out_kind} " +
                 (f"n_bands={self.n_bands} init_u={self.init_u}" if self.out_kind == "scalar"
                  else f"sh_bound={self.sh_bound}"))
        if self.use_dir:
            extra += f" dir_pe={self.pe_dir_bands} dist={self.use_dist}"
        return (f"app_dim={self.app_dim} c_in={self.trunk[0].in_features} "
                f"hidden={self.trunk[0].out_features} layers={nl} chunk={self.chunk} "
                + extra)
