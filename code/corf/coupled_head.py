# -*- coding: utf-8 -*-
"""前馈耦合学生：**一支共享主干 + 两个模态特有的外观头**。

与 `corf/rf_head.py` 的关系
--------------------------------------------------------------------------------
`RFHead` 只出射频外观，光学一半钉死取自种子 —— 那是「单模态前馈」的版本。
本文件是它向「耦合 + 输入可变」的扩展，结构上与 `corf/coupled_model.py`
（阶段 1 的逐场景耦合模型）同构：

    共享几何  = 高斯点云（MVSplat 前馈产出，全程冻结）
    共享主干  = 逐点 MLP，吃「点在哪 + 有哪些证据」
    模态外观  = 两个独立的输出头：光学球谐、射频球谐

证据（evidence）有四种：
    pos      只用位置（阴性对照：证据到底有没有用）
    opt      位置 + 12 通道光学统计（`make_student_features.py`）
    rf       位置 + 4 朝向射频近邻证据（`tools/make_rf_evidence.py`）
    opt_rf   两者都给

`modality_dropout=True` 时，训练时每步在「当前可用的子集」里随机抽一个组合
（空集除外），于是**同一个权重**要覆盖 {opt}、{rf}、{opt,rf} 三种输入。
这就是「不管输入组成是什么都能出耦合结果」在训练上的实现方式。

⚠ 缺一支时喂 0 向量，并额外喂两个 **存在标志**（1/0）让网络知道缺的是哪一支。
  没有标志的话，「光学特征恰好全 0」与「没有光学输入」在网络眼里无法区分。

输出参数化与老师逐字相同：球谐 16 系数 × 3 通道 + 1 不透明度（激活前）。
点序与老师 sidecar 逐个相同 ⇒ 可做逐点属性蒸馏。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from corf.rf_head import (PositionalEncoding, RF_SH_C, RF_SH_COEFFS,
                          VIS_CHANNELS, xyz_norm_stats)

EVIDENCE_KINDS = ("pos", "opt", "rf", "opt_rf")


class CoupledHead(nn.Module):
    def __init__(self, evidence: str = "opt_rf", hidden: int = 64, n_layers: int = 3,
                 pe_bands: int = 6, rf_dim: int = 0, chunk: int = 250_000,
                 sh_coeffs: int = RF_SH_COEFFS, sh_c: int = RF_SH_C,
                 xyz_mean: torch.Tensor | None = None,
                 xyz_std: torch.Tensor | None = None,
                 opacity_init: float = 0.1):
        super().__init__()
        if evidence not in EVIDENCE_KINDS:
            raise ValueError(f"evidence 必须是 {EVIDENCE_KINDS} 之一，收到 {evidence}")
        self.evidence = evidence
        self.has_opt = evidence in ("opt", "opt_rf")
        self.has_rf = evidence in ("rf", "opt_rf")
        self.rf_dim = int(rf_dim) if self.has_rf else 0
        if self.has_rf and self.rf_dim <= 0:
            raise ValueError("evidence 含 rf 时必须给 rf_dim > 0")
        self.sh_coeffs = int(sh_coeffs)
        self.sh_c = int(sh_c)
        self.chunk = int(chunk)

        self.pe = PositionalEncoding(pe_bands) if pe_bands > 0 else None
        pos_dim = self.pe.out_dim if self.pe is not None else 3
        if self.pe is not None:
            if xyz_mean is None or xyz_std is None:
                raise ValueError("pe_bands > 0 时须显式传 xyz_mean / xyz_std"
                                 "（用 corf.rf_head.xyz_norm_stats(xyz) 算一次）")
            self.register_buffer("_xyz_mean", torch.as_tensor(xyz_mean).float().reshape(3).clone())
            self.register_buffer("_xyz_std",
                                 torch.as_tensor(xyz_std).float().reshape(3).clamp_min(1e-6).clone())

        # +2 = 两个模态存在标志
        c_in = pos_dim + (VIS_CHANNELS if self.has_opt else 0) + self.rf_dim + 2
        dims = [c_in] + [hidden] * (n_layers - 1)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)

        out_dim = self.sh_coeffs * self.sh_c + 1
        self.head_opt = nn.Linear(dims[-1], out_dim)
        self.head_rf = nn.Linear(dims[-1], out_dim)
        # 外观系数零初始化（= 中灰），不透明度偏置设到 0.1（3DGS 惯例，
        # 也与阶段 1 的 `--rf_opacity_init 0.1` 对齐）。权重全 0 起手，
        # 但输入非零 ⇒ 第一步就有梯度，不会像 `requires_grad=False` 那样卡死。
        for h in (self.head_opt, self.head_rf):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
            with torch.no_grad():
                h.bias[0] = float(torch.log(torch.tensor(opacity_init / (1 - opacity_init))))

    # ------------------------------------------------------------------ 前向
    def forward(self, xyz: torch.Tensor,
                vis: torch.Tensor | None = None,
                rf: torch.Tensor | None = None,
                use_opt: bool | None = None,
                use_rf: bool | None = None):
        """返回 (opt_op, opt_sh, rf_op, rf_sh)，四个量都是**激活前**参数。

        `use_opt/use_rf` 只允许把「本臂可用的支路」关掉（模态丢弃），
        **不允许**打开本臂没有的支路 —— 那会变成另一条实验臂，静默越界。
        """
        use_opt = self.has_opt if use_opt is None else bool(use_opt)
        use_rf = self.has_rf if use_rf is None else bool(use_rf)
        if use_opt and not self.has_opt:
            raise ValueError(f"本臂 evidence={self.evidence} 没有光学支路，不能 use_opt=True")
        if use_rf and not self.has_rf:
            raise ValueError(f"本臂 evidence={self.evidence} 没有射频支路，不能 use_rf=True")
        if not (use_opt or use_rf) and (self.has_opt or self.has_rf):
            raise ValueError("两条证据支路都被关掉了 —— 至少留一条"
                             "（`pos` 阴性对照臂本来就没有证据支路，不在此列）")

        parts = []
        if self.pe is not None:
            parts.append(self.pe((xyz - self._xyz_mean) / self._xyz_std))
        else:
            parts.append(xyz)
        if self.has_opt:
            if use_opt:
                if vis is None:
                    raise ValueError("use_opt=True 但没给 vis")
                parts.append(vis)
            else:
                parts.append(xyz.new_zeros(xyz.shape[0], VIS_CHANNELS))
        if self.has_rf:
            if use_rf:
                if rf is None:
                    raise ValueError("use_rf=True 但没给 rf 证据")
                parts.append(rf)
            else:
                parts.append(xyz.new_zeros(xyz.shape[0], self.rf_dim))
        # 存在标志：两列常数，告诉网络这一支到底在不在
        parts.append(torch.stack([
            xyz.new_full((xyz.shape[0],), float(use_opt)),
            xyz.new_full((xyz.shape[0],), float(use_rf)),
        ], dim=-1))
        x = torch.cat(parts, dim=-1)

        p = x.shape[0]
        if p <= self.chunk:
            h = self.trunk(x)
            oo, ro = self.head_opt(h), self.head_rf(h)
        else:
            oo_l, ro_l = [], []
            for s in range(0, p, self.chunk):
                h = self.trunk(x[s:s + self.chunk])
                oo_l.append(self.head_opt(h))
                ro_l.append(self.head_rf(h))
            oo, ro = torch.cat(oo_l, dim=0), torch.cat(ro_l, dim=0)

        def split(t):
            return t[:, 0:1], t[:, 1:].reshape(p, self.sh_coeffs, self.sh_c)

        opt_op, opt_sh = split(oo)
        rf_op, rf_sh = split(ro)
        return opt_op, opt_sh, rf_op, rf_sh

    def extra_repr(self) -> str:
        nl = sum(1 for m in self.trunk if isinstance(m, nn.Linear))
        return (f"evidence={self.evidence} c_in={self.trunk[0].in_features} "
                f"hidden={self.trunk[0].out_features} layers={nl} "
                f"rf_dim={self.rf_dim} chunk={self.chunk}")


def build(evidence: str, cache: dict, rf_dim: int, **kw) -> CoupledHead:
    """从特征缓存装配一个头（训练与渲染两端共用，保证统计量一致）。"""
    xyz = cache["xyz"].float()
    mean, std = xyz_norm_stats(xyz)
    return CoupledHead(evidence=evidence, rf_dim=rf_dim,
                       xyz_mean=mean, xyz_std=std, **kw)
