# -*- coding: utf-8 -*-
"""逐高斯漫散射权重 d_i（静态，宽角度、实数功率）。

    z_total = z_specular + d_i · o_i      （d_i 从很小开始，镜面模型不被扰动）

- 输入只有 (embed, xyz)，不含 dir/tx：漫散射功率是表面属性；
- softplus + 很负的 bias 初值 ⇒ 第 0 步 d≈3e-4，相当于在 F20k 上加一个可学的小残差；
- 相位不建模（漫散射逐路径随机相位不可预测），只建模功率；CBF/MVDR/PDP 都是功率域。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffuseHead(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 64,
                 bias_init: float = -8.0):
        super().__init__()
        self.hidden = int(hidden)
        self.bias_init = float(bias_init)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.bias_init)

    def forward(self, embed: torch.Tensor, xyz: torch.Tensor,
                center=None, scale: float = 0.0) -> torch.Tensor:
        p = xyz
        if center is not None and scale:
            p = (xyz - torch.as_tensor(center, dtype=xyz.dtype,
                                       device=xyz.device)) / scale
        return F.softplus(self.net(torch.cat([embed, p], -1)).squeeze(-1))
