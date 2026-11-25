# -*- coding: utf-8 -*-
"""静态 RF 不透明度场。

α_i = sigmoid( logit(o_i) + δ_i )，δ = MLP(embed_i, xyz_i)，最后一层 bias 初始化为 bias_init（默认 0）

- 输入**不含** dir/tx/yaw/freq：RF 遮挡是场景的静态属性，不能随视角变；
- δ 零初始化 ⇒ 第 0 步 α = 光学 opacity，保留物理先验；
- `log_k` 控制体素密度的全局尺度（softplus）；初值很负时 v≈1，旧模型不被扰动，
  训练可以把它抬起来，让先验真正参与遮挡；
- 端点留 margin 由 `render_cir` 处理，避免散射体自己遮挡自己。
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RfAlphaHead(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 64,
                 gx: int = 64, gy: int = 64, gz: int = 32,
                 bias_init: float = 0.0):
        super().__init__()
        self.hidden = int(hidden)
        self.gx, self.gy, self.gz = int(gx), int(gy), int(gz)
        self.bias_init = float(bias_init)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.bias_init)
        self.log_k = nn.Parameter(torch.tensor(-2.0))

    def delta(self, embed: torch.Tensor, xyz: torch.Tensor,
              center=None, scale: float = 0.0) -> torch.Tensor:
        p = xyz
        if center is not None and scale:
            p = (xyz - torch.as_tensor(center, dtype=xyz.dtype, device=xyz.device)) / scale
        return self.net(torch.cat([embed, p], -1)).squeeze(-1)

    def alpha(self, embed: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor,
              center=None, scale: float = 0.0) -> torch.Tensor:
        o = opacity.clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.logit(o) + self.delta(embed, xyz, center, scale))
