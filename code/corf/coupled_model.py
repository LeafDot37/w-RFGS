# -*- coding: utf-8 -*-
"""耦合高斯：共享几何 + 双模态外观。

对应 CORF-GS 的耦合原语

    G_i = { mu_i, Sigma_i, o_i^opt, c_i^opt, o_i^rf, c_i^rf }

其中 { mu, Sigma } 是两个模态**共享**的几何，{o^opt, c^opt} 与 {o^rf, c^rf}
是各自的外观。

实现方式（关键：**不修改引擎核心**）
--------------------------------------------------------------------------------
`splat/gaussian_renderer.render()` 只要求第一个参数提供一组鸭子类型的成员：
`get_xyz / get_opacity / get_features / get_scaling / get_rotation /
 get_covariance / active_sh_degree / max_sh_degree / _features`。
所以：

- `CoupledGaussianModel` 子类化 `GaussianModel`，**复用基类的光学那一半**
  （`_opacity`、`_features_dc`、`_features_rest` 名字与语义一字不变），
  只**新增**射频那一半（`_opacity_rf`、`_features_rf_dc`、`_features_rf_rest`）。
- `ModalityView` 把耦合模型按某一模态呈现给渲染器，`render()` 一行不改。

这样既有的渲染、评测、可视化工具链对**光学**那一半仍然逐字可用。
"""
import torch
from torch import nn

from splat.scene.gaussian_model import GaussianModel
from splat.utils.general_utils import inverse_sigmoid


class CoupledGaussianModel(GaussianModel):
    """共享几何 + 光学外观（基类）+ 射频外观（新增）。"""

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        # 射频外观：与光学外观同构（SH 系数 + 透明度），但独立参数
        self._opacity_rf = torch.empty(0)
        self._features_rf_dc = torch.empty(0)
        self._features_rf_rest = torch.empty(0)

    # ------------------------------------------------------------------ 初始化
    def init_rf_appearance(self, opacity_init: float = 0.1,
                           feature_init: float = 0.0):
        """按当前点数建立射频外观参数。

        feature_init = 0 与既有阶段三（`train.py` 里 `_features_dc.fill_(0.0)`）一致：
        SH 直流分量 0 对应颜色 0.5（中灰）。射频谱是 jet 伪彩，中灰是一个中性起点。
        """
        P = self.get_xyz.shape[0]
        n_rest = (self.max_sh_degree + 1) ** 2 - 1
        dev = self.get_xyz.device
        self._opacity_rf = nn.Parameter(
            inverse_sigmoid(opacity_init * torch.ones((P, 1), dtype=torch.float,
                                                      device=dev)),
            requires_grad=True)
        self._features_rf_dc = nn.Parameter(
            torch.full((P, 1, 3), float(feature_init), dtype=torch.float,
                       device=dev),
            requires_grad=True)
        self._features_rf_rest = nn.Parameter(
            torch.full((P, n_rest, 3), float(feature_init), dtype=torch.float,
                       device=dev),
            requires_grad=True)

    # ------------------------------------------------------------------ 属性
    @property
    def get_opacity_rf(self):
        return self.opacity_activation(self._opacity_rf)

    @property
    def get_features_rf(self):
        return torch.cat((self._features_rf_dc, self._features_rf_rest), dim=1)

    # ------------------------------------------------------------------ 视图
    def view(self, modality: str):
        assert modality in ("opt", "rf"), modality
        return ModalityView(self, modality)


class ModalityView:
    """把 `CoupledGaussianModel` 按一个模态呈现给 `splat.gaussian_renderer.render`。

    几何（xyz / scaling / rotation）两个模态共用 ⇒ 射频损失回传的梯度
    落在**同一批**几何参数上，这就是「耦合」在代码上的全部含义。
    """

    def __init__(self, cm: CoupledGaussianModel, modality: str):
        self.cm = cm
        self.modality = modality

    # —— 共享几何 ——
    @property
    def get_xyz(self):
        return self.cm.get_xyz

    @property
    def get_scaling(self):
        return self.cm.get_scaling

    @property
    def get_rotation(self):
        return self.cm.get_rotation

    def get_covariance(self, scaling_modifier=1):
        return self.cm.get_covariance(scaling_modifier)

    # —— 模态特有外观 ——
    @property
    def get_opacity(self):
        return (self.cm.get_opacity if self.modality == "opt"
                else self.cm.get_opacity_rf)

    @property
    def get_features(self):
        return (self.cm.get_features if self.modality == "opt"
                else self.cm.get_features_rf)

    # —— 渲染器还会读到这两个 ——
    @property
    def active_sh_degree(self):
        return self.cm.active_sh_degree

    @property
    def max_sh_degree(self):
        return self.cm.max_sh_degree
