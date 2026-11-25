# -*- coding: utf-8 -*-
"""CORF 复现：耦合光学-射频高斯（老师）。

本包**不修改引擎核心**（`splat/scene/gaussian_model.py`、`splat/gaussian_renderer/`、
`train.py`、CUDA 光栅化器均一字不动）。全部通过子类化 + 鸭子类型的模态视图实现，
和本仓既有的 `train_rrf_scalar.py` / `splat/color_scalar.py` 是同一套做法。

对应外部工作：CORF-GS（arXiv 2607.25569），
耦合原语 G_i = {mu_i, Sigma_i, o_i^opt, c_i^opt, o_i^rf, c_i^rf}。
"""
