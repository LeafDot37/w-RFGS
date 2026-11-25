# -*- coding: utf-8 -*-
"""baselines —— 外部基线模型的 vendor 与适配层

本包只放**不是我们写的**、被拉进来当对照的模型。与 `code/splat/`（引擎）、
`code/feedforward/`（前馈视觉）并列，互不 import。

为什么 vendor 而不是重写
========================
基线最怕的一件事是「你把它写弱了」——只要是我们照论文重写的，审稿人就有权
怀疑容量、初始化、优化细节被无意削弱。因此凡是要跑的数，网络与体渲染**逐字**
取上游源码，sha256 钉进各子包的 `UPSTREAM.md`；我们只写**数据适配**与**评测对接**。

为什么与 gs3d 同环境
====================
`code/tools/diag/metrics_fast.py` / `metrics.py` 的评测链（PSNR/SSIM/LPIPS-VGG）全部
住在 `code/splat/` 下。给基线另起 conda 环境会让基线拿不到同一把尺子，
而「同一把尺子」正是这份对照存在的唯一理由。故一律沿用 gs3d。

子包
====
- `nerf2` —— NeRF²（XPengZhao/NeRF2, MIT）。无视觉、纯从 RF 测量学场的代表，
  是本项目「视觉引导」论点的正对面基线。
"""
