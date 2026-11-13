# -*- coding: utf-8 -*-
"""发射源朝向的旋转矩阵（纯 numpy，不依赖 Sionna）。

与 Sionna 2.2 的 `sionna.rt.utils.geometry.rotation_matrix` 逐项一致：
TR 38.901 (7.1-4)，参数 `(alpha, beta, gamma)` 分别是绕 **z / y / x** 的旋转（弧度）。
本仓只需要 `gamma = 0`（数据里 roll 固定为 0）。

R = Rz(alpha)·Ry(beta)·Rx(gamma) ：
    [ cosα cosβ,  cosα sinβ sinγ − sinα cosγ,  cosα sinβ cosγ + sinα sinγ ]
    [ sinα cosβ,  sinα sinβ sinγ + cosα cosγ,  sinα sinβ cosγ − cosα sinγ ]
    [ −sinβ,      cosβ sinγ,                   cosβ cosγ                  ]

局部 **+x** 是阵列/阵元的波束轴（Sionna 的 `look_at_orientation` 就是让局部 x 指向目标）。
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(alpha: float, beta: float, gamma: float = 0.0) -> np.ndarray:
    """→ (3,3) float64，TR 38.901 (7.1-4)。"""
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    return np.array([
        [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
        [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
        [-sb, cb * sg, cb * cg],
    ], dtype=np.float64)


def base_orientation(position, target):
    """让局部 +x 指向 target 的 (alpha, beta)，与 Sionna `look_at_orientation` 同式。

    alpha = atan2(v_y, v_x)，beta = atan2(hypot(v_x, v_y), v_z) − π/2，
    其中 v = normalize(target − position)。竖直方向退化时 alpha=0（v 水平分量为 0）。
    """
    v = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("target 与 position 重合，朝向无定义")
    v = v / n
    alpha = 0.0 if np.hypot(v[0], v[1]) < 1e-12 else float(np.arctan2(v[1], v[0]))
    theta = float(np.arctan2(np.hypot(v[0], v[1]), v[2]))
    return alpha, theta - np.pi / 2


def offset_to_orientation(base_alpha, base_beta, d_yaw, d_pitch, sign=+1.0):
    """基准帧 + 偏移角 → 绝对 (alpha, beta)，弧度。`sign` 只为消融符号约定。"""
    return float(base_alpha + sign * d_yaw), float(base_beta + sign * d_pitch)
