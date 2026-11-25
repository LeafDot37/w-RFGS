# -*- coding: utf-8 -*-
"""Delay 算子：按方向着色的归一化时延图（RF-3DGS 的 Delay 通道口径）。

RF-3DGS 的定义（论文 §V-B）：每个角度像素记录该方向到达 MPC 相对**首达路径**的
归一化时延，范围 0–200 ns，超过丢弃。它不是一维 PDP，而是空间谱的第三个通道。

从 CIR 计算：
    P(θ, τ) = |a(θ)ᴴ x[:, τ]|²
    τ*(θ)   = argmax_τ P
    delay(θ) = clip(τ*(θ) − min_θ τ*, 0, 200) / 200

输入 x 为 [M², T] 复 CIR；返回 [H, W] 的 u ∈ [0,1]（无效像素置 1）。
"""
from __future__ import annotations

import numpy as np


def delay_map(x: np.ndarray, a_mat: np.ndarray, dt_ns: float,
              hw: tuple[int, int], max_delay_ns: float = 200.0,
              rel_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """x: [M², T] complex；a_mat: [M², HW] 复流形；→ (u [H,W], valid [H,W])。"""
    x = np.asarray(x)
    T = x.shape[1]
    HW = a_mat.shape[1]
    best = np.full(HW, -1.0, np.float32)
    tau = np.zeros(HW, np.float32)
    for t in range(T):
        p = np.abs(x[:, t] @ a_mat).astype(np.float32) ** 2
        m = p > best
        best[m] = p[m]
        tau[m] = t * dt_ns
    valid = best > rel_floor * max(float(best.max()), 1e-30)
    if valid.any():
        t0 = float(tau[valid].min())
        rel = np.where(valid, tau - t0, max_delay_ns)
    else:
        rel = np.full(HW, max_delay_ns, np.float32)
    u = np.clip(rel / max_delay_ns, 0.0, 1.0).reshape(hw)
    return u, valid.reshape(hw)


def cir_from_paths_np(a: np.ndarray, tau_ns: np.ndarray, valid: np.ndarray,
                      T: int, dt_ns: float) -> np.ndarray:
    """逐路径响应 → [M², T] CIR（与 train_csi_cbf.build_gt_cir 同一分箱）。"""
    a = np.asarray(a)
    tau = np.asarray(tau_ns)
    valid = np.asarray(valid, bool)
    tap = np.clip(np.floor(tau[valid].astype(np.float64) / dt_ns).astype(np.int64),
                  0, T - 1)
    x = np.zeros((a.shape[0], T), np.complex64)
    np.add.at(x, (slice(None), tap), a[:, valid])
    return x


def path_pixels(theta_r: np.ndarray, phi_r: np.ndarray, hw: tuple[int, int],
                fov_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sionna 的 AoA (theta_r, phi_r) → 角域图像素 (row, col, ok)。

    图像网格来自 `compute_angle_matrices`：d = [x/f, y/f, 1]，φ=atan2(−d_x,d_z)，
    θ=π/2−asin(d_y)。阵列在 y–z 平面，只有 (v,w)=(sinθsinφ, cosθ) 可观测；
    反解得 d_x=−v，d_y=w，d_z=sqrt(1−v²−w²)。
    """
    H, W = hw
    th = np.asarray(theta_r, np.float64)
    ph = np.asarray(phi_r, np.float64)
    v = np.sin(th) * np.sin(ph)
    w = np.cos(th)
    d2 = 1.0 - v * v - w * w
    ok = d2 > 1e-8
    dz = np.sqrt(np.clip(d2, 1e-12, None))
    f = W / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    col = np.round(f * (-v / dz) + W / 2.0).astype(np.int64)
    row = np.round(H / 2.0 - f * (w / dz)).astype(np.int64)
    ok &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
    return row, col, ok


def delay_map_paths(thr, phr, tau_ns, pwr, valid, hw=(200, 300),
                    fov_deg: float = 90.0, sigma: float = 3.0,
                    max_delay_ns: float = 200.0, relative: bool = True,
                    rel_floor: float = 1e-6):
    """RF-3DGS 的 Delay 通道：逐路径把归一化时延 splat 到到达角像素。

    每个像素取「功率加权」的时延（等效于强路径主导）；返回 u [H,W] 与 valid。
    """
    H, W = hw
    row, col, ok = path_pixels(thr, phr, hw, fov_deg)
    ok = ok & np.asarray(valid, bool)
    if not ok.any():
        return np.ones(hw, np.float32), np.zeros(hw, bool)
    tau = np.asarray(tau_ns, np.float64)
    p = np.asarray(pwr, np.float64)
    if relative:
        tau = tau - tau[ok].min()
    r = int(np.ceil(3 * sigma))
    offs = np.arange(-r, r + 1)
    gy, gx = np.meshgrid(offs, offs, indexing="ij")
    ker = np.exp(-(gy ** 2 + gx ** 2) / (2 * sigma ** 2)).ravel()
    num = np.zeros(H * W, np.float64)
    den = np.zeros(H * W, np.float64)
    for rr, cc, tt, pp in zip(row[ok], col[ok], tau[ok], p[ok]):
        ys = np.clip(rr + gy.ravel(), 0, H - 1)
        xs = np.clip(cc + gx.ravel(), 0, W - 1)
        idx = ys * W + xs
        num[idx] += ker * pp * tt
        den[idx] += ker * pp
    m = den > rel_floor * max(den.max(), 1e-30)
    u = np.ones(H * W, np.float32)
    u[m] = np.clip(num[m] / den[m] / max_delay_ns, 0.0, 1.0)
    return u.reshape(H, W), m.reshape(H, W)
