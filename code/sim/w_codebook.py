# -*- coding: utf-8 -*-
"""发射阵元几何与波束权码本（w 轴的唯一定义处）。

为什么码本必须**与数据无关、可枚举**
--------------------------------------------------------------------------------
「留出 w」要有意义，前提是留出的是**方向**而不是随机数：训练域给一组方向，
测试给内插带里没出现过的方向。若按实例随机取 w，留出就没有可解释的边界。

阵元序（实测，不是假设）
--------------------------------------------------------------------------------
`rt.PlanarArray(num_rows=nr, num_cols=nc, vertical_spacing=0.5,
horizontal_spacing=0.5).positions(1.0)` 在 4×4 上给出

    y = [-0.75]*4 + [-0.25]*4 + [0.25]*4 + [0.75]*4      （慢轴，升序）
    z = [0.75, 0.25, -0.25, -0.75] * 4                   （快轴，降序）

即 n = iy*nc + iz。`spectrum_gen._element_grid(4)` 恰好是同一种排法
（`y` 取 `[-0.75,-0.25,0.25,0.75]`、`z` 取 `[0.75,0.25,-0.25,-0.75]`，
y 在展平后是慢轴），所以 RX 转向矢量与 TX 阵元序是同一套约定。

符号约定（**实测反了，已改**）
--------------------------------------------------------------------------------
原以为是 exp(-j2π p_n·û)，取共轭即 w = exp(+j...)。`check_sign()` 在真实求迹上
扫 (θ,φ) 网格，峰落在**反方向**（θ→180°−θ、φ→φ+180°），说明 `paths.a` 的发射
阵元相位实际是 exp(+j2π p_n·û)。因此匹配发射权取负号：

    w_n = exp(-j2π (y_n·sinθ·sinφ + z_n·cosθ)) / √N

`check_sign()` 是这条的判据：峰位角误差应 < 5°。改号后重测通过。
"""
from __future__ import annotations

import numpy as np

V_SPACING = 0.5          # 波长单位，与生成器里的 PlanarArray 一致
H_SPACING = 0.5

# 训练域 / 留出带（都是**内插**，不外推）
THETA_TRAIN = (15.0, 30.0, 60.0, 75.0)
THETA_HOLD = (45.0,)
PHI_TRAIN = (0.0, 30.0, 60.0, 120.0, 150.0, 180.0, 240.0, 270.0, 300.0)
PHI_HOLD = (90.0, 210.0, 330.0)

UNIFORM = "uniform"


def element_positions(nr: int, nc: int):
    """返回 (y, z)[N] —— 与 Sionna `PlanarArray.positions()` 同序（y 慢、升；z 快、降）。"""
    ys = (np.arange(nr) - (nr - 1) / 2.0) * V_SPACING          # 升序
    zs = ((nc - 1) / 2.0 - np.arange(nc)) * H_SPACING          # 降序
    Y, Z = np.meshgrid(ys, zs, indexing="ij")                  # Y[i,j] = ys[i]
    return Y.ravel().astype(np.float64), Z.ravel().astype(np.float64)


def direction_weight(theta_deg: float, phi_deg: float, nr: int, nc: int) -> np.ndarray:
    """指向 (θ,φ) 的匹配发射权，‖w‖₂ = 1。"""
    y, z = element_positions(nr, nc)
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    phase = 2 * np.pi * (y * np.sin(t) * np.sin(p) + z * np.cos(t))
    return np.exp(-1j * phase).astype(np.complex128) / np.sqrt(nr * nc)


def uniform_weight(nr: int, nc: int) -> np.ndarray:
    return np.ones(nr * nc, np.complex128) / np.sqrt(nr * nc)


def label_of(theta_deg: float, phi_deg: float) -> str:
    return f"t{theta_deg:g}_p{phi_deg:03.0f}"


def train_labels() -> list[str]:
    return [UNIFORM] + [label_of(t, p) for t in THETA_TRAIN for p in PHI_TRAIN]


def holdout_labels() -> list[str]:
    """内插带里的方向：θ 落在 30° 与 60° 之间，或 φ 落在训练网格的空档。"""
    out = []
    for t in THETA_HOLD:
        for p in PHI_TRAIN:
            out.append(label_of(t, p))
    for t in THETA_TRAIN:
        for p in PHI_HOLD:
            out.append(label_of(t, p))
    return out


def parse_label(label: str):
    if label == UNIFORM:
        return None
    t, p = label.split("_")
    return float(t[1:]), float(p[1:])


def vector_for(label: str, nr: int, nc: int) -> np.ndarray:
    """按标签取权向量；未知标签直接报错（不静默回退到 uniform）。"""
    if label == UNIFORM:
        return uniform_weight(nr, nc)
    tp = parse_label(label)
    if tp is None:
        return uniform_weight(nr, nc)
    return direction_weight(tp[0], tp[1], nr, nc)


def assign_train_labels(ids, seed: int = 0, labels=None) -> dict[int, str]:
    """给一批实例号分配训练域 w（固定种子，可复现；同号永远同标签）。"""
    labels = list(labels or train_labels())
    out = {}
    for i in ids:
        rng = np.random.default_rng([seed, int(i)])
        out[int(i)] = labels[int(rng.integers(0, len(labels)))]
    return out


def beam_gain(aw_weights: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Σ_n a_n w_n（复）。a 为 [N_tx, P]，w 为 [N_tx]。"""
    return aw_weights @ a


def beam_scan(av: np.ndarray, nr: int, nc: int, step: float = 2.0,
              theta_range=(5.0, 175.0), phi_range=(-180.0, 180.0)):
    """在 (θ,φ) 网格上扫描波束形成增益，返回 (θ_s, φ_s, 增益图)。

    `av` 为 [M², N_tx]（单条路径的阵元响应）。增益 = 该路径上 |Σ_n a_n w_n|
    对 M² 个接收阵元的平均。一次 matmul 算完整个网格。
    """
    theta_s = np.arange(theta_range[0], theta_range[1] + 1e-9, step)
    phi_s = np.arange(phi_range[0], phi_range[1] + 1e-9, step)
    T, P = np.meshgrid(theta_s, phi_s, indexing="ij")
    y, z = element_positions(nr, nc)
    v = np.sin(np.deg2rad(T)).ravel() * np.sin(np.deg2rad(P)).ravel()
    w_ = np.cos(np.deg2rad(T)).ravel()
    W = np.exp(-1j * 2 * np.pi * (np.outer(v, y) + np.outer(w_, z))) / np.sqrt(nr * nc)
    g = np.abs(W @ av.T).mean(axis=1).reshape(T.shape)
    i, j = np.unravel_index(int(np.argmax(g)), g.shape)
    return theta_s[i], phi_s[j], g


def tx_orientation_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float = 0.0):
    """Sionna 的 [yaw, pitch, roll] 朝向：R = Rz·Ry·Rx（局部 → 世界）。"""
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _dir_to_angles(u: np.ndarray):
    u = u / max(np.linalg.norm(u), 1e-30)
    th = float(np.rad2deg(np.arccos(np.clip(u[2], -1, 1))))
    ph = float(np.rad2deg(np.arctan2(u[1], u[0])))
    return th, ph


def check_sign(paths, rx: int = 0, n_paths: int = 3, step: float = 2.0,
               yaw_rad: float | None = None, pitch_rad: float | None = None):
    """**决定性**地验证 `direction_weight` 的相位号：看波束形成的峰落在哪。

    做法：取该接收机上最强的 `n_paths` 条路径，对每一条的真实发射方向 (θ_t,φ_t)，
    用 `beam_scan` 在整个上半球找「用 `direction_weight` 加权后增益最大」的方向。
    符号正确 ⇒ 峰就在 (θ_t,φ_t)；符号反了 ⇒ 峰落到 p·û 反号的另一处（通常
    相距很远）。这条判据不依赖「哪条路径最强」，只看峰位，无法被弱路径带偏。

    返回 [(θ_true, φ_true, θ_peak, φ_peak, 角误差°), ...]
    """
    ave = paths.a
    if isinstance(ave, (tuple, list)) and len(ave) == 2:
        a = np.asarray(ave[0][rx, :, 0, :, :]) + 1j * np.asarray(ave[1][rx, :, 0, :, :])
    else:
        aa = np.asarray(ave)
        if aa.ndim == 6:
            aa = aa[..., 0]
        a = aa[rx, :, 0, :, :]
    n_ant = a.shape[1]          # 轴 0 是接收阵元 M²=100，轴 1 才是发射阵元
    nr = int(round(np.sqrt(n_ant)))
    if nr * nr != n_ant:
        raise ValueError(f"阵元数 {n_ant} 不是方阵")
    nc = nr

    if paths.theta_t is None or paths.phi_t is None:
        raise RuntimeError("paths 上没有 theta_t/phi_t，无法验证符号")
    tt = np.asarray(paths.theta_t)
    pp = np.asarray(paths.phi_t)
    if tt.ndim == 4:
        tt, pp = tt[..., 0], pp[..., 0]
    if tt.ndim != 3 or tt.shape[-1] != a.shape[-1]:
        raise RuntimeError(f"theta_t 形状 {tt.shape} 与 a {a.shape} 对不上")
    theta_t, phi_t = tt[rx, 0, :], pp[rx, 0, :]

    # TX 有朝向时，`theta_t/phi_t` 若是世界系，必须转到 TX 本地系才能与阵元坐标
    # 比较（实例 0 的 yaw/pitch 是 −37.9°/48.7°，不转会差几十度）。
    R = (tx_orientation_matrix(yaw_rad, pitch_rad)
         if yaw_rad is not None and pitch_rad is not None else None)

    pw = np.abs(a).sum(axis=(0, 1))
    top = np.argsort(pw)[-n_paths:]
    out = []
    for p in top:
        th = float(np.rad2deg(theta_t[p]))
        ph = float(np.rad2deg(phi_t[p]))
        if R is not None:
            u = np.array([np.sin(np.deg2rad(th)) * np.cos(np.deg2rad(ph)),
                          np.sin(np.deg2rad(th)) * np.sin(np.deg2rad(ph)),
                          np.cos(np.deg2rad(th))])
            th, ph = _dir_to_angles(R.T @ u)
        th_p, ph_p, _ = beam_scan(a[:, :, p], nr, nc, step=step)
        # 判据必须建在 (v, w) 上而不是三维方向角上：阵元全在 y-z 平面上（x ≡ 0），
        # 波束方向只由 v = sinθ·sinφ 与 w = cosθ 决定。实测有两条路径 (62°,−52°)
        # 与 (63°,−128°) 的三维角差 66°，但 (v,w) 只差 0.004 —— 那是阵**看不到**
        # 的方向，拿三维角差当判据会误判成符号错。
        v = np.sin(np.deg2rad(th)) * np.sin(np.deg2rad(ph))
        w = np.cos(np.deg2rad(th))
        v_p = np.sin(np.deg2rad(th_p)) * np.sin(np.deg2rad(ph_p))
        w_p = np.cos(np.deg2rad(th_p))
        d = float(np.rad2deg(np.arcsin(np.clip(np.hypot(v - v_p, w - w_p), 0, 1))))
        out.append((th, ph, float(th_p), float(ph_p), d))
    return out
