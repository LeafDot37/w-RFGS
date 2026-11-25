# -*- coding: utf-8 -*-
"""ray_geom.py —— 像素 ↔ 世界系射线的解析式（**纯 numpy，无 torch、无数据集依赖**）

单独成一个模块的理由不是整洁，是**可测**：这份代数在服务器上写错的话，症状是
「训练能跑、指标很差」，而那种错极难从日志里看出来。抽出来之后它能在本机用
numpy 直接对拍，不需要 GPU、不需要数据集、不需要 torch。

本仓的验证历史已经证明这类错误是真实风险：相机约定栽过一次，而
`tools/verify_cam_projection.py` 记着更狠的一条 —— 曾经想用图像 IoU 来自证口径，
结果**四种错法给出同一个数**（0.879），掩码越框越测不出。所以口径必须离开图像、
直接和解析式对拍。本模块就是那个解析式。

约定（与引擎一致）
==================
R 是 **camera-to-world**，来自 `dataset_readers.readColmapCameras`：
    R = qvec2rotmat(qvec).T          T = tvec
    p_cam = Rᵀ·p_world + T           ⇒ 相机中心 C = −R·T
三根轴：
    R[:,0] = 像右（+u）、R[:,1] = 像下（+v）、R[:,2] = 光轴（前方）
`verify_cam_projection.py` 的 `basis_axes(R,"col")` 取的就是这三根，且已验证它与
渲染器实际用的 `P @ Rt` 在浮点舍入级一致（≤0.02 px），四种错法被拉开 100 倍以上。

    d_cam = (x/z, y/z, 1)
    x/z   = (i + half − cx) / fx
    y/z   = (j + half − cy) / fy
    d_world = R @ d_cam        （**再归一化**，见下）

**必须归一化**：上游 `Renderer.sample_points` 是 `pts = o + d·t`，且
`raw2outputs` 里 `dists *= ‖rays_d‖`、`path_loss = 0.025/path`。这些式子把 t 当作
**米**。若 d 未归一化，near/far 不再是米、path_loss 的物理含义整个错掉 —— 而且
**不会报任何错**，只是指标悄悄变差。
"""
import numpy as np

# 半像素：像素中心取 (i+0.5, j+0.5)。关掉它只为让 G1 能证明
# 「差半个像素」确实会被 0.02 px 的容差抓住（半像素 = 0.5 px 偏差，超容差 25 倍）。
HALF_PIXEL = 0.5


def _axes(R, sign_h=1.0, sign_v=1.0):
    """取出三根轴并按符号修正。sign 是**待判定量**，不是已知量 —— G1 的职责就是定它。"""
    R = np.asarray(R, dtype=np.float64)
    return R[:, 0] * sign_h, R[:, 1] * sign_v, R[:, 2]


def rays_from_camera(R, fx, fy, cx, cy, W, H, sign_h=1.0, sign_v=1.0,
                     half_pixel=True, swap_uv=False, out=None):
    """像素 → 世界系**单位**射线方向。

    返回 (H, W, 3) float32；`out` 给定时写进它（4 朝向只需算 4 次，可复用缓冲）。

    `swap_uv` 是**错法开关**，只给 G1 的对照用；正路径必须为 False。
    """
    if out is None:
        out = np.empty((H, W, 3), dtype=np.float32)
    half = HALF_PIXEL if half_pixel else 0.0
    xz = (np.arange(W, dtype=np.float64) + half - cx) / fx      # (W,)
    yz = (np.arange(H, dtype=np.float64) + half - cy) / fy      # (H,)
    if swap_uv:
        xz, yz = yz, xz
    u, v, f = _axes(R, sign_h, sign_v)
    d = (xz[None, :, None] * u[None, None, :]
         + yz[:, None, None] * v[None, None, :]
         + f[None, None, :])
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    out[...] = d
    return out


def project_to_pixel(R, fx, fy, cx, cy, d_world, sign_h=1.0, sign_v=1.0,
                     half_pixel=True, swap_uv=False):
    """世界系方向 → 像素坐标 (…,2)。`rays_from_camera` 的严格逆。G1 用它。

    不要求 d_world 归一化（比值 x/z、y/z 与长度无关）。
    """
    d = np.asarray(d_world, dtype=np.float64)
    half = HALF_PIXEL if half_pixel else 0.0
    if swap_uv:
        # 交换轴之后：u 轴用 fy 的尺度、v 轴用 fx —— 这正是这个错法会露馅的地方
        u, v, f = _axes(R, sign_h, sign_v)
        z = d @ f
        return np.stack([(d @ u) / z * fy + cy - half, (d @ v) / z * fx + cx - half], axis=-1)
    u, v, f = _axes(R, sign_h, sign_v)
    z = d @ f
    return np.stack([(d @ u) / z * fx + cx - half,
                     (d @ v) / z * fy + cy - half], axis=-1)


def pixels_grid(W, H):
    """返回 (H,W,2) 的**整数像素索引** (i, j)。

    注意这里给的是索引 i（0..W−1），不是带半像素的 i+0.5 —— 因为
    `project_to_pixel` 的返回值就是**索引**：把 `rays_from_camera` 用的
    `xz = (i+half−cx)/fx` 代回它，得到 `xz·fx + cx − half = i`。
    两者互为逆是**恒等式**，G1 正是靠这个恒等式把「半像素是否一致」钉死。
    """
    ii, jj = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return np.stack([ii, jj], axis=-1)


def roundtrip_max_px(R, fx, fy, cx, cy, W, H, **kw):
    """像素索引 → 方向 → 像素索引，报 (u 方向最大偏差, v 方向最大偏差)，单位 px。

    这是个**恒等式检验**：对任何自洽的 (rays, project) 对都成立。
    它抓「两者不互为逆」（比如半像素只在一侧加），
    抓不到「两者一致地错」—— 后者靠 G1 与渲染器的 `P @ Rt` 对拍。
    """
    d = rays_from_camera(R, fx, fy, cx, cy, W, H, **kw)
    uv = project_to_pixel(R, fx, fy, cx, cy, d, **kw)
    g = pixels_grid(W, H)
    return float(np.abs(uv[..., 0] - g[..., 0]).max()), float(np.abs(uv[..., 1] - g[..., 1]).max())
