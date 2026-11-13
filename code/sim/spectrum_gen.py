# -*- coding: utf-8 -*-
"""按 RF-3DGS 教程的**原口径**算「角域射频空间谱」图（CBF 起手）。

来源：教程 notebook `_external/rf3dgs_tutorial/out/.../RF-3DGS-tutorial.ipynb`
的 cell 54（CBF_spectrum）、57（compute_angle_matrices）、36（转向矢量）、
44（paths→CIR 的时间网格合并）、87（jet 上色）。本文件是这些公式在
**numpy + Sionna 2.2** 下的等价实现，供本仓生成别的发射源的谱图。

与原实现的对应关系
--------------------------------------------------------------------------------
- AoA 网格：`compute_angle_matrices(width=300, height=200, fov=90)`，针孔方向
  网格 → (phi, theta)；与数据集 300×200 的像素一一对应。
- 阵列：10×10 UPA（M=10），半波长间距，阵元方向图 tr38901（由 Sionna 的
  `rx_array` 在 CIR 里给出），`synthetic_array=True`。
- 转向矢量：只用相位 exp(+j2π(y·sinθsinφ + z·cosθ))，y/z 以半波长为单位；
  与教程 `steering_vector` 逐项一致（教程的 array_manifold_vector 额外乘了
  阵元方向图，CBF 不用它）。
- CIR：把 Sionna 的路径按 0.5 ns 的时间网格合并（`time_interval=0.5`），
  同一格里的复幅度直接相加（教程的 merge_paths_to_time_grid）。
- CBF：P(θ,φ) = |Σ_m Σ_l conj(x[m,l]) · a_m(θ,φ)|，再逐图 min-max 归一化 →
  matplotlib jet → uint8。教程 `jet_colormap_convert` 就是逐图 min-max。

为什么保留「逐图归一化」
--------------------------------------------------------------------------------
教程的数据集就是这样生成的（每张谱图独立 min-max）。这是 RF-3DGS 的评测
口径，改掉就不可比了。
"""
from __future__ import annotations

import numpy as np

# 教程 cell 106 的四张图角度顺序（同一位置下的 4 个阵列朝向）
ANGLES = (-np.pi / 2, 0.0, np.pi / 2, np.pi)


def compute_angle_matrices(width: int = 300, height: int = 200, fov: float = 90.0):
    """像素 → (theta, phi)。逐行照抄教程 cell 57。"""
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(height / 2, -height / 2, height)
    xx, yy = np.meshgrid(x, y)
    d = np.stack([xx / focal, yy / focal, np.ones_like(xx)], -1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    phi = np.arctan2(-d[..., 0], d[..., 2])
    theta = np.pi / 2 - np.arcsin(d[..., 1])
    return theta.astype(np.float32), phi.astype(np.float32)


def _element_grid(M: int):
    values = 0.25 + 0.5 * np.arange(M / 2)
    y_i = np.concatenate((-values[::-1], values))
    z_i = np.concatenate((values[::-1], -values))
    y_i_grid, z_i_grid = np.meshgrid(y_i, z_i)
    y = np.reshape(y_i_grid.transpose(), M ** 2)
    z = np.reshape(z_i_grid.transpose(), M ** 2)
    return y.astype(np.float32), z.astype(np.float32)


def steering_vector(M: int, theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """[M², H, W] 复转向矢量（纯相位，教程 cell 36 的 steering_vector）。"""
    y, z = _element_grid(M)
    v = np.sin(theta_grid) * np.sin(phi_grid)
    w = np.cos(theta_grid)
    phase = 2 * np.pi * (y[:, None, None] * v[None] + z[:, None, None] * w[None])
    return np.exp(1j * phase).astype(np.complex64)


def paths_complex(paths) -> np.ndarray:
    """paths.a → numpy 复张量 [num_rx, num_rx_ant, num_tx, num_tx_ant, P]。

    Sionna 2.x 的 `paths.a` 是 (实部, 虚部) 二元组，且没有时间步轴；旧版是带
    最后一维 T 的复张量。两种都归一成 5 维、丢掉时间轴（此处 num_time_steps=1）。
    """
    a = paths.a
    if isinstance(a, (tuple, list)) and len(a) == 2:          # Sionna 2.x 给 (实部, 虚部)
        a = np.asarray(a[0]) + 1j * np.asarray(a[1])
    else:
        a = np.asarray(a)
        if not np.iscomplexobj(a):
            a = a.astype(np.complex64)
    if a.ndim == 6:
        a = a[..., 0]
    if a.ndim != 5:
        raise RuntimeError(f"paths.a 形状认不出：{a.shape}")
    return a


def cir_from_paths(paths, time_interval_ns: float = 0.5, rx: int = 0):
    """把 Sionna 路径合并到时间网格，返回 [M², L] 复 CIR（教程 paths_to_response）。

    `paths.a` 形状 [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    `paths.tau` 形状 [num_rx, num_tx, num_paths]（秒）
    无效路径（valid=False）丢弃；其余按 searchsorted 落到 0.5 ns 网格上相加。
    """
    a = paths_complex(paths)[rx, :, 0, 0, :]                   # [M², P]
    tau = np.asarray(paths.tau)[rx, 0, :].astype(np.float64) / 1e-9   # ns
    valid = np.ones(len(tau), bool)
    if getattr(paths, "valid", None) is not None:
        valid = np.asarray(paths.valid)[rx, 0, :].astype(bool)
    tau, a = tau[valid], a[:, valid]
    t_max = float(np.max(tau)) if tau.size else 0.0
    grid = np.arange(0.0, max(np.ceil(t_max), time_interval_ns),
                     time_interval_ns, dtype=np.float32)
    idx = np.clip(np.searchsorted(grid, tau, side="left") - 1, 0, len(grid) - 1)
    # 所有阵元共用同一组时间格 ⇒ 一次排序 + reduceat，比逐阵元 np.add.at 快两个量级
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    bins, starts = np.unique(idx_s, return_index=True)
    merged = np.add.reduceat(a[:, order], starts, axis=1)
    out = np.zeros((a.shape[0], len(grid)), np.complex64)
    out[:, bins] = merged
    return out


def cbf_spectrum(paths, M: int, theta_grid, phi_grid, time_interval_ns: float = 0.5):
    """返回 |Σ_m Σ_l conj(x[m,l])·a_m(θ,φ)|，形状 [H, W]。"""
    x = cir_from_paths(paths, time_interval_ns)                # [M², L]
    a = steering_vector(M, theta_grid, phi_grid)               # [M², H, W]
    x_sum = np.conj(x).sum(1)                                  # [M²]
    spec = np.abs(np.einsum("m,mhw->hw", x_sum, a))
    return spec.astype(np.float32)


def cir_from_response(aw: np.ndarray, tau_ns: np.ndarray,
                      valid: np.ndarray | None = None,
                      time_interval_ns: float = 0.5):
    """`cir_from_paths` 的「响应已在主机上」版本（预编码 w 用），返回 [M², L]。

    `cir_from_paths` 把 tx-ant 轴切成 0（只取单根发射天线），4×4 发射阵 + 波束权
    w 的合成必须在调用方先做完再进来。时间格那段是**逐行复制**而不是抽公共函数：
    `cir_from_paths` 是生产路径，重构它要重验与旧数据逐位一致，不值得为 10 行冒险。
    """
    tau = np.asarray(tau_ns, np.float64)
    aw = np.asarray(aw)
    if valid is not None and not np.all(valid):
        tau, aw = tau[valid], aw[:, valid]
    t_max = float(np.max(tau)) if tau.size else 0.0
    grid = np.arange(0.0, max(np.ceil(t_max), time_interval_ns),
                     time_interval_ns, dtype=np.float32)
    idx = np.clip(np.searchsorted(grid, tau, side="left") - 1, 0, len(grid) - 1)
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    bins, starts = np.unique(idx_s, return_index=True)
    merged = np.add.reduceat(aw[:, order], starts, axis=1)
    out = np.zeros((aw.shape[0], len(grid)), np.complex64)
    out[:, bins] = merged
    return out


class CBFMapper:
    """批量生成用：转向矩阵只算一次，逐图只剩「CIR→点积」。

    `cbf_spectrum` 每次重新算 [M²,H,W] 的复指数（6M 次 exp，~0.2 s/图），
    在要生成十万张图时是主要开销。这里把它提到构造器里。
    """

    def __init__(self, M: int, theta_grid, phi_grid, time_interval_ns: float = 0.5):
        self.M = M
        self.hw = theta_grid.shape
        self.interval = time_interval_ns
        self.a_mat = steering_vector(M, theta_grid, phi_grid).reshape(M * M, -1)

    def __call__(self, paths, rx: int = 0) -> np.ndarray:
        x = cir_from_paths(paths, self.interval, rx=rx)
        x_sum = np.conj(x).sum(1)
        return np.abs(x_sum @ self.a_mat).reshape(self.hw).astype(np.float32)

    def from_response(self, aw: np.ndarray, tau_ns: np.ndarray,
                      valid: np.ndarray | None = None) -> np.ndarray:
        """与 `__call__` 同一条计算链，输入换成已在主机上的合成响应 aw [M², P]。

        `__call__` 只能取单根发射阵元；4×4 发射阵 + 波束权 w 的 Σ_n a_n·w_n
        必须由调用方先做完（见 `gen_cbf_params._aw_with_w`）。
        """
        x = cir_from_response(aw, tau_ns, valid, self.interval)
        x_sum = np.conj(x).sum(1)
        return np.abs(x_sum @ self.a_mat).reshape(self.hw).astype(np.float32)


def tr38901_gain(theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """阵元方向图幅度 C_θ(θ,φ)（3GPP TR 38.901 表 7.3-1，垂直极化）。

    等价于 Sionna 1.x 的 `tr38901_pattern(theta, phi, slant_angle=0,
    polarization_model=2)[0]`：model 2 在 slant=0 时只做 c_θ = cos(0)·C̃_θ = C̃_θ，
    c_φ = 0。教程 notebook 里 MVDR 用的就是这个量。

    本仓装的 Sionna RT 2.x 把该函数放在 `sionna.rt.antenna_pattern.
    v_tr38901_pattern`（Mitsuba 算子，签名和返回值都不同），所以这里照公式重写，
    免得被版本差绑住；数值与库内实现一致（a_v/a_h/a_max/g_e_max 全按 TR 38.901）。
    """
    t3db = np.deg2rad(65.0)
    a_max = 30.0
    g_e_max = 8.0
    phi = np.asarray(phi_grid, np.float64)
    phi = phi + np.pi
    phi = phi - np.floor(phi / (2 * np.pi)) * 2 * np.pi
    phi = phi - np.pi
    a_v = -np.minimum(12.0 * ((np.asarray(theta_grid, np.float64) - np.pi / 2) / t3db) ** 2,
                      a_max)
    a_h = -np.minimum(12.0 * (phi / t3db) ** 2, a_max)
    a_db = -np.minimum(-(a_v + a_h), a_max) + g_e_max
    return np.sqrt(10.0 ** (a_db / 10.0)).astype(np.float32)


def array_manifold_vector(M: int, theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """[M², H, W] 复流形 = 阵元方向图 × 相位（教程 cell 17 的 array_manifold_vector）。

    与 `steering_vector` 的唯一差别是**乘了阵元方向图 C_θ**（tr38901）。CBF 用
    纯相位那份，MVDR 用这一份——教程两条路径用的就是不同的流形，照抄，不统一。
    """
    y, z = _element_grid(M)
    v = np.sin(theta_grid) * np.sin(phi_grid)
    w = np.cos(theta_grid)
    phase = 2 * np.pi * (y[:, None, None] * v[None] + z[:, None, None] * w[None])
    gain = tr38901_gain(theta_grid, phi_grid)[None]
    return (gain * np.exp(1j * phase)).astype(np.complex64)


class MVDRMapper:
    """批量生成用：MVDR（Capon）空间谱，教程 cell 68 `MVDR_spectrum` 的等价实现。

        x   = paths_to_response(paths, time_interval)        [M², L]
        R   = x·xᴴ                                           [M², M²]（对时延格求和）
        p   = 1 / |aᴴ·R⁻¹·a|                                 → 可选再取 10log10

    注意事项
    ---------------------------------------------------------------------------
    * **R 需要满秩**：L ≥ M²（教程 cell 68 的注释也这么说）。时延格太少或路径
      太少时 R 病态，`inv` 会给出一片噪声而不是报错——所以要记条件数。
    * 教程的 MVDR 用 `array_manifold_vector`（含阵元方向图），CBF 用纯相位
      `steering_vector`。两者**不通用**，别拿来互换。
    * 教程既演示过线性 `p_mvdr`，也演示过 `p_mvdr_db`；本类默认输出线性值，
      由调用方决定是否 `10log10` 与逐图 min-max（`db=True`）。
    """

    def __init__(self, M: int, theta_grid, phi_grid, time_interval_ns: float = 0.5,
                 db: bool = False, loading: float = 0.0, method: str = "solve"):
        self.M = M
        self.hw = theta_grid.shape
        self.interval = time_interval_ns
        self.db = db
        self.loading = loading
        self.method = method      # "solve" = 照抄教程（R⁻¹ 再四次型）；"chol" = 快速等价式
        self.a_mat = array_manifold_vector(M, theta_grid, phi_grid).reshape(M * M, -1)
        self.cond = np.nan        # 上一次调用的 R 条件数，供自检记录
        self.track_cond = True    # cond 要一次 SVD（8 ms/图），正式生成可只在每组首图算
        # 时间格合并缓存：同位置 4 个朝向共用一组 tau，one-hot 矩阵只需建一次
        self._key = None
        self._G = None

    # ------------------------------------------------------------------
    # 三段式：prepare（主线程，碰 DrJit）→ merge_matrix（主线程，建 G）
    #         → apply（可放线程池，纯 numpy）
    #
    # 为什么拆开：单图后处理约 83 ms，其中取回主机 30 ms、纯 numpy 53 ms。
    # 纯 numpy 那 53 ms 在多线程下实测加速 2.74×（4 线程），而 DrJit 的
    # CUDA→主机拷贝不确定是否线程安全，所以留在主线程串行做。
    # ------------------------------------------------------------------
    def prepare(self, paths, rx: int = 0) -> dict:
        """取单个接收机的复幅度与时间格索引回主机。**主线程调用**。

        三个坑，都是实测出来的：

        1. **先切片再传回主机**。`paths.a` 是 DrJit 的 CUDA 张量，`np.asarray`
           一次拷整组接收机（4×100×10 万 × 4 B = 160 MB/分量，56 ms）。改成
           `paths.a[0][rx, :, 0, 0, :]` 先切再拷，单接收机 15 ms + 6 ms。
        2. **不拼复数数组**。拼 complex64 要再读写 160 MB（+44 ms）；R 完全可以用
           实/虚部各做一次实数乘法拼出来（见 `apply`），省掉这一整趟。
        3. **合并用 one-hot 稀疏矩阵**。`argsort`+`reduceat` 69 ms/接收机，稀疏
           乘 14.5 ms，二者数值一致（差 1e-11 量级）。G 只由 tau 决定，而同一位置
           4 个朝向的 tau 完全相同（已实测），所以一组只建一次。
        """
        ave = paths.a
        if isinstance(ave, (tuple, list)) and len(ave) == 2:       # Sionna 2.x：(实, 虚)
            re = np.asarray(ave[0][rx, :, 0, 0, :], np.float32)
            im = np.asarray(ave[1][rx, :, 0, 0, :], np.float32)
        else:
            a = np.asarray(ave)
            if a.ndim == 6:
                a = a[..., 0]
            a = a[rx, :, 0, 0, :]
            re, im = a.real.astype(np.float32), a.imag.astype(np.float32)

        tau = np.asarray(paths.tau)[rx, 0, :].astype(np.float64) / 1e-9
        if getattr(paths, "valid", None) is not None:
            keep = np.asarray(paths.valid)[rx, 0, :].astype(bool)
            if not keep.all():
                tau, re, im = tau[keep], np.ascontiguousarray(re[:, keep]), \
                    np.ascontiguousarray(im[:, keep])

        t_max = float(np.max(tau)) if tau.size else 0.0
        grid = np.arange(0.0, max(np.ceil(t_max), self.interval),
                         self.interval, dtype=np.float32)
        idx = np.clip(np.searchsorted(grid, tau, side="left") - 1, 0, len(grid) - 1)
        return {"re": re, "im": im, "idx": idx, "n_bins": len(grid)}

    def prepare_from_response(self, aw: np.ndarray, tau_ns: np.ndarray,
                              valid: np.ndarray | None = None) -> dict:
        """从**已在主机上的**合成响应 `aw` [M², P] 生成 prep（给预编码 w 用）。

        `prepare` 只能取单根发射天线的响应（`paths.a[rx, :, 0, 0, :]` 把 tx-ant
        轴切成 0），所以 4×4 发射阵 + 波束权 w 的合成要在调用方做完再进来。
        时间格那段与 `prepare` 逐行相同——**故意复制**而不是抽公共函数：`prepare`
        是生产路径，任何重构都要重验它与旧数据逐位一致，不值得为 6 行冒这个险。
        """
        re = np.ascontiguousarray(aw.real.astype(np.float32))
        im = np.ascontiguousarray(aw.imag.astype(np.float32))
        tau = np.asarray(tau_ns, np.float64)
        if valid is not None and not np.all(valid):
            tau = tau[valid]
            re = np.ascontiguousarray(re[:, valid])
            im = np.ascontiguousarray(im[:, valid])
        t_max = float(np.max(tau)) if tau.size else 0.0
        grid = np.arange(0.0, max(np.ceil(t_max), self.interval),
                         self.interval, dtype=np.float32)
        idx = np.clip(np.searchsorted(grid, tau, side="left") - 1, 0, len(grid) - 1)
        return {"re": re, "im": im, "idx": idx, "n_bins": len(grid)}

    def merge_matrix(self, prep: dict):
        """按 `prepare` 给出时间格索引建 one-hot 稀疏矩阵 G（[L, P]）。"""
        import scipy.sparse as sp
        idx = prep["idx"]
        key = (prep["n_bins"], len(idx), hash(idx.tobytes()))
        if key != self._key:
            self._G = sp.csr_matrix(
                (np.ones(len(idx), np.float32), (idx, np.arange(len(idx)))),
                shape=(prep["n_bins"], len(idx)))
            self._key = key
        return self._G

    def apply(self, prep: dict, G, track_cond: bool | None = None) -> np.ndarray:
        """纯 numpy 段：合并 → R → cholesky → 三角解 → dB 谱。**可多线程**。

        `track_cond` 单独传参而不是读 `self.track_cond`：多线程里共享属性会被
        别的线程改掉；条件数写回 `prep["cond"]`（每个 prep 是线程私有的）。
        """
        if track_cond is None:
            track_cond = self.track_cond
        re, im = prep["re"], prep["im"]
        # G @ vᵀ 里 vᵀ 若是非连续视图，scipy 会慢一倍（31.6 → 14.5 ms）
        xr = np.ascontiguousarray((G @ np.ascontiguousarray(re.T)).T)
        xi = np.ascontiguousarray((G @ np.ascontiguousarray(im.T)).T)
        xr = xr.astype(np.float64)
        xi = xi.astype(np.float64)
        # R = x·xᴴ 展开成四次实数乘法，省掉复数数组：R = (xr·xrᵀ + xi·xiᵀ)
        #                                                    + j(xi·xrᵀ − xr·xiᵀ)
        R = (xr @ xr.T + xi @ xi.T) + 1j * (xi @ xr.T - xr @ xi.T)   # [M², M²]
        if self.loading > 0.0:
            R = R + self.loading * (np.trace(R).real / (self.M * self.M)) * np.eye(len(R))
        if track_cond:
            self.cond = float(np.linalg.cond(R))
            prep["cond"] = self.cond
        if self.method == "chol":
            # R = L·Lᴴ ⇒ aᴴR⁻¹a = ‖L⁻¹a‖²。只做一次三角解，比 solve 快 5 倍；
            # 数学上与教程的 1/|aᴴR⁻¹a| 等价，差别只在舍入。complex64 又快一倍
            #（29 → 16 ms），本场景 cond(R) 1e3~1e4，单精度够。
            import scipy.linalg as sla
            try:
                L = sla.cholesky(R, lower=True, check_finite=False).astype(np.complex64)
            except np.linalg.LinAlgError:
                L = None
            if L is not None:
                v = sla.solve_triangular(L, self.a_mat, lower=True, check_finite=False)
                denom = (v.real ** 2 + v.imag ** 2).sum(0)
            else:                       # R 数值上不正定（极少见）：退回 LU
                u = np.linalg.solve(R, self.a_mat)
                denom = np.abs(np.einsum("mp,mp->p", np.conj(self.a_mat), u))
        else:
            u = np.linalg.solve(R, self.a_mat)                   # R⁻¹·a，比显式 inv 稳
            denom = np.abs(np.einsum("mp,mp->p", np.conj(self.a_mat), u))
        p = 1.0 / np.maximum(denom, 1e-30)
        if self.db:
            p = 10.0 * np.log10(p)
        return p.reshape(self.hw).astype(np.float32)

    def __call__(self, paths, rx: int = 0) -> np.ndarray:
        """单图入口：prepare → merge_matrix → apply（串行，与拆开前逐位一致）。"""
        prep = self.prepare(paths, rx)
        return self.apply(prep, self.merge_matrix(prep))


def normalize_spectrum(arr: np.ndarray) -> np.ndarray:
    """逐图 min-max 到 [0,1]（教程 jet_colormap_convert 的第一步）。"""
    return (arr - np.min(arr)) / max(float(np.max(arr) - np.min(arr)), 1e-12)


def jet_from_unit(a: np.ndarray) -> np.ndarray:
    """[0,1] 灰度 → jet uint8 RGB。"""
    import matplotlib
    rgb = matplotlib.colormaps["jet"](np.clip(a, 0.0, 1.0))[..., :3]
    return (rgb * 255.0).round().astype(np.uint8)


def jet_convert(arr: np.ndarray) -> np.ndarray:
    """逐图 min-max → matplotlib jet → uint8 RGB（教程 jet_colormap_convert）。"""
    return jet_from_unit(normalize_spectrum(arr))


def read_colmap_poses(data_root) -> dict:
    """读作者发布的 images.txt → {img_id: (相机中心 C, 阵列 yaw)}。

    教程把 **c2w 四元数**写进 images.txt，平移写的是 `-R_c2w·C`。按 3DGS /
    COLMAP 的读法 `C = -Rᵀ t` 正好还原出 RX 位置（同一位置的 4 张图中心相同，
    已核对）。四元数是标量在前 [qw,qx,qy,qz]。
    """
    import os
    out = {}
    path = os.path.join(str(data_root), "sparse", "0", "images.txt")
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) < 10:
            continue
        i = int(f[0])
        w, x, y, z = (float(v) for v in f[1:5])
        t = np.array([float(v) for v in f[5:8]], np.float64)
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        out[i] = ((-R.T @ t).astype(np.float32), ANGLES[(i - 1) % 4])
    return out
