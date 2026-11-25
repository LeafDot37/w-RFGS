# -*- coding: utf-8 -*-
"""从**光学点云**算几何可见性特征：发射源 → 点的射线在场景里走多远被挡住。

为什么需要它
--------------------------------------------------------------------------------
NIST 走廊数据上的误差分解（`tools/diag_nist_learnability.py` 与一次性脚本）
显示：把测试发射源换成最近的训练源时，**只有 9.9% 的点可见性翻转**，但这些点
贡献了 **68.7%** 的绝对误差（翻转点 36.7 dB，非翻转点 1.85 dB）。也就是说，
「新发射源」这一轴上的难点几乎全是**遮挡关系改变**，而不是场强水平本身。

遮挡关系是几何问题，而光学点云就是几何。所以这里把「射线被挡」做成逐点特征，
交给前馈头当输入（拼进 `app` 那段，引擎与头都不用改）。

特征（每个 (发射源, 点) 一条）
--------------------------------------------------------------------------------
    seen        1.0 = 整条射线在栅格里都空；0.0 = 被挡
    first_hit   首个占据体素沿射线的归一化位置 ∈ [0,1]（1 = 全程畅通）
    occ_frac    采样点里占据体素的比例（粗糙的"透过率"）

注意
--------------------------------------------------------------------------------
- 栅格用 **占据**（有高斯点的体素=1），不是 SDF；点云只有可见面，
  所以 `dilate` 默认开一格来补洞，代价是表面变厚 1 个体素。
- 两端各跳过一段：发射源在自由空间里（skip_tx），点本身是沿法线偏移
  `query_offset_delta` 得到的，跳得太短会把自身所在的面当遮挡物。
- 这是**近似**：真实路径含镜面反射，无视线也可能有路径（实测约 20–30%），
  有视线也可能无路径（栅格空洞误判）。它不追求与 Sionna 完全一致，
  只作为头的一个强特征；判别力由 `eval_vis_feature` 量出来。
"""
from __future__ import annotations

import numpy as np


def _fill_dilate(occ: np.ndarray, dim: np.ndarray) -> np.ndarray:
    """3×3×3 膨胀一次（补点云空洞）。用移位或实现，避免 scipy 依赖。"""
    o = occ.reshape(dim)
    out = o.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                out |= np.roll(np.roll(np.roll(o, dx, 0), dy, 1), dz, 2)
    return out.reshape(-1)


class OccupancyGrid:
    """点云占据栅格。坐标一律是世界系（与 field/meta 同系）。"""

    def __init__(self, xyz: np.ndarray, voxel: float = 0.10,
                 dilate: bool = True, lo=None, hi=None):
        xyz = np.asarray(xyz, np.float32)
        self.voxel = float(voxel)
        self.lo = (xyz.min(0) - 2 * self.voxel) if lo is None else np.asarray(lo, np.float32)
        self.hi = (xyz.max(0) + 2 * self.voxel) if hi is None else np.asarray(hi, np.float32)
        self.dim = np.ceil((self.hi - self.lo) / self.voxel).astype(np.int64)
        idx = np.floor((xyz - self.lo) / self.voxel).astype(np.int64)
        idx = np.clip(idx, 0, self.dim - 1)
        key = self._key(idx)
        occ = np.zeros(int(self.dim.prod()), bool)
        occ[key] = True
        self.occ = _fill_dilate(occ, self.dim) if dilate else occ
        self.n_occ = int(self.occ.sum())

    def _key(self, i: np.ndarray) -> np.ndarray:
        return i[..., 0] + self.dim[0] * (i[..., 1] + self.dim[1] * i[..., 2])

    def ray_features(self, tx, pts: np.ndarray, skip_tx: float = 0.25,
                     skip_pt: float = 0.25, step: float = 0.05,
                     chunk: int = 6000) -> dict[str, np.ndarray]:
        """返回逐点特征。`tx` 是 [3]（世界系），`pts` 是 [P,3]。"""
        tx = np.asarray(tx, np.float32).reshape(3)
        pts = np.asarray(pts, np.float32)
        P = len(pts)
        seen = np.ones(P, bool)
        first = np.ones(P, np.float32)
        occf = np.zeros(P, np.float32)
        for s in range(0, P, chunk):
            p = pts[s:s + chunk]
            d = p - tx
            L = np.linalg.norm(d, axis=1)
            usable = L > (skip_tx + skip_pt)
            u = d / np.maximum(L, 1e-6)[:, None]
            a0 = np.minimum(skip_tx, L * 0.5)
            a1 = np.maximum(L - skip_pt, a0 + 1e-3)
            ns = np.maximum(2, np.ceil((a1 - a0) / step).astype(np.int64))
            nmax = int(ns.max())
            t = np.arange(nmax)[None, :] / np.maximum(ns - 1, 1)[:, None]   # [n, nmax] ∈ [0,1]
            dist = a0[:, None] + (a1 - a0)[:, None] * t
            valid = (t <= 1.0) & usable[:, None]
            pos = tx[None, None, :] + u[:, None, :] * dist[:, :, None]
            ii = np.floor((pos - self.lo) / self.voxel).astype(np.int64)
            inb = ((ii[..., 0] >= 0) & (ii[..., 0] < self.dim[0]) &
                   (ii[..., 1] >= 0) & (ii[..., 1] < self.dim[1]) &
                   (ii[..., 2] >= 0) & (ii[..., 2] < self.dim[2]))
            ii = np.clip(ii, 0, self.dim - 1)
            hit = self.occ[self._key(ii)] & valid & inb
            has = hit.any(1)
            seen[s:s + chunk] = ~has & usable
            occf[s:s + chunk] = hit.sum(1) / np.maximum(valid.sum(1), 1)
            # 首个命中处的归一化弧长
            arg = np.argmax(hit, axis=1)
            fh = t[np.arange(len(p)), arg]
            fh = np.where(has, fh, 1.0)
            first[s:s + chunk] = np.where(usable, fh, 1.0)
        return {"seen": seen.astype(np.float32), "first_hit": first, "occ_frac": occf}

    @classmethod
    def from_ply(cls, path: str, voxel: float = 0.10, opacity_thr: float = 0.5,
                 dilate: bool = True) -> "OccupancyGrid":
        """直接读 3DGS 的 ply（只取 xyz + opacity；opacity 是 logit）。"""
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        if "opacity" in v.data.dtype.names and opacity_thr is not None:
            op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
            xyz = xyz[op >= opacity_thr]
        return cls(xyz, voxel=voxel, dilate=dilate)
