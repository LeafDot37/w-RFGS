# -*- coding: utf-8 -*-
"""空间射频场数据集（`gen_sionna_field.py --mode gen`）的读取与渲染口径。

数据形状
--------------------------------------------------------------------------------
    点云       cloud.ply：房间 6 个面的格心，72000 点（几何冻结）
    场         field/<j>.npy：float16 dBm，形状 [72000]，第 j 个发射源
    位姿       meta["rx_local"]：36 个接收机位置；meta["views"]["cameras"]：6 个面
    GT 图      不落盘，评测/训练时由「逐点标量场 + 同一光栅化器」现渲

为什么 GT 现渲而不预存
--------------------------------------------------------------------------------
场是逐点的，GT 图只是它在某个接收机视角下的**可视化**。现渲保证：
    预测图与 GT 图走同一份点云、同一套尺度/不透明度、同一台相机、同一条 jet 链，
两者唯一的差别就是喂进去的逐点标量，因此图像差可以直接归因到场误差。
预存 PNG 反而会引入量化与重采样口径分歧。

像素-点严格对齐
--------------------------------------------------------------------------------
点云就是 6 个测量面的格心（`gen_sionna_field.plane_specs`），GT 场也在这同一批
格心上取值 ⇒ 不存在「场在某处、点云在别处」的插值误差。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.gaussian_renderer import render                     # noqa: E402
from splat.scene.cameras import Camera                         # noqa: E402
from splat.scene.gaussian_model import GaussianModel           # noqa: E402
from rfgs.coupled.train_student import StudentView                          # noqa: E402

# 与 gen_sionna_field.write_cloud_ply 写进 ply 的 logit 一致（sigmoid(4)=0.982）
CLOUD_OPACITY_LOGIT = 4.0


def freeze(gm: GaussianModel) -> None:
    for n in ("_xyz", "_scaling", "_rotation", "_features_dc", "_features_rest", "_opacity"):
        getattr(gm, n).requires_grad = False


class SpatialScene:
    """一个场景（目前只有 SionnaRoom）的只读视图。"""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        m = self.meta
        if m.get("format") != "spatial_rf_v1":
            raise ValueError(f"不是空间场数据集：format={m.get('format')!r}")

        self.gm = GaussianModel(3)
        self.gm.load_ply(os.path.join(root, m["cloud_ply"]))
        freeze(self.gm)
        self.view = StudentView(self.gm)
        self.xyz = self.gm.get_xyz.detach().float()
        self.P = int(self.xyz.shape[0])
        if self.P != int(m["n_points"]):
            raise ValueError(f"点云点数 {self.P} != meta.n_points {m['n_points']}")

        self.tx = torch.tensor(m["tx_local"], dtype=torch.float32)
        self.rx = torch.tensor(m["rx_local"], dtype=torch.float32)
        self.lo = float(m["db_map"]["lo_dbm"])
        self.hi = float(m["db_map"]["hi_dbm"])
        if not self.hi > self.lo:
            raise ValueError("meta.db_map 的 hi 必须大于 lo")

        v = m["views"]
        self.n_face = int(v["n"])
        self.H = self.W = int(v["img"])
        # fov 存在每个 camera 上（meta 顶层只有一个 fov 汇总值）
        fx = {float(c["fovx"]) for c in v["cameras"]}
        fy = {float(c["fovy"]) for c in v["cameras"]}
        if len(fx) != 1 or len(fy) != 1:
            raise ValueError(f"6 个朝向的 fov 不一致：fovx={fx} fovy={fy}")
        self.fovx = float(np.radians(fx.pop()))
        self.fovy = float(np.radians(fy.pop()))
        self._R = [np.asarray(c["R"], dtype=np.float32) for c in v["cameras"]]
        self.face_names = [c["name"] for c in v["cameras"]]

        # 相机名 -> 它主要看向哪个面（几何闸用）
        self.cam_center = self.xyz.mean(dim=0)

    # ------------------------------------------------------------ 场
    def dbm(self, j: int, device="cuda") -> torch.Tensor:
        p = os.path.join(self.root, "field", f"{int(j):05d}.npy")
        a = np.load(p).astype(np.float32)
        if a.shape != (self.P,):
            raise ValueError(f"{p} 形状 {a.shape} != ({self.P},)")
        return torch.from_numpy(a).to(device)

    def u(self, j: int, device="cuda") -> torch.Tensor:
        """逐点标量 ∈ [0,1]：u = clip((dBm−lo)/(hi−lo))."""
        d = self.dbm(j, device)
        return ((d - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)

    def u_to_dbm(self, u: torch.Tensor) -> torch.Tensor:
        return u * (self.hi - self.lo) + self.lo

    # ------------------------------------------------------------ 相机
    def camera(self, rx_idx: int, face: int, uid: int | None = None) -> Camera:
        """接收机第 rx_idx 个位姿、第 face 个朝向。R 是 c2w（列 = right, down, fwd）。"""
        R = self._R[int(face)]
        C = self.rx[int(rx_idx)].numpy().astype(np.float32)
        T = -(R.T @ C)
        return Camera(uid if uid is not None else int(rx_idx) * self.n_face + int(face),
                      R, T.astype(np.float32), self.fovx, self.fovy,
                      torch.zeros(3, self.H, self.W), None,
                      f"rx{rx_idx:02d}_{self.face_names[int(face)]}", int(face))

    # ------------------------------------------------------------ 渲染
    def render_scalar(self, cam: Camera, u_point: torch.Tensor, pipe,
                      background: torch.Tensor, opacity_logit: float = CLOUD_OPACITY_LOGIT):
        """逐点标量 → 光栅化混合后的标量图 (1,H,W)。u_point: [P] 或 [P,1]。"""
        if u_point.dim() == 1:
            u_point = u_point[:, None]
        if u_point.shape[0] != self.P:
            raise ValueError(f"u_point 点数 {u_point.shape[0]} != {self.P}")
        self.view.set_rf_scalar(torch.full((self.P, 1), float(opacity_logit),
                                           device=u_point.device))
        mixed = render(cam, self.view, pipe, background,
                       override_color=u_point.repeat(1, 3))["render"]
        return mixed[0:1]


def make_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))
