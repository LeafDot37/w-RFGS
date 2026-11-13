# -*- coding: utf-8 -*-
"""角域谱图的训练数据面：作者相机 + 逐发射源 GT 谱图。

为什么不用 `splat.scene.Scene`
--------------------------------------------------------------------------------
`Scene` 会把 3200 张 GT 图全部 `Image.open().convert("RGB")` 并搬成 float 张量
（3200×3×200×300×4 B ≈ 2.3 GB），而本任务**每张图的 GT 依赖发射源**：
37 个发射源 × 3200 张 = 118,400 张，一次性载入不现实。这里只从作者的
`sparse/0/{cameras,images}.txt` 还原**相机位姿**（与 `readColmapCameras` 逐字同口径），
图像按 (tx, 图号) 现取现读。

位姿口径（与 `splat/scene/dataset_readers.py:readColmapCameras` 完全一致）
--------------------------------------------------------------------------------
    R = qvec2rotmat(q).T          # 世界→相机 的旋转（Camera 内部再转置回去）
    T = tvec
    FoVx/FoVy = focal2fov(fx, W) / focal2fov(fy, H)
    camera_center = -R_c2wᵀ·t = 作者 images.txt 里的接收位置

作者是把 **c2w 四元数**写进 images.txt 的（tvec = −R_c2w·C，见
`sim/spectrum_gen.read_colmap_poses` 的说明），上面的读法正好还原出接收点，
与 Sionna 仿真里用的 RX 位置逐图对齐。

GT 的表示
--------------------------------------------------------------------------------
磁盘上存的是**逐图 min-max 归一化后的 8bit 灰度**（`sim/gen_cbf_spectra.py` 的输出）。
读取端查 256 级 jet 表还原成 RGB —— 与作者 `jet_colormap_convert` 同一张表，
也就是 `splat.color_scalar.make_jet_lut(256)`。训练与评测都用这一条通路，
避免训练（灰度）与评测（RGB）两套口径。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.scene.cameras import Camera                          # noqa: E402
from splat.scene.colmap_loader import (qvec2rotmat,             # noqa: E402
                                       read_extrinsics_text,
                                       read_intrinsics_text)
from splat.utils.graphics_utils import focal2fov                # noqa: E402
from splat.color_scalar import make_jet_lut                     # noqa: E402
from splat.param_frame import rotation_matrix                   # noqa: E402


def load_cameras(poses_root: str, data_device: str = "cpu"):
    """→ (cams: {image_name: Camera}, train_names, test_names)。

    图像只用来给 `Camera` 一个占位张量（宽高必须对，渲染分辨率 = 300×200），
    构造完立刻释放 ⇒ 3200 个相机不占 2.3 GB。GT 由 `SpectrumStore` 另取。
    """
    poses_root = str(poses_root)
    intr = read_intrinsics_text(os.path.join(poses_root, "sparse", "0", "cameras.txt"))
    extr = read_extrinsics_text(os.path.join(poses_root, "sparse", "0", "images.txt"))

    cams = {}
    for key, e in extr.items():
        it = intr[e.camera_id]
        if it.model == "SIMPLE_PINHOLE":
            fx, fy = it.params[0], it.params[0]
        elif it.model == "PINHOLE":
            fx, fy = it.params[0], it.params[1]
        else:
            raise ValueError(f"不支持相机模型 {it.model}")
        R = np.transpose(qvec2rotmat(e.qvec))
        T = np.array(e.tvec)
        name = os.path.basename(e.name).split(".")[0]
        dummy = torch.zeros(3, it.height, it.width)
        cam = Camera(colmap_id=it.id, R=R, T=T,
                     FoVx=focal2fov(fx, it.width), FoVy=focal2fov(fy, it.height),
                     image=dummy, gt_alpha_mask=None, image_name=name,
                     uid=len(cams), data_device=data_device)
        cam.original_image = None                     # 不参与任何计算，释放占位
        cams[name] = cam

    def _read_ids(p):
        return [Path(x).stem for x in Path(p).read_text().split() if x.strip()]

    root = Path(poses_root)
    train_names = [n for n in _read_ids(root / "train_index.txt") if n in cams]
    test_names = [n for n in _read_ids(root / "test_index.txt") if n in cams]
    if not train_names or not test_names:
        raise RuntimeError(f"train/test index 读空：{root}")
    missing = set(cams) - set(train_names) - set(test_names)
    if missing:
        raise RuntimeError(f"{len(missing)} 个相机既不在 train 也不在 test（示例 "
                           f"{sorted(missing)[:3]}）⇒ 划分文件与 images.txt 不对应")
    return cams, train_names, test_names


class SpectrumStore:
    """逐发射源/实例 GT 谱图：`<root>/<key>/<image_name>.png`（8bit 灰度）→ (3,H,W)。

    `key_fmt` 决定目录名：历史数据集是 `tx{idx:03d}`（默认，所有旧调用点不变），
    参数化数据集是 `inst{idx:03d}`。

    查表用**与作者同源的 256 级 jet**（`make_jet_lut`），dtype 保持 float32；
    返回的图在 [0,1]，与 `apply_jet()` 的输出同一量纲，可直接算 L1/SSIM。
    """

    def __init__(self, root: str, device="cuda", key_fmt: str = "tx{idx:03d}"):
        self.root = Path(root)
        self.device = device
        self.key_fmt = key_fmt
        lut = make_jet_lut(256).clamp(0, 1)
        self.lut = (lut * 255.0).round().to(torch.uint8).cpu().numpy()   # (256,3) uint8
        self._cache: dict[tuple[int, str], torch.Tensor] = {}
        self._ucache: dict[tuple[int, str], torch.Tensor] = {}

    def dir(self, tx: int) -> Path:
        return self.root / self.key_fmt.format(idx=int(tx))

    def count(self, tx: int) -> int:
        d = self.dir(tx)
        return len(list(d.glob("*.png"))) if d.is_dir() else 0

    def path(self, tx: int, name: str) -> Path:
        return self.dir(tx) / f"{name}.png"

    def has(self, tx: int, name: str) -> bool:
        return self.path(tx, name).exists()

    def __call__(self, tx: int, name: str) -> torch.Tensor:
        key = (int(tx), name)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = self.path(tx, name)
        if not p.exists():
            raise FileNotFoundError(f"缺谱图 {p}（发射源 {tx} 还没生成完？）")
        u8 = np.asarray(Image.open(p).convert("L"))
        rgb = self.lut[u8]                                   # (H,W,3) uint8
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        if len(self._cache) > 256:                            # 小 LRU，避免随机视角下吃满内存
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = t
        if len(self._ucache) > 256:
            self._ucache.pop(next(iter(self._ucache)))
        self._ucache[key] = torch.from_numpy(u8).float().div_(255.0).to(
            self.device).unsqueeze(0)                         # (1,H,W)，顺带缓存省一次读盘
        return t

    def unit(self, tx: int, name: str) -> torch.Tensor:
        """原始归一化谱值 u∈[0,1]，形状 (1,H,W)；只给亮区加权损失用。"""
        key = (int(tx), name)
        hit = self._ucache.get(key)
        if hit is not None:
            return hit
        p = self.path(tx, name)
        if not p.exists():
            raise FileNotFoundError(f"缺谱图 {p}（发射源 {tx} 还没生成完？）")
        u8 = np.asarray(Image.open(p).convert("L"))
        t = torch.from_numpy(u8).float().div_(255.0).to(self.device).unsqueeze(0)
        if len(self._ucache) > 256:
            self._ucache.pop(next(iter(self._ucache)))
        self._ucache[key] = t
        return t


@dataclass
class InstanceTable:
    """参数化数据集的实例表（`meta.json` 里有 `instances` 时才有）。

    下标 = 实例 id（加载时已校验连续）。`xyz` 是位置，`yaw_off/pitch_off` 是
    **相对场景朝向基准帧的偏移角**（弧度，头吃这个），`yaw_abs/pitch_abs` 是绝对
    角度（用来造旋转矩阵），`rot` 是 (N,3,3) 的 TR38901 绝对旋转。
    """

    ids: np.ndarray
    position_id: np.ndarray
    xyz: np.ndarray
    yaw_off: np.ndarray
    pitch_off: np.ndarray
    yaw_abs: np.ndarray
    pitch_abs: np.ndarray
    freq_hz: np.ndarray
    role: np.ndarray
    views: np.ndarray
    rot: np.ndarray
    w_theta: np.ndarray          # 波束指向 θ（度，发射源本地系）；uniform 时无定义
    w_phi: np.ndarray            # 波束指向 φ（度）
    w_uniform: np.ndarray        # bool：该实例的权是不是 uniform
    w_label: np.ndarray          # 原始标签（码本里的字符串，溯源用）
    base_instance: np.ndarray    # 派生实例（如 w_holdout）克隆自哪个实例 id；-1=无
    roles: dict
    meta: dict
    key_fmt: str

    def __len__(self):
        return len(self.ids)

    def mask(self, roles):
        want = {roles} if isinstance(roles, str) else set(roles)
        return np.array([r in want for r in self.role], bool)


def load_instances(meta_path: str, key_fmt: str = "inst{idx:03d}") -> InstanceTable | None:
    """读参数化实例表。**旧 meta（无 `instances`）返回 None**，调用方走老路径。"""
    m = json.load(open(meta_path, encoding="utf-8"))
    raw = m.get("instances")
    if raw is None:
        return None
    raw = sorted(raw, key=lambda i: int(i["id"]))
    ids = np.array([int(i["id"]) for i in raw], np.int64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("实例 id 有重复")
    # 行号 = 排序后的位置；数组按行索引，目录名用 ids[row]。
    # 正式实例表 id 连续 0..N-1（两者相等），闸测表用 100 段编号也照常工作。
    n = len(raw)
    yaw_abs = np.array([float(i.get("yaw_abs_rad", 0.0)) for i in raw], np.float64)
    pitch_abs = np.array([float(i.get("pitch_abs_rad", 0.0)) for i in raw], np.float64)
    xyz = np.array([i["xyz"] for i in raw], np.float32)
    rot = np.stack([rotation_matrix(yaw_abs[k], pitch_abs[k], 0.0) for k in range(n)])
    # 偏移角的三种写法（正式表 / 闸测表 / 兼容旧写法）；都没有则退化为 0
    def _off(i, rad_keys, deg_keys):
        for k in rad_keys:
            if k in i:
                return float(i[k])
        for k in deg_keys:
            if k in i:
                return float(np.deg2rad(float(i[k])))
        return 0.0

    yaw_off = np.array([_off(i, ("offset_yaw_rad", "yaw_off_rad"), ("delta_yaw_deg",))
                        for i in raw], np.float64)
    pitch_off = np.array([_off(i, ("offset_pitch_rad", "pitch_off_rad"),
                               ("delta_pitch_deg",)) for i in raw], np.float64)
    roles = {}
    for k, i in enumerate(raw):
        roles.setdefault(i.get("role", "train"), []).append(k)
    roles = {k: np.array(v, np.int64) for k, v in roles.items()}
    # 波束标签：老表（无 w 轴）没有这个字段 ⇒ 一律当 uniform，训练端按 use_tx_w
    # 开关决定要不要读它（关着的时候也就不构成 in_dim 的一部分）。
    from sim.w_codebook import UNIFORM, parse_label
    w_label = np.array([str(i.get("w_label", UNIFORM)) for i in raw])
    w_theta = np.zeros(n, np.float64)
    w_phi = np.zeros(n, np.float64)
    w_uniform = np.ones(n, bool)
    for k in range(n):
        tp = parse_label(str(w_label[k]))
        if tp is not None:
            w_theta[k], w_phi[k] = float(tp[0]), float(tp[1])
            w_uniform[k] = False
    split = m.get("split", {})
    orig = int(m.get("orig_tx_index", -1))
    if orig >= 0 and split.get("test") and orig not in [int(x) for x in split["test"]]:
        raise RuntimeError(f"原发射源下标 {orig} 不在 test 里")
    return InstanceTable(
        ids=ids,
        position_id=np.array([int(i.get("position_id", -1)) for i in raw], np.int64),
        xyz=xyz, yaw_off=yaw_off, pitch_off=pitch_off,
        yaw_abs=yaw_abs, pitch_abs=pitch_abs,
        freq_hz=np.array([float(i["freq_hz"]) for i in raw], np.float64),
        role=np.array([i.get("role", "train") for i in raw]),
        views=np.array([i.get("views", "all") for i in raw]),
        rot=rot.astype(np.float64),
        w_theta=w_theta, w_phi=w_phi, w_uniform=w_uniform, w_label=w_label,
        base_instance=np.array([int(i.get("base_instance", -1)) for i in raw],
                               np.int64),
        roles=roles, meta=m, key_fmt=key_fmt)


def load_tx_table(meta_path: str):
    """→ (tx_world [T,3] float32, train_idx, test_idx, orig_idx)。

    `orig_idx` 是作者原 RF-3DGS 发射源在本表里的下标（meta 的 `orig_tx_index`），
    它必须落在 test 里 —— 这是「原发射源留作测试」这条要求的机器可查形式。
    """
    m = json.load(open(meta_path, encoding="utf-8"))
    tx = np.asarray(m["tx_world"], np.float32)
    tr = [int(i) for i in m["split"]["train"]]
    te = [int(i) for i in m["split"]["test"]]
    orig = int(m.get("orig_tx_index", -1))
    if orig >= 0 and orig not in te:
        raise RuntimeError(f"原发射源下标 {orig} 不在 test 里 ⇒ 与「原发射源留作测试」冲突")
    return tx, tr, te, orig
