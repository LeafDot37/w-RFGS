# -*- coding: utf-8 -*-
"""双模态场景：一个进程里同时持有视觉相机与射频相机，共用一套高斯几何。

为什么要新写这一个类
--------------------------------------------------------------------------------
引擎的 `splat.scene.Scene` 一次只吃**一个** source_path，且视觉半边是 Blender 格式
（`transforms_train.json`）、射频半边是 COLMAP 格式（`sparse/0/`），
两种加载器二选一。既有管线是靠**两个进程**分开跑的（阶段一读视觉、阶段二读频谱），
所以从来不冲突。耦合优化必须在同一个进程里同时渲染两个模态 ⇒ 需要把两套相机
装进同一个对象。

做法：**视觉半边原样复用引擎的 `Scene`**（让它照旧建相机、写 `input.ply` /
`cameras.json`、并把共享几何 `create_from_pcd` 起来），射频半边用引擎自己的
COLMAP 加载器另建一份相机表。`splat/scene/*` 与 `splat/gaussian_renderer/*`
**一行不改**。

两个模态必须在同一个世界坐标系 —— 这一点本仓已实测（见
`code/tools/diag/probe_visual_rf_pose_align.py`）：两条轨迹绕同一个中心、半径同量级；
既有管线拿视觉几何去射频位姿上渲染本来就能出图。
"""
import os
from argparse import Namespace

from splat.scene import Scene
from splat.scene.dataset_readers import readColmapSceneInfo
from splat.utils.camera_utils import cameraList_from_camInfos


def load_rf_cameras(path, images="images", resolution=-1, data_device="cpu",
                    verbose=True, limit=None):
    """只把射频相机装起来，不碰高斯。

    走引擎既有的 COLMAP 加载器 + 引擎既有的相机装配。注意 `readColmapSceneInfo`
    读的是数据集自带的 `train_index.txt` / `test_index.txt`（2560 / 640），
    不是 COLMAP 惯例的 llffhold 划分，与既有阶段三同源。

    训练（`DualScene`）与渲染（`render_corf.py`）两侧共用本函数 ⇒ 相机口径不会分叉。
    返回 (train_cameras, test_cameras)。

    `limit`：只装前 N 个训练相机（**仅供探针/闸使用**；默认 None = 全量，
    训练与渲染两侧都不传，口径不变）。装 3200 张 300×200 图约 2–4 分钟，
    调参时每次等这一下很浪费。
    """
    if not path:
        raise ValueError("射频数据目录为空")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"射频数据目录不存在：{path}")
    info = readColmapSceneInfo(path, images, False)
    # 射频图像原生 300×200，**不做降采样**（与既有阶段三的 --resolution -1 口径一致）
    ns = Namespace(resolution=resolution, data_device=data_device)
    train_infos = info.train_cameras if limit is None else info.train_cameras[:int(limit)]
    train = cameraList_from_camInfos(train_infos, 1.0, ns)
    test = cameraList_from_camInfos(info.test_cameras, 1.0, ns)
    if verbose and train:
        c = train[0]
        print(f"[corf] 射频相机 {len(train)} 训练 / {len(test)} 测试  "
              f"{c.image_width}×{c.image_height}  device={data_device}")
    del info   # 让 PIL 句柄与 cam_info 一起释放
    return train, test


class DualScene:
    """视觉相机 + 射频相机 + 共享高斯。接口尽量与引擎的 `Scene` 同形。"""

    def __init__(self, dataset, gaussians, *, rf_source_path,
                 rf_images="images", rf_resolution=-1, rf_data_device="cpu",
                 shuffle=False, resolution_scales=(1.0,), load_rf=True):
        # —— 视觉半边：完全复用引擎的 Scene（它顺带把共享几何建起来）——
        self.opt = Scene(dataset, gaussians, shuffle=shuffle,
                         resolution_scales=list(resolution_scales))
        self.model_path = self.opt.model_path
        self.gaussians = gaussians
        self.cameras_extent = self.opt.cameras_extent
        self.source_path = dataset.source_path
        self.rf_source_path = rf_source_path
        self.rf_train = []
        self.rf_test = []
        if load_rf:
            self.rf_train, self.rf_test = load_rf_cameras(
                rf_source_path, rf_images, rf_resolution, rf_data_device)
        else:
            print("[dual-scene] 未加载射频相机（--rf_source_path 为空）")

    # —— 与引擎 Scene 同名的取用接口 ——
    def getTrainCameras(self, scale=1.0):
        return self.opt.getTrainCameras(scale)

    def getTestCameras(self, scale=1.0):
        return self.opt.getTestCameras(scale)

    def getRfTrainCameras(self):
        return self.rf_train

    def getRfTestCameras(self):
        return self.rf_test

    def save(self, iteration):
        """按引擎惯例落盘（写的是**光学**那一半的标准点云格式）。"""
        self.opt.save(iteration)
