"""init_cloud_cov.py —— 带【世界系协方差】的初始化点云导出。

为什么需要这个脚本
------------------
`create_from_pcd`（splat/scene/gaussian_model.py）**只吃 xyz 与 colors**：
  scales = log(sqrt(distCUDA2))[...].repeat(1, 3)   ← 各向同性球，与 MVSplat 无关
  rots   = 单位四元数                                ← 朝向信息为零
  opacities = inverse_sigmoid(0.1)                   ← 常数
⇒ MVSplat 已经预测出来的**协方差（各向异性尺度 + 局部表面朝向）被整段丢掉**。

这不是「种下去被洗掉」，是**从来没种下去过**。本脚本把这条通道接上。

与 init_cloud.py 的关系
-----------------------
位姿 / 聚类 / 前向 / 过滤 / 导出**逐字复用** init_cloud.py 的 helper（直接 import），
只改三件事：

  1. **记录每个高斯的来源视角下标**。init_cloud.py 算了 `view_idx` 但只用于算深度，
     没导出；把相机系四元数搬到世界系必须要它。

  2. **rot_0..3 导出为世界系四元数**。MVSplat 的 GaussianAdapter 返回的
     scales/rotations 是**相机系**的 —— 它自己的注释就承认了：
         NOTE: These aren't yet rotated into world space, but they're only used for
         exporting Gaussians to ply files. This needs to be fixed...
     协方差矩阵被 `c2w @ cov @ c2wᵀ` 搬到世界系了，rotations 没有。
     本脚本逐字复刻那一步：
         R_world = R_c2w @ R_cam
     依据 `feedforward/model/encoder/common/gaussians.py:build_covariance`：
         cov = R diag(s²) Rᵀ   ⇒  R 的第 i **列** 是标准差 s_i 的轴
     所以世界系旋转就是同一个左乘。**尺度是旋转不变量，不需要变换。**
     四元数顺序注意：adapter 的 `quaternion_to_matrix` 用的是 **xyzw**（scipy 序，
     该文件注释「Order changed to match scipy format!」），而 3DGS 的 ply 用 wxyz
     ⇒ 导出时 `np.roll(..., 1, axis=1)`（export_ply_3dgs 内部做）。

  3. **体素降采样时属性取「代表点」而不是分量平均**。xyz 仍取质心
     ⇒ **与 init.ply 逐位相同**（同一个 np.unique 分组、同一个质心算法）。
     属性平均会毁掉本脚本要保住的信息：voxel = 0.0929 m 而高斯最长轴只有 0.036 m
     ⇒ 一个格子里平均掉若干个点，四元数还被**跨相机系**的平均毁掉。

输出
----
  <out_dir>/init_cov.ply   **simple 格式 + 额外字段**：x/y/z/nx/ny/nz/red/green/blue
                           与 init.ply **逐位相同**，另带 opacity / scale_0..2（log）/
                           rot_0..3（wxyz，**世界系**）。
                           为什么是「simple + 额外」而不是纯 3DGS 格式：fetchPly
                           （splat/scene/dataset_readers.py:107-113）**只读
                           red/green/blue，不认 f_dc_***。做成这种格式，同一个文件
                           既是 RF_POINTCLOUD 的合法输入、又带协方差 ⇒ 对照臂与处理臂
                           可以用**同一个文件**，跨文件行序一致性这个无声事故源被消掉。
  <out_dir>/init_cov_3dgs.ply  标准 3DGS 格式，**仅供 supersplat 查看**，不进训练。
  <out_dir>/init_cov.npz   scale_log / rot_wxyz / rot_cam_wxyz / rgb_u8 /
                           cluster_id / view_id（闸与诊断用，不进训练）

**不覆盖** init.ply / init_view.ply —— 历史工件只读。

用法（cwd = code/；环境变量必须与产出 init.ply 的那一次完全一致，
否则 xyz 不会逐位相同，单变量对照就废了）
    RF_OUT_DIR=<新目录> RF_NUM_CLUSTERS=16 RF_VIEWS_PER_CLUSTER=4 \
    RF_GAZES_PER_CLUSTER=2 RF_MAX_GAUSSIANS=1500000 RF_MAX_DEPTH=15 \
    RF_SOURCE_PATH=<blender_visual_dataset> \
    python init_cloud_cov.py mode=test +experiment=re10k \
        checkpointing.load=<re10k.ckpt>
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
import init_cloud as IC  # noqa: E402  复用其位姿/聚类/前向/过滤/导出 helper

from scipy.spatial.transform import Rotation as _R  # noqa: E402


# ---------------------------------------------------------------------------
# 本脚本新增的数值工具
# ---------------------------------------------------------------------------

def quat_xyzw_to_mat(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐字复刻 feedforward/.../common/gaussians.py:quaternion_to_matrix。

    顺序是 **xyzw**（scipy 序），不是 wxyz —— 那个文件自己注释了这一点。
    逐字复刻（而不是另写一个等价的）是刻意的：一旦这里和 adapter 分叉，
    朝向就会整体错一个旋转，而且**在指标上完全看不出来**（初始缩放下
    各向异性的方向错了，物理阶段照样能训出一个云）。
    """
    i, j, k, r = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / ((q * q).sum(-1) + eps)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(-1, 3, 3)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数，**wxyz**（3DGS ply 的顺序）。走 scipy，避免自己写分支。"""
    return np.roll(_R.from_matrix(R).as_quat(), 1, axis=1)  # scipy 给 xyzw


def export_ply_with_cov(means, rgb_u8, scales_log, opac, q_xyzw, path: Path):
    """一个文件同时装「fetchPly 认的字段」与「协方差字段」。

    为什么必须塞进**同一个**文件
    ---------------------------
    `splat/scene/dataset_readers.py:fetchPly` 只读 x/y/z + red/green/blue + nx/ny/nz
    （**不认 f_dc_\\***）。所以处理臂的 `RF_POINTCLOUD` 必须指向本文件，
    `create_from_pcd` 才能拿到与 init.ply **完全相同**的 xyz 与 colors；
    而 scale/rot 再从同一文件的同一行读出来。
    同文件 + 同行序 ⇒ 结构上不可能错位。跨文件的行序一致性是一类
    **只在无声中出错**的事故（行序被任何一步改写过就会错配，而指标上看不出来）。
    （另存一份 `init_cov_3dgs.ply` 走 IC.export_ply_3dgs，只供 supersplat 看。）

    字段：x/y/z、nx/ny/nz（与 init.ply 一样是假零）、red/green/blue（同 init.ply）、
          opacity、scale_0..2（**log** 半径，3DGS 约定）、rot_0..3（**wxyz**）。
    入参 `scales_log` 必须是**已取对数**的（3DGS 约定）；线性半径由调用方取 log。
    """
    n = means.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    el = np.empty(n, dtype=dtype)
    el["x"], el["y"], el["z"] = means[:, 0], means[:, 1], means[:, 2]
    el["nx"] = el["ny"] = el["nz"] = 0.0
    el["red"], el["green"], el["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    el["opacity"] = opac
    el["scale_0"], el["scale_1"], el["scale_2"] = scales_log[:, 0], scales_log[:, 1], scales_log[:, 2]
    # 3DGS 的 ply 用 wxyz；本函数的 q_xyzw 是 model 序（scipy 序）
    #   np.roll(xyzw, 1) = wxyz  ⇒  rot_0=w=原第3列, rot_1=x=原第0列, ...
    q_wxyz = np.roll(q_xyzw, 1, axis=1)
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = (
        q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def voxel_downsample_rep(points: np.ndarray, voxel: float, attrs: list):
    """xyz 取质心（与 IC.voxel_downsample 逐位相同），属性取**离质心最近的代表点**。

    返回 (centroids, rep_attrs, counts, rep_index)。

    分组用与 IC.voxel_downsample 完全相同的 `np.unique(keys, axis=0, return_inverse=True)`
    ⇒ 输出顺序与质心都逐位相同，所以「position 与 init.ply 完全一致」是可以断言的。
    唯一的区别是属性不再 `np.add.at(...)/counts` 平均。
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    cells = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=cells)
    sums = np.zeros((cells, points.shape[1]))
    np.add.at(sums, inv, points)
    cent = sums / counts[:, None]

    # 每个格子内离质心最近的点；并列时取下标小的（lexsort 的第三键）⇒ 确定性
    d = ((points - cent[inv]) ** 2).sum(1)
    order = np.lexsort((np.arange(len(d)), d, inv))   # 主键 inv，次键 d，末键下标
    first = np.unique(inv[order], return_index=True)[1]
    rep = order[first]
    return cent, [a[rep] for a in attrs], counts, rep


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict) -> None:
    cfg = IC.load_typed_root_config(cfg_dict)
    IC.set_cfg(cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    # --- 超参：**必须与产出 init.ply 的那一次一致**（否则 xyz 不逐位相同） ----
    K = IC.env_int("RF_NUM_CLUSTERS", 8)
    V = IC.env_int("RF_VIEWS_PER_CLUSTER", 8)
    G = IC.env_int("RF_GAZES_PER_CLUSTER", 1)
    target = IC.env_int("RF_MAX_GAUSSIANS", 500_000)
    voxel_override = IC.env_float("RF_VOXEL_SIZE", 0.0)
    opac_th = IC.env_float("RF_OPACITY_THRESHOLD", 0.05)
    max_depth = IC.env_float("RF_MAX_DEPTH", 15.0)
    limit = IC.env_int("RF_CLUSTERS_LIMIT", 0)
    bg_name = os.environ.get("RF_BACKGROUND", "black").lower()
    near, far = IC.env_float("RF_NEAR", 1.0), IC.env_float("RF_FAR", 100.0)
    src = Path(os.environ.get(
        "RF_SOURCE_PATH", "datasets/RF-3DGS_dataset/blender_visual_dataset"
    ))
    out_dir = Path(os.environ.get("RF_OUT_DIR", "outputs/rf_init_cov"))
    ref_ply = os.environ.get("RF_REF_PLY", "").strip()   # 可选：对拍 xyz 的参照 init.ply

    bg = (np.array([1.0, 1.0, 1.0], np.float32) if bg_name == "white"
          else np.array([0.0, 0.0, 0.0], np.float32))
    assert src.exists(), f"source path not found: {src.resolve()}"

    print("=" * 70)
    print("init_cloud_cov：导出带世界系协方差的初始化点云")
    print(f"  K={K} V={V} G={G}  target={target}  voxel_override={voxel_override}")
    print(f"  opac_th={opac_th}  max_depth={max_depth}  bg={bg_name}")
    print(f"  源数据 {src}")
    print(f"  输出   {out_dir}")
    print("=" * 70)

    # --- 位姿（与 init_cloud.py 逐字相同，含 Blender→MVSplat 的符号翻转） -----
    with (src / "transforms_train.json").open() as f:
        frames = json.load(f)["frames"]
    n = len(frames)
    c2w = np.stack([np.asarray(fr["transform_matrix"], np.float32) for fr in frames])
    c2w[:, :3, 1:3] *= -1
    paths = [src / fr["file_path"] for fr in frames]
    pos = c2w[:, :3, 3]
    look = c2w[:, :3, 2]
    print(f"loaded {n} visual poses from {src.name}")

    if K <= 0:
        per = IC.env_int("RF_TARGET_FRAMES_PER_CLUSTER", 100)
        K = max(1, int(round(n / per)))
        print(f"auto K = round({n}/{per}) = {K}")

    labels, centers = IC.kmeans(pos, K, seed=0)
    for i in range(K):
        m = labels == i
        print(f"  cluster {i:2d}: {int(m.sum()):3d} frames, "
              f"max span {float(np.ptp(pos[m], axis=0).max()):.2f}")

    # --- 模型（同 init_cloud.py） --------------------------------------------
    print(f"loading checkpoint: {cfg.checkpointing.load}")
    if V != 2:
        cfg_dict["dataset"]["view_sampler"]["num_context_views"] = V
        cfg = IC.load_typed_root_config(cfg_dict)
        IC.set_cfg(cfg_dict)
        print(f"rebuilt encoder with num_context_views={V}")
    checkpoint_path = IC.update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    encoder, _ = IC.get_encoder(cfg.model.encoder)
    decoder = IC.get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = IC.ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=None,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval().to(device)
    step = model_wrapper.global_step
    print(f"peak VRAM estimate for this run: ~{0.66 * V:.1f} GiB (V={V})")

    fx, fy, cx, cy = 800.0, 800.0, 800.0, 450.0
    W, H = 1600.0, 900.0

    # --- 逐簇前向（新增 c2w_R 与 clust 两条轨迹） ---------------------------
    means, opac, colors, scales, rots, depths = [], [], [], [], [], []
    # 第 7 条轨迹：逐点深度标准差（MVSplat 内部张量「深度分布的宽度」）。
    # 它**不是**本脚本原先导出的任何量 —— 原版 encoder 算完就丢，本分支把它带出来。
    # 与 depths 同序、逐点一一对应；走完全相同的三道处理（两个 mask + 体素代表点）。
    dstd = []
    c2w_R, clust, gview = [], [], []
    sel_info = {}
    n_forward = 0
    for i in range(K):
        if limit and i >= limit:
            break
        idx = np.where(labels == i)[0]
        k = IC.quantize_gaze(look[idx])
        counts = np.bincount(k, minlength=8)
        order = np.argsort(-counts)
        sectors = [int(s) for s in order if counts[s] >= 2][:G]
        if not sectors:
            print(f"  [skip] cluster {i}: no gaze sector with >= 2 frames")
            continue
        for g in sectors:
            pool = idx[k == g]
            views = pool[IC.fps(pos[pool], min(V, len(pool)))]
            if len(views) < V:
                print(f"  [skip] cluster {i} gaze {g}: pool {len(pool)} < V={V}")
                continue
            n_forward += 1
            sel_info[f"{i}:g{g}"] = [int(v) for v in views]

            imgs = torch.stack([IC.load_composite(paths[v], bg) for v in views])
            intrins = torch.zeros(len(views), 3, 3)
            intrins[:, 0, 0] = fx / W
            intrins[:, 1, 1] = fy / H
            intrins[:, 0, 2] = cx / W
            intrins[:, 1, 2] = cy / H
            intrins[:, 2, 2] = 1.0
            imgs, intrins = IC.rescale_and_crop(imgs, intrins, (256, 256))

            context = {
                "image": imgs.unsqueeze(0).to(device),
                "intrinsics": intrins.unsqueeze(0).to(device),
                "extrinsics": torch.from_numpy(c2w[views]).unsqueeze(0).to(device),
                "near": torch.full((1, len(views)), near).to(device),
                "far": torch.full((1, len(views)), far).to(device),
            }
            with torch.no_grad():
                dump = {}
                gs = encoder.forward(context, step, False, visualization_dump=dump)

            gmeans = gs.means[0].detach().cpu().numpy()
            n_per_view = gmeans.shape[0] // len(views)
            # 每个高斯由 V 个视角中的一个发出，view-major 顺序
            view_idx = np.repeat(np.arange(len(views)), n_per_view)
            cam_centers = c2w[views][:, :3, 3]
            gdepth = np.linalg.norm(gmeans - cam_centers[view_idx], axis=1)

            means.append(gmeans)
            opac.append(gs.opacities[0].detach().cpu().numpy())
            colors.append(gs.harmonics[0, :, :, 0].detach().cpu().numpy())
            scales.append(dump["scales"][0].detach().cpu().numpy())
            rots.append(dump["rotations"][0].detach().cpu().numpy())
            depths.append(gdepth)
            # dump["depth_std"] 是 [1, V, h, w, srf, s]；拍平到一维后的行序与
            # gs.means[0] 的 (v r srf spp) 完全一致（gpp=srf=spp=1 时两者都是 v-major）。
            dstd.append(dump["depth_std"][0].detach().cpu().numpy().reshape(-1))
            # 发出该高斯的那个视角的 camera-to-world 旋转（MVSplat 约定，含符号翻转）
            c2w_R.append(c2w[views][view_idx][:, :3, :3].astype(np.float64))
            clust.append(np.full(gmeans.shape[0], i, dtype=np.int32))
            gview.append(views[view_idx].astype(np.int32))
            print(f"  cluster {i:2d} gaze {g}: {gmeans.shape[0]:7d} gaussians  "
                  f"(V={len(views)}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB)")

            del gs, dump, context, imgs
            torch.cuda.empty_cache()

    means = np.concatenate(means)
    opac = np.concatenate(opac)
    colors = np.concatenate(colors)
    scales = np.concatenate(scales)
    rots = np.concatenate(rots)
    depths = np.concatenate(depths)
    dstd = np.concatenate(dstd)
    assert dstd.shape[0] == means.shape[0], "depth_std 与 xyz 行数不等 ⇒ 逐点对齐不成立"
    c2w_R = np.concatenate(c2w_R)
    clust = np.concatenate(clust)
    gview = np.concatenate(gview)
    print(f"provenance: {len(gview)} gaussians, 来源视角 {len(np.unique(gview))} 个 "
          f"(应等于 {n_forward} × {V})")
    print(f"\nraw union: {means.shape[0]} gaussians from {n_forward} forwards "
          f"(each 256x256 pixel -> {V*256*256} max/forward)")

    # --- 世界系旋转（**这一步是 init_cloud.py 缺的**） ----------------------
    # 逐字复刻 adapter 对协方差做的事：R_world = R_c2w @ R_cam
    R_cam = quat_xyzw_to_mat(rots.astype(np.float64))
    R_world = c2w_R @ R_cam
    q_world_wxyz = mat_to_quat_wxyz(R_world)
    q_world_xyzw = np.roll(q_world_wxyz, -1, axis=1)     # 交回 model 序给导出函数
    print(f"world rotations composed: |det| p50 "
          f"{np.median(np.abs(np.linalg.det(R_world))):.6f}  "
          f"|q| p50 {np.median(np.linalg.norm(q_world_wxyz, axis=1)):.6f}")

    # --- 不透明度 / 深度过滤（同 init_cloud.py，mask 也要作用到 c2w_R） ----
    q = np.quantile(opac, [0.5, 0.9, 0.99, 1.0])
    print(f"opacity quantiles p50/p90/p99/max: {np.round(q, 3)}")
    mask = opac >= opac_th
    print(f"opacity filter (>= {opac_th}): keep {mask.mean()*100:.1f}% "
          f"({mask.sum()}/{means.shape[0]})")
    means, opac, colors, scales, q_world_xyzw, depths, clust, gview, rots, dstd = \
        [a[mask] for a in (means, opac, colors, scales, q_world_xyzw, depths,
                           clust, gview, rots, dstd)]

    dq = np.quantile(depths, [0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"depth-from-own-view quantiles p1/p5/p50/p95/p99/max: {np.round(dq, 2)}")
    if max_depth > 0:
        m = depths <= max_depth
        print(f"depth filter (<= {max_depth}): keep {m.mean()*100:.1f}% ({m.sum()})")
        means, opac, colors, scales, q_world_xyzw, depths, clust, gview, rots, dstd = \
            [a[m] for a in (means, opac, colors, scales, q_world_xyzw, depths,
                            clust, gview, rots, dstd)]

    # --- 体素降采样：xyz 取质心（同原版），属性取代表点（本脚本的改动） -----
    if voxel_override > 0:
        vox = voxel_override
    else:
        vox = max(2.0 * float(np.median(scales)), 1e-3)
        print(f"auto voxel size {vox:.4f} (median gaussian scale "
              f"{np.median(scales):.4f}, target cap {target})")

    # --- 可选：把**体素降采样之前**的全部候选点存下来 ------------------------
    # 为什么需要：降采样把 700 万候选压成 128 万代表点，若只留代表点，
    # 「改变点的**密度分配**」这一类干预就没有候选池可用（想给某个格子多放点，
    # 手上却没有那个格子的其余成员）。由 RF_DUMP_PREVOXEL=<完整文件名> 打开。
    # 默认关闭 ⇒ 既有行为逐字不变（连文件句柄都不开）。
    dump_pre = os.environ.get("RF_DUMP_PREVOXEL", "").strip()
    # ⚠ 此刻的 `vox` 只是**起始**体素，下面那个循环会一路 ×1.25 涨到满足 target 为止。
    # 所以本块的 voxel_size 记的是起始值，**不是**最终用于分格的值 —— 想拿最终值请在
    # 循环之后记。第一次落盘时踩过这个坑（prevox.npz 里是 0.0195，实际是 0.0929）。
    if dump_pre:
        with open(dump_pre, "wb") as fh:
            np.savez(
                fh,
                xyz=means.astype(np.float32),
                depth_std=dstd.astype(np.float32),
                opacity=opac.astype(np.float32),
                rgb_u8=(IC.sh_dc_to_rgb(colors) * 255.0 + 0.5).astype(np.uint8),
                scale_linear=scales.astype(np.float32),
                rot_wxyz=np.roll(q_world_xyzw, 1, axis=1).astype(np.float32),
                cluster_id=clust.astype(np.int32),
                view_id=gview.astype(np.int32),
                voxel_size=np.float64(vox),
                target=np.int64(target),
            )
        print(f"pre-voxel dump -> {dump_pre}  ({means.shape[0]} candidates, "
              f"voxel={vox:.4f})")

    for _ in range(12):
        pts, (o2, c2, s2, q2, d2), counts, rep = voxel_downsample_rep(
            means, vox, [opac, colors, scales, q_world_xyzw, dstd]
        )
        if len(pts) <= target:
            break
        vox *= 1.25
    print(f"voxel(vox={vox:.4f}): {len(pts)} gaussians kept from {means.shape[0]} "
          f"(每格平均 {means.shape[0]/max(len(pts),1):.2f} 个点被压成 1 个)")
    means, opac, colors, scales, q_world_xyzw = pts, o2, c2, s2, q2
    dstd = d2                    # 与上面 5 条同一次代表点选取 ⇒ 行序必然一致
    rep_clust = clust[rep]
    rep_view = gview[rep]
    # 代表点的**相机系**四元数（未做任何平均）—— 供闸 G3 作对照臂：
    # 同一批点、同一套代表点选取，只有「有没有搬到世界系」这一处不同。
    q_cam_rep_wxyz = np.roll(rots[rep], 1, axis=1)

    # --- 导出 ----------------------------------------------------------------
    # rgb_u8 的公式与 init_cloud.py:448 **逐字相同** ⇒ init_cov.ply 的
    # red/green/blue 与 init.ply 逐位相同（G1 的另一半：颜色也是单变量对照的一部分）。
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (IC.sh_dc_to_rgb(colors) * 255.0 + 0.5).astype(np.uint8)
    # scales 是**线性**米制半径（adapter 的输出），py 与 3DGS 约定都要 log
    scales_log = np.log(np.clip(scales, 1e-6, None)).astype(np.float32)
    out_ply = out_dir / "init_cov.ply"
    export_ply_with_cov(means, rgb_u8, scales_log, opac, q_world_xyzw, out_ply)
    IC.export_ply_3dgs(means, colors, opac, scales, q_world_xyzw,
                       out_dir / "init_cov_3dgs.ply")
    np.savez_compressed(out_dir / "init_cov.npz",
                        cluster_id=rep_clust.astype(np.int32),
                        view_id=rep_view.astype(np.int32),
                        scale_log=scales_log,
                        scale_linear=scales.astype(np.float32),
                        opacity=opac.astype(np.float32),
                        rot_wxyz=np.roll(q_world_xyzw, 1, axis=1).astype(np.float32),
                        rot_cam_wxyz=q_cam_rep_wxyz.astype(np.float32),
                        rgb_u8=rgb_u8,
                        # 逐点深度标准差（MVSplat 内部量，原版丢弃）。行序与
                        # init_cov.ply 的 vertex 行**逐行对应**。
                        depth_std=dstd.astype(np.float32),
                        voxel_size=np.float64(vox))
    with (out_dir / "clusters.json").open("w") as f:
        json.dump({"num_clusters": K, "views_per_cluster": V,
                   "gazes_per_cluster": G, "sizes": [int((labels == i).sum()) for i in range(K)],
                   "selected_views": sel_info,
                   "gaussians": {"raw_union": int(len(clust)), "final": int(len(means))},
                   "voxel_size": float(vox), "max_depth": max_depth,
                   "opacity_threshold": opac_th,
                   "note": "x/y/z 与 init.ply 逐位相同；scale/rot 为世界系且取代表点（未平均）",
                   "source": str(src), "bg": bg_name}, f, indent=2)

    # --- 新产物的自陈与「免费反查」 -----------------------------------------
    # ⚠ 这一段**放在落盘之后**：它只是读数，即使自身出错也不该把几十分钟的前向白跑。
    # 免费反查：如果 depth_std 与网络自己已经吐出来的 scale_linear 强相关，
    # 说明 MVSplat 早就把这件事做完了 ⇒ 再喂回去不可能带新信息，当场判死、不必花 GPU。
    dq2 = np.quantile(dstd, [0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"\ndepth_std quantiles p1/p5/p50/p95/p99/max: {np.round(dq2, 4)}")
    # 无信息基线：pdf 若**完全没信息**（均匀铺满 D 个候选），std 就是这个数。
    # 观测中位数显著小于它才说明这个量带信息；接近它 ⇒ 是「一片糊」的退化分布。
    Dc = int(encoder.cfg.num_depth_candidates)
    _dc = 1.0 / (1.0 / far + np.linspace(0.0, 1.0, Dc) * (1.0 / near - 1.0 / far))
    null_std = float(np.sqrt(((_dc - _dc.mean()) ** 2).mean()))
    print(f"无信息基线（pdf 均匀铺在 {Dc} 个候选上）std = {null_std:.4f} m ⇒ "
          f"p50 基线比 = {np.median(dstd) / null_std:.3f}（<1 才有信息）")
    # scales 是 (N,3) 的**逐轴**半径，depth_std 是标量 ⇒ 必须先把三轴压成一个当量半径。
    # 这里同时给算术均值与几何均值两个口径（几何均值对应体积等效），避免口径挑出来的假相关。
    _ld = np.log1p(dstd.astype(np.float64))
    _sc = np.clip(scales.astype(np.float64), 1e-12, None)
    rho_mean = float(np.corrcoef(_ld, np.log(_sc.mean(axis=1)))[0, 1])
    rho_geom = float(np.corrcoef(_ld, np.log(np.exp(np.log(_sc).mean(axis=1))))[0, 1])
    rho_opac = float(np.corrcoef(_ld, opac.astype(np.float64))[0, 1])
    print(f"免费反查 ρ(log1p(depth_std), log(当量半径)) = "
          f"{rho_mean:+.4f}（算术均值） / {rho_geom:+.4f}（几何均值）"
          f"   ρ(·, opacity) = {rho_opac:+.4f}")
    print("  （|ρ| 接近 1 ⇒ 网络早已把该量编码进输出，喂回去是冗余，应判死；"
          "接近 0 ⇒ 是网络没给过的独立信息）")

    # --- 自查：xyz 与 colors 是否与参照 init.ply 逐位相同（单变量对照的前提） -
    # create_from_pcd 只吃这两样 ⇒ 这两样逐位相同，处理臂与对照臂的**唯一**差别
    # 才是协方差。颜色也别漏：它同样是「输入」的一部分。
    if ref_ply:
        ref = IC.PlyData.read(ref_ply)["vertex"]
        rx = np.stack([np.asarray(ref[c], dtype=np.float64) for c in ("x", "y", "z")], 1)
        print("\n--- 对拍 init.ply ---")
        same_n = rx.shape == means.shape
        print(f"  N  参照 {rx.shape[0]}  本脚本 {means.shape[0]}  {'✓' if same_n else '✗ 不等'}")
        if not same_n:
            print("  ⚠ 点数不等 ⇒ 单变量对照不成立（连比都不用比了）")
        else:
            dx = np.abs(rx - means).max()
            print(f"  xyz 最大绝对差        {dx:.3e}  "
                  f"{'✓ 逐位相同' if dx == 0 else '✗ 不是逐位相同'}")
            if hasattr(ref, "data") and "red" in [p.name for p in ref.properties]:
                rr = np.stack([np.asarray(ref[c], dtype=np.uint8)
                               for c in ("red", "green", "blue")], 1)
                dc = int(np.abs(rr.astype(np.int16) - rgb_u8.astype(np.int16)).max())
                print(f"  red/green/blue 最大差 {dc}  "
                      f"{'✓ 逐位相同' if dc == 0 else '✗ 不是逐位相同'}")
                if dc != 0:
                    print("  ⚠ 颜色变了 ⇒ 即便位置相同，单变量对照也破了（输入里有两项在动）")
            if dx != 0:
                print("  ⚠ 单变量对照前提被破坏：位置变了，协方差的效应无法与位置效应分开")

    print(f"\nDONE -> {out_ply.resolve()}")
    print(f"  npz -> {(out_dir / 'init_cov.npz').resolve()}")


if __name__ == "__main__":
    main()
