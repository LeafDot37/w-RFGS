# -*- coding: utf-8 -*-
"""data_adapter.py —— 把本仓的 RF 频谱数据集适配成 NeRF² 要的 (ray_o, ray_d, tx) → 标量

上游要的数据布局是 `gateway_info.yml + tx_pos.csv + spectrum/*.png`（单网关 + 90×360
极坐标谱图），与我们这份 `images/ + sparse/0/ + train_index.txt` 完全不同。本文件
是两者之间的**唯一**桥梁。

三条设计原则
============
1. **相机约定不自己推**。R/T/FovX/FovY 全部走 `splat.scene.dataset_readers.
   readColmapSceneInfo` —— 与引擎用的是同一个函数。自己按 COLMAP 文档重推一遍
   看着更「独立」，实则是**多了一个可能不一致的地方**，而相机约定本仓已经栽过一次。
2. **射线解析式只有一份**。`rays_from_camera()` 与 `project_to_pixel()` 互为逆，
   且 G1 闸与训练路径**共用同一对函数**。各写一份是这类 bug 的标准来源。
3. **靶子全量上卡，训练循环里零磁盘 IO**。3200 张 300×200×3 的 uint8 是 576 MB，
   射线方向只有 4 份（见下）—— 这是把 8192×30000 次采样跑得动的前提。

射线方向只需算 4 份（实测事实）
===============================
3200 个位姿 = 800 个位置 × **恰好 4 个朝向**（纯偏航，`R[i] ≡ R[i mod 4]` 在 1e-6 内
对全部 3200 成立）。所以世界系射线方向只有 4 种，每张图按自己的朝向索引取一份即可。
这省掉的不只是算力，还有 3200×200×300×3 = 5.76 GB 的显存。

划分是**不可协商**的
====================
`train_index.txt`(2560) / `test_index.txt`(640) 是唯一划分来源。本模块**不提供任何
划分开关** —— 因为 `dataset_readers.py:271-277` 的 llffhold 分支被注释掉了，
`--eval` 对频谱集完全无效，暴露开关只会招来误用。
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 复用引擎的相机加载链 —— 刻意不自己推
from splat.scene.dataset_readers import readColmapSceneInfo

from .ray_geom import project_to_pixel, rays_from_camera   # noqa: F401  （再导出给 G1）

VIEWS_PER_POS = 4        # 实测：3200 = 800 位置 × 4 朝向


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class Nerf2Cfg:
    """一个训练节点的全部数据侧配置。会被原样写进 `nerf2_cfg.json`。"""
    data_dir: str
    method: str = ""
    # 射线口径（由 G1 判定后固定；默认即本仓已验证的那一支）
    sign_h: float = 1.0        # R[:,0] 前的符号
    sign_v: float = 1.0        # R[:,1] 前的符号
    swap_uv: bool = False      # 是否交换两轴（错法对照用）
    half_pixel: bool = True    # 取 (i+0.5, j+0.5) 为像素中心
    # 采样
    near: float = 0.10
    far: float = 0.0           # 0 ⇒ 按场景实算 1.05·max‖C−tx‖
    n_samples: int = 64
    far_margin: float = 1.05
    # tx：单 Tx ⇒ 固定常量。理由见 UPSTREAM.md §4.2（常量被首层 bias 吸收，
    # 不可辨识、不提供容量；设为可学习只会与 bias 退化，产不出可信的「反解位置」）
    tx: list = None
    # 杂项
    lut_n: int = 256
    seed: int = 0


# 射线 ↔ 像素的解析式全在 `ray_geom.py`（纯 numpy、可本机对拍），此处不再重复实现。


# --------------------------------------------------------------------------
# 主构建
# --------------------------------------------------------------------------
def build(cfg: Nerf2Cfg, verbose=True):
    """读数据集，返回一个 dict（可直接喂给 train/render）。

    返回键：
        names_test (640,) 升序 image_name —— 落盘顺序必须照它
        names_train (2560,)
        pos      (V,3) float64  相机中心
        R        (V,3,3)
        orient   (V,) int       朝向索引 ∈ {0,1,2,3}
        tx       (3,) float64
        near, far float
        targets  (V,H,W,3) uint8（可 fp16 存盘）
        dirs     (4,H,W,3) float32  世界系单位方向
        meta     dict
    """
    root = Path(cfg.data_dir)
    if not root.is_dir():
        raise FileNotFoundError("数据目录不存在：%s" % root)

    # --- 复用引擎的相机链。注意它会按需写 sparse/0/points3D.ply（已存在则不写）---
    si = readColmapSceneInfo(str(root), None, eval=False)
    test_infos = si.test_cameras
    train_infos = si.train_cameras
    if not test_infos or not train_infos:
        raise RuntimeError("划分读为空：train=%d test=%d" % (len(train_infos), len(test_infos)))

    # 顺序 = image_name 升序（readColmapSceneInfo 已排好）。render.py 走 shuffle=False，
    # 所以既有臂的 {k:05d}.png 就是这个顺序 —— 我们必须一致，否则 G2 配对全错。
    test_infos = sorted(test_infos, key=lambda c: c.image_name)
    train_infos = sorted(train_infos, key=lambda c: c.image_name)

    ci0 = test_infos[0]
    W, H = int(ci0.width), int(ci0.height)
    fx = W / (2.0 * np.tan(ci0.FovX / 2.0))
    fy = H / (2.0 * np.tan(ci0.FovY / 2.0))
    cx, cy = W / 2.0, H / 2.0

    all_infos = sorted(test_infos + train_infos, key=lambda c: c.image_name)
    V = len(all_infos)
    names = [c.image_name for c in all_infos]
    # 全局序号映射。**不能**用「test 段 + train 段」的拼接偏移 —— 那与自然顺序不同，
    # 而「4 朝向」是按自然顺序成立的，配错朝向会静默喂错射线（形状对、数值全错）。
    name2g = {n: i for i, n in enumerate(names)}
    test_gidx = np.array([name2g[c.image_name] for c in test_infos], dtype=np.int64)
    train_gidx = np.array([name2g[c.image_name] for c in train_infos], dtype=np.int64)

    R = np.stack([np.asarray(c.R, np.float64) for c in all_infos])          # (V,3,3)
    T = np.stack([np.asarray(c.T, np.float64) for c in all_infos])          # (V,3)
    # getWorld2View2 给的是 p_cam = Rᵀ·p_world + T ⇒ 相机中心 C 满足 Rᵀ·C + T = 0
    pos = np.einsum('vij,vj->vi', -R, T)                                   # C = −R·T

    # --- 朝向只取 4 类（实测 R[i] ≡ R[i mod 4]）---
    # 前提：自然顺序就是**连续编号**。先钉住这条，否则「模 4」毫无意义。
    #
    # **编号是 1-based**（2026-09-16 实测）：`sparse/0/images.txt` 的 3200 条位姿，
    # 名字恰为 00001.png..03200.png；`train_index.txt`/`test_index.txt` 里写的也是
    # 这套 1-based **图号**（字面量就是 "00005"，直接对应 images/00005.png）。
    # 本仓既有 reader 的 `image_name` 去扩展名后是 "00005" 字符串，与索引文件逐字相等。
    # 别处一律用**位置序号**（0-based），只有这里做「名字↔位置」的连续性检查。
    want = ["%05d" % i for i in range(1, V + 1)]
    if names != want:
        bad = next((i for i, (a, b) in enumerate(zip(names, want)) if a != b), 0)
        raise AssertionError(
            "自然顺序不是连续编号：第 %d 个是 %r，期望 %r。"
            "「3200 = 800×4 同朝向」依赖连续编号，编号一变模 4 就配错朝向。"
            % (bad, names[bad] if bad < len(names) else None, want[bad]))

    orient = (np.arange(V, dtype=np.int64) % VIEWS_PER_POS)
    dev = np.abs(R - R[orient]).max(axis=(1, 2))
    if dev.max() > 1e-6:
        k = int(dev.argmax())
        raise AssertionError(
            "R[%d] 与 R[%d] 差 %.3e —— 「V = 800×4 同朝向」的假设不成立，"
            "射线方向不能再按 4 份复用。请重新检查数据集结构。"
            % (k, k % VIEWS_PER_POS, dev.max()))

    # --- tx：固定常量 ---
    if cfg.tx is not None:
        tx = np.asarray(cfg.tx, np.float64)
    else:
        # 无上游真值时取测量区质心 —— 只是**占位常量**，不是「反解出的 Tx」，
        # 不得当物理结论报告（见 UPSTREAM.md §4.2）
        tx = pos.mean(0)

    # --- near/far：按本数据集实算，不抄上游的 far=5 ---
    dist = np.linalg.norm(pos - tx[None, :], axis=1)
    far = cfg.far if cfg.far > 0 else float(cfg.far_margin * dist.max())
    near = float(cfg.near)
    if not (0 < near < far):
        raise ValueError("near=%g far=%g 非法" % (near, far))

    # --- 世界系射线方向：只算 4 份 ---
    dirs = np.empty((VIEWS_PER_POS, H, W, 3), dtype=np.float32)
    for k in range(VIEWS_PER_POS):
        rays_from_camera(R[k], fx, fy, cx, cy, W, H,
                         cfg.sign_h, cfg.sign_v, cfg.half_pixel, cfg.swap_uv, out=dirs[k])

    # --- 靶子全量上内存 ---
    targets = np.empty((V, H, W, 3), dtype=np.uint8)
    for i, c in enumerate(all_infos):
        im = Image.open(c.image_path).convert("RGB")
        if im.size != (W, H):
            raise ValueError("%s 尺寸 %s != (%d,%d)" % (c.image_name, im.size, W, H))
        targets[i] = np.asarray(im, dtype=np.uint8)

    meta = {
        "data_dir": str(root), "method": cfg.method,
        "W": W, "H": H, "fx": float(fx), "fy": float(fy), "cx": cx, "cy": cy,
        "fovx_deg": float(np.degrees(ci0.FovX)), "fovy_deg": float(np.degrees(ci0.FovY)),
        "n_total": V, "n_test": len(test_infos), "n_train": len(train_infos),
        "pos_min": pos.min(0).tolist(), "pos_max": pos.max(0).tolist(),
        "dist_max": float(dist.max()), "dist_min": float(dist.min()),
        "orient_checked": True,
    }
    if verbose:
        print("[adapter] %s" % root, flush=True)
        print("  分辨率 %dx%d  fx=fy=%.3f  HFOV=%.2f° VFOV=%.2f°"
              % (W, H, fx, meta["fovx_deg"], meta["fovy_deg"]), flush=True)
        print("  视角 %d（train %d / test %d）朝向 4 类已核验"
              % (V, len(train_infos), len(test_infos)), flush=True)
        print("  near=%.3f  far=%.3f（实算 = %.2f×max‖C−tx‖=%.3f）"
              % (near, far, cfg.far_margin, dist.max()), flush=True)

    return {
        "names_all": names,
        "names_test": [c.image_name for c in test_infos],
        "names_train": [c.image_name for c in train_infos],
        "test_gidx": test_gidx, "train_gidx": train_gidx,
        "pos": pos, "R": R, "orient": orient,
        "tx": tx, "near": near, "far": far,
        "targets": targets, "dirs": dirs, "meta": meta,
    }


# --------------------------------------------------------------------------
# 缓存（避免每次训练都读 3200 张图）
# --------------------------------------------------------------------------
CACHE_VERSION = 2


def save_cache(path, data, cfg: Nerf2Cfg):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(p),
        version=np.array([CACHE_VERSION]),
        cfg_json=np.array([json.dumps(asdict(cfg), ensure_ascii=False)]),
        names_test=np.array(data["names_test"]),
        names_train=np.array(data["names_train"]),
        test_gidx=data["test_gidx"], train_gidx=data["train_gidx"],
        pos=data["pos"], R=data["R"], orient=data["orient"], tx=data["tx"],
        near=np.array([data["near"]]), far=np.array([data["far"]]),
        targets=data["targets"], dirs=data["dirs"],
        meta_json=np.array([json.dumps(data["meta"], ensure_ascii=False)]),
    )
    return str(p)


def load_cache(path):
    z = np.load(str(path), allow_pickle=False)
    if int(z["version"][0]) != CACHE_VERSION:
        raise ValueError("cache 版本 %d != %d，请重建" % (int(z["version"][0]), CACHE_VERSION))
    return {
        "names_test": [str(x) for x in z["names_test"]],
        "names_train": [str(x) for x in z["names_train"]],
        "test_gidx": z["test_gidx"], "train_gidx": z["train_gidx"],
        "pos": z["pos"], "R": z["R"], "orient": z["orient"], "tx": z["tx"],
        "near": float(z["near"][0]), "far": float(z["far"][0]),
        "targets": z["targets"], "dirs": z["dirs"],
        "meta": json.loads(str(z["meta_json"][0])),
        "cfg": json.loads(str(z["cfg_json"][0])),
    }


# 缓存里存的是 `build()` **解算后**的 near/far（`far=0` 会被解成
# `1.05·max‖C−tx‖`），而 `Nerf2Cfg` 里的是**原始**取值。拿两者直接比永远不等 ——
# 2026-09-16 实测踩到：DATA 段用默认 `--far 0` 建缓存 ⇒ npz 里存 8.691441；
# TRAIN 段也传 `--far 0`，检查却报「near/far 与本次配置不符」，4 个节点全停。
# 教训：**要比同一层的量**。`save_cache` 已经把**原始** cfg 写进 `cfg_json`，比它。
CACHE_CFG_FIELDS = ("data_dir", "sign_h", "sign_v", "swap_uv", "half_pixel",
                    "near", "far", "far_margin")


def check_cache_cfg(cached_cfg: dict, cfg: "Nerf2Cfg"):
    """缓存是**哪套配置**建的？返回差异列表（空 = 一致，可直接复用）。

    只比**真的会改变缓存张量**的字段：
        targets ← data_dir
        dirs    ← sign_h / sign_v / swap_uv / half_pixel
        near/far ← near / far / far_margin（后者参与解算）
    刻意**不比** `n_samples` / `lut_n` / `seed` —— 它们不落进 npz，
    比了只会造出「必须重建缓存」的假阳性（重建一次要读 3200 张 PNG）。
    """
    bad = []
    for k in CACHE_CFG_FIELDS:
        if k not in cached_cfg:
            continue
        a, b = cached_cfg[k], getattr(cfg, k)
        if isinstance(a, float) or isinstance(b, float):
            same = abs(float(a) - float(b)) <= 1e-9
        else:
            same = a == b
        if not same:
            bad.append("%s: 缓存 %r ≠ 本次 %r" % (k, a, b))
    return bad


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------
def sample_batch(data, batch, gen, device, split="train", targets=None):
    """抽一批射线。返回 (rays_o, rays_d, tx, label) —— 全部在 device 上。

    label 形状 (B,3) float32 ∈ [0,1]（RGB 域）。N2 的标量靶子由调用方就地反解，
    不在这里做 —— 反解是 `scalar_io` 的事，采样层不该知道输出参数化。

    随机抽「图」再抽「图内像素」而不是全域乱抽：让一个 batch 里的射线有空间相关性，
    与上游 `DataLoader` 按谱图分块的行为一致（`:105` 是一整张谱图的射线连续存放）。
    """
    g = data["train_gidx_t"] if split == "train" else data["test_gidx_t"]
    k = torch.randint(0, g.numel(), (batch,), generator=gen, device=device)
    gidx = g[k]                                    # → 全局自然序号

    H, W = data["meta"]["H"], data["meta"]["W"]
    py = torch.randint(0, H, (batch,), generator=gen, device=device)
    px = torch.randint(0, W, (batch,), generator=gen, device=device)

    o = data["pos_t"][gidx]                                     # (B,3)
    d = data["dirs_t"][data["orient_t"][gidx], py, px]          # (B,3)
    tx = data["tx_t"].expand(batch, 3)
    tgt = targets if targets is not None else data["targets_t"]
    lab = tgt[gidx, py, px].float().div_(255.0)                 # (B,3) ∈ [0,1]
    return o, d, tx, lab


def sample_view(data, k, device, split="test", targets=None):
    """整张视角的全部射线 + 标签。k 是**该 split 内**的序号。
    返回 (rays_o(HW,3), rays_d(HW,3), tx(HW,3), label(HW,3))。
    """
    gm = data["test_gidx_t"] if split == "test" else data["train_gidx_t"]
    g = int(gm[k])
    H, W = data["meta"]["H"], data["meta"]["W"]
    o = data["pos_t"][g].expand(H * W, 3)
    d = data["dirs_t"][data["orient_t"][g]].reshape(H * W, 3)
    tx = data["tx_t"].expand(H * W, 3)
    tgt = targets if targets is not None else data["targets_t"]
    lab = tgt[g].reshape(H * W, 3).float().div_(255.0)
    return o, d, tx, lab


def to_device(data, device, fp16_targets=True):
    """把 numpy 侧的量搬到 device。targets 默认 fp16 省一半显存（576 MB → 288 MB）。"""
    data["pos_t"] = torch.as_tensor(data["pos"], dtype=torch.float32, device=device)
    data["R_t"] = torch.as_tensor(data["R"], dtype=torch.float32, device=device)
    data["orient_t"] = torch.as_tensor(data["orient"], dtype=torch.long, device=device)
    data["tx_t"] = torch.as_tensor(data["tx"], dtype=torch.float32, device=device)
    data["dirs_t"] = torch.as_tensor(data["dirs"], dtype=torch.float32, device=device)
    data["test_gidx_t"] = torch.as_tensor(data["test_gidx"], dtype=torch.long, device=device)
    data["train_gidx_t"] = torch.as_tensor(data["train_gidx"], dtype=torch.long, device=device)
    t = torch.as_tensor(data["targets"], device=device)
    data["targets_t"] = t.half() if fp16_targets else t
    return data
