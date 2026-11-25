# -*- coding: utf-8 -*-
"""空间场头的**逐点附加输入**：最近训练源参考场 + 光学几何可见性。

为什么是这两个
--------------------------------------------------------------------------------
NIST 走廊数据的误差分解（`tools/diag_head_errors.py`、`tools/eval_vis_feature.py`）：

    参考场（最近训练源整场）          测试 MAE 5.31 dB；非翻转点 1.85 dB
    纯坐标头（当前）                测试 MAE 10.87 dB；非翻转点 9.64 dB、翻转只占误差能量 19%
    参考场 + 几何遮挡硬置零（5 cm）   测试 MAE 3.68 dB

结论：头缺两样东西——**逐点场强记忆**（自己从坐标重建学不出来）和**遮挡几何**
（沿 TX→点射线被挡与否，光学点云就能算）。这里把两样都做成逐点输入，
由调用方拼进 `app`；头的结构、引擎、渲染器一律不改。

约定
--------------------------------------------------------------------------------
- 参考场 `u_ref`：训练源里**离目标源最近的另一个源**的整场 u（训练时留一，
  绝不用自己；测试时取训练集里最近的）。这是"几次训练"允许使用的射频记忆。
- 可见性：两个体素尺度各 3 通道，见 `corf/visibility.py` 与 `tools/gen_vis_features.py`。
"""
from __future__ import annotations

import json
import os

import numpy as np

VIS_KEYS = ("seen05", "first_hit05", "occ_frac05",
            "seen10", "first_hit10", "occ_frac10")


def load_vis(data_root: str, keys=VIS_KEYS, tx_ids=None) -> dict[int, np.ndarray]:
    """读 `vis/<tx>.npz`，返回 {tx: [P,K] float32}。缺文件直接报错（不要静默补零）。"""
    vdir = os.path.join(data_root, "vis")
    meta_p = os.path.join(vdir, "meta.json")
    if not os.path.isfile(meta_p):
        raise FileNotFoundError(f"缺可见性特征：{meta_p}（先跑 tools/gen_vis_features.py）")
    if tx_ids is None:
        tx_ids = range(len(json.load(
            open(os.path.join(data_root, "meta.json"), encoding="utf-8"))["tx_world"]))
    out = {}
    for j in tx_ids:
        p = os.path.join(vdir, f"{int(j):05d}.npz")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺可见性特征：{p}")
        z = np.load(p)
        out[int(j)] = np.stack([z[k] for k in keys], 1).astype(np.float32)
    return out


def nearest_other(tx: np.ndarray, ids) -> dict[int, int]:
    """ids 内每个源 → ids 内最近的**另一个**源。用于训练时留一参考场。"""
    ids = [int(i) for i in ids]
    if len(ids) < 2:
        # 只有一个训练源就没有"另一个"可借；绝不能退化成自引用（那是把答案喂给头）
        raise ValueError("参考场需要至少 2 个训练源（n_tx>=2）")
    t = tx[ids]
    D = np.linalg.norm(t[:, None] - t[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return {ids[i]: ids[int(np.argmin(D[i]))] for i in range(len(ids))}


def nearest_in(tx: np.ndarray, j: int, ids) -> int:
    """任意源 j → ids 中最近的一个（测试源用）。"""
    ids = [int(i) for i in ids]
    d = np.linalg.norm(tx[ids] - tx[int(j)], axis=1)
    return ids[int(np.argmin(d))]


def build_extra(tx_j: int, ref_id: int, vis: dict[int, np.ndarray],
                u_all, P: int, device="cuda"):
    """拼一段 [P, 1+6]：参考场 + 可见性。`u_all` 是 [T,P] 的 CPU 张量。"""
    import torch
    vis_j = torch.from_numpy(vis[int(tx_j)]).to(device)
    ref = u_all[int(ref_id)].to(device).reshape(P, 1)
    return torch.cat([ref, vis_j], dim=1)
