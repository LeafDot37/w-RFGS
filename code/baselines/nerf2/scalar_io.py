# -*- coding: utf-8 -*-
"""scalar_io.py —— jet 色标的**严格反解**，以及标量↔RGB 的往返

正向链的唯一真源是 `splat.color_scalar`（`make_jet_lut` / `apply_jet`），本模块
**不重写 LUT**，只写它的逆。任何关于「jet 长什么样」的疑问都以那边为准。

为什么需要一个新反解，而不直接用 `tools/lut_project_probe.py`
============================================================
`project_curve()` 做的是同一件事的**近似版**，但它有两个刻意的取舍，正好都不适合这里：

1. **它丢掉了参数 u**。它返回的是「投影后的颜色」，因为它的用途是诊断
   （把渲染图吸附回调色板再看指标）。这里要的是 u 本身 —— 色标坐标才是 N2 头
   的输出语义，也是 `--target scalar_inverted` 的靶子。
2. **它先找最近格点、再只看相邻两段**。这是个启发式：对「想要个颜色」够用，
   但对「想要准确的 u」在某些像素上会落到局部最优而非全局最近段。
   本模块改为**遍历全部 255 段取全局最近**。代价可忽略（255 段/像素），
   换来的是精确 —— 而且不需要 KD 树。

数学上二者是同一条曲线
======================
`apply_jet` 的实现是：`x = clamp(u,0,1)·(n-1)`；`i0 = floor(x)`；`frac = x-i0`；
`col = lut[i0] + frac·(lut[i0+1] − lut[i0])`。
所以映射出去的那条曲线**就是**由 `lut` 的 256 个锚点连成的折线。
反解 = 把颜色投影到这条折线上，再读回参数。

    u = (i0 + t) / (n − 1)        t ∈ [0,1] 为段内参数

注意：曲线不自交（jet 是单一色相扫掠），但**存在大量不在曲线上的颜色**
（MLP 渲出的像素实测约 41% 漂在调色板外）。对本模块而言那些像素照样有投影，
只是残差大 —— 残差本身就是有用的诊断量，见 `roundtrip_report()` 的 `off` 项。
"""
import numpy as np
import torch

from splat.color_scalar import apply_jet, make_jet_lut


def build_curve(n: int = 256, device="cpu") -> torch.Tensor:
    """色标折线的锚点 (n,3) ∈ [0,1]。就是 `make_jet_lut` 本身。"""
    return make_jet_lut(n).to(device)


@torch.no_grad()
def invert_jet(rgb01: torch.Tensor, lut: torch.Tensor = None,
               chunk: int = 1 << 16) -> torch.Tensor:
    """把颜色投到 jet 曲线上，返回**色标坐标** u ∈ [0,1]。

    参数
    ----
    rgb01 : (..., 3)，值域 [0,1]。形状任意，返回同形状去掉末维（即 (...)）。
    lut   : (n,3) 锚点；None 则 `make_jet_lut(256)`。
    chunk : 一次处理的像素数（显存控制）。默认 65536 ⇒ 65536×255×3×4 B ≈ 200 MB。

    实现
    ----
    对全部 n−1 段做全局最近点搜索（不是启发式取相邻两段）：

        段 i: a = lut[i], b = lut[i+1], ab = b−a
        t  = clip( ⟨g−a, ab⟩ / ‖ab‖², 0, 1 )
        p  = a + t·ab
        d  = ‖g−p‖²

    取 d 最小的段，u = (i + t)/(n−1)。

    退化段（‖ab‖²=0，jet 里不会出现，但别让除零静默传播）按 t=0 处理。
    """
    if lut is None:
        lut = make_jet_lut(256).to(rgb01.device)
    n = lut.shape[0]
    if n < 2:
        raise ValueError("lut 至少要有 2 个锚点，收到 %d" % n)

    a = lut[:-1]                                   # (S,3)
    ab = lut[1:] - lut[:-1]                        # (S,3)
    denom = (ab * ab).sum(-1)                      # (S,)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    S = a.shape[0]

    flat = rgb01.reshape(-1, 3)
    out = torch.empty(flat.shape[0], dtype=lut.dtype, device=lut.device)
    for s in range(0, flat.shape[0], chunk):
        g = flat[s:s + chunk]                      # (m,3)
        # (m,1,3) − (1,S,3) → (m,S,3)；S=255 时 m×255×3 尚可控
        diff = g[:, None, :] - a[None, :, :]
        t = (diff * ab[None, :, :]).sum(-1) / denom[None, :]
        t = t.clamp_(0.0, 1.0)
        p = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d = ((g[:, None, :] - p) ** 2).sum(-1)     # (m,S)
        idx = d.argmin(-1)                         # (m,)
        ti = t.gather(-1, idx[:, None]).squeeze(-1)
        out[s:s + chunk] = (idx.to(lut.dtype) + ti) / (n - 1)

    return out.reshape(rgb01.shape[:-1])


def forward_jet(u01: torch.Tensor, lut: torch.Tensor = None) -> torch.Tensor:
    """u (H,W) 或 (1,H,W) → RGB (3,H,W)。直接走 `splat.color_scalar.apply_jet`。"""
    if lut is None:
        lut = make_jet_lut(256).to(u01.device)
    return apply_jet(u01, lut)


def u_to_rgb(u_hw: torch.Tensor, lut: torch.Tensor = None) -> torch.Tensor:
    """N2 渲染出口用：u (H,W) 或 (1,H,W) → (3,H,W) ∈ [0,1]。"""
    return forward_jet(u_hw, lut)


def u_flat_to_rgb(u_flat: torch.Tensor, lut: torch.Tensor = None) -> torch.Tensor:
    """(P,) 的标量 → (P,3) 的 RGB。给**没有图像结构**的往返统计用。

    **为什么需要它**：`apply_jet` 的返回是 **(3,H,W)，没有 batch 维**。统计代码里
    常见的写法 `apply_jet(u.unsqueeze(0), lut).squeeze(0)` 看着像「去掉 batch」，
    但 dim0 是 **3** ⇒ `squeeze(0)` 是 **no-op**，紧随其后的 `.permute(1, 0)` 直接抛
    `RuntimeError: permute(...): input.dim() = 3 is not equal to len(dims) = 2`。
    本仓在 `roundtrip_report` / `ceiling_metrics` / G4b 三处同时栽了这个（2026-09-16
    首次跑到 G4 时才暴露 —— 之前只验过 `--stage local` 的 G1/G7，G4 从没跑过）。

    本函数把这条转换收成唯一出口，杜绝再写第二次。
    """
    if u_flat.dim() != 1:
        raise ValueError("u_flat_to_rgb 期望 (P,)，收到 %s" % (tuple(u_flat.shape),))
    return apply_jet(u_flat.reshape(1, -1), lut)[:, 0, :].transpose(0, 1)   # (P,3)


def rgb_to_u(rgb_chw: torch.Tensor, lut: torch.Tensor = None) -> torch.Tensor:
    """反解入口：RGB (3,H,W) 或 (H,W,3) → u (1,H,W)。"""
    if rgb_chw.dim() != 3:
        raise ValueError("rgb_to_u 期望 (3,H,W)，收到 %s" % (tuple(rgb_chw.shape),))
    if rgb_chw.shape[0] != 3:
        raise ValueError("rgb_to_u 期望通道在前且为 3，收到 %s" % (tuple(rgb_chw.shape),))
    u = invert_jet(rgb_chw.permute(1, 2, 0).contiguous(), lut)
    return u.unsqueeze(0)


# --------------------------------------------------------------------------
# G4：往返误差与 (b) 版分数天花板
# --------------------------------------------------------------------------
@torch.no_grad()
def roundtrip_report(u8_hwc: np.ndarray, lut: torch.Tensor = None,
                     max_images: int = 0) -> dict:
    """对 uint8 RGB 图（(N,H,W,3) 或 (H,W,3)）报 jet 反解往返误差。

    返回各项**以 0–255 为单位**（好和 G4 的阈值直接比）：

        mean / p99 / max : ‖jet(invert(g)) − g‖₂ 的逐像素统计
        off_frac         : 残差 > 2 的像素占比（≈ 漂在曲线外的比例）
        n_pix            : 参与统计的像素数

    `off_frac` 是关键诊断：GT 实测 100% 落在调色板上（残差≈0），
    而模型渲出的图会有可观比例在曲线外 —— 这个数直接量出「输出有没有被调色板约束住」。
    """
    if lut is None:
        lut = make_jet_lut(256)
    dev = lut.device
    arr = u8_hwc
    if arr.ndim == 3:
        arr = arr[None]
    if arr.shape[-1] != 3:
        raise ValueError("期望末维为 3 的 RGB，收到 %s" % (arr.shape,))
    if max_images and arr.shape[0] > max_images:
        step = max(1, arr.shape[0] // max_images)
        arr = arr[::step][:max_images]

    g = torch.from_numpy(arr.reshape(-1, 3)).to(dev).float().div_(255.0)
    u = invert_jet(g, lut)
    back = u_flat_to_rgb(u.reshape(-1), lut)                          # (P,3)
    err = (back - g).mul(255.0).pow(2).sum(-1).sqrt()

    return {
        "mean": float(err.mean()),
        "p99": float(err.quantile(0.99)),
        "max": float(err.max()),
        "off_frac": float((err > 2.0).float().mean()),
        "n_pix": int(err.numel()),
    }


@torch.no_grad()
def ceiling_metrics(u8_hwc: np.ndarray, lut: torch.Tensor = None,
                    device: str = "cuda") -> dict:
    """(b) 版的**分数天花板**：`metrics(jet(invert(g)), g)`。

    意义：N2 的标量头是先出 u、再经 jet 查表成 RGB。只要经过这一遭，
    分数就不可能超过「把 GT 自己反解再重渲」的结果 —— 因为那已经是这条
    参数化路径能做到的**上确界**（靶子本身完美落在路径上）。

    所以报 N2 的数时**必须同时报这个天花板**，否则读者无法知道
    「离满分的差距里，有多少是参数化本身的损失、有多少是模型没学好」。
    """
    from splat.utils.image_utils import psnr
    from splat.utils.loss_utils import ssim

    if lut is None:
        lut = make_jet_lut(256)
    arr = u8_hwc if u8_hwc.ndim == 4 else u8_hwc[None]
    g = torch.from_numpy(arr.reshape(-1, 3)).to(lut.device).float().div_(255.0)
    u = invert_jet(g, lut)
    back = u_flat_to_rgb(u.reshape(-1), lut).transpose(0, 1)   # (3,P)


    H, W = arr.shape[1], arr.shape[2]
    P = H * W
    ps, ss = [], []
    for i in range(arr.shape[0]):
        a = back[:, i * P:(i + 1) * P].reshape(3, H, W).unsqueeze(0).to(device)
        b = g[i * P:(i + 1) * P].T.reshape(1, 3, H, W).to(device)
        ps.append(psnr(a, b).mean().item())
        ss.append(ssim(a, b).mean().item())
    return {"PSNR": float(np.mean(ps)), "SSIM": float(np.mean(ss)), "n": len(ps)}
