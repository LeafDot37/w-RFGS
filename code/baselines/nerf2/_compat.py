# -*- coding: utf-8 -*-
"""_compat.py —— NeRF² vendor 层的垫片

本文件是**唯一**允许改变上游行为的地方。上游 `model.py` / `renderer.py` 保持逐字
（sha256 见 `UPSTREAM.md`），需要扩展的一律子类化。理由：基线最怕「你改了它」，
而子类化让「你改过上游没有」永远能一句话回答 —— 没有。

三件事
======
1. `assert_single_visible_device()` —— 设备不变量。**训练前必调**。
2. `RendererSpectrumN` —— 把 `Renderer_spectrum` 从「1 个标量通道」推广到 N 个通道。
   N=1 时与上游**逐位一致**（G5b 卡这条）。
3. `verify_upstream_hashes()` —— 上游文件指纹校验（G7 用）。

为什么需要第 2 件
=================
上游 `Renderer_spectrum.raw2outputs` 的结构是：

    receive = |∑_n  s_a · exp(1j·s_p) · att_i · phase_i · path_loss|

其中 `att_i`（透射率）与 `phase_i`（累积相位）是**路径的属性**，与「用什么模态
去表达这条射线的接收量」无关；只有 `s_a, s_p`（每采样点的幅度与相位）才随通道变。
所以推广到 C 通道**只需复制 `s_a, s_p`，透射率与相位共享一份**：

    receive_c = |∑_n  s_a_c · exp(1j·s_p_c) · att_i · phase_i · path_loss|

这不是「凑出来的 3 通道」——它正是上游公式在没有理由区分通道时的自然形式，
且 C=1 时原样退化回上游。

输出头的通道数由 `model.py` 的构造参数决定，**`model.py` 一行都不用改**：
    sig_output_dims = 2·C      （(幅度, 相位) 交替：s_a_0, s_p_0, s_a_1, s_p_1, ...）
    attn_output_dims = 2       （衰减两通道，两臂共享）
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .renderer import Renderer_spectrum

_HERE = Path(__file__).resolve().parent

# 上游三件套指纹（与 UPSTREAM.md 同源）
UPSTREAM_SHA256 = {
    "model.py": "c63fa4246d6255035d03ebc1dbd7b5a1d40391b51e98805ed3d9969bc3546ad1",
    "renderer.py": "4f5e5b1ab617de27918e8d2bc540182a110d37eb6ac5e743449c34b3a8535c92",
    "LICENSE": "c308fc9f0bca53656b48b0821a44cb387063a020636e1cff2182aaf83a3e2757",
}


# --------------------------------------------------------------------------
# 1. 设备不变量
# --------------------------------------------------------------------------
def assert_single_visible_device(tag: str = "") -> str:
    """断言「只暴露一张 CUDA 卡」并返回该卡的名字。不满足就**大声崩**。

    为什么这条必须硬断言
    --------------------
    上游 `renderer.py` 有 7 处不带设备号的 `.cuda()`，其中 4 处在
    `Renderer_spectrum.raw2outputs` 里 —— 也就是**每次前向都会走到**。
    不带设备号的 `.cuda()` 恒取 cuda:0。若 `CUDA_VISIBLE_DEVICES` 没设或设成多张，
    cuda:0 就是**物理 0 号卡**，而不是我们被分配的那张。

    这台机器上 0 号卡被别的用户作业满载：后果不是报错就是抢卡，
    而抢卡会把**别人的**作业 OOM 掉 —— 一个不该由我们造成的破坏。

    注意「设晚了」也要能抓到：`CUDA_VISIBLE_DEVICES` 在 `import torch` 之后才设是
    **无效**的。本函数通过 `torch.cuda.device_count()` 反查实际可见数，
    所以设晚了一样的会被这里拦下（那时 count 会是真实卡数）。
    """
    prefix = ("[%s] " % tag) if tag else ""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.strip():
        raise RuntimeError(
            "%sCUDA_VISIBLE_DEVICES 未设置。上游 renderer.py 的 .cuda() 不带设备号，"
            "会落到物理 0 号卡（正被其它用户作业占用）。\n"
            "  必须形如：CUDA_VISIBLE_DEVICES=3 <python> ..." % prefix)

    ids = [s.strip() for s in vis.split(",") if s.strip()]
    if len(ids) != 1:
        raise RuntimeError(
            "%sCUDA_VISIBLE_DEVICES=%r 暴露了 %d 张卡；必须**恰好一张**，"
            "否则 .cuda() 恒取 cuda:0 = 物理 %s 号卡。"
            % (prefix, vis, len(ids), ids[0] if ids else "?"))

    if not torch.cuda.is_available():
        raise RuntimeError("%sCUDA_VISIBLE_DEVICES=%r 但 torch.cuda 不可用。" % (prefix, vis))

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            "%storch 实际可见 %d 张卡（CUDA_VISIBLE_DEVICES=%r）。"
            "多因该变量在 import torch **之后**才设置 —— 那样它是无效的，"
            "必须由驱动脚本在进程启动前导出。" % (prefix, n, vis))

    return torch.cuda.get_device_name(0)


# --------------------------------------------------------------------------
# 2. 上游指纹
# --------------------------------------------------------------------------
def verify_upstream_hashes(verbose: bool = False):
    """返回 (ok, 明细列表)。明细每项为 (文件名, 实际, 期望, 是否一致)。"""
    rows = []
    ok = True
    for name, want in sorted(UPSTREAM_SHA256.items()):
        p = _HERE / name
        if not p.is_file():
            rows.append((name, "<缺失>", want, False))
            ok = False
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        good = (got == want)
        ok &= good
        rows.append((name, got, want, good))
        if verbose:
            print("  %s %-12s %s" % ("✓" if good else "✗", name, got))
    return ok, rows


# --------------------------------------------------------------------------
# 3. N 通道谱渲染器
# --------------------------------------------------------------------------
class RendererSpectrumN(Renderer_spectrum):
    """`Renderer_spectrum` 的 N 通道推广。C=1 时与上游**逐位一致**。

    `raw` 的最后一维布局（长度 2 + 2C）：
        [att_a, att_p, s_a_0, s_p_0, s_a_1, s_p_1, ..., s_a_{C-1}, s_p_{C-1}]

    C=1 时就是上游的 `[att_a, att_p, s_a, s_p]`，逐位等价。

    通道数不存状态，**从 `raw.shape[-1]` 现推**（`C = (D-2)//2`）——
    这样模型换了 `sig_output_dims` 而渲染器忘了同步时不会静默按错的 C 去切。
    若 `D-2` 是奇数或为 0，直接报错。
    """

    @staticmethod
    def _n_channels(raw):
        d = raw.shape[-1]
        if d < 4 or (d - 2) % 2:
            raise ValueError(
                "raw 最后一维 = %d，应为 2 + 2C（C ≥ 1）。检查 model 的 "
                "attn_output_dims=2 / sig_output_dims=2C 是否成对。" % d)
        return (d - 2) // 2

    def raw2outputs(self, raw, r_vals, rays_d):
        """与上游逐行同构，只把 `s_a, s_p` 拆成 C 份。

        逐位一致性（C=1）：乘法的结合顺序、归约轴、归约顺序都与上游相同，
        广播只多出一个长度 1 的尾维，不改变任何一次浮点运算的结果。
        """
        C = self._n_channels(raw)

        raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists)
        raw2phase = lambda raw, dists: raw * dists

        dists = r_vals[..., 1:] - r_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # 衰减：路径的属性，C 个通道共享（上游原式）
        att_a, att_p = raw[..., 0], raw[..., 1]
        att_p = torch.sigmoid(att_p) * np.pi * 2
        att_a = abs(F.leaky_relu(att_a))

        # 信号：每通道一份 (幅度, 相位)。C=1 时 sig[...,0]/[...,1] 即上游的 s_a/s_p
        # 形状：sig = [..., N, C, 2]  ⇒  s_a, s_p = [..., N, C]
        sig = raw[..., 2:].reshape(raw.shape[:-1] + (C, 2))
        s_a, s_p = sig[..., 0], sig[..., 1]
        s_p = torch.sigmoid(s_p) * np.pi * 2
        s_a = abs(F.leaky_relu(s_a))

        alpha = raw2alpha(att_a, dists)
        phase = raw2phase(att_p, dists)

        ones = torch.ones((alpha.shape[0], 1)).cuda()
        att_i = torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], -1), -1)[:, :-1]
        path = torch.cat(
            [r_vals[..., 1:], torch.Tensor([1e10]).cuda().expand(r_vals[..., :1].shape)], -1)
        path_loss = 0.025 / path
        phase_i = torch.cumsum(torch.cat([ones, phase], -1), -1)[:, :-1]
        phase_i = torch.exp(1j * phase_i)

        # 逐通道求和：s_a/s_p 已是 [..., N, C]，把 [..., N] 的三个量补一维广播到 C，
        # 再对 **N 轴**（-2）归约，保留通道轴。
        # C=1 时 s_a 形如 [..., N, 1]，比上游多一个长度 1 的尾维 ——
        # 逐元素乘法与归约顺序一字不变，故逐位一致。
        term = (s_a
                * torch.exp(1j * s_p)
                * att_i.unsqueeze(-1)
                * phase_i.unsqueeze(-1)
                * path_loss.unsqueeze(-1))
        receive_signal = torch.sum(term, -2)          # [..., C]
        receive_signal = abs(receive_signal)

        if C == 1:
            receive_signal = receive_signal.squeeze(-1)   # 与上游同形 [B]
        return receive_signal

    def render_ss(self, tx, rays_o, rays_d):
        """与上游同名同形，只是多通道时返回 [B, C] 而非 [B]。"""
        pts, t_vals = self.sample_points(rays_o, rays_d)
        view = rays_d[:, None].expand(pts.shape)
        tx = tx[:, None].expand(pts.shape)
        raw = self.network_fn(pts, view, tx)
        return self.raw2outputs(raw, t_vals, rays_d)


# --------------------------------------------------------------------------
# 4. 构造器
# --------------------------------------------------------------------------
def build_network(D=8, W=256, skips=(4,), n_channels=1,
                  multires=None, is_embeded=None, device="cuda"):
    """按上游默认值造 `NeRF2`；只把 `sig_output_dims` 换成 2·C。

    **不改 `model.py`** —— 上游构造签名已经把两个输出头的维度参数化了
    (`attn_output_dims` / `sig_output_dims`)，见 UPSTREAM.md §3.5。
    """
    from .model import NeRF2

    if multires is None:
        multires = {"pts": 10, "view": 10, "tx": 10}
    if is_embeded is None:
        # 上游 rfid-spectrum.yml 三个都是 True（注意 model.py 的**默认参数**写的是
        # tx: False，那是另一回事，别照默认值来）
        is_embeded = {"pts": True, "view": True, "tx": True}

    net = NeRF2(
        D=D, W=W, skips=list(skips),
        input_dims={"pts": 3, "view": 3, "tx": 3},
        multires=multires,
        is_embeded=is_embeded,
        attn_output_dims=2,
        sig_output_dims=2 * n_channels,
    ).to(device)
    return net


def build_renderer(net, near, far, n_samples):
    """造 `RendererSpectrumN`。near/far/n_samples 由调用方按**本数据集实算**传入。"""
    return RendererSpectrumN(networks_fn=net, near=near, far=far, n_samples=n_samples)


# --------------------------------------------------------------------------
# 5. 分块前向
# --------------------------------------------------------------------------
def render_rays_chunked(renderer, rays_o, rays_d, tx, chunk=8192):
    """把 `renderer.render_ss` 按射线分块调用再拼回，**不改变数值**。

    为什么必须分块（显存账，别按「一张图才 18 万个数」估）
    ----------------------------------------------------
    整图 300×200 = 60000 条射线 × 64 采样点 = 384 万个点。衰减网每层产生
    (3.84e6, 256) 的激活 = **3.93 GB**，8 层就是 31 GB —— 单这一项就超了 24 GB 卡。
    输出图本身只有 300×200×3 个 float（720 KB），**完全不是瓶颈**；
    瓶颈是中间激活，而它随射线数线性增长。

    分块后峰值由 chunk 决定：8192 射线 × 64 = 52.4 万点 ⇒ 每层 537 MB。
    分块**只影响显存峰值，不影响结果**：`sample_points` 与网络都是逐射线独立的
    （`torch.linspace` 的 t 采样与射线无关；网络无跨样本运算），
    所以分块拼接与整批前向在浮点上是同一次运算的同一顺序，逐位相同。

    调用方负责 `torch.no_grad()` / `inference_mode` —— 本函数不加，
    因为 `torch.no_grad()` 是作用**全局**的上下文管理器，在这里加会让
    「误在训练路径上调用它」变成静默断梯度。
    """
    n = int(rays_o.shape[0])
    if n == 0:
        raise ValueError("射线数为 0")
    if chunk < 1:
        raise ValueError("chunk 必须 ≥ 1，收到 %d" % chunk)
    outs = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        outs.append(renderer.render_ss(tx[s:e], rays_o[s:e], rays_d[s:e]))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)
