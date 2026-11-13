# -*- coding: utf-8 -*-
"""
splat/color_scalar.py —— RRF 阶段二「标量头 + 固定 jet 查表」颜色表达（形态 A' / A×）

设计动机（为什么不是又一个 RGB 头）
--------------------------------------------------------------------------------
原版 RRF 与我们的 MLP 头都存**逐高斯 RGB**、由光栅化在 **RGB 域混合**：
    image = Σ wᵢ·cᵢ + (1−Σ wᵢ)·bg
而实测（见 docs/RF前向模型规格.md §5、.diag/palette_*）：
  - CBF/MVDR 的 GT 像素 **100% 落在 matplotlib jet 调色板上**，色↔标量严格双射；
  - jet **不是仿射映射** ⇒ 调色板色的凸组合**不在调色板上**。
两条合起来：只要 Σ wᵢ < 1（边缘、半透明区、背景），原版**必然**渲染出调色板外的
颜色，而 GT 从不偏离。实测作者自己的渲染：MVDR 93% 像素离板、CBF 65% 离板；
且 MVDR 上"逐像素吸附回调色板"后 LPIPS 反而变好（65.5% 视角），CBF 则不变差也不变好。

本模块改混合的**空间**：头输出**标量** s ∈ (0,1)，光栅化混合标量得
    u = Σ wᵢ·sᵢ + (1−Σ wᵢ)·u_bg        （u_bg = 0 ⇒ u = Σ wᵢ·sᵢ）
再对该**混合后的标量**查表：
    image = jet(u)
因 jet 是固定映射，输出**按构造永远在调色板上** ⇒ 与原版是**不同的函数类**，
不是同一个类的不同参数化。注意这**不等于**原版做法：原版是 Σ wᵢ·LUT(sᵢ)
（先查表再混合），非线性映射与线性混合不可交换。

实现边界（不动引擎内核）
--------------------------------------------------------------------------------
- 本模块 **不 import、不修改** gaussian_renderer / gaussian_model / train.py。
- 查表发生在 `render()` **之外**（入口变体里对返回的渲染图做后处理），
  故 gaussian_renderer 一行不改。
- 标量经 override_color 进光栅化，**必须复制成 (P,3)**：实测 (P,1) 张量传入
  **不报错但静默产出垃圾**（逐通道尺寸不匹配被广播/错位），见 compute_scalar 的注释。
- 背景色：render 用 bg=[0,0,0]（RRF 数据 white_background=False）。标量域里
  未覆盖像素得 u=0 → jet(0) = 深蓝，**仍在调色板上**；而 RGB 域里同样的
  bg=0 给出纯黑 **（调色板外）**。实测 GT 纯黑像素占比 0.00% —— GT 背景本就是
  调色板的暗端，故 A' 在这里自动对上。

位置编码与 LR 调度
--------------------------------------------------------------------------------
- PE 三个开关（pe_xyz_L / pe_dir_L / use_dist）**完全复用 color_mlp 的实现**
  （直接 import 其 fourier_feat / make_dir / _dist / set_scene_norm），保证与 MLP 臂
  是**单变量对比**：唯一差异 = 输出标量而非 RGB、以及查表位置。
- 输入布局同 color_mlp：cat[ embed(D), γ_xyz(6L_xyz|缺), dir(3), γ_dir(6L_dir|缺), r(1|缺) ]。

形态 A×：多频段标量分支（n_bands=2, band_mix="mult"）
--------------------------------------------------------------------------------
动机：A' 在 CBF 上渲染通道 std 37.05 明显低于 GT 51.08（过平滑），MVDR 则几乎持平
（52.96 vs 54.12）。**先验推断**是"单标量头的容量不够"；本形态就是去兑现或证伪它。

**为什么必须是乘法、且只做 2 个 band**（两条都有硬理由，不是取舍）：
1. 渲染器对颜色**严格线性且逐通道独立**（`diff-gaussian-rasterization-modified/
   cuda_rasterizer/forward.cu:355-373`：alpha/T 在通道循环外算，`C[ch] += f[id*C+ch]·α·T`）
   ⇒ `ū_k(p) = Σᵢ wᵢ(p)·s_{k,i}`，权重 `wᵢ` 与颜色无关。于是**任何"各 band 混合后线性求和
   再查表"都等价于把头输出层加宽**，不是新机制。**只有非线性合成能逃出该塌缩。**
2. 统一输入 + 对称初值（`W_k` 全 0、`b_k` 全 logit(init_u)）下 `s_1 ≡ s_2`；初值时
   `∂u/∂ū_k = ū_0·λ_k` 对各 band 相同 ⇒ Adam 逐参数确定性 ⇒ `W_1 ≡ W_2` **永远**。
   ⇒ 第 3 个 band 是纯规范冗余，**n_bands 定为 2**。（要真做 3 个必须让各 band 输入不同，
   那会破坏与 A' 的嵌套性，本步不做。）

数据流（一次 render 出两个场，零额外开销）：
    s = head(...) → (P,2)  ∈ (0,1)²
    override_color = cat[s_0, s_1, s_1]          # (P,3)，借光栅化器逐通道线性
    mixed = render(...)["render"]                # (3,H,W)，通道 0/1 = ū_0 基底场 / ū_1 细节场
    u = ū_0 · ( 1 + λ·(ū_1 − init_u) )           # 标量域非线性合成，仍是单一标量图
    image = apply_jet(u, lut)                    # 只查一次表 ⇒ 输出仍严格在 jet 曲线上

性质：
- **λ=0 时精确退化成 A'**（`u = ū_0`）⇒ A× **严格包含** A'，对照是**嵌套**的而非并列的。
- **不塌缩**：两个混合场之积 ≠ 任何一个单一混合场。
- 成本与 A' 相同；头只多 1 个输出行 + 1 个标量 λ（+129 参数）。
- 直击症状：乘性细节项 = 可学的**局部对比度调制**。

SH 输入与空间上下文块（2026-09-13 加，默认关，旧档逐字不变）
--------------------------------------------------------------------------------
两个可选输入块**一律追加在输入的最后**（`r` 之后）：

    cat[ embed, γ(xyz)?, dir, γ(dir)?, r?,  sh_block?,  ctx_block? ]

**只有追加在最后**，旧 config 的 `in_dim` 才逐字节不变、旧 sidecar 才照样严格 load。

1. `use_sh ∈ {none, dc, eval}` —— 把**物理阶段 SH**（在 blender_visual_dataset 上训出来的
   RGB 外观）喂进来。取值逐字照抄引擎渲染器（`gaussian_renderer/__init__.py:74-78`）：

       sh_block = clamp_min(eval_sh(deg, get_features.transpose(1,2), dir) + 0.5, 0)

   `dc` 即 `deg=0`（只取基色，视角无关）。**不再减 0.5** —— 减了只会引入一个
   relu 之后不可被下游 bias 吸收的常数偏移。
   **degree 以 sidecar cfg 为唯一真源**（`head.sh_degree`）：`load_ply`
   （gaussian_model.py:256）会把 `active_sh_degree` 强设成 `max_sh_degree`，
   而训练路径是从 ckpt 第 0 位恢复的 ⇒ 不钉死就会出现训练/渲染 degree 静默不一致。

2. `ctx_knn` K > 0 —— 在**固定 kNN 图**上做单层单头注意力，聚合邻居的 `(embed, SH)`
   得到空间上下文块。图由几何决定，而 RRF 阶段几何冻结 + `--densify_until_iter 0`
   ⇒ 数量与顺序恒定 ⇒ 图**只建一次**，落 `<model_path>/knn_k<K>.npz`，
   文件名与建树所用 xyz 的 sha1 记进 sidecar cfg（load 时校验，见 `check_knn_xyz`）。

**两个新块在第 0 层的权重列零初始化。** 这不是为了保第 0 步 —— 第 0 步**本来就**相同
（`out.weight` 已零初始化 ⇒ `W_out=0` ⇒ 首步 `∂L/∂net = W_outᵀ·upstream = 0`）。
它保的是一条更强的性质：**给定 A2 的权重，新通路的贡献恰好是 0**
⇒ A4/A5 在初始化处是 A2 的**严格嵌套函数类**，新通路只有自己挣出来才会被用上。

⚠ **反方论据（必须与结论一并报出）**：几何冻结 ⇒ 任何逐点特征，**包括邻居聚合出来的**，
都是该点的定值；而头里已有自由可训的 `embed_p`（P×32）能把它吸收掉
⇒ 这两个块带来的是**归纳偏置 / 优化路径，不是新信息**。
零结果 = 「优化器选择不开这条路」，是一个干净的结论。
**v1 的上下文块是视角无关的**（token 只含 embed 与 SH，`dir` 不进注意力）——
反方论据表明加视角并不会换来非冗余性，先做最便宜的形式。

用法（由入口 train_rrf_scalar.py / render_rrf_scalar.py 调用）
--------------------------------------------------------------------------------
    head = ScalarColorHead(embed_dim=32, hidden=128, depth=2,
                           pe_xyz_L=5, pe_dir_L=3, use_dist=True,
                           n_bands=2, band_mix="mult").cuda()
    s = compute_scalar(head, embed, xyz, cam_center, spatial_lr_scale)  # (P,K) ∈ (0,1)
    mixed = render(..., override_color=to_override_color(s))["render"]  # (3,H,W)
    image = apply_jet(combine_bands(mixed, head), lut)                  # → 调色板上的 RGB
**打包与合成必须走 to_override_color / combine_bands 两个助手**：光栅化器硬要求
`NUM_CHANNELS == 3`（rasterizer_impl.cu:242-245），K∉{1,2,3} 的张量形状不被校验，
会静默读越界或 backward 崩；而合成方式写错（如对 A' 臂套用 mult）会静默污染评测。
"""
import hashlib
import os

import numpy as np
import torch
from torch import nn

# 与 color_mlp 共用同一套 PE/方向/距离语义（单变量对比的前提）
from splat.color_mlp import fourier_feat, make_dir, _dist, set_scene_norm  # noqa: F401
from splat.utils.sh_utils import eval_sh

# 上下文块沿 N 分块的长度。**训练与渲染必须用同一个值** —— matmul 的 kernel 选择随
# batch 变，块长不等会破坏「渲染路径与训练路径逐位相同」这条既有纪律。
# 它随 sidecar 一起存档并在 load 时断言（见 load_scalar_state），
# 改了它 ⇒ 旧档拒绝加载，而不是安静地换一种数值。
_CTX_CHUNK = 131072

# ── matplotlib `jet` 的分段控制点（_cm.py 的 _jet_data，连续色标内部点 y0==y1）──
# 纯 numpy/torch 复现，**不依赖 matplotlib**（训练环境无需装它）。
# 实测与 matplotlib jet 的解析输出最大逐通道差 4.4e-16。
_JET_SEG = {
    "red":   [(0.00, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.00, 0.5)],
    "green": [(0.00, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0),
              (0.91, 0.0), (1.00, 0.0)],
    "blue":  [(0.00, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.00, 0.0)],
}


def make_jet_lut(n: int = 256) -> torch.Tensor:
    """构造 jet 查表 (n,3) ∈ [0,1]。n=256 时逐项等于 matplotlib 的 256 级 LUT。"""
    u = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    cols = []
    for key in ("red", "green", "blue"):
        pts = _JET_SEG[key]
        xs = torch.tensor([p[0] for p in pts], dtype=torch.float64)
        ys = torch.tensor([p[1] for p in pts], dtype=torch.float64)
        idx = torch.searchsorted(xs, u, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        w = (u - x0) / (x1 - x0)
        cols.append((y0 + w * (y1 - y0)).clamp(0.0, 1.0))
    return torch.stack(cols, dim=-1).to(torch.float32)      # (n,3)


def apply_jet(u01: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """把（已在标量域混合好、**已合成为单一标量**的）标量图查表成 RGB。

    u01: (H,W) 或 (1,H,W) → 返回 (3,H,W) ∈ [0,1]。
    表内线性插值：对 u 可导（梯度经插值权重回传），索引本身不可导（正确，查表语义）。
    越界一律 clamp 到 [0,1] —— jet 定义域外无意义，且 clamp 的零梯度段正是饱和区。

    **硬护栏（2026-09-12 加）**：旧实现接受 (3,H,W) 并**静默只用通道 0** —— 多频段形态
    下这会把 band 1 悄悄丢掉而**不报任何错**。现在非单通道直接 raise：多带必须先经
    `combine_bands()` 合成再查表。
    """
    if u01.dim() == 2:
        u01 = u01.unsqueeze(0)
    if u01.dim() != 3:
        raise ValueError(f"apply_jet 期望 (H,W) 或 (1,H,W)，收到 {tuple(u01.shape)}")
    if u01.shape[0] != 1:
        raise ValueError(
            f"apply_jet 只接受单通道标量图，收到 {u01.shape[0]} 通道 "
            f"{tuple(u01.shape)}。多频段必须先 combine_bands() 合成成单一标量再查表；"
            "直接传多通道会静默丢弃通道 1+（旧版行为，已封死）。")
    u = u01
    n = lut.shape[0]
    x = torch.clamp(u, 0.0, 1.0) * (n - 1)
    i0 = x.detach().floor().long().clamp(0, n - 2)
    frac = x - i0.to(x.dtype)
    c0 = lut[i0]                                    # (1,H,W,3)
    c1 = lut[i0 + 1]
    col = c0 + frac.unsqueeze(-1) * (c1 - c0)
    return col[0].permute(2, 0, 1).contiguous()     # (3,H,W)


def to_override_color(s: torch.Tensor) -> torch.Tensor:
    """把头的 (P,K) 标量输出**打包**成光栅化器能吃的 (P,3)。

    为什么要打包而不是 repeat：光栅化器按 3 通道**逐通道独立**地线性混合
    （`C[ch] += f[id*C+ch]·α·T`，见 forward.cu:355-373），所以一次 render 就能同时
    得到 K 个 band 各自的混合场 —— 前提是每个 band 只占一个通道。
      K=1 → repeat(1,3)：三通道相等，任一通道即 ū_0（与 A' 逐字一致）
      K=2 → cat[s_0, s_1, s_1]：通道 0 = 基底场，通道 1 = 细节场（通道 2 是占位副本）
      K=3 → 原样
    **K∉{1,2,3} 必须报错**：光栅化器 `rasterizer_impl.cu:242-245` 硬要求 NUM_CHANNELS==3，
    形状不被校验 ⇒ 会静默读越界或 backward 崩。
    """
    if s.dim() != 2:
        raise ValueError(f"to_override_color 期望 (P,K)，收到 {tuple(s.shape)}")
    k = s.shape[1]
    if k == 1:
        return s.repeat(1, 3)
    if k == 2:
        return torch.cat([s[:, 0:1], s[:, 1:2], s[:, 1:2]], dim=1)
    if k == 3:
        return s
    raise ValueError(f"band 数 {k} 无法打包进 3 通道（光栅化器硬要求 NUM_CHANNELS==3）")


def combine_bands(mixed: torch.Tensor, head) -> torch.Tensor:
    """把 render 出来的 (3,H,W) **各 band 混合场**按 head.band_mix 合成**单一标量图** (1,H,W)。

    非线性合成发生在**标量域**（查表之前），故输出仍过同一条 jet ⇒ A' 的"输出永远在
    调色板上"这一不变量不变。

      single(K=1)：u = ū_0                                  —— 与 A' 逐字一致
      mult  (K=2)：u = ū_0 · (1 + λ·(ū_1 − init_u))         —— λ=sigmoid(lam_raw) ∈ (0,1)
                    λ 初值 0.5；**λ=0 精确退化成 A'**（嵌套对照）。
                    调制中心取 head.init_u 而非硬编码 0.5，保证 --init_u 语义不变。
      coh   (K=2)：u = sqrt(ū_0² + ū_1²)                    —— **相干双通道**（Re/Im 复场）
                    光栅化对颜色是逐通道线性叠加 ⇒ 通道 0/1 的两张混合场就是
                    逐高斯复场的实/虚部之和（跨高斯相干求和）；逐点先取模等于
                    在逐高斯处丢掉跨点相位。见 `color_scalar_coh.py`。
    """
    if mixed.dim() != 3:
        raise ValueError(f"combine_bands 期望 (3,H,W)，收到 {tuple(mixed.shape)}")
    c = mixed.shape[0]
    mix = getattr(head, "band_mix", "single")
    if mix == "single":
        out = mixed[0:1]
    elif mix == "mult":
        if c < 2:
            raise ValueError(f"band_mix='mult' 需要 >=2 个 band 通道，渲染只给了 {c} 个 ——"
                             "检查 to_override_color 是否与该 head 的 n_bands 匹配")
        lam = torch.sigmoid(head.lam_raw).to(mixed.dtype).reshape(1, 1, 1)
        out = mixed[0:1] * (1.0 + lam * (mixed[1:2] - float(head.init_u)))
    elif mix == "coh":
        if c < 2:
            raise ValueError(f"band_mix='coh' 需要 2 个实/虚通道，渲染只给了 {c} 个 ——"
                             "检查 to_override_color 是否与该 head 的 n_bands 匹配")
        out = torch.sqrt(mixed[0:1].square() + mixed[1:2].square() + 1e-12)
    else:
        raise ValueError(f"未知 band_mix={mix!r}")

    # 头可选的**渲染后**处理。目前只有 `TxFoldColorHead` 定义：
    # 它的输出是未归一化的逐点幅度，必须按可见像素做逐图 min-max 才能对上 GT
    # 的口径（归一化写进头里会把可见区按全点云极值压扁，见该模块头）。
    # 历史头没有这个方法 ⇒ 行为逐字不变。
    fn = getattr(head, "post_render", None)
    return out if fn is None else fn(out)


def _xyz_sha1(xyz_f32: np.ndarray) -> str:
    """xyz 的 float32 字节摘要。把「近邻图与这份点序不符」从静默错位变成硬错误。"""
    return hashlib.sha1(
        np.ascontiguousarray(xyz_f32, dtype=np.float32).tobytes()).hexdigest()


def sh_appearance(sh_coeff: torch.Tensor, dirv: torch.Tensor,
                  degree: int) -> torch.Tensor:
    """物理阶段 SH 在视线 `dirv` 下的外观值 → (P,3) ≥ 0。**逐字照抄引擎渲染器的口径**
    （`gaussian_renderer/__init__.py:74-78`）。

    sh_coeff: (P,3,(degree+1)²) —— 即 `get_features.transpose(1,2)`
              （`eval_sh` 要 `[..., C, coeff]`，见 sh_utils.py:65）
    dirv:     (P,3) **单位**方向 —— 即 make_dir(xyz, cam_center)

    `eval_sh` 返回**未加 0.5** 的原始值（sh_utils.py:74-80），渲染器在 :78 才 `+0.5`
    ⇒ 这里必须同样 `+0.5`，否则喂进头的量与渲染器算的不是同一个。
    """
    return torch.clamp_min(eval_sh(int(degree), sh_coeff, dirv) + 0.5, 0.0)


def build_or_load_knn(model_path: str, xyz: torch.Tensor, k: int):
    """建/读**固定近邻图**，返回 `(knn int64 (P,k+1) 在 xyz 设备上, xyz_sha1, 文件名)`。

    形状契约：**(P, k+1)，第 0 列是自身。** `cKDTree.query(k=k+1)` 给的是按距离升序的
    [自身, 近邻₁ … 近邻_k]，但**并列距离时顺序不唯一**，两个高斯重合时自身还未必落
    第 0 列 ⇒ 建完**强制**把第 0 列写成 `arange(P)`，load 侧再断言一次。

    幂等：文件存在就直接读。几何冻结（`_xyz` requires_grad_(False) + `densify_until_iter 0`
    ⇒ 数量与顺序恒定）⇒ 图恒定，重跑/续跑不必重建。
    落 `<model_path>/knn_k<k>.npz`，内含 knn/xyz_sha1/k 三项 ⇒ **自校验**，
    混用了别的模型的图会当场报错而不是静默换邻居。

    **只由 `training()` 调用**（它拥有 `model_path`）。渲染侧只读，见 `load_scalar_state`。
    """
    from scipy.spatial import cKDTree     # 延迟 import：不用上下文块的环境不该被要求装 scipy
    k = int(k)
    fname = f"knn_k{k}.npz"
    path = os.path.join(model_path, fname)
    pts = np.ascontiguousarray(xyz.detach().to("cpu", torch.float32).numpy(),
                               dtype=np.float32)
    sha = _xyz_sha1(pts)

    if os.path.isfile(path):
        z = np.load(path)
        prev_k, prev_sha = int(z["k"]), str(z["xyz_sha1"])
        if prev_k != k:
            raise ValueError(f"{path} 里记的是 k={prev_k}，本次要 k={k} —— 拒绝复用")
        if prev_sha != sha:
            raise ValueError(
                f"{path} 是用另一份 xyz 建的（记 {prev_sha[:12]}…，实测 {sha[:12]}…）。"
                "几何冻结的前提下这不该发生 ⇒ 拒绝复用，删掉该文件重建。")
        knn = z["knn"].astype(np.int64)
    else:
        tree = cKDTree(pts.astype(np.float64))
        _, idx = tree.query(pts.astype(np.float64), k=k + 1, workers=-1)
        knn = idx.astype(np.int64)
        np.savez(path, knn=knn.astype(np.int32), xyz_sha1=sha, k=k)

    P = pts.shape[0]
    if knn.shape != (P, k + 1):
        raise ValueError(f"近邻图形状 {knn.shape} 与 (P={P}, k+1={k+1}) 不符")
    knn[:, 0] = np.arange(P, dtype=np.int64)      # 强制自身落第 0 列（并列距离/重合点）
    # 返回**计算设备上的 int64 张量**：`knn` 是普通属性，不会被 module.to() 搬走，
    # 所以挂载时就得放对设备（`_context` 里对此有断言）。放 CPU 会让每步 H2D 搬 75 MB。
    return torch.as_tensor(knn).to(device=xyz.device), sha, fname


def _find_model_root(sidecar_path: str) -> str:
    """从 sidecar 路径**上溯**到含 `cfg_args` 的目录 = `<model_path>`。

    上溯而不是数 `..` 的层数：目录布局将来若变，数层数会**静默指错目录**，
    上溯只会失败（找不到 `cfg_args` 就报错）。
    """
    d = os.path.dirname(os.path.abspath(sidecar_path))
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "cfg_args")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"从 {sidecar_path} 上溯 6 层没找到含 cfg_args 的目录 ⇒ 定位不到 model_path，"
        "无法按 cfg.knn_file 找到近邻图。")


def check_knn_xyz(head, xyz: torch.Tensor) -> None:
    """校验 `head.knn` 是用这份 xyz 建的。几何冻结 ⇒ 加载后做**一次**即可。

    走 sha1 而非逐字节比较：ply 是 float32 往返、本身可逆，但行序可能被别的工具
    （如 `bake_scalar_to_ply.py`）改写过 —— 那种情况下邻居会**静默错位**。
    """
    if getattr(head, "knn", None) is None:
        return
    if getattr(head, "xyz_sha1", None) is None:
        raise ValueError("sidecar 缺 xyz_sha1，无法校验近邻图归属（该档是旧版写的？）")
    got = _xyz_sha1(xyz.detach().to("cpu", torch.float32).numpy())
    if got != head.xyz_sha1:
        raise ValueError(
            f"近邻图与当前 xyz 不符：cfg 记 {head.xyz_sha1[:12]}…，实测 {got[:12]}…。"
            "邻居会静默错位 ⇒ 拒绝加载。常见原因：ply 行序被改写（如 bake_scalar_to_ply），"
            "或 model_path 下混用了别的模型的 knn 文件。")


class ScalarColorHead(nn.Module):
    """共享标量头：cat[embed, γ_xyz?, dir, γ_dir?, r?] → Linear+ReLU×depth → K → sigmoid。

    与 color_mlp.NeuralColorHead 结构同构，唯一差异 = 输出 K 维标量（K=1 即形态 A'）
    而非 3 维 RGB。
    init 末层置零 ⇒ logit=0 ⇒ 每个 band 都是 init_u（缺省 0.5）⇒ jet(0.5) ≈ 亮绿。
    注意这与 RGB 头的"0.5 中性灰"不同：**灰不在调色板上**，而 jet(0.5) 在。

    n_bands / band_mix：
      n_bands=1, band_mix="single"  → 形态 A'（缺省，与旧 sidecar 逐字兼容）
      n_bands=2, band_mix="mult"    → 形态 A×（乘法调制，见模块头部）
    末层对 K 行**同时**置零 + 偏置 logit(init_u)，这恰好也是 λ 冷启动所需的"两带对齐"初值。
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 0, pe_dir_L: int = 0, use_dist: bool = False,
                 init_u: float = 0.5, n_bands: int = 1, band_mix: str = "single",
                 use_sh: str = "none", sh_degree: int = 3,
                 ctx_knn: int = 0, ctx_dim: int = 16):
        super().__init__()
        n_bands = int(n_bands)
        if n_bands not in (1, 2, 3):
            # 光栅化器硬要求 NUM_CHANNELS == 3 ⇒ 最多只能借 3 个通道带 3 个 band
            raise ValueError(f"n_bands 须 ∈ {{1,2,3}}（受光栅化器 NUM_CHANNELS==3 限制），"
                             f"收到 {n_bands}")
        if band_mix not in ("single", "mult", "coh"):
            raise ValueError(f"band_mix 须 ∈ {{'single','mult','coh'}}，收到 {band_mix!r}")
        if band_mix in ("mult", "coh") and n_bands < 2:
            raise ValueError(f"band_mix={band_mix!r} 需要 n_bands >= 2")
        # 新块（默认关）的参数一律**当场校验**：错值写进 sidecar 就成陈档，事后很难查
        if use_sh not in ("none", "dc", "eval"):
            raise ValueError(f"use_sh 须 ∈ {{'none','dc','eval'}}，收到 {use_sh!r}")
        if not (0 <= int(sh_degree) <= 3):
            raise ValueError(f"sh_degree 须 ∈ [0,3]（本仓 SH 是 degree 3），收到 {sh_degree}")
        if int(ctx_knn) < 0:
            raise ValueError(f"ctx_knn 须 >= 0（0 = 关），收到 {ctx_knn}")
        if int(ctx_dim) <= 0:
            raise ValueError(f"ctx_dim 须 > 0，收到 {ctx_dim}")
        self.embed_dim = embed_dim
        self.mlp_hidden = hidden
        self.mlp_depth = depth
        self.pe_xyz_L = int(pe_xyz_L)
        self.pe_dir_L = int(pe_dir_L)
        self.use_dist = bool(use_dist)
        self.init_u = float(init_u)
        self.n_bands = n_bands
        self.band_mix = band_mix
        self.xyz_center = None
        self.xyz_scale = 0.0
        # —— 新块（见模块头「SH 输入与空间上下文块」）——
        self.use_sh = use_sh
        self.sh_degree = int(sh_degree)
        self.ctx_knn = int(ctx_knn)      # = **邻居数**（不含自身）；0 = 关。不再另设 ctx_k
        self.ctx_dim = int(ctx_dim)
        # 近邻图走**普通属性**，绝不 register_buffer：进 state_dict() 就会让旧 sidecar
        # 报 unexpected key 而读不进来（本文件 :258-261 已记过这类事故）。
        self.knn = None
        self.knn_file = None
        self.xyz_sha1 = None

        in_dim = embed_dim
        if self.pe_xyz_L > 0:
            in_dim += 6 * self.pe_xyz_L
        in_dim += 3
        if self.pe_dir_L > 0:
            in_dim += 6 * self.pe_dir_L
        if self.use_dist:
            in_dim += 1

        # 两个新块**一律追加在最后**，顺序 [sh_block, ctx_block]。
        # 只有追加在最后，旧 config 的 in_dim 才逐字节不变、旧 sidecar 才照旧严格 load。
        self.sh_dims = 3 if self.use_sh != "none" else 0
        self.ctx_dims = self.ctx_dim if self.ctx_knn > 0 else 0
        base_in_dim = in_dim                     # = A2（两开关全关）的输入宽度
        self.in_dim = base_in_dim + self.sh_dims + self.ctx_dims

        # 第 0 层：**先按 A2 的宽度建**（消耗掉与 A2 完全相同的 RNG 抽样），再零填充加宽。
        # 为什么不能直接 nn.Linear(self.in_dim, hidden)：
        #   safe_state（splat/utils/general_utils.py:130-132）设了 torch.manual_seed(0)
        #   ⇒ 同配置重跑逐位确定；改宽度会同时改 kaiming 的界 1/√fan_in 与抽样个数
        #   ⇒ net[0] 的旧列、net[2]、net[4] 全部跟着变 ⇒ A2/A4/A5 会拿到三份**互不相关**
        #   的初始化，零结果就分不清是「新输入没用」还是「初始化不同」。
        # ⚠ 由此得一条**易碎不变式**：新增子模块必须在 net / out / lam_raw **之后**构造，
        #   否则又挪动 RNG 流。验证闸 V7a 专门守它。
        first = nn.Linear(base_in_dim, hidden)
        if self.sh_dims + self.ctx_dims > 0:
            w = first.weight.detach()
            first.weight = nn.Parameter(torch.cat(
                [w, w.new_zeros(hidden, self.sh_dims + self.ctx_dims)], dim=1))
            first.in_features = self.in_dim
            # 新增的那些列**就是**零（上面的 new_zeros）—— 给定 A2 的权重，新通路的
            # 贡献恰好是 0 ⇒ A4/A5 在初始化处是 A2 的**严格嵌套**函数类。
        layers = [first, nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, n_bands)

        # init：权重置零 + 偏置取 sigmoid⁻¹(init_u) ⇒ 每个 band 初始 = init_u（默认 0.5）
        nn.init.zeros_(self.out.weight)
        u0 = min(max(self.init_u, 1e-4), 1 - 1e-4)
        nn.init.constant_(self.out.bias, float(torch.log(torch.tensor(u0 / (1.0 - u0)))))

        # λ 只在本形态存在，**必须条件创建**：旧 A' sidecar 用严格 load_state_dict，
        # 多出一个 key 会直接报 unexpected key 而读不进来。
        if n_bands > 1:
            self.lam_raw = nn.Parameter(torch.zeros(1))   # λ = sigmoid(0) = 0.5 ∈ (0,1)

        # ⚠ 上下文子模块**最后建**（RNG 不变式，见上）。LayerNorm 不吃 RNG（weight=1/bias=0）。
        if self.ctx_dims > 0:
            self.ctx_tok = nn.Linear(embed_dim + self.sh_dims, self.ctx_dim)
            self.ctx_Wq = nn.Linear(self.ctx_dim, self.ctx_dim)
            self.ctx_Wk = nn.Linear(self.ctx_dim, self.ctx_dim)
            self.ctx_Wv = nn.Linear(self.ctx_dim, self.ctx_dim)
            self.ctx_Wo = nn.Linear(self.ctx_dim, self.ctx_dim)
            self.ctx_ln = nn.LayerNorm(self.ctx_dim)

    def band_lambda(self) -> float:
        """当前 λ（仅 mult 有意义）；single 返回 None。用于日志/溯源。"""
        if not hasattr(self, "lam_raw"):
            return None
        return float(torch.sigmoid(self.lam_raw.detach()).reshape(-1)[0].item())

    def _context(self, embed: torch.Tensor, sh_block: torch.Tensor) -> torch.Tensor:
        """单层单头注意力：在**固定 kNN 图**上聚合邻居的 (embed, SH) → (P, ctx_dim)。

        v1 **视角无关**（`dir` 不进注意力）：token 只含「我是谁 + 我长什么样」，
        邻接由几何决定 ⇒ 整块是该点的定值。这与几何冻结的事实一致，
        也让它在初始化处是 A4 的严格嵌套（新列在 `net[0]` 里为零 ⇒ 贡献恰好 0）。

        分块沿 P 切（`_CTX_CHUNK`）只压住**单次 gather 的瞬态峰值** —— autograd 仍会把
        每个 chunk 的 K_g/V_g 留到 backward，实际增量约 1.2 GB（见模块头评估）。
        训练与渲染**必须**用同一块长：matmul 的 kernel 随 batch 变，块长不等会破坏
        「渲染路径与训练路径逐位相同」这条既有纪律。
        """
        if self.knn is None:
            raise ValueError(
                "ctx_knn>0 但 head.knn 未挂载。训练路径要经 build_or_load_knn() 建图，"
                "渲染路径要经 load_scalar_state() 从 sidecar 的 knn_file 读回。")
        P = embed.shape[0]
        knn = self.knn
        if tuple(knn.shape) != (P, self.ctx_knn + 1):
            raise ValueError(
                f"近邻图形状 {tuple(knn.shape)} != (P={P}, ctx_knn+1={self.ctx_knn + 1})。"
                f"sidecar 的 ctx_knn 与 .npz 里的数组对不上 ⇒ 邻居会静默错位。")
        if knn.device != embed.device:
            raise ValueError(f"近邻图在 {knn.device} 而输入在 {embed.device}；"
                             "knn 是普通属性，不会被 module.to() 搬走，挂载时就该放对设备。")
        # 第 0 列必须是自身：cKDTree.query 在并列距离/重合点下不保证自身排第一，
        # 建图时已强制过，这里再断言一次 —— 图被别的工具改写时**没有别的地方能发现**。
        self_col = knn[:, 0]
        if not torch.equal(self_col, torch.arange(P, device=self_col.device,
                                                  dtype=self_col.dtype)):
            raise ValueError("近邻图第 0 列不是自身索引 ⇒ 图被改写/截断过，邻居会静默错位")
        nbr = self.ctx_knn + 1
        tok_in = embed if self.sh_dims == 0 else torch.cat([embed, sh_block], dim=1)
        token = self.ctx_tok(tok_in)          # (P, d)
        Q = self.ctx_Wq(token)
        Kg_all = self.ctx_Wk(token)
        Vg_all = self.ctx_Wv(token)
        scale = 1.0 / float(self.ctx_dim) ** 0.5
        outs = []
        for s in range(0, P, _CTX_CHUNK):
            e = min(s + _CTX_CHUNK, P)
            idx = knn[s:e].reshape(-1)        # 拍平索引 ⇒ 从 (P,d) 上一把取
            Kg = Kg_all.index_select(0, idx).view(e - s, nbr, self.ctx_dim)
            Vg = Vg_all.index_select(0, idx).view(e - s, nbr, self.ctx_dim)
            w = torch.softmax(torch.einsum("bkd,bd->bk", Kg, Q[s:e]) * scale, dim=1)
            outs.append(torch.einsum("bk,bkd->bd", w, Vg))
        c = torch.cat(outs, dim=0)
        return self.ctx_ln(token + self.ctx_Wo(c))

    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor = None, dist: torch.Tensor = None,
                sh_block: torch.Tensor = None) -> torch.Tensor:
        """拼接块序与 __init__ 的 in_dim 计算一致；返回 (P,K) ∈ (0,1)，K = n_bands。

        `sh_block` = 物理阶段 SH 在 `dirv` 下的外观值 (P,3)，见 `sh_appearance()`。
        `use_sh != "none"` 时**必传**：缺了会静默换一种输入语义，所以这里硬报错。
        """
        if self.sh_dims > 0 and sh_block is None:
            raise ValueError(
                f"use_sh={self.use_sh!r} 需要 sh_block（(P,3)，来自 sh_appearance()），"
                "但收到 None。三个调用点都要传：train_rrf_scalar.py（循环外提升 sh_coeff）、"
                "render_rrf_scalar.py（render_sets 里提升）、"
                "bake_scalar_to_ply.py（该脚本对 SH 臂语义可疑，见其头部警告）。")
        if self.sh_dims == 0 and sh_block is not None:
            raise ValueError("use_sh='none' 却传了 sh_block ⇒ 调用点与 sidecar cfg 不一致")
        parts = [embed]
        if self.pe_xyz_L > 0:
            if xyz is None:
                raise ValueError("pe_xyz_L>0 但未传入 xyz")
            xin = xyz
            if self.xyz_center is not None and self.xyz_scale > 0:
                cen = torch.as_tensor(self.xyz_center, dtype=xyz.dtype,
                                      device=xyz.device)
                xin = (xyz - cen) / self.xyz_scale
            parts.append(fourier_feat(xin, self.pe_xyz_L))
        parts.append(dirv)
        if self.pe_dir_L > 0:
            parts.append(fourier_feat(dirv, self.pe_dir_L))
        if self.use_dist:
            if dist is None:
                raise ValueError("use_dist=True 但未传入 dist")
            parts.append(dist)
        # 两个新块**一律追加在最后**，顺序与 __init__ 的 sh_dims/ctx_dims **共用同一对变量**
        # （避免魔法偏移写错）。旧配置两者皆 0 ⇒ 这一段的图与改动前逐位相同。
        if self.sh_dims > 0:
            parts.append(sh_block)
        if self.ctx_dims > 0:
            parts.append(self._context(embed, sh_block))
        x = torch.cat(parts, dim=1)
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"拼接宽度 {x.shape[1]} != in_dim {self.in_dim}"
                f"（embed={self.embed_dim}, pe_xyz_L={self.pe_xyz_L}, pe_dir_L={self.pe_dir_L},"
                f" use_dist={self.use_dist}, sh_dims={self.sh_dims}, ctx_dims={self.ctx_dims}）。"
                "多半是 sidecar cfg 与调用点传参不一致 —— 继续跑会静默产出错图。")
        x = self.net(x)
        return torch.sigmoid(self.out(x))


def compute_scalar(head: ScalarColorHead, embed: torch.Tensor,
                   xyz: torch.Tensor, cam_center: torch.Tensor,
                   spatial_lr_scale: float = None,
                   sh_coeff: torch.Tensor = None) -> torch.Tensor:
    """全高斯标量 = head(embedding, γ(xyz), dir, γ(dir), r) → (P,K)，K = head.n_bands。

    **调用方必须经 `to_override_color()` 打包成 (P,3) 再喂 override_color**：
    (P,K) 直传不报错但静默产出错误图像（光栅化按 3 通道读，末维不符时错位/广播）。
    本函数故意原样返回 (P,K)，把"必须打包"这一步显式留在调用点，避免隐式陷阱。

    `sh_coeff`（(P,3,16) = `get_features.transpose(1,2)`）**是尾随参数**，因为它有三个
    调用点（train_rrf_scalar.py / render_rrf_scalar.py / bake_scalar_to_ply.py），
    加在最后才不打乱既有的位置参数调用（`render_rrf_walk.py` 就是这么调的）。
    `head.use_sh != "none"` 时**必传**。
    """
    dirv = make_dir(xyz, cam_center)
    dist = None
    if head.use_dist:
        dist = _dist(xyz, cam_center, spatial_lr_scale)
    sh_block = None
    if head.use_sh != "none":
        if sh_coeff is None:
            raise ValueError(
                f"该头 use_sh={head.use_sh!r}，需要 sh_coeff=(P,3,16)，但调用点没传。"
                "训练侧：从 gaussians.get_features.transpose(1,2).contiguous() 提升到循环外；"
                "渲染侧：从 ply 的 f_dc/f_rest 重建（注意 load_ply 会把 active_sh_degree "
                "强设为 max_sh_degree，degree 一律以 sidecar 的 head.sh_degree 为准）；"
                "bake_scalar_to_ply.py：该脚本对 SH 臂语义可疑，直接报错即可。")
        if sh_coeff.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"sh_coeff 行数 {sh_coeff.shape[0]} != xyz 行数 {xyz.shape[0]} ⇒ "
                "SH 与几何行序不对齐，继续跑会把别的高斯的外观贴到每个点上。")
        # 'dc' 就是 deg=0（只取基色、视角无关）；'eval' 用 sidecar 钉住的 head.sh_degree
        deg = 0 if head.use_sh == "dc" else head.sh_degree
        sh_block = sh_appearance(sh_coeff, dirv, deg)
    return head(embed, dirv, xyz, dist, sh_block)


def save_scalar_state(path: str, head: ScalarColorHead, embed: torch.Tensor,
                      iters_done: int, spatial_lr_scale: float = None,
                      lut_n: int = 256):
    """写 sidecar（embedding + 标量头权重 + 结构/PE 配置 + LUT 规模），供评测重建。
    与 point_cloud/iteration_<i>/point_cloud.ply 平行存放；embedding 行序 = ply 行序。
    lut_n 一并存档只为溯源（jet 是固定的，重建时按同参数构造即可）。
    """
    cfg = {
        "head_type": "scalar_jet",
        "embed_dim": head.embed_dim,
        "mlp_hidden": head.mlp_hidden,
        "mlp_depth": head.mlp_depth,
        "pe_xyz_L": head.pe_xyz_L,
        "pe_dir_L": head.pe_dir_L,
        "use_dist": head.use_dist,
        "init_u": head.init_u,
        # 形态 A× 溯源（旧 load 侧不读这两个键也能跑；新 load 侧用 cfg.get 带默认值读）
        "n_bands": int(head.n_bands),
        "band_mix": head.band_mix,
        "band_center": float(head.init_u),
        # λ 终值随档位永久留痕（K=1 时为 None）——"λ 到底动没动"是形态 A× 的关键待查项
        "band_lambda": head.band_lambda(),
        "lut_n": int(lut_n),
        "spatial_lr_scale": float(spatial_lr_scale or 0.0),
        "xyz_center": list(head.xyz_center) if head.xyz_center else None,
        "xyz_scale": float(head.xyz_scale or 0.0),
        # —— SH 输入块 / 空间上下文块（2026-09-13；旧 load 侧不读这些键也能跑）——
        "use_sh": head.use_sh,
        # degree **以 cfg 为唯一真源**：load_ply 会把 active_sh_degree 强设成 max_sh_degree，
        # 渲染侧从 ply 重建 SH 时拿不到真值 ⇒ 不钉死就会出现训练/渲染 degree 静默不一致。
        "sh_degree": int(head.sh_degree),
        "ctx_knn": int(head.ctx_knn),
        "ctx_dim": int(head.ctx_dim),
        "knn_file": head.knn_file,          # 相对 <model_path> 的文件名；None = 无图
        "ctx_chunk": int(_CTX_CHUNK),       # 渲染侧断言相等，见 _CTX_CHUNK 的注释
        "xyz_sha1": head.xyz_sha1,          # 建图所用 xyz 的摘要；None = 无图
    }
    torch.save({
        "cfg": cfg,
        "embed_dim": head.embed_dim,
        "mlp_hidden": head.mlp_hidden,
        "mlp_depth": head.mlp_depth,
        "iters_done": iters_done,
        "embedding": embed.detach().cpu(),
        "head": head.state_dict(),
    }, path)


def load_scalar_state(path: str, device: str = "cuda"):
    """读 sidecar → (head, embedding, spatial_lr_scale)。embedding 行序应对齐 ply。"""
    sd = torch.load(path, map_location=device)
    cfg = sd.get("cfg") or {}
    if cfg.get("head_type") not in (None, "scalar_jet"):
        raise ValueError(f"sidecar 不是标量头档（head_type={cfg.get('head_type')}）: {path}")
    head = ScalarColorHead(
        embed_dim=cfg.get("embed_dim", sd.get("embed_dim") or 32),
        hidden=cfg.get("mlp_hidden", sd.get("mlp_hidden") or 128),
        depth=cfg.get("mlp_depth", sd.get("mlp_depth") or 2),
        pe_xyz_L=cfg.get("pe_xyz_L", 0),
        pe_dir_L=cfg.get("pe_dir_L", 0),
        use_dist=bool(cfg.get("use_dist", False)),
        init_u=cfg.get("init_u", 0.5),
        # 旧 A' sidecar 没这两个键 ⇒ 默认 K=1/single ⇒ 行为逐字不变
        n_bands=cfg.get("n_bands", 1),
        band_mix=cfg.get("band_mix", "single"),
        # 旧 sidecar 没这些键 ⇒ 默认全关 ⇒ 结构与行为逐字不变
        use_sh=cfg.get("use_sh", "none"),
        sh_degree=cfg.get("sh_degree", 3),
        ctx_knn=cfg.get("ctx_knn", 0),
        ctx_dim=cfg.get("ctx_dim", 16),
    ).to(device)
    head.load_state_dict(sd["head"])
    embedding = sd["embedding"].to(device)
    scale = float(cfg.get("spatial_lr_scale", 0.0))
    head.xyz_center = cfg.get("xyz_center") or None
    head.xyz_scale = float(cfg.get("xyz_scale", 0.0) or 0.0)
    # 顺带把 lut_n 挂到 head 上：重建侧要按同参数造 LUT，省掉再 torch.load 一遍 sidecar
    head.lut_n = int(cfg.get("lut_n", 256))
    # —— 近邻图：**只读**。渲染侧绝不建图（它没有 model_path 的写权限语义），
    #    找不到就报错 —— 静默退回「无上下文」会让 A5 的输出看起来像 A4。
    head.xyz_sha1 = cfg.get("xyz_sha1")
    head.knn_file = cfg.get("knn_file")
    if head.ctx_knn > 0:
        if int(cfg.get("ctx_chunk", _CTX_CHUNK)) != _CTX_CHUNK:
            raise ValueError(
                f"sidecar 的 ctx_chunk={cfg.get('ctx_chunk')} != 本文件 _CTX_CHUNK={_CTX_CHUNK}。"
                "块长不等会让 matmul 走不同 kernel ⇒ 渲染与训练不再逐位一致。"
                "要么改回常量，要么重训（两者必须同源）。")
        if not head.knn_file:
            raise ValueError(f"ctx_knn={head.ctx_knn} 但 sidecar 没记 knn_file: {path}")
        root = _find_model_root(path)
        kp = os.path.join(root, head.knn_file)
        if not os.path.isfile(kp):
            raise ValueError(
                f"近邻图文件不存在: {kp}（sidecar 记的是 {head.knn_file}，"
                f"从 {path} 上溯到的 model_path = {root}）。渲染侧只读、不建图 —— "
                "训练时该文件会落在 model_path 下并与 sidecar 同目录。")
        z = np.load(kp)
        knn = torch.as_tensor(z["knn"].astype(np.int64)).to(device)
        if int(z["k"]) != head.ctx_knn:
            raise ValueError(f"近邻图 k={int(z['k'])} 与 sidecar ctx_knn={head.ctx_knn} 不符")
        if str(z["xyz_sha1"]) != (head.xyz_sha1 or ""):
            raise ValueError(
                f"近邻图 xyz_sha1={str(z['xyz_sha1'])[:12]}… 与 sidecar "
                f"{str(head.xyz_sha1)[:12]}… 不符 ⇒ 图不是用这份点序建的。")
        P = int(embedding.shape[0])
        if tuple(knn.shape) != (P, head.ctx_knn + 1):
            raise ValueError(f"近邻图形状 {tuple(knn.shape)} != (P={P}, ctx_knn+1="
                             f"{head.ctx_knn + 1})")
        self_col = knn[:, 0]
        if not torch.equal(self_col, torch.arange(P, device=device, dtype=self_col.dtype)):
            raise ValueError("近邻图第 0 列不是自身索引 ⇒ 图被别的工具改写或截断过")
        # 普通属性，**不是 register_buffer**：进 state_dict() 就会让旧 sidecar 读不进来
        head.knn = knn
        head.knn_npz = kp        # 只为日志/溯源
    return head, embedding, scale
