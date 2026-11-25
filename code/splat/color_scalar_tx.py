# -*- coding: utf-8 -*-
"""角域前馈标量头：逐场景 ScalarColorHead 的**跨发射源**版本。

为什么要新写一个头，而不是给 ScalarColorHead 加开关
--------------------------------------------------------------------------------
`ScalarColorHead` 是历史基线（形态 A'/A×）的**逐场景**头：同一份点云、同一个发射源
训一个头，输入里没有任何「哪个发射源」的信息。本文件要的是反过来：

    一个头 + 一份逐点 embedding，吃 (几何, 视线, 发射源位置) → 标量
    发射源在训练集里换 36 个，测试时换一个**没见过**的位置

把发射源条件硬塞进旧头会破坏它「旧 sidecar 严格 load、新列零初始化」的不变式
（见 `color_scalar.py` 模块头对 RNG 流的讨论），故单开一个类，旧类一行不改。

输入布局（与旧头同构，只在尾部追加）
--------------------------------------------------------------------------------
    cat[ embed,                      # (E) 逐点可学身份
         γ(xyz_norm)?,               # 6·pe_xyz_L，场景归一化后编码
         dir,                        # 3，视线方向 = normalize(xyz − cam_center)
         γ(dir)?,                    # 6·pe_dir_L
         r_cam?,                     # 1，||xyz − cam|| / spatial_lr_scale
         Δ, |Δ|, Δ̂, γ(Δ̂)?,          # 3+1+3+6·pe_rel_L，发射源相对几何
         t̂x, γ(t̂x)?, |tx|, sin/cos az, sin/cos el, log1p|tx|,  # 发射源绝对位置
         sinψ, cosψ, sin2ψ, cos2ψ?,  # 4，接收阵列朝向（可选）
         sin/cos Δyaw, sin/cos Δpitch?,      # 4，发射源朝向（可选，场景基准帧的偏移角）
         Rᵀ·Δ̂,                               # 3，逐点：点在发射源局部系中的方向
         f_norm?,                            # 1，频率（可选，声明范围线性映射）
         sinθ_w, cosθ_w, sinφ_w, cosφ_w, uni?  # 5，发射波束指向（可选，见下）
    ]  →  Linear+ReLU×depth  →  K  →  sigmoid  →  (P,K) ∈ (0,1)

参数块只在 `use_tx_orient / use_tx_local_dir / use_tx_freq / use_tx_w` 打开时出现，
四个开关默认全关 ⇒ `in_dim` 与旧检查点逐位一致（148），`load_state_dict` 不受影响。

波束块（`use_tx_w`）为什么只给**指向**、不给阵元增益
--------------------------------------------------------------------------------
发射阵是 4×4，权向量 w 在本仓的码本里就是「某个 (θ_w, φ_w) 方向的匹配权」或 uniform
（`sim/w_codebook.py` 是唯一定义处），所以 w 可以用 (θ_w, φ_w, 是否 uniform) 无损地
表达，5 维足够，不必塞 32 维复数原始权。

**故意不喂 g = wᴴ·a(û)**（那才是逐点的解析阵元增益）。`synthetic_array=True` 下
路径集合与 w 无关，`w → 图` 对 CBF 是闭式重加权；若把 g 直接算好喂进去，模型只需
学一个 O(#paths) 的线性重加权，实验就退化成「复现闭式解」，测不出高斯场自身的表达
能力。给指向、让头自己从数据里学 `|wᴴa(û)|` 这条角度响应，才是「蒸馏进高斯场」的
那条判据。若头部学不动，再开解析增益作对照 —— 那时结论要改写成「对标闭式重加权」。

**坐标系**：码本的 (θ,φ) 是**发射源本地系**（`direction_weight` 用 `element_positions`
的 y/z 轴），所以 `use_tx_w` 强制要求 `use_tx_local_dir` 同时打开，否则波束指向与
逐点方向分属两套坐标系。uniform 时 θ/φ 无定义，方向四维乘 (1−uni) 归零、只留标志位，
避免模型把「uniform」与「指向 θ=0,φ=0」混为一谈。

`P` 可达 1e6（本数据集 1,014,142 个光学高斯），全部输入都是逐点/常量张量，
没有 kNN 图、没有近邻 gather。

为什么 `dir` 是「角域」的那一半
--------------------------------------------------------------------------------
原 RF-3DGS 的谱图是**每个接收位置 × 每个阵列朝向**一张 300×200 的 AoA 谱。光栅化器把
逐高斯标量按透视投到像素上，因此「某个高斯落在哪个像素」由相机位姿决定；而**同一位置
的 4 个阵列 yaw 共享同一个相机中心**，`dir` 对它们完全相同。旧管线（SH 或逐场景 MLP）
因此在结构上无法区分这 4 张图 —— 它只能渲染出彼此相差一个滚转的同一个场。
本文件默认 (`use_yaw=False`) 保持这个结构；`use_yaw=True` 把阵列朝向显式喂进去，
作为「是否值得多给一个输入」的消融开关（该臂与原管线不再逐字同构，须单独标注）。

发射源条件为什么用「相对 + 绝对」两段
--------------------------------------------------------------------------------
- 相对段 Δ 是主变化轴：自由空间里逐点场强几乎只由 |Δ| 决定（本仓 `空间场` 实验实测
  相对特征 +7.4 dB）。它也是唯一能对**没见过**的发射源位置做外推的表示。
- 绝对段 t̂x 让头能学到「这个位置的发射源在房间里属于哪一类」（贴墙 / 居中 / 走廊），
  代价是 36 个样本下可能被用来记忆。两段同时给，消融再决定砍哪一段。

与旧管线共享的机制
--------------------------------------------------------------------------------
- 末层权重**零初始化**、偏置 = logit(init_u) ⇒ 第 0 步所有高斯都是 u=init_u，
  与形态 A' 的冷启动一致。
- 输出是 (P,K) 标量，调用点必须经 `to_override_color()` 打包成 (P,3)，渲染后再
  `combine_bands()` + `apply_jet()`（全部复用 `splat/color_scalar.py`），
  保证「先混合再查表」的口径与历史臂逐字一致。
"""
from __future__ import annotations

import json
import torch
import torch.nn as nn

from splat.color_mlp import fourier_feat, make_dir, _dist, set_scene_norm  # noqa: F401


class TxScalarColorHead(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 pe_xyz_L: int = 5, pe_dir_L: int = 3,
                 pe_rel_L: int = 4, pe_tx_L: int = 4,
                 use_dist: bool = True, use_rel: bool = True,
                 use_tx: bool = True, use_yaw: bool = False,
                 use_rx_dir: bool = False,
                 use_tx_orient: bool = False, use_tx_local_dir: bool = False,
                 use_tx_freq: bool = False, use_tx_w: bool = False,
                 use_tx_w_complex: bool = False,
                 use_tx_w_diff: bool = False, use_geom: bool = False,
                 film: bool = False, film_pos: bool = False,
                 use_pos_embed: bool = False, n_pos: int = 0,
                 pos_dim: int = 8, pos_dropout: float = 0.0,
                 freq_lo_hz: float = 2.0e9, freq_hi_hz: float = 4.5e9,
                 init_u: float = 0.5, n_bands: int = 1,
                 band_mix: str = "single"):
        super().__init__()
        n_bands = int(n_bands)
        if n_bands not in (1, 2, 3):
            raise ValueError(f"n_bands 须 ∈ {{1,2,3}}（光栅化器 NUM_CHANNELS==3），收到 {n_bands}")
        if band_mix not in ("single", "mult", "coh"):
            raise ValueError(f"band_mix 须 ∈ {{'single','mult','coh'}}，收到 {band_mix!r}")
        if band_mix in ("mult", "coh") and n_bands < 2:
            raise ValueError(f"band_mix={band_mix!r} 需要 n_bands >= 2")
        for name, v in (("pe_xyz_L", pe_xyz_L), ("pe_dir_L", pe_dir_L),
                        ("pe_rel_L", pe_rel_L), ("pe_tx_L", pe_tx_L)):
            if int(v) < 0:
                raise ValueError(f"{name} 须 >= 0，收到 {v}")
        if depth < 1:
            raise ValueError(f"depth 须 >= 1，收到 {depth}")
        if not (0.0 < float(init_u) < 1.0):
            raise ValueError(f"init_u 须在 (0,1) 内，收到 {init_u}")

        self.embed_dim = int(embed_dim)
        self.mlp_hidden = int(hidden)
        self.mlp_depth = int(depth)
        self.pe_xyz_L = int(pe_xyz_L)
        self.pe_dir_L = int(pe_dir_L)
        self.pe_rel_L = int(pe_rel_L)
        self.pe_tx_L = int(pe_tx_L)
        self.use_dist = bool(use_dist)
        self.use_rel = bool(use_rel)
        self.use_tx = bool(use_tx)
        self.use_yaw = bool(use_yaw)
        # 接收阵本体系方向：AoA 图是**阵列本体系**里的量，而 `dir` 用的是相机中心
        # （同一位置的 4 个视角 camera_center 完全相同 ⇒ dir 逐位相同）。把 dir 按
        # 阵列 yaw 反旋进本体系，元素方向图（tr38901，65° 波束）那部分增益才学得动。
        self.use_rx_dir = bool(use_rx_dir)
        if self.use_rx_dir and not self.use_yaw:
            raise ValueError("use_rx_dir=True 需要 use_yaw=True：旋转角来自 yaw")
        self.use_tx_orient = bool(use_tx_orient)
        self.use_tx_local_dir = bool(use_tx_local_dir)
        self.use_tx_freq = bool(use_tx_freq)
        self.use_tx_w = bool(use_tx_w)
        self.use_tx_w_complex = bool(use_tx_w_complex)
        self.use_tx_w_diff = bool(use_tx_w_diff)
        # 高斯基元自身的几何：最小轴（≈表面法向）+ 它对两条射线的夹角 + 镜面平分线
        # 对齐度 + 各向异性。三轴尺度比中位数 32（smin 1.2 mm），法向是真实可用的信号，
        # 而现有一切输入（xyz / dir / Δ）都推不出来。见模块头「几何块」。
        self.use_geom = bool(use_geom)
        self.geom_dims = 7 if self.use_geom else 0
        # FiLM：由**实例级**参数（朝向/频率/波束，不含逐点的 local_dir）预测每个隐层的
        # 仿射 scale/shift。零初始化 ⇒ 第 0 步与原头逐位相同，之后的变化只可能来自参数。
        # 动机见 §6.2：感知项末段买的是一条**全局色调曲线**（out.bias 位移 17 倍），
        # FiLM 让它变成「由参数预测的、逐实例的色调/形状族」。
        self.film = bool(film)
        self.use_pos_embed = bool(use_pos_embed)
        # or 0：旧检查点的 cfg 里 n_pos 可能是 None（基础头当时没有这个属性，
        # 训练端写的是 getattr(head, "n_pos", None)），不能直接 int(None)。
        self.n_pos = int(n_pos or 0)
        self.pos_dim = int(pos_dim or 8)
        self.pos_dropout = float(pos_dropout or 0.0)
        if self.use_pos_embed:
            if self.n_pos < 1:
                raise ValueError(f"use_pos_embed=True 需要 n_pos >= 1，收到 {n_pos}")
            if self.pos_dim < 1:
                raise ValueError(f"pos_dim 须 >= 1，收到 {pos_dim}")
            if not (0.0 <= self.pos_dropout < 1.0):
                raise ValueError(f"pos_dropout 须在 [0,1)，收到 {pos_dropout}")
        if float(freq_hi_hz) <= float(freq_lo_hz):
            raise ValueError(f"freq_hi_hz 须大于 freq_lo_hz，收到 "
                             f"[{freq_lo_hz}, {freq_hi_hz}]")
        self.freq_lo_hz = float(freq_lo_hz)
        self.freq_hi_hz = float(freq_hi_hz)
        self.init_u = float(init_u)
        self.n_bands = n_bands
        self.band_mix = band_mix
        # 场景归一化参数：训练时 set_scene_norm() 写入，随 cfg 进检查点
        self.xyz_center = None
        self.xyz_scale = 0.0

        in_dim = self.embed_dim
        if self.pe_xyz_L > 0:
            in_dim += 6 * self.pe_xyz_L
        in_dim += 3                                   # dir
        if self.pe_dir_L > 0:
            in_dim += 6 * self.pe_dir_L
        if self.use_dist:
            in_dim += 1
        self.rel_dims = (3 + 1 + 3 + (6 * self.pe_rel_L if self.pe_rel_L > 0 else 0)
                         ) if self.use_rel else 0
        self.tx_dims = (3 + (6 * self.pe_tx_L if self.pe_tx_L > 0 else 0) + 1 + 4 + 1
                        ) if self.use_tx else 0
        self.yaw_dims = 4 if self.use_yaw else 0
        self.rx_dir_dims = (3 + 6 * self.pe_dir_L) if self.use_rx_dir else 0
        # 发射源参数块（尾部追加，顺序固定：朝向 4 → 逐点局部方向 3 → 频率 1）
        self.orient_dims = 4 if self.use_tx_orient else 0
        self.local_dir_dims = 3 if self.use_tx_local_dir else 0
        self.freq_dims = 1 if self.use_tx_freq else 0
        self.w_dims = 5 if self.use_tx_w else 0
        self.w_complex_dims = 32 if self.use_tx_w_complex else 0
        # 波束差向量 2 维：Δv = u_y − sinθ_w·sinφ_w，Δw = u_z − cosθ_w。见模块头
        # 「波束差」说明；只在 use_tx_w 与 use_tx_local_dir 都打开时可用。
        self.w_diff_dims = 2 if self.use_tx_w_diff else 0
        self.param_dims = (self.orient_dims + self.local_dir_dims + self.freq_dims
                           + self.w_dims + self.w_complex_dims + self.w_diff_dims)
        # 位置条件化：每个训练位置一行可学隐向量；第 n_pos 行是「未知位置」回退行。
        # 位置 id 由实例表的 `position_id` 给出，渲染端只对训练过的位置传 id，
        # 其余（位置留出组）一律走回退行 —— 否则读数会偷偷用上没训练过的随机行。
        self.pos_dims = self.pos_dim if self.use_pos_embed else 0
        if self.use_tx_w and not self.use_tx_local_dir:
            raise ValueError(
                "use_tx_w=True 必须同时打开 use_tx_local_dir：码本的波束指向 (θ,φ) 是"
                "发射源本地系，逐点方向 Rᵀ·Δ̂ 也必须是本地系，否则两套坐标对不上")
        if self.use_tx_w_diff and not (self.use_tx_w and self.use_tx_local_dir):
            raise ValueError(
                "use_tx_w_diff=True 需要 use_tx_w 与 use_tx_local_dir 同时打开："
                "差向量是「波束指向」与「逐点本地方向」两者之差")
        self.in_dim = (in_dim + self.rel_dims + self.tx_dims + self.yaw_dims
                       + self.rx_dir_dims
                       + self.param_dims + self.geom_dims + self.pos_dims)

        if self.use_pos_embed:
            # 小方差随机初值：第 0 步输出由 out 的零权重决定，与 init_u 冷启动不冲突；
            # 随机是为了让不同位置在第 0 步就互不相同（全零会让它们永远靠梯度分开）。
            self.pos_emb = nn.Embedding(self.n_pos + 1, self.pos_dim)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        layers = [nn.Linear(self.in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, n_bands)

        # FiLM 调制器：只吃**非位置**的实例级参数（朝向、频率、波束、RX yaw），
        # 输出 depth 组 (scale, shift)。位置条件化已由 v1 的位置码判负过
        # （见记忆 model-averages-across-positions），这里不再重复。
        # 最后一层零初始化 ⇒ scale=1、shift=0，与未开 FiLM 的旧档第 0 步等价。
        # film_pos：把发射源**位置**也喂给调制器。理由：位置是这批数据里最强的
        # 变量（不同位置 14–15 dB 差，参数轴只有 45 dB 量级的图内效应），而干扰也
        # 来自位置——144 个实例共用同一个头。只按朝向/频率调制等于「相邻位置用同
        # 一套调制」，修不了位置干扰。逐位置**码表**判负过（`_位置码` 臂），但那是
        # 离散查表；这里是对连续 tx 位置的 γ 编码，第 0 步同样零初始化、恒等。
        self.film_pos = bool(film_pos and self.film)
        self.film_pos_dims = (3 + (6 * self.pe_tx_L if self.pe_tx_L > 0 else 0)
                              ) if self.film_pos else 0
        self.film_in = ((self.orient_dims + self.freq_dims + self.w_dims
                         + self.yaw_dims + self.film_pos_dims)
                        if self.film else 0)
        if self.film:
            if self.film_in < 1:
                raise ValueError("film=True 但没有任何实例级参数（朝向/频率/波束）"
                                 "可以喂调制器")
            self.film_mlp = nn.Sequential(
                nn.Linear(self.film_in, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * hidden * self.mlp_depth))
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)

        # 冷启动：权重 0 + 偏置 logit(init_u) ⇒ 每个 band 初值 = init_u
        nn.init.zeros_(self.out.weight)
        u0 = min(max(self.init_u, 1e-4), 1 - 1e-4)
        nn.init.constant_(self.out.bias, float(torch.log(torch.tensor(u0 / (1.0 - u0)))))
        if n_bands > 1:
            self.lam_raw = nn.Parameter(torch.zeros(1))   # λ = sigmoid(0) = 0.5

    def band_lambda(self):
        if not hasattr(self, "lam_raw"):
            return None
        return float(torch.sigmoid(self.lam_raw.detach()).reshape(-1)[0].item())

    # ------------------------------------------------------------------ #
    def _norm_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.xyz_center is None or not self.xyz_scale:
            return xyz
        cen = torch.as_tensor(self.xyz_center, dtype=xyz.dtype, device=xyz.device)
        return (xyz - cen) / float(self.xyz_scale)

    def _tx_block(self, tx: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """tx: (3,) 或 (P,3) → 逐点行向量（广播到 like 的行数）。"""
        if tx.dim() == 1:
            tx = tx.unsqueeze(0)
        if tx.shape[0] == 1 and like.shape[0] != 1:
            tx = tx.expand(like.shape[0], 3)
        parts = []
        if self.use_tx:
            t = self._norm_xyz(tx)
            r = tx.norm(dim=-1, keepdim=True)
            d = tx / r.clamp_min(1e-6)
            az = torch.atan2(d[:, 1], d[:, 0])
            el = torch.asin(d[:, 2].clamp(-1.0, 1.0))
            parts.append(t)
            if self.pe_tx_L > 0:
                parts.append(fourier_feat(t, self.pe_tx_L))
            parts.append(r)
            parts.append(torch.stack([torch.sin(az), torch.cos(az),
                                      torch.sin(el), torch.cos(el)], dim=1))
            parts.append(torch.log1p(r))
        if self.use_rel:
            dlt = like - tx
            r = dlt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            parts.append(dlt / max(float(self.xyz_scale or 1.0), 1e-6))
            parts.append(r / max(float(self.xyz_scale or 1.0), 1e-6))
            parts.append(dlt / r)
            if self.pe_rel_L > 0:
                parts.append(fourier_feat(dlt / max(float(self.xyz_scale or 1.0), 1e-6),
                                          self.pe_rel_L))
        return torch.cat(parts, dim=1)

    def _scalar_col(self, v, like: torch.Tensor) -> torch.Tensor:
        """标量/张量 → (P,1)，广播到 like 的行数。"""
        if torch.is_tensor(v):
            t = v.reshape(-1).to(dtype=like.dtype, device=like.device)
        else:
            t = torch.tensor([float(v)], dtype=like.dtype, device=like.device)
        if t.numel() == 1:
            return t.expand(like.shape[0], 1)
        if t.numel() != like.shape[0]:
            raise ValueError(f"参数长度 {t.numel()} 与点数 {like.shape[0]} 不符")
        return t.reshape(-1, 1)

    def _film_vec(self, like: torch.Tensor, yaw=None, tx_yaw=None,
                  tx_pitch=None, tx_freq=None, tx_w=None,
                  tx=None) -> torch.Tensor:
        """(1, film_in)：实例级参数（不含位置、不含逐点 local_dir）。

        取第 0 行即实例值——一次 forward 只对应一个发射源实例。
        """
        parts = []
        if self.yaw_dims > 0:
            if yaw is None:
                raise ValueError("use_yaw=True 但未传入 yaw")
            y = self._scalar_col(yaw, like)
            parts.append(torch.cat([torch.sin(y), torch.cos(y),
                                    torch.sin(2 * y), torch.cos(2 * y)], dim=1))
        if self.orient_dims:
            y = self._scalar_col(tx_yaw, like)
            p = self._scalar_col(tx_pitch, like)
            parts.append(torch.cat([torch.sin(y), torch.cos(y),
                                    torch.sin(p), torch.cos(p)], dim=1))
        if self.freq_dims:
            f = self._scalar_col(tx_freq, like)
            span = self.freq_hi_hz - self.freq_lo_hz
            parts.append(2.0 * (f - self.freq_lo_hz) / span - 1.0)
        if self.w_dims:
            if not torch.is_tensor(tx_w):
                tx_w = torch.tensor(tx_w, dtype=like.dtype, device=like.device)
            tw = tx_w.reshape(-1).to(dtype=like.dtype, device=like.device)
            if tw.numel() != 3:
                raise ValueError(f"tx_w 须为 3 个数 [θ_deg, φ_deg, is_uniform]，"
                                 f"收到 {tw.numel()} 个")
            th = torch.deg2rad(tw[0]).reshape(1, 1)
            ph = torch.deg2rad(tw[1]).reshape(1, 1)
            uni = tw[2].reshape(1, 1)
            s = 1.0 - uni
            parts.append(torch.cat([s * torch.sin(th), s * torch.cos(th),
                                    s * torch.sin(ph), s * torch.cos(ph), uni], dim=1))
        if self.film_pos_dims:
            if tx is None:
                raise ValueError("film_pos=True 但未传入 tx（发射源位置）")
            t = tx.reshape(-1, 3).to(dtype=like.dtype, device=like.device)
            if t.shape[0] > 1:
                t = t[:1]
            if t.shape[0] == 1 and like.shape[0] != 1:
                t = t.expand(like.shape[0], 3)
            tn = self._norm_xyz(t)
            parts.append(tn)
            if self.pe_tx_L > 0:
                parts.append(fourier_feat(tn, self.pe_tx_L))
        v = torch.cat(parts, dim=1)
        return v[:1]

    def _apply_net(self, x: torch.Tensor, film_vec) -> torch.Tensor:
        """跑 `self.net`；开 FiLM 时在每个 ReLU 后做 (scale, shift) 调制。

        仍逐模块遍历原 `self.net`，**不重建结构** ⇒ 旧检查点的
        `net.0.weight` / `net.2.weight` 键名逐字保留。
        """
        if not self.film:
            return self.net(x)
        g = self.film_mlp(film_vec).view(self.mlp_depth, 2, self.mlp_hidden)
        h, li = x, 0
        for m in self.net:
            h = m(h)
            if isinstance(m, nn.ReLU):
                h = h * (1.0 + g[li, 0]) + g[li, 1]
                li += 1
        return h

    def _param_block(self, like: torch.Tensor, tx: torch.Tensor,
                     tx_yaw=None, tx_pitch=None, tx_freq=None,
                     tx_rot=None, tx_w=None) -> torch.Tensor:
        """发射源参数块（13 维以内的尾段，见模块头「参数块」说明）。

        - 朝向 4：`sin/cos` 编码的**偏移角**（相对场景朝向基准帧），避免 0/2π 断点；
        - 局部方向 3：`u_local = Rᵀ·Δ̂`，R 为发射源绝对朝向的 TR38901 旋转矩阵。
          这是唯一逐点变化、且直接决定 tr38901 阵元增益方向的量；
        - 频率 1：`f_norm = 2(f−f_lo)/(f_hi−f_lo) − 1`，用**声明范围**而非观测范围，
          这样留出带仍可表示；不夹取。
        - 波束 5：`[sin/cos θ_w, sin/cos φ_w, uni]`，θ/φ 为度、发射源本地系；
          uniform 时前四维乘 0（见模块头「波束块」）。
        """
        parts = []
        uloc = None       # (P,3) 逐点在发射源本地系里的方向；波束差要用
        if self.orient_dims:
            if tx_yaw is None or tx_pitch is None:
                raise ValueError("use_tx_orient=True 但未传入 tx_yaw/tx_pitch")
            y = self._scalar_col(tx_yaw, like)
            p = self._scalar_col(tx_pitch, like)
            parts.append(torch.cat([torch.sin(y), torch.cos(y),
                                    torch.sin(p), torch.cos(p)], dim=1))
        if self.local_dir_dims:
            if tx is None:
                raise ValueError("use_tx_local_dir=True 但未传入 tx")
            if tx_rot is None:
                raise ValueError("use_tx_local_dir=True 但未传入 tx_rot")
            if tx.dim() == 1:
                tx = tx.unsqueeze(0)
            if tx.shape[0] == 1 and like.shape[0] != 1:
                tx = tx.expand(like.shape[0], 3)
            dlt = like - tx
            u = dlt / dlt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            R = torch.as_tensor(tx_rot, dtype=like.dtype, device=like.device)
            if R.shape != (3, 3):
                raise ValueError(f"tx_rot 须为 (3,3)，收到 {tuple(R.shape)}")
            uloc = u @ R                        # 行向量：(Rᵀu)ᵀ，列 = (x,y,z)
            parts.append(uloc)
        if self.freq_dims:
            if tx_freq is None:
                raise ValueError("use_tx_freq=True 但未传入 tx_freq")
            f = self._scalar_col(tx_freq, like)
            span = self.freq_hi_hz - self.freq_lo_hz
            parts.append(2.0 * (f - self.freq_lo_hz) / span - 1.0)
        if self.w_dims:
            if tx_w is None:
                raise ValueError("use_tx_w=True 但未传入 tx_w（[θ_deg, φ_deg, is_uniform]）")
            if not torch.is_tensor(tx_w):
                tx_w = torch.tensor(tx_w, dtype=like.dtype, device=like.device)
            tw = tx_w.reshape(-1).to(dtype=like.dtype, device=like.device)
            if tw.numel() != 3:
                raise ValueError(f"tx_w 须为 3 个数 [θ_deg, φ_deg, is_uniform]，"
                                 f"收到 {tw.numel()} 个")
            tw = tw.unsqueeze(0).expand(like.shape[0], 3)
            th = torch.deg2rad(tw[:, 0:1])
            ph = torch.deg2rad(tw[:, 1:2])
            uni = tw[:, 2:3]
            s = 1.0 - uni                       # uniform 时方向维归零，只留标志位
            parts.append(torch.cat([s * torch.sin(th), s * torch.cos(th),
                                    s * torch.sin(ph), s * torch.cos(ph), uni], dim=1))
            if self.w_diff_dims:
                # 阵元排在 y–z 平面（x≡0），匹配权的相位是 2π(y·v0 + z·w0)，
                # 所以增益只依赖二维差。把它显式给出来，头要学的是**所有配置共享**
                # 的一条窗函数，而不是「w 与逐点方向各自一段」再去拼乘积。
                if uloc is None:
                    raise RuntimeError("w_diff 需要本地方向，但 uloc 未计算")
                v0 = torch.sin(th) * torch.sin(ph)      # sinθ_w·sinφ_w
                w0 = torch.cos(th)                      # cosθ_w
                parts.append(torch.cat([s * (uloc[:, 1:2] - v0),
                                        s * (uloc[:, 2:3] - w0)], dim=1))
        if self.w_complex_dims:
            if tx_w is None:
                raise ValueError("use_tx_w_complex=True 但未传入 tx_w（32 维 Re/Im）")
            if not torch.is_tensor(tx_w):
                tx_w = torch.tensor(tx_w, dtype=like.dtype, device=like.device)
            tw = tx_w.reshape(-1).to(dtype=like.dtype, device=like.device)
            if tw.numel() != 32:
                raise ValueError(f"tx_w 复数输入须为 32 个数（16 个 Re/Im），收到 {tw.numel()}")
            parts.append(tw.unsqueeze(0).expand(like.shape[0], 32))
        return torch.cat(parts, dim=1)

    def _pos_block(self, n: int, pos_id, xyz: torch.Tensor) -> torch.Tensor:
        """(n, pos_dim)。pos_id=None/-1 ⇒ 全部走「未知位置」回退行。

        训练期按 `pos_dropout` 随机把整批位置换成回退行：这样回退行也被训练，
        位置留出组（只看得到回退行）才有意义；不丢的话回退行停在随机初值，
        pos_holdout 的读数会是「随机向量进了网络」，不可读。
        整批一起丢（而不是逐点丢）：同一次前向里所有高斯必须用同一个位置码。
        """
        fb = self.n_pos
        if pos_id is None:
            ids = torch.full((n,), fb, dtype=torch.long, device=xyz.device)
        else:
            ids = pos_id if torch.is_tensor(pos_id) else torch.as_tensor(
                pos_id, dtype=torch.long, device=xyz.device)
            ids = ids.reshape(-1).long()
            if ids.shape[0] == 1 and n != 1:
                ids = ids.expand(n)
            # -1 是「显式未知」的写法，与 None 等价
            ids = torch.where(ids < 0, torch.full_like(ids, fb), ids)
            if int(ids.max()) > fb:
                raise ValueError(
                    f"位置 id {int(ids.max())} 越界：码表只有 {self.n_pos} 行 + 1 回退行"
                    f"（回退行下标 {fb}；未知位置请传 None 或 -1）")
        if self.training and self.pos_dropout > 0.0:
            if float(torch.rand((), device=ids.device)) < self.pos_dropout:
                ids = torch.full_like(ids, fb)
        return self.pos_emb(ids)

    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor, dist: torch.Tensor = None,
                tx: torch.Tensor = None, yaw: torch.Tensor = None,
                tx_yaw=None, tx_pitch=None, tx_freq=None,
                tx_rot=None, tx_w=None, pos_id=None,
                normal=None, aniso=None) -> torch.Tensor:
        """→ (P,K) ∈ (0,1)。`yaw`：(P,) 或标量（弧度），仅 use_yaw=True 时必传。
        `tx_yaw/tx_pitch/tx_freq`：标量或 (P,)，`tx_rot`：(3,3)，
        `tx_w`：(3,) = [θ_w, φ_w, is_uniform]，见 `_param_block`。
        `pos_id`：发射源位置 id（标量或 (P,)），None/-1 = 未知位置。
        `normal`：(P,3) 逐高斯法向（最小轴）；`aniso`：(P,1) log(smax/smin)。"""
        x = self._features(embed, dirv, xyz, dist=dist, tx=tx, yaw=yaw,
                           tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq,
                           tx_rot=tx_rot, tx_w=tx_w, pos_id=pos_id,
                           normal=normal, aniso=aniso)
        fv = (self._film_vec(x, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                             tx_freq=tx_freq, tx_w=tx_w, tx=tx)
              if self.film else None)
        return torch.sigmoid(self.out(self._apply_net(x, fv)))

    def _geom_block(self, xyz, dirv, tx, normal, aniso) -> torch.Tensor:
        """(P,7)：法向、法向与 TX 射线夹角余弦、与 RX 射线夹角余弦、
        与两者平分线的对齐度（镜面反射条件 |n·b|）、各向异性 log(smax/smin)。

        `max_depth=1` 且 `diffuse=off` ⇒ 目标谱图的每个峰都对应一条**单次镜面反射**
        路径；峰的位置只由 (TX, RX, 表面法向) 决定。前两者早已在输入里，法向没有——
        这正是「峰位移 71.6 px」最直接的候选解释。
        """
        if normal is None:
            raise ValueError("use_geom=True 但未传入 normal（逐高斯最小轴）")
        n = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        t = tx if tx.dim() == 2 else tx.unsqueeze(0)
        if t.shape[0] == 1 and n.shape[0] != 1:
            t = t.expand(n.shape[0], 3)
        utx = xyz - t
        utx = utx / utx.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        urx = dirv / dirv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        b = utx + urx
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if aniso is None:
            aniso = torch.zeros(n.shape[0], 1, dtype=n.dtype, device=n.device)
        return torch.cat([n,
                          (n * utx).sum(-1, keepdim=True),
                          (n * urx).sum(-1, keepdim=True),
                          # 镜面条件：入射 (TX→点) 与出射 (点→RX) 的角平分线 ∝ ±n
                          (n * b).sum(-1, keepdim=True).abs(),
                          aniso.reshape(-1, 1)], dim=1)

    def _features(self, embed: torch.Tensor, dirv: torch.Tensor,
                  xyz: torch.Tensor, dist: torch.Tensor = None,
                  tx: torch.Tensor = None, yaw: torch.Tensor = None,
                  tx_yaw=None, tx_pitch=None, tx_freq=None,
                  tx_rot=None, tx_w=None, pos_id=None,
                  normal=None, aniso=None) -> torch.Tensor:
        """拼出 (P, in_dim) 输入矩阵。**只做拼装，不出 sigmoid**。

        抽出来是为了让复场头（`color_scalar_tx_cplx.py`）复用同一套几何编码：
        两种头唯一的差别在「输出层之后怎么进光栅化器」，输入必须逐位一致，
        否则两臂不可比。原 `forward` 行为完全不变。
        """
        parts = [embed]
        if self.pe_xyz_L > 0:
            parts.append(fourier_feat(self._norm_xyz(xyz), self.pe_xyz_L))
        parts.append(dirv)
        if self.pe_dir_L > 0:
            parts.append(fourier_feat(dirv, self.pe_dir_L))
        if self.use_dist:
            if dist is None:
                raise ValueError("use_dist=True 但未传入 dist")
            parts.append(dist)
        if self.rel_dims + self.tx_dims > 0:
            if tx is None:
                raise ValueError("use_rel/use_tx 打开但未传入 tx（[3] 或 [P,3]）")
            parts.append(self._tx_block(tx, xyz))
        if self.yaw_dims > 0:
            if yaw is None:
                raise ValueError("use_yaw=True 但未传入 yaw")
            y = yaw.reshape(-1, 1) if torch.is_tensor(yaw) else torch.tensor(
                [[float(yaw)]], dtype=xyz.dtype, device=xyz.device)
            if y.shape[0] == 1 and xyz.shape[0] != 1:
                y = y.expand(xyz.shape[0], 1)
            parts.append(torch.cat([torch.sin(y), torch.cos(y),
                                    torch.sin(2 * y), torch.cos(2 * y)], dim=1))
        if self.rx_dir_dims > 0:
            if yaw is None:
                raise ValueError("use_rx_dir=True 但未传入 yaw")
            y = yaw.reshape(-1, 1) if torch.is_tensor(yaw) else torch.tensor(
                [[float(yaw)]], dtype=dirv.dtype, device=dirv.device)
            if y.shape[0] == 1 and dirv.shape[0] != 1:
                y = y.expand(dirv.shape[0], 1)
            c, s = torch.cos(y), torch.sin(y)
            # Rz(−yaw)·dir：世界系方向 → 接收阵本体系
            drx = torch.cat([c * dirv[:, :1] + s * dirv[:, 1:2],
                             -s * dirv[:, :1] + c * dirv[:, 1:2],
                             dirv[:, 2:3]], dim=1)
            parts.append(drx)
            if self.pe_dir_L > 0:
                parts.append(fourier_feat(drx, self.pe_dir_L))
        if self.param_dims > 0:
            parts.append(self._param_block(xyz, tx, tx_yaw=tx_yaw,
                                           tx_pitch=tx_pitch, tx_freq=tx_freq,
                                           tx_rot=tx_rot, tx_w=tx_w))
        if self.geom_dims > 0:
            if tx is None:
                raise ValueError("use_geom=True 需要 tx")
            parts.append(self._geom_block(xyz, dirv, tx, normal, aniso))
        if self.pos_dims > 0:
            parts.append(self._pos_block(xyz.shape[0], pos_id, xyz))
        x = torch.cat(parts, dim=1)
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"拼接宽度 {x.shape[1]} != in_dim {self.in_dim}："
                f"embed={self.embed_dim} γxyz={self.pe_xyz_L} γdir={self.pe_dir_L} "
                f"dist={self.use_dist} rel={self.use_rel}/{self.pe_rel_L} "
                f"tx={self.use_tx}/{self.pe_tx_L} yaw={self.use_yaw} "
                f"param={self.param_dims} pos={self.pos_dims}")
        return x


def compute_scalar_tx(head: TxScalarColorHead, embed: torch.Tensor,
                      xyz: torch.Tensor, cam_center: torch.Tensor,
                      spatial_lr_scale: float, tx: torch.Tensor,
                      yaw=None, tx_yaw=None, tx_pitch=None,
                      tx_freq=None, tx_rot=None, tx_w=None,
                      pos_id=None, normal=None, aniso=None) -> torch.Tensor:
    """全高斯标量 = head(embed, γ(xyz), dir, γ(dir), r, tx, yaw?, 参数块?, 位置码?)。返回 (P,K)。

    按 `head_type` 分派：复值/二次型波束头（`color_scalar_tx_cplx.py`）的 w 走
    输出侧的闭式收缩，签名与调用点完全一致，训练与渲染两端因此不必分叉。
    `pos_id` 只对基础标量头有效（其余头不接受位置码，传了会静默丢列，故早报错）。
    """
    if getattr(head, "pos_dims", 0):
        if getattr(head, "head_type", "scalar_jet_tx") != "scalar_jet_tx":
            raise NotImplementedError(
                f"位置条件化只接在基础标量头上，当前 head_type="
                f"{getattr(head, 'head_type', '?')} 不支持 pos_id")
    if getattr(head, "head_type", "") == "scalar_jet_tx_cplx":
        from splat.color_scalar_tx_cplx import compute_scalar_cplx
        return compute_scalar_cplx(head, embed, xyz, cam_center, spatial_lr_scale,
                                   tx, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                                   tx_freq=tx_freq, tx_rot=tx_rot, tx_w=tx_w)
    if getattr(head, "head_type", "") == "scalar_jet_tx_fold":
        from splat.color_scalar_tx_fold import compute_scalar_fold
        return compute_scalar_fold(head, embed, xyz, cam_center, spatial_lr_scale,
                                   tx, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                                   tx_freq=tx_freq, tx_rot=tx_rot, tx_w=tx_w)
    if getattr(head, "head_type", "") == "scalar_jet_tx_af":
        from splat.color_scalar_af import compute_scalar_af
        return compute_scalar_af(head, embed, xyz, cam_center, spatial_lr_scale,
                                 tx, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                                 tx_freq=tx_freq, tx_rot=tx_rot, tx_w=tx_w)
    if getattr(head, "head_type", "") == "scalar_jet_tx_coh":
        from splat.color_scalar_coh import compute_scalar_coh
        return compute_scalar_coh(head, embed, xyz, cam_center, spatial_lr_scale,
                                  tx, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                                  tx_freq=tx_freq, tx_rot=tx_rot, tx_w=tx_w)
    if getattr(head, "head_type", "") == "scalar_jet_tx_bisg":
        from splat.color_scalar_bisg import compute_scalar_bisg
        return compute_scalar_bisg(head, embed, xyz, cam_center, spatial_lr_scale,
                                   tx, yaw=yaw, tx_yaw=tx_yaw, tx_pitch=tx_pitch,
                                   tx_freq=tx_freq, tx_rot=tx_rot, tx_w=tx_w)
    dirv = make_dir(xyz, cam_center)
    dist = _dist(xyz, cam_center, spatial_lr_scale) if head.use_dist else None
    return head(embed, dirv, xyz, dist, tx=tx, yaw=yaw,
                tx_yaw=tx_yaw, tx_pitch=tx_pitch, tx_freq=tx_freq, tx_rot=tx_rot,
                tx_w=tx_w, pos_id=pos_id, normal=normal, aniso=aniso)


def gaussian_geom(gaussians):
    """(P,3) 最小轴单位法向、(P,1) log(smax/smin)；几何冻结 ⇒ 只需算一次。

    **符号消歧是必须的**：最小轴作为特征向量本身没有朝向，3DGS 的四元数初值是随机的
    ⇒ 一半高斯的 n 与另一半反号。直接拿去做 cos(n, u) 相当于往输入里灌随机符号噪声。
    本仓是封闭房间，统一取**朝内**（指向点云质心）的那一侧。
    """
    from splat.utils.general_utils import build_rotation
    rot = gaussians.get_rotation
    if rot.dim() == 1:
        rot = rot.unsqueeze(0)
    R = build_rotation(rot)
    sc = gaussians.get_scaling
    imin = sc.argmin(dim=1)
    idx = torch.arange(sc.shape[0], device=sc.device)
    n = R[idx, :, imin]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    xyz = gaussians.get_xyz.detach()
    c = xyz.mean(dim=0, keepdim=True)
    outward = (n * (xyz - c)).sum(-1, keepdim=True)
    n = torch.where(outward > 0, -n, n)          # 指向质心 = 房间内侧
    aniso = torch.log(sc.max(1).values / sc.min(1).values.clamp_min(1e-12)).unsqueeze(1)
    return n.detach(), aniso.detach()


def _base_head_cfg(head: TxScalarColorHead) -> dict:
    """两种头共用的结构字段。`head_type` 由各自的 cfg 函数覆盖。"""
    return {
        "head_type": "scalar_jet_tx",
        "embed_dim": head.embed_dim, "mlp_hidden": head.mlp_hidden,
        "mlp_depth": head.mlp_depth,
        "pe_xyz_L": head.pe_xyz_L, "pe_dir_L": head.pe_dir_L,
        "pe_rel_L": head.pe_rel_L, "pe_tx_L": head.pe_tx_L,
        "use_dist": head.use_dist, "use_rel": head.use_rel,
        "use_tx": head.use_tx, "use_yaw": head.use_yaw,
        "use_rx_dir": head.use_rx_dir,
        "use_tx_orient": head.use_tx_orient,
        "use_tx_local_dir": head.use_tx_local_dir,
        "use_tx_freq": head.use_tx_freq, "use_tx_w": head.use_tx_w,
        "use_tx_w_complex": head.use_tx_w_complex,
        "use_tx_w_diff": head.use_tx_w_diff,
        "use_geom": head.use_geom,
        "use_los": bool(getattr(head, "use_los", False)),
        "los_param": (("log" if hasattr(head, "los_log") else "direct")
                      if getattr(head, "use_los", False) else "direct"),
        "los_kappa0": ([float(head.los_kappa0.real), float(head.los_kappa0.imag)]
                       if hasattr(head, "los_kappa0") else None),
        "use_rf_alpha": bool(hasattr(head, "rf_alpha")),
        "rf_alpha_hidden": int(getattr(getattr(head, "rf_alpha", None), "hidden", 64)),
        "rf_gx": int(getattr(getattr(head, "rf_alpha", None), "gx", 64)),
        "rf_gy": int(getattr(getattr(head, "rf_alpha", None), "gy", 64)),
        "rf_gz": int(getattr(getattr(head, "rf_alpha", None), "gz", 32)),
        "rf_bias_init": float(getattr(getattr(head, "rf_alpha", None),
                                      "bias_init", 0.0)),
        "rf_use_amp": bool(getattr(head, "rf_use_amp", False)),
        "use_diffuse": bool(hasattr(head, "diffuse")),
        "diffuse_hidden": int(getattr(getattr(head, "diffuse", None), "hidden", 64)),
        "film": bool(getattr(head, "film", False)),
        "film_pos": bool(getattr(head, "film_pos", False)),
        "film_in": int(getattr(head, "film_in", 0)),
        "use_pos_embed": head.use_pos_embed, "n_pos": head.n_pos,
        "pos_dim": head.pos_dim, "pos_dropout": head.pos_dropout,
        "freq_lo_hz": head.freq_lo_hz, "freq_hi_hz": head.freq_hi_hz,
        "param_spec": {
            "order": ["orient_sin_cos_yaw", "orient_sin_cos_pitch",
                      "local_dir_xyz", "freq_norm", "w_dir_sin_cos", "w_uniform",
                      "w_diff_vw"],
            "orient": "sin/cos of Δyaw, Δpitch (rad) in the scene-facing base frame",
            "local_dir": "Rᵀ·normalize(xyz − tx), R = TR38901 R(alpha,beta,0)",
            "freq": "2(f−f_lo)/(f_hi−f_lo) − 1, no clip",
            "w": "sin/cos(θ_w), sin/cos(φ_w) (rad, TX local frame) × (1−uni), then uni",
            "w_diff": "(1−uni)·[u_y − sinθ_w·sinφ_w, u_z − cosθ_w]，u 为本地系逐点方向；"
                      "阵元在 y–z 平面 ⇒ 增益只依赖这个二维差",
            "dims": {"orient": head.orient_dims,
                     "local_dir": head.local_dir_dims,
                     "freq": head.freq_dims, "w": head.w_dims,
                     "w_diff": head.w_diff_dims, "pos": head.pos_dims},
        },
        "init_u": head.init_u, "n_bands": int(head.n_bands),
        "band_mix": head.band_mix, "band_lambda": head.band_lambda(),
        "xyz_center": None if head.xyz_center is None else list(head.xyz_center),
        "xyz_scale": float(head.xyz_scale),
        "in_dim": int(head.in_dim),
    }


def tx_head_cfg(head: TxScalarColorHead, extra: dict | None = None) -> dict:
    """按 `head_type` 分派；复值/二次型波束头另加 n_ant / n_pos / n_neg。"""
    if getattr(head, "head_type", "scalar_jet_tx") == "scalar_jet_tx_cplx":
        from splat.color_scalar_tx_cplx import cplx_head_cfg
        return cplx_head_cfg(head, extra)
    if getattr(head, "head_type", "scalar_jet_tx") == "scalar_jet_tx_fold":
        from splat.color_scalar_tx_fold import fold_head_cfg
        return fold_head_cfg(head, extra)
    if getattr(head, "head_type", "scalar_jet_tx") == "scalar_jet_tx_af":
        from splat.color_scalar_af import af_head_cfg
        return af_head_cfg(head, extra)
    if getattr(head, "head_type", "scalar_jet_tx") == "scalar_jet_tx_coh":
        from splat.color_scalar_coh import coh_head_cfg
        return coh_head_cfg(head, extra)
    if getattr(head, "head_type", "scalar_jet_tx") == "scalar_jet_tx_bisg":
        from splat.color_scalar_bisg import bisg_head_cfg
        return bisg_head_cfg(head, extra)
    cfg = _base_head_cfg(head)
    if extra:
        cfg.update(extra)
    return cfg


def _attach_los(head, cfg: dict):
    return _attach_extra(head, cfg)


def _attach_extra(head, cfg: dict):
    if cfg.get("use_los"):
        head.use_los = True
        if cfg.get("los_param") == "log":
            k0 = cfg.get("los_kappa0") or [1.0, 0.0]
            head.los_kappa0 = complex(float(k0[0]), float(k0[1]))
            head.los_log = nn.Parameter(torch.zeros(2))
        else:
            head.los_kappa = nn.Parameter(
                torch.tensor(1.0 + 0.0j, dtype=torch.complex64))
    if cfg.get("use_rf_alpha"):
        from splat.rf_alpha import RfAlphaHead
        head.rf_alpha = RfAlphaHead(
            embed_dim=int(cfg.get("embed_dim", 32)),
            hidden=int(cfg.get("rf_alpha_hidden", 64)),
            gx=int(cfg.get("rf_gx", 64)), gy=int(cfg.get("rf_gy", 64)),
            gz=int(cfg.get("rf_gz", 32)),
            bias_init=float(cfg.get("rf_bias_init", 0.0)))
        head.rf_use_amp = bool(cfg.get("rf_use_amp", False))
    if cfg.get("use_diffuse"):
        from splat.diffuse import DiffuseHead
        head.use_diffuse = True
        head.diffuse = DiffuseHead(embed_dim=int(cfg.get("embed_dim", 32)),
                                   hidden=int(cfg.get("diffuse_hidden", 64)))
    return head


def head_from_cfg(cfg: dict) -> TxScalarColorHead:
    if cfg.get("head_type") == "scalar_jet_tx_cplx":
        from splat.color_scalar_tx_cplx import TxComplexColorHead
        return _attach_extra(TxComplexColorHead.from_cfg(cfg), cfg)
    if cfg.get("head_type") == "scalar_jet_tx_fold":
        from splat.color_scalar_tx_fold import TxFoldColorHead
        return _attach_extra(TxFoldColorHead.from_cfg(cfg), cfg)
    if cfg.get("head_type") == "scalar_jet_tx_af":
        from splat.color_scalar_af import TxArrayFactorColorHead
        return _attach_extra(TxArrayFactorColorHead.from_cfg(cfg), cfg)
    if cfg.get("head_type") == "scalar_jet_tx_coh":
        from splat.color_scalar_coh import TxCoherentColorHead
        return _attach_extra(TxCoherentColorHead.from_cfg(cfg), cfg)
    if cfg.get("head_type") == "scalar_jet_tx_bisg":
        from splat.color_scalar_bisg import TxBiSGColorHead
        return _attach_extra(TxBiSGColorHead.from_cfg(cfg), cfg)
    h = TxScalarColorHead(
        embed_dim=cfg["embed_dim"], hidden=cfg["mlp_hidden"], depth=cfg["mlp_depth"],
        pe_xyz_L=cfg["pe_xyz_L"], pe_dir_L=cfg["pe_dir_L"],
        pe_rel_L=cfg.get("pe_rel_L", 0), pe_tx_L=cfg.get("pe_tx_L", 0),
        use_dist=cfg["use_dist"], use_rel=cfg.get("use_rel", False),
        use_tx=cfg.get("use_tx", False), use_yaw=cfg.get("use_yaw", False),
        use_rx_dir=cfg.get("use_rx_dir", False),
        use_tx_orient=cfg.get("use_tx_orient", False),
        use_tx_local_dir=cfg.get("use_tx_local_dir", False),
        use_tx_freq=cfg.get("use_tx_freq", False),
        use_tx_w=cfg.get("use_tx_w", False),
        use_tx_w_complex=cfg.get("use_tx_w_complex", False),
        use_tx_w_diff=cfg.get("use_tx_w_diff", False),
        use_geom=cfg.get("use_geom", False),
        film=cfg.get("film", False),
        film_pos=cfg.get("film_pos", False),
        use_pos_embed=cfg.get("use_pos_embed", False),
        # 旧检查点：n_pos 可能显式存成 None（见 __init__ 的注释），or 兜住
        n_pos=cfg.get("n_pos") or 0,
        pos_dim=cfg.get("pos_dim") or 8,
        pos_dropout=cfg.get("pos_dropout") or 0.0,
        freq_lo_hz=cfg.get("freq_lo_hz", 2.0e9),
        freq_hi_hz=cfg.get("freq_hi_hz", 4.5e9),
        init_u=cfg["init_u"], n_bands=cfg.get("n_bands", 1),
        band_mix=cfg.get("band_mix", "single"))
    h.xyz_center = None if cfg.get("xyz_center") is None else tuple(cfg["xyz_center"])
    h.xyz_scale = float(cfg.get("xyz_scale", 0.0))
    return _attach_los(h, cfg)


def save_tx_state(path: str, head: TxScalarColorHead, embed: torch.Tensor,
                  iteration: int, extra: dict | None = None,
                  opacity: torch.Tensor | None = None) -> None:
    """存「头 + 逐点 embedding +（可选）训练过的 opacity + 全部结构配置」。

    opacity 必须一起存：本任务几何冻结、densify=0，唯一会变的逐点参数就是它；
    只存头会让渲染端静默用回种子检查点的 opacity，指标与训练时**不是同一个模型**。
    """
    cfg = tx_head_cfg(head, extra)
    blob = {"iteration": int(iteration),
            "head": head.state_dict(),
            "embed": embed.detach().cpu(),
            "cfg": cfg}
    if opacity is not None:
        blob["opacity"] = opacity.detach().cpu().reshape(-1)
    torch.save(blob, path)


def load_tx_state(path: str, device="cuda"):
    """→ (head, embed, opacity 或 None, cfg, iteration)。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = head_from_cfg(ck["cfg"])
    head.load_state_dict(ck["head"])
    head = head.to(device)
    embed = ck["embed"].to(device)
    op = ck.get("opacity")
    if op is not None:
        op = op.reshape(-1, 1).to(device)
    return head, embed, op, ck["cfg"], int(ck.get("iteration", -1))
