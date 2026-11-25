# -*- coding: utf-8 -*-
"""heads.py —— NeRF² 两臂的末端映射

两臂的**唯一**差别是输出参数化，所以这里刻意把「末端映射」做成同一个类、同一份
代码、同一套初始化规则：

    N1（RGB 头）  : sig_output_dims = 6  ⇒ 渲染器给 [B,3]
    N2（标量头）  : sig_output_dims = 2  ⇒ 渲染器给 [B]（与上游逐字同）
    两臂共用      : v̂ = σ(α ∘ v + β)      （α, β 为可学习**逐通道**标量）
    N2 额外       : rgb = apply_jet(v̂)     （固定查表，无参数）

为什么必须有这个末端映射，而不是直接拿上游的 abs() 输出当靶子
==============================================================
上游的 `receive = |∑_n s_a·e^{i·s_p}·att_i·phase_i·path_loss|` 是**非负、无界**的
标量；上游靠 `path_loss = 0.025/path` 的尺度让它自然落在 [0,1] 附近，再用 `/255`
的标签配 MSE 去压 —— **它的后面没有任何 sigmoid**（见 UPSTREAM.md §3.4）。

我们的 GT 是 [0,1] 且逐图满量程（UPSTREAM.md §4.1 / §6 边界 5），不存在一个固定
仿射能把渲染标量映到靶子 ⇒ 必须有一个**可学**的末端映射。把它设计成两臂共用同一
形式，于是它在 Δ(N2 − N1) 里自动抵消。

这条推翻了方案初稿 §1 表格里「N1 保持线性、N2 加 sigmoid」的写法 —— 那会让两臂
差两件事（线性/非线性 + 通道数），Δ 就不再干净。以 UPSTREAM.md §4.3 为准。

为什么 α 用 softplus 参数化
==========================
`v` 恒非负（`abs()` 的输出），而「接收功率越大 ⇒ 色标坐标越靠红端」是生成侧的物理
方向（u=0 深蓝、u=1 深红）。α > 0 把这个单调方向钉成先验。

softplus 还消掉一个**静默的死点**：α=0 时 `σ(0·v + β) ≡ σ(β)` 与 `v` 无关，
`∂L/∂v ≡ 0` ⇒ 渲染器那一整条支路的梯度全断，而 loss 照常下降（只是降到常数解），
日志上看不出任何异常。softplus 让 α 恒 > 0。

为什么需要 calibrate()
======================
`v` 的初始量级由最后一个 Linear 的随机初始化与 `path_loss` 的尺度共同决定，
**事前不可知**。若 α 初值取 1 而 v 的实际量级是 1e-3，则 `σ(αv+β)` 恒 ≈ σ(β)，
要靠优化把 α 抬到 1e3 —— 而 Adam 每步只动 ~lr，8e-4 × 3 万步也追不上一个
量级差 3 个数量级的初值。这不是理论担忧：`path_loss = 0.025/path`（path 是米）
与 64 个采样点的复数累加，会让 v 落在很小的量级上。

`calibrate()` 用一次**无梯度**前向，把 α、β 设成「输出分位数对齐靶子分位数」。
它是纯粹的初始化选择，两臂同一套规则，不增加任何可学参数。

    α = (logit(hi) − logit(lo)) / (q_hi(v) − q_lo(v))
    β = logit(lo) − α · q_lo(v)

N1 的 `lo/hi` 分位取**靶子自身**的逐通道分位数（jet 的 R 通道只到 0.5、G 通道
满量程，逐通道对齐才合理）；N2 无对应的逐通道靶子（靶子在 jet 之后），取 [0.05, 0.95]
让初值铺满整条色标，由训练去收。

本模块**不碰上游**：`model.py` 的两个输出头维度本来就是构造参数
（`attn_output_dims` / `sig_output_dims`，见 UPSTREAM.md §3.5）。
"""
import torch
import torch.nn.functional as F
from torch import nn

from splat.color_scalar import apply_jet, make_jet_lut

MODES = ("rgb", "scalar")
ENDS = ("sigmoid", "clamp", "leaky")

# leaky 形式在带外的斜率。**不能取 0**（那就退化成 clamp，见下方 docstring 的实测），
# 也不必大：它是「逃逸通道」而不是工作区，取 1e-2 让带外像素每 100 步挪 ≈ lr·100 个
# 输出单位，足以在几百步内走回带内，同时对带内像素的读数没有影响。
LEAK = 1e-2

# N2 无逐通道靶子时用的默认分位区间
DEFAULT_LO, DEFAULT_HI = 0.05, 0.95


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """softplus 的逆。要求 y > 0；y 接近 0 时趋于 −inf（调用方保证 α 下界）。"""
    return y + torch.log(-torch.expm1(-y))


class EndMap(nn.Module):
    """末端映射 `v ↦ 值`，逐通道可学习标量。两臂共用同一形式（`--head_end` 选）。

        end="sigmoid"   v̂ = σ(α ∘ v + β)，α = softplus(alpha_raw) > 0
        end="clamp"     v̂ = clamp(a ∘ v + b, 0, 1)，a **带符号自由**
        end="leaky"     同 clamp，但带外**保留 LEAK 的斜率**（见"为什么还要 leaky"）

    `v : (..., C)` ⇒ `(..., C)`。C = n_channels。

    为什么加 clamp（2026-09-16 MVDR/G6 实测，不是审美）
    =================================================
    σ 的尾部**指数平坦**：`∂L/∂v = ∂L/∂u · σ'(αv+β) · α` 三个因子同时趋零，
    所以一个通道只要漂到带外，就**连同它自己的 α、β 一起永久冻结**（σ'=0 ⇒ 参数
    梯度也是 0，没有回头路）。三条实测：

    ① 同一份 v（N1 的 6000 步存档）上，逐通道 `clamp(a·v+b)` 的 oracle 是
       **22.325 dB**，而训练出来的 σ 头只有 **10.212 dB** —— 12.1 dB 卡在 σ 上。
    ② oracle 的最优斜率**带负号**（a_R = −1.66），而 `softplus` 把 α 钉死为正
       ⇒ 那个解在**函数类层面**就表达不了。「功率越大越靠红端」这条先验在本数据上
       不成立：实测 r(v, u_gt) = **−0.41**。
    ③ 冻结的样子：N1 的 R 通道全图饱和在 1.0（v ∈ [2.33, 44.4]，而靶子需要的区间是
       [−0.085, 1.08]），输出恒 1.000、Pearson **+0.0000**，α 6000 步只从 18.42 挪到 15.82。

    clamp 的中段是**线性**的，且 (a, b) 对固定 v 的 L1 损失是**凸**的 ⇒ 没有这个陷阱。
    代价是 [0,1] 之外梯度为零，所以锚点（`calibrate`）要保证初值时绝大多数像素在带内。

    为什么还要 leaky（2026-09-16 23:0x 实测，clamp 也栽了同一个坑）
    =============================================================
    **clamp 的带外梯度恒为 0 —— 与 σ 的尾部是同一种死区**，只是形状不同。
    锚点是在**初始 v** 上算的（窗口 W=0.04），而场会把 v 长出这个带子：

        N1（log-clamp，W=0.04，1500 步）逐通道截断占比实测 **14.1% / 88.0% / 100.0%**
        B 通道 v ∈ [0.106, 3.54] 而带只有 [−0.0004, 0.041] ⇒ **100% 截断** ⇒
        B 通道输出恒为常数、梯度恒为 0 ⇒ 该通道**永久冻结**，PSNR 卡在 10.642 dB，
        而同一份 v 上的逐通道仿射 oracle 是 23.8 dB。

    ⇒ 「带内线性、损失凸」只保证**带内**好优化，不保证**能回到带内**。
    带外给一个 ‖LEAK‖ 的小斜率，通道就永远有路走回来；代价是输出不再严格 ∈[0,1]
    （超出量的 LEAK 倍，LEAK=1e-2 时可忽略），且**它仍然是一个良定的末端映射**。

    三种形式**共用 `alpha_raw` / `beta` 这两个参数名**：state_dict 的键一字不变，
    存档可以互载（数值含义不同 —— 靠 `head_cfg()["end"]` 判，`train_nerf2.py` 续跑时对拍）。
    """

    def __init__(self, C: int, end: str = "sigmoid",
                 alpha: float = 1.0, beta: float = 0.0):
        super().__init__()
        if C < 1:
            raise ValueError("C 必须 ≥ 1，收到 %d" % C)
        if end not in ENDS:
            raise ValueError("end 必须是 %s 之一，收到 %r" % (ENDS, end))
        self.end = end
        a = torch.full((C,), float(alpha))
        if end == "sigmoid":
            self.alpha_raw = nn.Parameter(_softplus_inv(a))
        else:
            # **斜率必须按对数走**：`a = sign · exp(alpha_raw)`。
            # 这不是美感问题，是量纲问题：Adam 每步把参数挪 ≈lr，对 a 而言**相对**变化
            # 只有 lr/|a|。而锚点把 a 定成 (t_hi−t_lo)/spread 的量级 —— 实测 N1 的
            # a ≈ −25 ⇒ 相对步 3.2e-5 ⇒ 跑满 1500 步只能改 5%，头被自己的尺度焊死
            # （实测 a 从 −24.99 只挪到 −24.5）。同时场的 v 长出带外 5 倍、大面积截断，
            # 头却来不及把 a 收小 —— 这是 N1「场给得起 23.8 dB、头只拿到 10.65」的直接原因。
            # 取对数后 Adam 的相对步恒为 lr（1500 步可改 e^{1.2} ≈ 3.3 倍，够用）。
            # 代价是**不能跨零**，所以符号由 `calibrate` 按数据定、存进 `sign`。
            self.alpha_raw = nn.Parameter(a.abs().clamp_min(1e-12).log())
            self.register_buffer("sign", a.sign().masked_fill(a == 0, 1.0))
        self.beta = nn.Parameter(torch.full((C,), float(beta)))

    @property
    def alpha(self) -> torch.Tensor:
        """sigmoid 形式下是恒正的 α；clamp 形式下是**带符号**的 a（可正可负）。"""
        if self.end == "sigmoid":
            return F.softplus(self.alpha_raw)
        return self.sign * self.alpha_raw.exp()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        z = self.alpha * v + self.beta
        if self.end == "sigmoid":
            return torch.sigmoid(z)
        c = z.clamp(0.0, 1.0)
        # clamp 的带外斜率为 0（= 死区）；leaky 用 LEAK 保底，见类 docstring
        return c if self.end == "clamp" else c + LEAK * (z - c)


class RfHead(nn.Module):
    """一个臂的全部输出侧状态：末端映射 + （仅 N2）固定 jet 查表。

    `mode='rgb'`    ⇒ n_channels = 3，value 直接就是 RGB
    `mode='scalar'` ⇒ n_channels = 1，value 是色标坐标 u，再过 `apply_jet` 成 RGB

    LUT **不进 optimizer、不存档**（由 `make_jet_lut(lut_n)` 确定性重建）——
    与 `train_rrf_scalar.py:147-148` 同一条纪律：色标是常数，不是可学量。
    """

    def __init__(self, mode: str = "rgb", lut_n: int = 256, device="cuda",
                 end: str = "sigmoid"):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode 必须是 %s 之一，收到 %r" % (MODES, mode))
        self.mode = mode
        self.lut_n = int(lut_n)
        self.n_channels = 3 if mode == "rgb" else 1
        self.end = EndMap(self.n_channels, end=end)
        # 固定查表：register_buffer 让它随 .to(device) 走但不进 state_dict 的 optimizer
        self.register_buffer("lut", make_jet_lut(self.lut_n), persistent=False)
        self.to(device)

    # ---------------------------------------------------------------- 前向
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """渲染器的原始输出 ⇒ 末端映射之后的值。形状不变。

        rgb 臂：(B,3) → (B,3)，三通道就是 RGB
        scalar 臂：(B,) → (B,)，单通道是色标坐标 u
        """
        if self.mode == "rgb":
            if v.dim() < 1 or v.shape[-1] != 3:
                raise ValueError("rgb 臂期望末维为 3，收到 %s" % (tuple(v.shape),))
            return self.end(v)
        # 渲染器若返回 (B,1) 也接受。**必须判 dim ≥ 2** —— B==1 时 (1,) 的末维也等于 1，
        # 按末维判会把整批压成 0 维标量，形状对不上却不报错。
        if v.dim() >= 2 and v.shape[-1] == 1:
            v = v.squeeze(-1)
        return self.end(v.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------- 值 → RGB 三通道
    def value_to_rgb(self, value: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """整图用：value (H·W, C) ⇒ RGB (3,H,W) ∈ [0,1]。

        对齐 `render.py:32` 的 `["render"]` 形状，供 `torchvision.utils.save_image`。
        """
        if value.shape[0] != H * W:
            raise ValueError("射线数 %d != H·W = %d" % (value.shape[0], H * W))
        if self.mode == "rgb":
            return value.reshape(H, W, 3).permute(2, 0, 1).contiguous()
        # apply_jet 的硬护栏要求首维恰为 1：多通道/多 batch 会 raise，不许塞 (B,H,W)
        return apply_jet(value.reshape(1, H, W), self.lut)

    def patch_to_rgb(self, value: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        """块用（训练损失）：value (ph·pw, C) ⇒ (1,3,ph,pw) —— `ssim` 要的四维形状。"""
        if value.shape[0] != ph * pw:
            raise ValueError("射线数 %d != ph·pw = %d" % (value.shape[0], ph * pw))
        if self.mode == "rgb":
            return value.reshape(1, ph, pw, 3).permute(0, 3, 1, 2).contiguous()
        return apply_jet(value.reshape(1, ph, pw), self.lut).unsqueeze(0)

    def batch_to_rgb(self, value: torch.Tensor) -> torch.Tensor:
        """散点用：value (B, C) ⇒ (B,3)。查表走 (1,1,B) 再转回来。"""
        if self.mode == "rgb":
            return value
        rgb = apply_jet(value.reshape(1, 1, -1), self.lut)      # (3,1,B)
        return rgb.squeeze(1).t().contiguous()                  # (B,3)

    # ---------------------------------------------------------------- 校准
    @torch.no_grad()
    def calibrate(self, v_cal: torch.Tensor, target_cal: torch.Tensor = None,
                  lo: float = DEFAULT_LO, hi: float = DEFAULT_HI,
                  min_v_window: float = 1.0) -> dict:
        """用一次无梯度前向的输出把 α、β 设成「分位数对齐」。

        v_cal        : (N, C) 渲染器在若干条射线上的**原始**输出（σ 之前）
        target_cal   : (N, C) 靶子；rgb 臂应给，scalar 臂给 None（见模块 docstring）
        min_v_window : `spread` 的下限，即**窗口 span/α 的下限**。见下方长注释 ——
                       它是这个基线最容易写错的一个数，**必须由调用方按 v 的量级显式给**，
                       默认值 1.0 只是为兼容旧行为保留，不要当成推荐值。

        返回一个可打印的 dict（进日志）。**幂等**：同一份 v_cal 调用两次结果相同。
        """
        v = v_cal.detach().reshape(-1, self.n_channels)
        if v.shape[0] < 16:
            raise ValueError("校准样本太少（%d），分位数不可信" % v.shape[0])

        # 靶子的 lo/hi 分位，**统一在值空间算**（sigmoid 再取 logit）。
        # 这样两种 end 形式共用同一条锚定逻辑，只在「按什么铺开」上分叉。
        if target_cal is not None:
            t = target_cal.detach().reshape(-1, self.n_channels)
            if t.shape[0] != v.shape[0]:
                raise ValueError("靶子行数 %d != 输出行数 %d" % (t.shape[0], v.shape[0]))
            t_lo = torch.quantile(t, lo, dim=0).clamp(1e-4, 1 - 1e-4)
            t_hi = torch.quantile(t, hi, dim=0).clamp(1e-4, 1 - 1e-4)
        else:
            t_lo = torch.full((self.n_channels,), float(lo), device=v.device, dtype=v.dtype)
            t_hi = torch.full((self.n_channels,), float(hi), device=v.device, dtype=v.dtype)

        v_lo = torch.quantile(v, lo, dim=0)
        v_hi = torch.quantile(v, hi, dim=0)
        # `min_v_window` 是这个基线**最容易写错的一个数**，2026-09-16 前后踩了两次，
        # 两次的教训合起来才是完整的：
        #
        # ① 原值 1e-8（形同虚设）⇒ 地板从不接管，α 拿到的是真 spread 除出来的值。
        #    未训练的 v 极小（实测 rgb 臂 q05≈8.5e-6、q95≈9.7e-4 ⇒ spread≈9.7e-4），
        #    于是 α = (y_hi−y_lo)/spread ≈ **1.9e4**，窗口 span/α = 9.7e-4 ≈ **1.2× lr**
        #    ⇒ Adam 一步就跨过整个窗口 ⇒ 输出饱和、梯度消失、永久卡死。
        #    症状：loss 仍在缓降（L1 0.47→0.20）**看起来正常**，但抽检 PSNR 在
        #    it 2500/5000/7500 **逐位相同**（5.309 / SSIM 0.5982），且 5.309 恰等于
        #    全白图的 PSNR（同口径：全白 5.309、全黑 3.012、噪声 6.341）。
        #
        # ② 改成 1.0（第一版修法）⇒ 窗口 1250×lr，不再饱和，**但这个值同样是错的**：
        #    窗口**远大于 v 实际能到达的范围**。锚点 `beta = y_lo − alpha·v_lo` 把靶子的
        #    lo 分位钉在 v = v_lo，hi 分位就落到 v = v_lo + 1.0 —— 而 v 实测 1500 步
        #    只长到 **0.37**（G6 单视角过拟合，N2 存档实测）⇒ 模型只能摸到输出区间
        #    最下面的三分之一。N2 实测 u_model ∈ [0.05, 0.32] 而 u_gt ∈ [0.22, 1.00]，
        #    分数（PSNR 3.03）**比最佳单色常数图（11.83 dB）还差** —— 输出成了与靶子
        #    反相的图。N1 同病：α 被夹到 18.4 后需要用 v≈0.5 才能走到靶子中段。
        #
        # ⇒ 窗口 span/α 必须与「v 实际能到达的范围」**匹配**，而它
        #    **既不能太小（饱和）也不能太大（够不到）**。所以它不该是硬编码常数，
        #    必须由调用方按 v 的量级与 lr 显式给出。
        #
        # 锚点性质与窗口取值**无关、始终精确**：v = v_lo 处输出恰为 t_lo。
        spread = (v_hi - v_lo).clamp_min(float(min_v_window))

        if self.end == "sigmoid":
            y_lo, y_hi = torch.logit(t_lo), torch.logit(t_hi)
            # α 用 clamp_min(1e-3) 兜底，只为防 spread 是 inf 时除出 0。
            alpha = ((y_hi - y_lo) / spread).clamp_min(1e-3)
            beta = y_lo - alpha * v_lo
            window = (y_hi - y_lo) / alpha
            sign = "恒正（softplus）"
        else:
            # clamp：把靶子的 [lo, hi] 分位**正好铺到 [0, 1]**，斜率的**符号由数据选**。
            #   a0 = (t_hi − t_lo)/spread > 0
            #   正向  a=+a0, b=t_lo − a0·v_lo   ⇒ v_lo→t_lo、v_hi→t_hi
            #   反向  a=−a0, b=t_hi + a0·v_lo   ⇒ v_lo→t_hi、v_hi→t_lo
            # 为什么不能让先验定符号：「接收功率越大越靠红端」在本数据上**不成立**
            # （实测 r(v, u_gt) = −0.41），而 N1 的 oracle 最优 a_R 就是负的（−1.66）。
            # 无靶子可判时（scalar 臂没传靶子）只能退回正向，并把这个事实写进返回值。
            a0 = (t_hi - t_lo) / spread
            fwd = (a0, t_lo - a0 * v_lo)
            rev = (-a0, t_hi + a0 * v_lo)
            if target_cal is None:
                alpha, beta = fwd
                sign = "正向（**无靶子可判**，退回物理先验）"
            else:
                err = []
                for a_c, b_c in (fwd, rev):
                    u = (a_c * v + b_c).clamp(0.0, 1.0)
                    err.append(float((u - t).abs().mean()))
                take_rev = err[1] < err[0]
                alpha, beta = rev if take_rev else fwd
                sign = ("**反向**（靶子 L1 %.5f < 正向 %.5f）" % (err[1], err[0])
                        if take_rev else
                        "正向（靶子 L1 %.5f ≤ 反向 %.5f）" % (err[0], err[1]))
            window = spread

        if self.end.end == "sigmoid":
            self.end.alpha_raw.copy_(_softplus_inv(alpha))
        else:
            # a = sign · exp(alpha_raw)：把符号单独存，模长取对数（见 EndMap.__init__）
            self.end.alpha_raw.copy_(alpha.abs().clamp_min(1e-12).log())
            self.end.sign.copy_(alpha.sign().masked_fill(alpha == 0, 1.0))
        self.end.beta.copy_(beta)
        return {
            "end": self.end.end,
            "v_q05": [float(x) for x in v_lo], "v_q95": [float(x) for x in v_hi],
            "v_spread": [float(x) for x in (v_hi - v_lo)],
            "min_v_window": float(min_v_window),
            "window": [float(x) for x in window],
            "alpha": [float(x) for x in self.end.alpha],
            "beta": [float(x) for x in beta],
            "sign": sign,
            "targeted": target_cal is not None,
        }

    # ---------------------------------------------------------------- 杂项
    def describe(self) -> str:
        a = ", ".join("%.4g" % float(x) for x in self.end.alpha)
        b = ", ".join("%.4g" % float(x) for x in self.end.beta)
        return "mode=%s end=%s C=%d lut_n=%d alpha=[%s] beta=[%s]" % (
            self.mode, self.end.end, self.n_channels, self.lut_n, a, b)

    def head_cfg(self) -> dict:
        """写进 experiment.json 的输出头配置。LUT 不进 —— 它由 lut_n 确定性重建。

        `end` 这一项是**续跑的安全带**：两种形式的参数名相同（alpha_raw/beta），
        state_dict 能互载而不报错，只有靠它才能发现「拿 sigmoid 的存档去续 clamp」。
        """
        return {"mode": self.mode, "end": self.end.end,
                "n_channels": self.n_channels, "lut_n": self.lut_n,
                "endmap": {"sigmoid": "sigmoid(softplus(a_raw)*v + b)",
                           "clamp": "clamp(a*v + b, 0, 1)",
                           "leaky": "clamp(a*v + b, 0, 1) + %.3g*(a*v + b - clamp)"
                                    % LEAK}[self.end.end],
                "n_params": self.n_params()}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_head(mode: str, lut_n: int = 256, device="cuda",
              end: str = "sigmoid") -> RfHead:
    """构造入口。`n_channels` 直接喂给 `_compat.build_network(n_channels=...)`。"""
    return RfHead(mode=mode, lut_n=lut_n, device=device, end=end)
