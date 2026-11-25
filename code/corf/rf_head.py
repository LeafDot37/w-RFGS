"""前馈射频头 —— 阶段 2 的「学生」。

设计与判死的支路见 `docs/CORF复现与前馈蒸馏_实施记录.md` §2。

**几何与光学外观全程钉死**（取 30 000 步纯光学种子，从未接触射频梯度），
学生只学一件事：把「每个高斯 + 它长什么样」映射成逐高斯射频外观 `{o^rf, c^rf}`。
参数化与老师**逐字相同**（球谐 16 系数 × 3 通道 + 1 不透明度），
且点序与阶段 1 三条臂逐个相同 ⇒ 可与老师**同序号**直接回归。

输入只有两支：

    位置支路   x_i                                (3)   永远有
    视觉支路   v_i                                (12)  「有视觉耦合」臂才有

⚠ **没有「频谱关键帧」支路。** 原设计想把每个高斯投到频谱关键帧上采样，
**已判死**：频谱图的像素与投影方向之间没有可验证的几何约束
（`docs/RF前向模型规格.md:149` 的既有实测 + `code/tools/diag/probe_projection.py` 本次新写的闸，
两者独立同向）。同一条闸在**光学相机**上是 PASS 的（A=+0.583 ≫ B=+0.235 ≫ C=+0.108）
⇒ 投影数学没错，是频谱数据本身没有这个关系。

「关键帧数 N」不是头的输入，而是**监督的视角数**（见文档 §2.3 的表）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from splat.utils.graphics_utils import fov2focal

# 射频外观的通道定义，必须与 CoupledGaussianModel 一致
RF_SH_COEFFS = 16          # (3+1)^2，球谐 0..3 阶
RF_SH_C = 3                # 频谱伪彩图的三个通道

# 视觉支路聚合后的通道数（改这里要同步改预计算脚本）
VIS_CHANNELS = 12          # mean rgb(3) + max rgb(3) + std rgb(3) + 该点自己的光学 DC(3)


class PositionalEncoding(nn.Module):
    """NeRF 式位置编码：`x -> [x, sin(2⁰πx), cos(2⁰πx), …, sin(2^(L-1)πx), cos(2^(L-1)πx)]`。

    输入须先归一化（见 `RFHead.forward`）。每轴输出 `1 + 2L` 维，三轴合计 `3 + 6L`。

    **为什么要有它**：位置支路原本是裸的 3 维坐标。频域上，一个 MLP 对裸坐标的
    高频表达能力很弱 —— 这一项若不打开，「无视觉」那条线可能**根本拟合不动**，
    于是两条线都差、比较变得没有信息量。打开它，是把**对照组做强**，
    而不是给实验组加分（两条臂拿到的设置逐字相同）。
    """

    def __init__(self, n_bands: int):
        super().__init__()
        self.n_bands = int(n_bands)
        self.out_dim = 3 + 6 * self.n_bands
        # 频率表是常量，注册成 buffer ⇒ 跟着 state_dict 走，训练与渲染两端逐位一致
        self.register_buffer("freqs", torch.tensor(
            [2.0 ** i * math.pi for i in range(self.n_bands)], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [P,3] 已归一化
        s = x.unsqueeze(-1) * self.freqs.to(x.dtype)         # [P,3,L]
        return torch.cat([x, s.sin().flatten(-2), s.cos().flatten(-2)], dim=-1)

    def extra_repr(self) -> str:
        return f"n_bands={self.n_bands} out_dim={self.out_dim}"


class RFHead(nn.Module):
    """逐高斯的射频外观头。MLP。

    ⚠ 关于分块：`forward` 按 `chunk` 切块**只压瞬时峰值，不压反向要保存的量**。
    各块经 `torch.cat` 连成一张图 ⇒ ReLU 的保存张量在整个前向期间都不得释放，
    累计仍是 2·P·hidden·4 字节。P=1.01e6、hidden=256 时约 2.08 GB，这才是显存瓶颈；
    **调大 `chunk` 只会更糟，调小几乎无益**。真要压得改「前向-反向-累积」，
    但本头的输出要喂下游光栅化器（损失在渲染之后才产生），做不到逐块反传。
    """

    def __init__(self, use_visual: bool = True, hidden: int = 256, n_layers: int = 3,
                 sh_coeffs: int = RF_SH_COEFFS, sh_c: int = RF_SH_C, chunk: int = 200_000,
                 pe_bands: int = 6, xyz_mean: torch.Tensor | None = None,
                 xyz_std: torch.Tensor | None = None):
        super().__init__()
        self.use_visual = bool(use_visual)
        self.sh_coeffs = int(sh_coeffs)
        self.sh_c = int(sh_c)
        self.chunk = int(chunk)
        self.pe_bands = int(pe_bands)

        # 位置支路：`pe_bands > 0` 时先归一化再做位置编码，否则就是裸 xyz（3 维）
        self.pe = PositionalEncoding(self.pe_bands) if self.pe_bands > 0 else None
        pos_dim = self.pe.out_dim if self.pe is not None else 3

        # 归一化统计量：**显式传入**（两个入口脚本都拿得到 `cache["xyz"]`，
        # 用 `xyz_norm_stats()` 算一次即可），并注册成 buffer 跟着 state_dict 走。
        #
        # ⚠ 试过「首次前向时惰性推定 + register_buffer(name, None)」的写法，**不行**：
        #   `load_state_dict(strict=True)` 面对 None 注册的 buffer 会抛
        #   `Unexpected key(s) in state_dict`（实测），于是渲染端在载入头时就崩。
        #   所以这里要求显式传入，pe_bands>0 而没给就直接报错，不静默地用错统计量。
        if self.pe is not None:
            if xyz_mean is None or xyz_std is None:
                raise ValueError(
                    "pe_bands > 0 时必须显式传入 xyz_mean / xyz_std"
                    "（用 corf.rf_head.xyz_norm_stats(xyz) 从缓存的 xyz 算一次）。"
                    "训练与渲染两端必须用**同一批**统计量，否则位置编码的尺度两边不同。")
            self.register_buffer("_xyz_mean", torch.as_tensor(xyz_mean).float().reshape(3).clone())
            self.register_buffer("_xyz_std",
                                 torch.as_tensor(xyz_std).float().reshape(3).clamp_min(1e-6).clone())

        c_in = pos_dim + (VIS_CHANNELS if self.use_visual else 0)
        dims = [c_in] + [hidden] * (n_layers - 1)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], self.sh_coeffs * self.sh_c + 1)   # +1 = 不透明度

        # 末层零初始化 ⇒ 起点是「射频外观全为 0」，与老师侧 k 初始化的起点一致，
        # 读数可直接对照（不会出现「学生起点就比老师终点好」这种口径错位）。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xyz: torch.Tensor,
                vis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (opacity_rf_logit [P,1], features_rf_raw [P, sh_coeffs, sh_c])。

        两者都是**激活前**的原始参数，与老师 sidecar 里存的量同空间，可直接回归。
        """
        pos = xyz
        if self.pe is not None:
            pos = self.pe((xyz - self._xyz_mean) / self._xyz_std)

        if self.use_visual:
            if vis is None:
                raise ValueError("use_visual=True 但没给 vis 特征")
            x = torch.cat([pos, vis], dim=-1)
        else:
            if vis is not None:
                raise ValueError(
                    "use_visual=False 却给了 vis 特征 —— 这会变成第三条臂，不是本阶段的两线口径")
            x = pos

        p = x.shape[0]
        if p <= self.chunk:
            out = self.head(self.trunk(x))
        else:
            # 分块累积：head 是逐点算子，分块与整块**逐位等价**
            outs = []
            for s in range(0, p, self.chunk):
                outs.append(self.head(self.trunk(x[s:s + self.chunk])))
            out = torch.cat(outs, dim=0)

        op = out[:, :1]
        sh = out[:, 1:].reshape(p, self.sh_coeffs, self.sh_c)
        return op, sh

    def extra_repr(self) -> str:
        nl = sum(1 for m in self.trunk if isinstance(m, nn.Linear))
        return (f"use_visual={self.use_visual} pe_bands={self.pe_bands} "
                f"c_in={self.trunk[0].in_features} "
                f"hidden={self.trunk[0].out_features} layers={nl} chunk={self.chunk}")


def xyz_norm_stats(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """位置编码的归一化统计量：逐轴均值与标准差。

    **必须从缓存的 xyz 算**（`train_student.py` / `render_student.py` 都持有同一个张量），
    不能在头内部惰性推定 —— 见 `RFHead.__init__` 里记的那个 `load_state_dict` 坑。
    返回的是 [3] 的 CPU 张量，与输入是否在 GPU 上无关。
    """
    x = xyz.detach().float()
    return x.mean(dim=0).cpu(), x.std(dim=0).cpu().clamp_min(1e-6)


# ---------------------------------------------------------------------------
# 投影采样 + 聚合（一次性预计算用；不进训练循环）
# ---------------------------------------------------------------------------

def project_and_sample(points: torch.Tensor, cams, images_feat: torch.Tensor,
                       min_depth: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """把点投到一组相机里，双线性采样图像特征。

    参数
      points      [P, 3]   世界坐标
      cams        `splat.scene.cameras.Camera` 列表（由 `load_rf_cameras` / `Scene` 装配）
      images_feat [V, C, h, w]  float，与 cams 同序

    返回
      sampled [P, V, C]  相机外的点填 0
      visible [P, V]     bool

    相机约定（**已用光学相机对拍验证，见 `code/tools/diag/probe_projection.py`**）：

    - `Camera.R` / `Camera.T` 就是喂给 `getWorld2View2(R, T)` 的那两个量
      （`splat/utils/graphics_utils.py:38-49` 里 `Rt[:3,:3] = R.T`、`Rt[:3,3] = T`），
      即 **R 是 camera-to-world**，世界点变相机坐标： `x_cam = Rᵀ·x + T`。
    - `Camera` **没有** `fx/fy/cx/cy` 属性，内参只能从视场角反推：
      `fx = fov2focal(FoVx, W)`（引擎自带，`graphics_utils.py:78`），主点取图像中心。
    - 光栅化器的 NDC→像素是 `(ndc + 1)·S/2 - 0.5`，x/y 同号不翻转
      ⇒ 相机 +y 指向图像**下方**（OpenCV 口径），与 `P[3,2] = +1`（+z 朝前）自洽。

    实测判别力（光学相机 4 个，199 140 个 opacity > 0.5 的点）：
    A 正确 r=+0.583、B y 翻转 r=+0.235、C 主点偏 20% r=+0.108。
    """
    v = len(cams)
    p = points.shape[0]
    c = images_feat.shape[1]
    out = points.new_zeros(p, v, c)
    vis = points.new_zeros(p, v, dtype=torch.bool)

    for k, cam in enumerate(cams):
        # trans / scale 非默认时 R/T 不再是完整变换；本管线两者都是默认值，先钉住
        if abs(float(cam.scale) - 1.0) > 1e-9 or float(torch.as_tensor(cam.trans).abs().sum()) > 1e-9:
            raise NotImplementedError(
                "本函数只支持 trans=0 / scale=1 的相机；该数据集不满足，需改用 world_view_transform")

        r = torch.as_tensor(cam.R, dtype=torch.float32, device=points.device)
        t = torch.as_tensor(cam.T, dtype=torch.float32, device=points.device)
        pc = points @ r + t                                   # [P,3]

        w = int(cam.image_width)
        h = int(cam.image_height)
        f = images_feat[k].unsqueeze(0)                        # [1,C,h',w']
        fh, fw = f.shape[-2], f.shape[-1]

        fx = fov2focal(float(cam.FoVx), w)
        fy = fov2focal(float(cam.FoVy), h)
        z = pc[:, 2].clamp_min(min_depth)

        # 先算「原图连续像素坐标」（光栅化器口径：`(ndc+1)·S/2 - 0.5`，中心在 S/2-0.5），
        # 再按特征图尺寸重标定。特征图与原图同视场角 ⇒ 把「原图中心」映到「特征图中心」：
        #     u_feat = (u_orig - (w/2 - 0.5))·(fw/w) + (fw/2 - 0.5)
        #            = u_orig·(fw/w) + (fw/(2w) - 0.5)
        # fw == w 时那个加数正好是 0（预计算走的死支路），fw ≠ w 时**不能省**，
        # 否则整体平移半个特征图像素。
        u_orig = fx * pc[:, 0] / z + 0.5 * w - 0.5
        v_orig = fy * pc[:, 1] / z + 0.5 * h - 0.5
        uu = u_orig * (fw / w) + (0.5 * fw / w - 0.5)
        vv = v_orig * (fh / h) + (0.5 * fh / h - 0.5)

        inside = (pc[:, 2] > min_depth) & (uu >= 0) & (uu <= fw - 1) & (vv >= 0) & (vv <= fh - 1)
        vis[:, k] = inside

        if inside.any():
            idx = inside.nonzero(as_tuple=True)[0]
            gx = (uu[idx] / (fw - 1) * 2 - 1).view(1, -1, 1)
            gy = (vv[idx] / (fh - 1) * 2 - 1).view(1, -1, 1)
            grid = torch.cat([gx, gy], dim=-1).unsqueeze(0)   # [1,1,n,2]
            s = torch.nn.functional.grid_sample(
                f, grid, mode="bilinear", padding_mode="border", align_corners=True)
            # grid_sample 的形状是 [1, C, 1, n]（网格是 1×n 的假空间维）
            out[idx, k] = s[0, :, 0, :].t()                   # [n, C]

    return out, vis


def visual_stats(sampled: torch.Tensor, visible: torch.Tensor,
                 opt_dc: torch.Tensor) -> torch.Tensor:
    """把 [P,V,3] 的光学采样聚合成视觉支路的 [P, VIS_CHANNELS]。

    参数 `opt_dc` 的名字是历史遗留：它并不是**原始**球谐 DC 系数，而是
    **激活之后的光学 RGB**，即 `(f_dc·C0 + 0.5).clamp(0,1)`（见
    `tools/make_student_features.py:44`）。钉死不变、无梯度。
    两种叫法在数值上差一个仿射变换，训练/评测共用同一份缓存 ⇒ 功能上无害，
    但**重新生成特征的人别把它当原始系数**。
    """
    m = visible.float()
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    x = sampled * m.unsqueeze(-1)

    mean_rgb = x.sum(dim=1) / denom
    mx = sampled.masked_fill(~visible.unsqueeze(-1), float("-inf")).amax(dim=1)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    var = ((sampled - mean_rgb.unsqueeze(1)) ** 2 * m.unsqueeze(-1)).sum(dim=1) / denom
    std = var.clamp_min(0).sqrt()

    return torch.cat([mean_rgb, mx, std, opt_dc], dim=-1)
