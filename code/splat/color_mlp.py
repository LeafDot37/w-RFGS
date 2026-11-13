# -*- coding: utf-8 -*-
"""
splat/color_mlp.py —— RRF 阶段二"embedding + 共享 MLP"颜色头（替代逐点 SH）

设计动机
--------------------------------------------------------------------------------
原版 RRF 把频谱伪彩图编码进"每高斯一份、点间互不共享"的 view-dependent SH 系数
（_features_dc/_features_rest，清零后重训）。本模块提供对照颜色表达：
    每高斯一个可训 embedding e_p (P,D)  +  全场景共享浅层 MLP(e_p, 视线方向 dir)
  → RGB，用于单变量对比"SH 形式是否用不好参数量"。
MLP 输入含 dir ⇒ 保留 SH 的视角相关表达能力，唯一变量 = 颜色表达形式。

实验扩展（LR 调度 × 位置编码，§6 实验设计，全部可选、缺省 = 现状行为）
--------------------------------------------------------------------------------
在默认 cat[e_p, dir] 之上可追加三类条件（对应编码后缀，缺省全部关闭 = 严格复现
现行 run，sidecar 向后兼容）：

  X（PE(xyz)，--pe_xyz_L）：xyz 谐波位置编码，L 个频带 l=0..L-1，每坐标 sin&cos
        ⇒ 6L 维（不含原始 xyz/DC —— 点的绝对身份已由每点 embedding 承担，谐波
        提供的是共享频域基，故 X 对现状是纯附加、单变量干净）。
  D（dir 高频化，--pe_dir_L）：在"始终输入的原始 dir(3)"之上追加 dir 谐波
        ⇒ +6L 维，对 3 维单位 dir 做 Fourier 特征化（SH 角度频率的 MLP 类比）。
  R（距离条件，--use_dist）：追加 1 维 r = ||xyz−cam_center|| / spatial_lr_scale
        （场景尺度归一化，≈O(1)，避免原始距离量级压垮输入）。
  L（LR 调度，train_rrf_mlp 侧）：MLP 组指数衰减（log-linear）或固定，见入口。

谐波公式（本模块统一，原始坐标一律不进谐波段）：
  γ(v, L) = concat_{l=0}^{L-1} [ sin(2^l ·π·v), cos(2^l ·π·v) ]     每坐标 → 2L 维
  输入块布局（构造与 forward 必须同序）：cat[ embed(D), γ_xyz(6L_xyz|缺), dir(3),
  γ_dir(6L_dir|缺), r(1|缺) ]。

与引擎的边界
--------------------------------------------------------------------------------
- 本模块**不 import、不修改** gaussian_renderer / gaussian_model / train.py 任何内核。
- 颜色通过现有 render(..., override_color=colors) → colors_precomp 通道进光栅化
  （见 splat/gaussian_renderer/__init__.py render(): override_color 非空时 shs=None）。
- 输出 sigmoid ∈ (0,1)，与频谱 PNG 归一化 [0,1] 一致（RRF 数据 white_background=False,
  黑底合成）。init 末层零偏 ⇒ 初始颜色 ≈ 0.5 中性灰（对标原 RRF SH 清零后从灰学）。

用法（由入口 train_rrf_mlp.py / render_rrf_mlp.py 调用）
--------------------------------------------------------------------------------
    head = NeuralColorHead(embed_dim=32, hidden=128, depth=2,
                           pe_xyz_L=5, pe_dir_L=3, use_dist=True).cuda()
    colors = compute_colors(head, embedding, xyz, cam_center,
                            spatial_lr_scale=1.6)   # (P,3)
"""
import torch
from torch import nn


def fourier_feat(v: torch.Tensor, L: int) -> torch.Tensor:
    """γ(v, L)：对末维坐标逐元素 append l=0..L-1 的 sin/cos(2^l·π·v)。

    v: (..., 3)（xyz 或单位 dir）→ 返回 (..., 6L)；L<=0 返回 None（不追加）。
    """
    if L is None or L <= 0:
        return None
    cols = []
    for l in range(L):
        a = (2.0 ** l) * torch.pi * v          # 2^l·π·v，(...,3)
        cols.append(a.sin())
        cols.append(a.cos())
    return torch.cat(cols, dim=-1)             # (...,6L)，每频带 6 列 [sin_v0..2, cos_v0..2]


def make_dir(xyz: torch.Tensor, cam_center: torch.Tensor) -> torch.Tensor:
    """复刻 gaussian_renderer/__init__.py:75-77 的视线方向算法：
    dir = normalize(xyz - camera_center)。camera_center 可为 (3,) 或 (1,3)。"""
    if cam_center.dim() == 1:
        cam_center = cam_center.unsqueeze(0)
    d = xyz - cam_center
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _dist(xyz: torch.Tensor, cam_center: torch.Tensor,
          spatial_lr_scale: float) -> torch.Tensor:
    """场景尺度归一化距离 r = ||xyz−cam||/scale（≈O(1)）。"""
    if cam_center.dim() == 1:
        cam_center = cam_center.unsqueeze(0)
    s = float(spatial_lr_scale) if spatial_lr_scale else 1.0
    return (xyz - cam_center).norm(dim=1, keepdim=True) / max(s, 1e-6)


def set_scene_norm(head: "NeuralColorHead", xyz: torch.Tensor):
    """训练时按冻结几何算 xyz-PE 的归一化参数并存入 head（随 cfg 落 sidecar）。

    γ(xyz) 需在"以场景中心为原点、单位半径"坐标上做才有 NeRF 式频率语义
    （l=0 约 1 个周期/场景、l=L-1 约 2^{L-1} 个周期）；直接对世界坐标编码（动辄 ±20）
    会让高频谐波跨场景数百周期、退化成逐点噪声。center/scale 冻结几何下恒定，
    评测重建时从 sidecar 读回同值即可复现同一编码。
    """
    with torch.no_grad():
        x = xyz.detach()
        c = x.mean(0)
        r = (x - c).norm(dim=1).max().item()
    head.xyz_center = tuple(c.tolist())
    head.xyz_scale = float(r)
    return head


class NeuralColorHead(nn.Module):
    """共享颜色解码器：cat[embed, γ_xyz?, dir, γ_dir?, r?] → Linear+ReLU×depth → sigmoid。

    缺省（pe_xyz_L=0, pe_dir_L=0, use_dist=False）时结构与参数量 = 原实现
    cat[e_p, dir] → net → sigmoid，故旧 sidecar 严格兼容。
    参数量小（与逐点 SH 的 P×3×(deg+1)² 相比是全局共享），容量集中于 MLP 权重。
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 128, depth: int = 2,
                 out: int = 3, pe_xyz_L: int = 0, pe_dir_L: int = 0,
                 use_dist: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden = hidden
        self.mlp_depth = depth
        self.pe_xyz_L = int(pe_xyz_L)
        self.pe_dir_L = int(pe_dir_L)
        self.use_dist = bool(use_dist)
        # xyz-PE 归一化参数（训练时 set_scene_norm 填充，随 cfg 进 sidecar）；缺省未设=直接编码
        self.xyz_center = None
        self.xyz_scale = 0.0

        # 输入块维度（布局与 forward 拼接同序）
        in_dim = embed_dim                       # embedding
        if self.pe_xyz_L > 0:
            in_dim += 6 * self.pe_xyz_L          # γ_xyz
        in_dim += 3                              # 原始 dir（始终有，同现状）
        if self.pe_dir_L > 0:
            in_dim += 6 * self.pe_dir_L          # γ_dir
        if self.use_dist:
            in_dim += 1                          # r 距离

        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out)

        # init：末层权重/偏置置零 ⇒ 初始输出 logit=0 ⇒ sigmoid=0.5 中性灰，
        # 对应原 RRF 清零 SH 后从灰学起的语义；中间层用默认 kaiming。
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, embed: torch.Tensor, dirv: torch.Tensor,
                xyz: torch.Tensor = None, dist: torch.Tensor = None) -> torch.Tensor:
        """拼接块序与 __init__ 的 in_dim 计算一致；未启用的条件可不传。"""
        parts = [embed]
        if self.pe_xyz_L > 0:
            if xyz is None:
                raise ValueError("pe_xyz_L>0 但未传入 xyz")
            xin = xyz
            if self.xyz_center is not None and self.xyz_scale > 0:
                cen = torch.as_tensor(self.xyz_center, dtype=xyz.dtype,
                                      device=xyz.device)
                xin = (xyz - cen) / self.xyz_scale      # 场景中心/半径 → 单位球
            parts.append(fourier_feat(xin, self.pe_xyz_L))
        parts.append(dirv)                       # 原始 dir
        if self.pe_dir_L > 0:
            parts.append(fourier_feat(dirv, self.pe_dir_L))
        if self.use_dist:
            if dist is None:
                raise ValueError("use_dist=True 但未传入 dist")
            parts.append(dist)
        x = torch.cat(parts, dim=1)
        x = self.net(x)
        return torch.sigmoid(self.out(x))


def compute_colors(head: NeuralColorHead, embed: torch.Tensor,
                   xyz: torch.Tensor, cam_center: torch.Tensor,
                   spatial_lr_scale: float = None) -> torch.Tensor:
    """对给定相机：全高斯颜色 = MLP(embedding, γ(xyz), dir, γ(dir), r) → (P,3)。

    spatial_lr_scale：距离归一化用场景尺度；use_dist=False 时无需传。训练侧传
    gaussians.spatial_lr_scale，评测侧传 sidecar 里存的同值（渲染时重载后无此量）。
    """
    dirv = make_dir(xyz, cam_center)
    dist = None
    if head.use_dist:
        dist = _dist(xyz, cam_center, spatial_lr_scale)
    return head(embed, dirv, xyz, dist)


def save_color_state(path: str, head: NeuralColorHead, embed: torch.Tensor,
                     iters_done: int, spatial_lr_scale: float = None):
    """写 sidecar 状态（embedding + MLP 权重 + 结构/实验配置），供评测重建。
    与 point_cloud/iteration_<i>/point_cloud.ply 平行存放。embedding 行序 = ply 行序
    （RRF 冻结几何 + densify=0 ⇒ 数量/顺序恒定）。

    结构参数放入顶层 + "cfg"（含新实验字段）。顶层键保持旧命名以便老读取逻辑
    （无 cfg 时按旧语义读）；cfg 是新读取逻辑的一等来源。spatial_lr_scale 在评测侧
    无法从 Scene 重获，必须随档存储（render_rrf_mlp 用它做 dist 归一化）。
    """
    cfg = {
        "embed_dim": head.embed_dim,
        "mlp_hidden": head.mlp_hidden,
        "mlp_depth": head.mlp_depth,
        "pe_xyz_L": head.pe_xyz_L,
        "pe_dir_L": head.pe_dir_L,
        "use_dist": head.use_dist,
        "spatial_lr_scale": float(spatial_lr_scale or 0.0),
        "xyz_center": list(head.xyz_center) if head.xyz_center else None,
        "xyz_scale": float(head.xyz_scale or 0.0),
    }
    torch.save({
        "cfg": cfg,
        # —— 下列为旧版兼容字段（老 color_mlp.py 的 load 只会读它们）——
        "embed_dim": head.embed_dim,
        "mlp_hidden": head.mlp_hidden,
        "mlp_depth": head.mlp_depth,
        "iters_done": iters_done,
        "embedding": embed.detach().cpu(),
        "head": head.state_dict(),
    }, path)


def load_color_state(path: str, device: str = "cuda"):
    """读 sidecar → (head, embedding, spatial_lr_scale)。embedding 行序应对齐 ply。

    新 sidecar：结构/实验配置取 cfg；旧 sidecar（无 cfg）按顶层 embed_dim/hidden/depth
    读取并把 pe_xyz_L/pe_dir_L/use_dist 置默认 0/0/False ⇒ 严格复现旧 run，向后兼容。
    """
    sd = torch.load(path, map_location=device)
    cfg = sd.get("cfg") or {}
    head = NeuralColorHead(
        embed_dim=cfg.get("embed_dim", sd.get("embed_dim") or 32),
        hidden=cfg.get("mlp_hidden", sd.get("mlp_hidden") or 128),
        depth=cfg.get("mlp_depth", sd.get("mlp_depth") or 2),
        pe_xyz_L=cfg.get("pe_xyz_L", 0),
        pe_dir_L=cfg.get("pe_dir_L", 0),
        use_dist=bool(cfg.get("use_dist", False)),
    ).to(device)
    head.load_state_dict(sd["head"])
    embedding = sd["embedding"].to(device)
    scale = float(cfg.get("spatial_lr_scale", 0.0))
    head.xyz_center = cfg.get("xyz_center") or None
    head.xyz_scale = float(cfg.get("xyz_scale", 0.0) or 0.0)
    return head, embedding, scale
