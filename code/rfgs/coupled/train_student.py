"""阶段 2 · 前馈学生 —— 只训一个权重共享的射频头。

设计与判死的支路见 `docs/CORF复现与前馈蒸馏_实施记录.md` §2。

**钉死的东西**（全程不参与梯度）：
  - 几何 `_xyz / _scaling / _rotation`      取种子（30 000 步纯光学模型）
  - 光学外观 `_opacity / _features*`        取种子
**唯一可训的**：`RFHead` 的 MLP 权重。它看「点在哪 + 这点长什么样」，
吐出逐高斯的 `{o^rf, c^rf}`（球谐 16 系数 × 3 通道 + 1 不透明度，与老师逐字同参化）。

点序与阶段 1 的三条臂**逐个相同**（阶段 1 全程不增密不剪枝，
`config.yaml:44` `densify_until_iter: 0`）⇒ 与老师**同序号**直接回归，不需要最近邻。

两类实验（见文档 §2.1）：

  甲 · 可抄性   靶子 = 老师的逐高斯属性（`--teacher_ckpt`），用全部频谱视角渲染损失
  乙 · 样本效率 靶子 = 只用 N 个频谱视角的渲染损失（`--n_keyframes`），**不接属性靶子**

用法示例（乙实验，N = 100，有视觉耦合）：
    python train_student.py -m <out> --feature_cache <cache.pt> \\
        --rf_source_path <spec_dir> --n_keyframes 100 --use_visual
"""

from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corf.dual_scene import load_rf_cameras                      # noqa: E402
from corf.rf_head import RFHead, RF_SH_COEFFS, RF_SH_C, xyz_norm_stats   # noqa: E402
from splat.gaussian_renderer import render                       # noqa: E402
from splat.scene.gaussian_model import GaussianModel             # noqa: E402
from splat.utils.general_utils import safe_state                 # noqa: E402
from splat.utils.loss_utils import l1_loss, ssim                 # noqa: E402

C0 = 0.28209479177387814


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def rgb_loss(image, gt, lambda_dssim):
    ll1 = l1_loss(image, gt)
    return (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim(image, gt)), ll1


# ---------------------------------------------------------------------------
# 鸭子类型视图：渲染器只认这几个接口，引擎核心一行不用改
# ---------------------------------------------------------------------------

class StudentView:
    """几何/光学直接转发种子（无梯度）；射频那一半由头每步写进来。"""

    def __init__(self, gm: GaussianModel):
        self.gm = gm
        self._op_rf = None
        self._sh_rf = None

    def set_rf(self, op_raw: torch.Tensor, sh_raw: torch.Tensor) -> None:
        """`op_raw` [P,1] 激活前；`sh_raw` [P, 16, 3]。与老师 sidecar 同空间。"""
        self._op_rf = op_raw
        self._sh_rf = sh_raw

    def set_rf_scalar(self, op_raw: torch.Tensor) -> None:
        """标量头模式：只写不透明度；颜色走 `render(..., override_color=...)`，
        不再读 `get_features`（渲染器只在 override_color is None 时取 SH）。"""
        self._op_rf = op_raw
        self._sh_rf = None

    # —— 几何/光学：转发 ——
    # ⚠ 下面三个**必须是 @property**：渲染器取它们时**不带括号**
    # （`splat/gaussian_renderer/__init__.py:26,53,65,66` 写的是 `pc.get_xyz`），
    # 写成普通方法会把「绑定方法对象」当张量喂进光栅化器。
    # `get_covariance` 相反，是按方法调的（`:63` `pc.get_covariance(scaling_modifier)`），保持方法。
    # 对着 `corf/coupled_model.py:93-104` 的 ModalityView 抄，别自己发明。
    @property
    def get_xyz(self):
        return self.gm.get_xyz

    @property
    def get_scaling(self):
        return self.gm.get_scaling

    @property
    def get_rotation(self):
        return self.gm.get_rotation

    def get_covariance(self, scaling_modifier=1.0):
        return self.gm.get_covariance(scaling_modifier)

    @property
    def active_sh_degree(self):
        return self.gm.active_sh_degree

    @property
    def max_sh_degree(self):
        return self.gm.max_sh_degree

    # —— 射频外观 ——
    @property
    def get_opacity(self):
        if self._op_rf is None:
            raise RuntimeError("射频外观还没写进来：先调 set_rf()")
        return torch.sigmoid(self._op_rf)

    @property
    def get_features(self):
        if self._sh_rf is None:
            raise RuntimeError("射频外观还没写进来：先调 set_rf()")
        return self._sh_rf


class _Pipe:
    """`PipelineParams` 的鸭子类型替身。值取自 `splat/arguments/__init__.py:65-68` 的默认。

    三个字段都是渲染器**必读**的（`gaussian_renderer/__init__.py:48,62,73`），少一个第一次
    渲染就 AttributeError。阶段 1 走的是真 `PipelineParams`（`render_corf.py:122`），
    本脚本没有完整 cfg 解析，才用替身 —— **两者的取值必须一致**，否则口径分叉。
    """
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


def load_seed(ckpt_path, gaussians, device):
    """解包顺序与 `train_rrf_scalar.py:76-78` 逐字相同。返回绝对起始迭代。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_params, first_iter = ckpt
    (active_sh_degree, xyz, f_dc, f_rest, scaling, rotation, opacity,
     _max_radii2D, _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params
    gaussians.active_sh_degree = int(active_sh_degree)
    gaussians._xyz = torch.nn.Parameter(xyz.detach().contiguous().to(device), requires_grad=False)
    gaussians._scaling = torch.nn.Parameter(scaling.detach().contiguous().to(device), requires_grad=False)
    gaussians._rotation = torch.nn.Parameter(rotation.detach().contiguous().to(device), requires_grad=False)
    gaussians._features_dc = torch.nn.Parameter(f_dc.detach().contiguous().to(device), requires_grad=False)
    gaussians._features_rest = torch.nn.Parameter(f_rest.detach().contiguous().to(device), requires_grad=False)
    gaussians._opacity = torch.nn.Parameter(opacity.detach().contiguous().to(device), requires_grad=False)
    gaussians.spatial_lr_scale = _spatial_lr_scale if _spatial_lr_scale is not None else 0
    # ⚠ `GaussianModel` **不是 nn.Module**（`splat/scene/gaussian_model.py:24` 是普通类），
    # 没有 `.parameters()` —— 用本仓自己的写法：按属性名遍历（`train_corf.py:71-74`）。
    # 上面六个 Parameter 本来就以 requires_grad=False 建的，这里再钉一次是**显式兜底**
    # （别处若改成可训的，这一行保证本阶段仍是全冻结）。
    for n in ("_xyz", "_scaling", "_rotation", "_features_dc", "_features_rest", "_opacity"):
        getattr(gaussians, n).requires_grad = False
    return int(first_iter)


def pick_keyframes(n_cams: int, n_key: int) -> list[int]:
    """按视角**均匀分层**抽 N 个（文档 §2.6 第 5 项）。

    随机抽在 N 小时会抽出位姿簇 —— 射频相机是环形轨迹，位置相邻的相机看到的场高度相关。
    均匀分层保证抽出来的 N 个在轨迹上铺得开。
    """
    if n_key <= 0 or n_key >= n_cams:
        return list(range(n_cams))
    if n_key == 1:
        return [0]
    return sorted({round(i * (n_cams - 1) / (n_key - 1)) for i in range(n_key)})


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True, help="输出目录")
    ap.add_argument("--feature_cache", required=True, help="make_student_features.py 的产物")
    ap.add_argument("--rf_source_path", required=True, help="频谱数据集目录")
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--teacher_ckpt", default="",
                    help="老师 sidecar（coupled_rf_*.pth）。给了才算甲实验的属性蒸馏项")
    ap.add_argument("--use_visual", action="store_true",
                    help="有视觉耦合臂。**不给就是无视觉耦合臂**（本阶段两条线唯一变量）")
    ap.add_argument("--shuffle_vis", action="store_true",
                    help="阴性对照：视觉特征打乱后仍喂进去（特征在、只是与高斯错配）")
    ap.add_argument("--n_keyframes", type=int, default=0, help="0 = 全部频谱视角")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--pe_bands", type=int, default=6,
                    help="位置编码的频带数，0 = 关（裸 xyz）。见 corf/rf_head.py:PositionalEncoding。"
                         "**两条臂必须用同一个值**，否则单一变量失效")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--w_attr", type=float, default=0.0, help="属性蒸馏权重；0 = 关（乙实验）")
    ap.add_argument("--attr_scale", choices=("rms", "none"), default="rms",
                    help="属性蒸馏三项的尺度处理。rms = 各除以靶子均方（= 零预测那条基线的 MSE，"
                         "默认，头零初始化时三项起手各为 1）；none = 原始 MSE（旧行为）。"
                         "见文档 §2.11 记的那个两数量级差")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_iterations", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--white_background", action="store_true",
                    help="必须与阶段 1 的 dataset.white_background 一致（现为 False）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0:
        raise SystemExit(
            "--iterations 必须 > 0。本参数是**头的训练步数**（头全新初始化、无绝对迭代语义），"
            "给 0 会空循环后照样存 head_0.pth 并报成功 —— 本仓明令防这种「成功退出、什么都没训」")

    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.model_path, exist_ok=True)

    if args.shuffle_vis and not args.use_visual:
        raise SystemExit("--shuffle_vis 是阴性对照，必须同时给 --use_visual")

    # —— 特征缓存 ——
    cache = torch.load(args.feature_cache, map_location="cpu")
    xyz = cache["xyz"].cuda().contiguous()
    seen = cache["seen_mask"]
    p = xyz.shape[0]
    vis = cache["vis"].float().cuda().contiguous() if args.use_visual else None
    if args.use_visual and args.shuffle_vis:
        g = torch.Generator(device="cpu").manual_seed(12345)
        vis = vis[torch.randperm(p, generator=g).to(vis.device)].contiguous()
        log("[student] **阴性对照**：视觉特征已打乱（与高斯错配）")

    # —— 种子（几何 + 光学，全冻结）——
    gaussians = GaussianModel(3)
    first_iter = load_seed(cache["meta"]["ckpt"], gaussians, "cuda")
    if gaussians.get_xyz.shape[0] != p:
        raise SystemExit(f"点数不一致：缓存 {p} vs 种子 {gaussians.get_xyz.shape[0]}")

    # —— 频谱相机 + 关键帧 ——
    cams, _ = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                              args.rf_data_device, verbose=not args.quiet)
    key_idx = pick_keyframes(len(cams), args.n_keyframes)
    keys = [cams[i] for i in key_idx]
    # 背景色：与阶段 1 同源（`train_corf.py:210` / `render_corf.py:95` 都读 dataset.white_background）
    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # —— 老师靶子（甲实验）——
    # 三项靶子的尺度处理。实测（§2.11）三项 MSE 之和起手就是 32.7 而渲染项只有约 0.34 ——
    # **差两个数量级** ⇒ 默认的 w_attr 其实由属性项主导，报出来的「可抄性」量的是权重不是可抄性。
    # `var` 模式把每一项除以该靶子自己的方差（detach 的常量），三项起手都在 1 附近，
    # w_attr 才真的是个可解释的权衡系数。
    teacher = None
    t_scale = None
    if args.teacher_ckpt:
        t = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher = {k: t[k].cuda().float().reshape(p, *t[k].shape[1:])
                   for k in ("opacity_rf", "features_rf_dc", "features_rf_rest")}
        if args.attr_scale == "rms":
            # 除以**靶子均方**，不是方差：头是零初始化的，零预测下
            #   mse(0, t) / mean(t²) = 1
            # ⇒ 三项起手各为 1，attr 的初值恒为 3，w_attr 才是个看得懂的系数。
            # （用方差归一的话，靶子均值一大就会得到 1 + mean²/var > 1，实测是 4.09 不是 3。）
            t_scale = {k: v.detach().pow(2).mean().clamp_min(1e-12) for k, v in teacher.items()}
            log("[student] 属性蒸馏用**均方归一**的尺度（零预测基线 = 1）："
                + "  ".join(f"{k}={float(v):.4g}" for k, v in t_scale.items()))
        if args.w_attr <= 0:
            log("[student] 警告：给了 --teacher_ckpt 但 --w_attr <= 0 ⇒ 靶子载了不用")
    if args.w_attr > 0 and teacher is None:
        raise SystemExit("--w_attr > 0 但没有 --teacher_ckpt")

    # 位置编码的归一化统计量：从缓存里的 xyz 算一次。渲染端会用同一个张量算出同一组值
    # ⇒ 两端尺度一致（这条一旦不一致，位置编码就变成了「训练时看不见的尺度漂移」）。
    pe_mean, pe_std = xyz_norm_stats(xyz)
    head = RFHead(use_visual=args.use_visual, hidden=args.hidden,
                  n_layers=args.n_layers, chunk=args.chunk,
                  pe_bands=args.pe_bands, xyz_mean=pe_mean, xyz_std=pe_std).cuda()
    n_param = sum(q.numel() for q in head.parameters())
    view = StudentView(gaussians)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-15)

    cfg = {**vars(args), "n_points": p, "first_iter_seed": first_iter,
           "n_rf_cameras": len(cams), "n_keyframes_used": len(keys),
           "key_idx": key_idx, "head_params": n_param,
           # 位置编码的归一化统计量也落盘（头里是 buffer，这里再存一份明文），
           # 好让渲染端能**对拍**而不是默默用一个不同的尺度。
           "pe_mean": [float(v) for v in pe_mean], "pe_std": [float(v) for v in pe_std]}
    with open(os.path.join(args.model_path, "student_experiment.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    log(f"[student] P={p}  头参数量={n_param}  use_visual={args.use_visual}"
        f"{' (打乱)' if args.shuffle_vis else ''}  w_attr={args.w_attr}")
    log(f"[student] 频谱相机 {len(cams)}，本实验用 {len(keys)} 个关键帧"
        f"（N={'全部' if args.n_keyframes <= 0 else args.n_keyframes}）")
    if args.use_visual and seen is not None:
        log(f"[student] 视觉支路覆盖 {float(seen.float().mean()) * 100:.1f}% 的点"
            f"（⚠ 这是**视锥**口径，不做遮挡剔除）")

    t0 = time.time()
    ema = None
    for it in range(1, args.iterations + 1):
        op_raw, sh_raw = head(xyz, vis)
        view.set_rf(op_raw, sh_raw)

        cam = keys[it % len(keys)] if len(keys) > 1 else keys[0]
        pkg = render(cam, view, _Pipe(), background)
        L, l1v = rgb_loss(pkg["render"], cam.original_image.cuda(), args.lambda_dssim)

        attr_v = 0.0
        if args.w_attr > 0:
            dc, rest = sh_raw[:, :1, :], sh_raw[:, 1:, :]
            pairs = ((op_raw, "opacity_rf"), (dc, "features_rf_dc"), (rest, "features_rf_rest"))
            la = 0.0
            for pred, key in pairs:
                term = F.mse_loss(pred, teacher[key])
                if t_scale is not None:
                    term = term / t_scale[key]
                la = la + term
            L = L + args.w_attr * la
            attr_v = float(la.item())

        L.backward()
        with torch.no_grad():
            opt.step()
        opt.zero_grad(set_to_none=False)

        ema = float(l1v.item()) if ema is None else 0.99 * ema + 0.01 * float(l1v.item())
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            msg = (f"[student] {it:6d}/{args.iterations}  L1={float(l1v.item()):.5f} "
                   f"ema={ema:.5f}")
            if args.w_attr > 0:
                msg += f"  attr={attr_v:.5f}"
            msg += f"  {el / it:.2f}s/it  剩 {(args.iterations - it) * el / it / 60:.0f} 分"
            log(msg)

        if it in args.save_iterations:
            torch.save({"iteration": it, "head": head.state_dict(), "config": cfg},
                       os.path.join(args.model_path, f"head_{it}.pth"))
            log(f"[student] 存头 -> head_{it}.pth")

    torch.save({"iteration": args.iterations, "head": head.state_dict(), "config": cfg},
               os.path.join(args.model_path, f"head_{args.iterations}.pth"))
    log(f"[student] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
