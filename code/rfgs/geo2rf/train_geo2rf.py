# -*- coding: utf-8 -*-
"""前馈「光学点云 → 射频点云」：从一份**已优化**的光学 3DGS 点云出发，
逐点预测射频不透明度与射频球谐。

与 `train_coupled_student.py` 的区别
--------------------------------------------------------------------------------
- 几何**不是** MVSplat 原始前馈点云，而是「MVSplat 初始化 + 光学 3DGS 短训」
  产出的点云（`--cloud_ply`）。位置/尺度/旋转/不透明度/光学 SH 全部**作为输入**，
  全部冻结。
- 只训**一个**射频头，不训光学头：光学外观是输入，不是输出。
- 损失只有射频渲染项（对该模态 GT 图像的 L1 + DSSIM）；没有老师，
  没有属性蒸馏（老师的点序与这份点云不同，逐点回归需要最近邻匹配，
  而 GT 渲染损失本来就是更强的信号）。

「前馈」的口径
--------------------------------------------------------------------------------
推理时：一份光学 3DGS 点云进，一次前向，逐点射频属性出，零逐场景优化。
训练时：只需要**一个场景**的射频测量（本数据集 640 个训练位置）。
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

from corf.dual_scene import load_rf_cameras                 # noqa: E402
from corf.geo_head import APP_DIM, GeoToRfHead, app_stats, build_app  # noqa: E402
from corf.rf_head import xyz_norm_stats                     # noqa: E402
from splat.color_mlp import _dist, make_dir                 # noqa: E402
from splat.color_scalar import (apply_jet, combine_bands,  # noqa: E402
                                make_jet_lut, to_override_color)
from splat.gaussian_renderer import render                  # noqa: E402
from splat.scene.gaussian_model import GaussianModel        # noqa: E402
from splat.utils.general_utils import safe_state            # noqa: E402
from rfgs.coupled.train_student import StudentView, _Pipe, rgb_loss      # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def freeze(gm: GaussianModel) -> None:
    """六个 Parameter 全部钉死（`GaussianModel` 不是 nn.Module，没有 .parameters()）。"""
    for n in ("_xyz", "_scaling", "_rotation", "_features_dc", "_features_rest", "_opacity"):
        getattr(gm, n).requires_grad = False


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--cloud_ply", required=True,
                    help="光学 3DGS 点云（标准 62 属性 ply，SH 3 阶）")
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--use_app", type=int, default=1,
                    help="1 = 位置 + 56 维光学属性；0 = 只用位置（阴性对照）")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--pe_bands", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--sh_bound", type=float, default=2.0,
                    help="球谐输出的 tanh 上界；<=0 表示不设界（无界版本实测会发散）")
    ap.add_argument("--out_kind", choices=("sh", "scalar"), default="sh",
                    help="sh = 逐点球谐（原臂）；scalar = 逐点标量、标量域混合后查 jet 表")
    ap.add_argument("--n_bands", type=int, default=1,
                    help="scalar 模式的标量带数：1 = 形态 A'，2 = 形态 A×（乘性调制）")
    ap.add_argument("--init_u", type=float, default=0.5,
                    help="scalar 模式的标量初值（jet(0.5) ≈ 亮绿，在调色板上）")
    ap.add_argument("--use_dir", type=int, default=0,
                    help="1 = 头额外吃单位视线方向 dir 与 γ(dir)（逐场景标量头就是这么做的）")
    ap.add_argument("--pe_dir_bands", type=int, default=0,
                    help="γ(dir) 的频带数 L（0 = 只用裸方向）")
    ap.add_argument("--use_dist", type=int, default=0,
                    help="1 = 额外吃场景尺度归一化距离 r（需 use_dir=1）")
    ap.add_argument("--scene_scale", type=float, default=0.0,
                    help="r 的归一化尺度；<=0 表示按点云到质心的最大距离自动估")
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--teacher_attr", default="",
                    help="tools/make_teacher_match.py 的产物（老师属性最近邻搬到本点云）")
    ap.add_argument("--lambda_attr", type=float, default=0.0)
    ap.add_argument("--attr_op_clip", type=float, default=8.0,
                    help="老师不透明度 logit 的目标裁剪（老师有 -17 这种退化点）")
    ap.add_argument("--opacity_init", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_iterations", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations 必须 > 0（本仓明令防「成功退出、什么都没训」）")
    if not os.path.isfile(args.cloud_ply):
        raise SystemExit(f"点云不存在：{args.cloud_ply}")

    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.model_path, exist_ok=True)

    # ---------------------------------------------------------------- 几何
    t0 = time.time()
    gm = GaussianModel(3)
    gm.load_ply(args.cloud_ply)
    freeze(gm)
    p = gm.get_xyz.shape[0]
    if p <= 0:
        raise SystemExit("点云点数为 0")
    log(f"[g2r] 点云 {os.path.basename(args.cloud_ply)}  P={p}  "
        f"（{time.time() - t0:.0f}s）")

    xyz = gm.get_xyz.detach().float().contiguous()
    # 距离特征的尺度：冻结几何下恒定，训练/渲染两端必须用同一个值（写进 cfg）
    scene_scale = float(args.scene_scale)
    if scene_scale <= 0:
        scene_scale = float((xyz - xyz.mean(dim=0)).norm(dim=1).max())
    app = build_app(gm) if args.use_app else None
    app_dim = APP_DIM if args.use_app else 0
    pe_mean, pe_std = xyz_norm_stats(xyz)
    if app is not None:
        a_mean, a_std = app_stats(app)
        log(f"[g2r] 光学属性 {tuple(app.shape)}  逐维 std 中位 {float(a_std.median()):.4g}  "
            f"min {float(a_std.min()):.4g}（≈0 说明该维全常数，等于没给）")
    else:
        a_mean = a_std = None

    # ---------------------------------------------------------------- 老师属性
    teacher = None
    t_scale = None
    t_mask = None
    if args.lambda_attr > 0 and args.out_kind != "sh":
        raise SystemExit("老师属性蒸馏只对 SH 头有意义（老师存的是 SH 系数），"
                         "scalar 头请勿开 --lambda_attr")
    if args.teacher_attr and args.lambda_attr > 0:
        ta = torch.load(args.teacher_attr, map_location="cpu")
        if int(ta["meta"]["student_points"]) != p:
            raise SystemExit(f"老师匹配文件的点云 {ta['meta']['student_points']} "
                             f"!= 当前点云 {p} —— 不是同一份点云")
        t_mask = ta["mask"].bool().cuda()
        teacher = {k: ta[k].float().cuda() for k in
                   ("opacity_rf", "features_rf_dc", "features_rf_rest")}
        teacher["opacity_rf"] = teacher["opacity_rf"].clamp(-args.attr_op_clip,
                                                            args.attr_op_clip)
        # 先按掩码压紧，避免每步在 1.4e6×49 上做两次布尔索引（≈0.5 GB 瞬态）
        teacher = {k: v[t_mask].contiguous() for k, v in teacher.items()}
        # 均方归一：零预测下每项 = 1，λ_attr 才是可解释的系数
        t_scale = {k: v.pow(2).mean().clamp_min(1e-12) for k, v in teacher.items()}
        log(f"[g2r] 老师属性蒸馏 保留 {int(t_mask.sum())}/{p} "
            f"({float(t_mask.float().mean()) * 100:.1f}%)  λ={args.lambda_attr}")
        log("[g2r] 属性尺度（零预测基线=1）："
            + "  ".join(f"{k}={float(v):.4g}" for k, v in t_scale.items()))
    elif args.lambda_attr > 0:
        log("[g2r] 没给 --teacher_attr ⇒ 属性蒸馏项关闭")

    # ---------------------------------------------------------------- 相机
    rf_train, rf_test = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                                        args.rf_data_device, verbose=not args.quiet)
    log(f"[g2r] 射频视角 {len(rf_train)} 训练 / {len(rf_test)} 测试")
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # ---------------------------------------------------------------- 头
    head = GeoToRfHead(app_dim=app_dim, hidden=args.hidden, n_layers=args.n_layers,
                       pe_bands=args.pe_bands, chunk=args.chunk,
                       pe_mean=pe_mean, pe_std=pe_std,
                       app_mean=a_mean, app_std=a_std,
                       opacity_init=args.opacity_init,
                       sh_bound=args.sh_bound,
                       out_kind=args.out_kind, n_bands=args.n_bands,
                       init_u=args.init_u,
                       use_dir=bool(args.use_dir), pe_dir_bands=args.pe_dir_bands,
                       use_dist=bool(args.use_dist)).cuda()
    lut = make_jet_lut(256).cuda() if args.out_kind == "scalar" else None
    n_param = sum(q.numel() for q in head.parameters())
    view = StudentView(gm)
    optm = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-15)

    cfg = {**vars(args), "n_points": p, "app_dim": app_dim,
           "scene_scale": scene_scale,
           "n_rf_train_cams": len(rf_train), "n_rf_test_cams": len(rf_test),
           "head_params": n_param,
           "pe_mean": [float(v) for v in pe_mean],
           "pe_std": [float(v) for v in pe_std],
           "app_mean": None if a_mean is None else [float(v) for v in a_mean],
           "app_std": None if a_std is None else [float(v) for v in a_std]}
    with open(os.path.join(args.model_path, "geo2rf_experiment.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    log(f"[g2r] 头参数量={n_param}  app_dim={app_dim}  c_in={head.trunk[0].in_features}")

    # ---------------------------------------------------------------- 训练
    t0 = time.time()
    ema = None
    for it in range(1, args.iterations + 1):
        cam = rf_train[it % len(rf_train)]
        if args.use_dir:
            dirv = make_dir(xyz, cam.camera_center)
            dist = (_dist(xyz, cam.camera_center, scene_scale)
                    if args.use_dist else None)
            op, attr = head(xyz, app, dirv, dist)
        else:
            op, attr = head(xyz, app)
        if args.out_kind == "sh":
            view.set_rf(op, attr)
            img = render(cam, view, _Pipe(), background)["render"]
        else:
            # 标量域：先把每个高斯的 K 个标量塞进光栅化器的 3 个通道，
            # 混合得到标量图，再**只查一次** jet 表 ⇒ 输出按构造在调色板上。
            view.set_rf_scalar(op)
            mixed = render(cam, view, _Pipe(), background,
                           override_color=to_override_color(attr))["render"]
            img = apply_jet(combine_bands(mixed, head), lut)
        L, l1 = rgb_loss(img, cam.original_image[:3].cuda(), args.lambda_dssim)
        if not L.requires_grad:
            raise SystemExit("损失没有梯度 —— 检查头是否真的接上了")

        attr_v = 0.0
        if teacher is not None:
            dc, rest = attr[:, :1, :], attr[:, 1:, :]
            la = xyz.new_zeros(())
            for pred, key in ((op, "opacity_rf"), (dc, "features_rf_dc"),
                              (rest, "features_rf_rest")):
                la = la + F.mse_loss(pred[t_mask], teacher[key]) / t_scale[key]
            L = L + args.lambda_attr * la
            attr_v = float(la.item())

        L.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        with torch.no_grad():
            optm.step()
        optm.zero_grad(set_to_none=False)

        v = float(l1.item())
        ema = v if ema is None else 0.99 * ema + 0.01 * v
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            with torch.no_grad():
                op_mean = float(torch.sigmoid(op).mean())
                if args.out_kind == "sh":
                    # 早警：|SH| 均值若越过 ~1 就是在发散（老师的上界只有 1.02）
                    extra = f"|SH|={float(attr.abs().mean()):.4f}"
                else:
                    # 标量域：看均值/标准差有没有塌成一个常数
                    extra = (f"s_mean={float(attr.mean()):.4f} "
                             f"s_std={float(attr.std()):.4f}")
            msg = (f"[g2r] {it:6d}/{args.iterations}  L1rf={v:.5f}(ema {ema:.5f})  "
                   f"{extra} op={op_mean:.4f}")
            if teacher is not None:
                msg += f" attr={attr_v:.4f}"
            msg += (f"  cam={it % len(rf_train):4d}  {el / it:.2f}s/it  "
                    f"剩 {(args.iterations - it) * el / it / 60:.0f} 分")
            log(msg)
            if it <= 5 * args.log_every:
                log(f"[g2r]   mem {torch.cuda.memory_allocated() / 2**30:.2f} GB "
                    f"alloc / {torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak")

        if it in args.save_iterations:
            torch.save({"iteration": it, "head": head.state_dict(), "config": cfg},
                       os.path.join(args.model_path, f"head_{it}.pth"))
            log(f"[g2r] 存头 -> head_{it}.pth")

    torch.save({"iteration": args.iterations, "head": head.state_dict(), "config": cfg},
               os.path.join(args.model_path, f"head_{args.iterations}.pth"))
    log(f"[g2r] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
