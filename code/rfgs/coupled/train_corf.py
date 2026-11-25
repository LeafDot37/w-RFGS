# -*- coding: utf-8 -*-
"""train_corf.py —— CORF 耦合光学-射频优化（**老师**）

对应外部工作 CORF-GS（arXiv 2607.25569）第 IV-E 节。本文件是**入口点变体**，
`splat/` 下的引擎核心（`gaussian_model.py` / `gaussian_renderer/` / CUDA 光栅化器）
一行不改。

一、耦合表示
--------------------------------------------------------------------------------
    G_i = { mu_i, Sigma_i, o_i^opt, c_i^opt, o_i^rf, c_i^rf }

`{mu, Sigma}` 共享，`{o^opt, c^opt}` 与 `{o^rf, c^rf}` 各自独立。
落在代码上是 `corf/coupled_model.py` 的 `CoupledGaussianModel`：它子类化引擎的
`GaussianModel`，复用基类光学的 `_opacity` / `_features_dc` / `_features_rest`，
只新增射频的 `_opacity_rf` / `_features_rf_dc` / `_features_rf_rest`。
两个模态共用的几何是**同一批参数对象** ⇒ 射频损失回传的梯度落在同一批几何上，
这就是「耦合」的全部含义。

二、损失
--------------------------------------------------------------------------------
    L_opt = (1-gamma)*L1(I_hat, I) + gamma*(1-SSIM(I_hat, I))
    L_rf  = (1-gamma)*L1(S_hat, S) + gamma*(1-SSIM(S_hat, S))
    L     = L_opt + lambda_rf * L_rf

gamma 取 `--lambda_dssim`（缺省 0.2，与 CORF 论文的 gamma=0.2 一致）。
**未实现 CORF 的深度正则项 eta*L_d** —— 它需要 Depth-Anything-2 的单目深度先验，
本仓没有该权重，且我们的初始化来自 MVSplat 前馈点云（多视角一致的深度已经隐含在里面）。
这一条是**已知的口径差异**，写在 `docs/CORF复现与前馈蒸馏_实施记录.md`。

三、三条消融臂（对应 CORF 论文 Table IV）
--------------------------------------------------------------------------------
    --mode joint      从头就 L_opt + lambda_rf*L_rf，几何与两套外观一起训   （= Ours）
    --mode rf_tuned   先纯光学训 --warmup_iters 步；之后只训射频外观，
                      但**几何仍可训**                                        （= w/o joint, RF-tuned geo.）
    --mode frozen     先纯光学训 --warmup_iters 步；之后**冻结几何**，
                      只训射频外观                                            （= w/o joint, frozen geo.）

`frozen` 臂在语义上等价于本仓既有管线（阶段一 + 阶段三）。

四、用法（服务器 code/ 目录，gs3d 环境）
--------------------------------------------------------------------------------
    RF_POINTCLOUD=<mvsplat init.ply> <PY> train_corf.py \
        -s <blender_visual_dataset> -m <out_dir> \
        --rf_source_path <training-rf-spectrum/3dgs_MVDR_100> \
        --resolution 2 --mode joint --iterations 34000 \
        --lambda_rf 0.1 --save_iterations 34000 --quiet
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from torch import nn

from corf.coupled_model import CoupledGaussianModel
from corf.dual_scene import DualScene
from splat.arguments import ModelParams, OptimizationParams, PipelineParams
from splat.gaussian_renderer import render
from splat.utils.general_utils import get_expon_lr_func, safe_state
from splat.utils.loss_utils import l1_loss, ssim
from tqdm import tqdm

# 参数名 -> 参数组归属。名字一律带前导下划线，与 GaussianModel 的字段同名。
GEO = ("_xyz", "_scaling", "_rotation")
OPT_APP = ("_opacity", "_features_dc", "_features_rest")
RF_APP = ("_opacity_rf", "_features_rf_dc", "_features_rf_rest")


def set_grad(gaussians, names, flag: bool):
    for n in names:
        getattr(gaussians, n).requires_grad = flag


def build_optimizer(gaussians, opt, rf_opacity_lr, rf_feature_lr):
    """九组参数：共享几何 3 组 + 光学外观 3 组 + 射频外观 3 组。"""
    s = float(gaussians.spatial_lr_scale or 1.0)
    groups = [
        {"params": [gaussians._xyz], "lr": opt.position_lr_init * s, "name": "xyz"},
        {"params": [gaussians._scaling], "lr": opt.scaling_lr, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": opt.rotation_lr, "name": "rotation"},
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [gaussians._features_dc], "lr": opt.feature_lr, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": opt.feature_lr / 20.0,
         "name": "f_rest"},
        {"params": [gaussians._opacity_rf], "lr": rf_opacity_lr,
         "name": "opacity_rf"},
        {"params": [gaussians._features_rf_dc], "lr": rf_feature_lr,
         "name": "f_rf_dc"},
        {"params": [gaussians._features_rf_rest], "lr": rf_feature_lr / 20.0,
         "name": "f_rf_rest"},
    ]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def set_group_lr(optimizer, name, lr):
    for g in optimizer.param_groups:
        if g.get("name") == name:
            g["lr"] = float(lr)
            return


def rgb_loss(image, gt, lambda_dssim):
    Ll1 = l1_loss(image, gt)
    return (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt)), Ll1


def phase_at(iteration, mode, warmup, lambda_rf):
    """返回该迭代该用哪些损失权重、哪些参数可训。"""
    if mode == "joint":
        return {"geo": True, "opt_app": True, "rf_app": True,
                "w_opt": 1.0, "w_rf": float(lambda_rf)}
    if iteration <= warmup:                      # 两阶段臂的第一段：纯光学
        return {"geo": True, "opt_app": True, "rf_app": False,
                "w_opt": 1.0, "w_rf": 0.0}
    return {"geo": (mode == "rf_tuned"),          # 第二段：只训射频外观
            "opt_app": False, "rf_app": True,
            "w_opt": 0.0, "w_rf": 1.0}


def load_seed(chkpnt_path, gaussians, sh_degree):
    """轻量 restore：只取几何/外观参数，**不重建优化器状态**。

    与 `train_rrf_scalar.py:load_visual_geometry()` 逐字同源（同一套解包顺序，
    必须与 `GaussianModel.capture()` 一致），因为本变体建的是 9 组 Adam，
    与引擎 `restore()` 里那 6 组的 state_dict 对不上。

    一律 `.detach()` 在前：`.contiguous()` 对非连续张量会 copy，copy 是非叶子节点，
    此时 `.requires_grad_(False)` 会抛「只能改叶子变量」。返回绝对起始迭代。
    """
    ckpt = torch.load(chkpnt_path, map_location="cuda")
    model_params, first_iter = ckpt
    (gaussians.active_sh_degree, xyz, f_dc, f_rest,
     scaling, rotation, opacity, _max_radii2D,
     _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params

    gaussians.max_sh_degree = sh_degree
    gaussians._xyz = nn.Parameter(xyz.detach().contiguous(), requires_grad=False)
    gaussians._scaling = nn.Parameter(scaling.detach().contiguous(), requires_grad=False)
    gaussians._rotation = nn.Parameter(rotation.detach().contiguous(), requires_grad=False)
    gaussians._features_dc = nn.Parameter(f_dc.detach().contiguous(), requires_grad=False)
    gaussians._features_rest = nn.Parameter(f_rest.detach().contiguous(), requires_grad=False)
    gaussians._opacity = nn.Parameter(opacity.detach().contiguous(), requires_grad=False)
    gaussians.spatial_lr_scale = (_spatial_lr_scale
                                  if _spatial_lr_scale is not None else 0)
    return int(first_iter)


def save_coupled(scene, gaussians, iteration, cfg):
    """光学那一半按引擎惯例落盘（标准点云格式，既有工具链可直接读）；
    射频那一半落 sidecar。"""
    scene.save(iteration)
    d = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
    torch.save({
        "iteration": iteration,
        "opacity_rf": gaussians._opacity_rf.detach().cpu(),
        "features_rf_dc": gaussians._features_rf_dc.detach().cpu(),
        "features_rf_rest": gaussians._features_rf_rest.detach().cpu(),
        "config": cfg,
    }, os.path.join(d, f"coupled_rf_{iteration}.pth"))


def training(dataset, opt, pipe, cfg, saving_iterations, start_checkpoint):
    log = lambda m: print(m, file=sys.stderr, flush=True)   # noqa: E731

    gaussians = CoupledGaussianModel(dataset.sh_degree)
    scene = DualScene(dataset, gaussians,
                      rf_source_path=cfg["rf_source_path"],
                      rf_images=cfg["rf_images"],
                      rf_resolution=cfg["rf_resolution"],
                      rf_data_device=cfg["rf_data_device"],
                      shuffle=False)
    if not scene.getRfTrainCameras():
        raise RuntimeError("射频相机为空 ⇒ 检查 --rf_source_path 是否指向 3dgs_<方法>_100")

    if start_checkpoint is None:
        first_iter = 0
        log("[corf] 起点 = MVSplat 前馈点云（RF_POINTCLOUD），从头联合优化")
    else:
        first_iter = load_seed(start_checkpoint, gaussians, dataset.sh_degree)
        log(f"[corf] 起点 = {start_checkpoint}（绝对迭代 {first_iter}）；"
            f"几何/光学外观由 seed 载入，射频外观全新初始化")
        # --iterations 一律按**绝对迭代**理解（同既有阶段三口径）。若它不大于 seed 的
        # 迭代数，训练循环是空区间 —— 会「成功退出、什么都不存」，最难查的一种失败。
        if int(cfg["iterations"]) <= first_iter:
            raise ValueError(
                f"--iterations ({cfg['iterations']}) 必须大于 seed 的绝对迭代 ({first_iter})，"
                f"否则训练循环为空。要跑 60 步就写 --iterations {first_iter + 60}")

    P = gaussians.get_xyz.shape[0]
    gaussians.init_rf_appearance(opacity_init=cfg["rf_opacity_init"])
    spatial_lr_scale = float(gaussians.spatial_lr_scale or 1.0)

    gaussians.optimizer = build_optimizer(gaussians, opt,
                                          cfg["rf_opacity_lr"], cfg["rf_feature_lr"])
    xyz_sched = get_expon_lr_func(
        lr_init=opt.position_lr_init * spatial_lr_scale,
        lr_final=opt.position_lr_final * spatial_lr_scale,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=int(cfg["iterations"]))

    rf_sched = None
    if cfg["rf_lr_schedule"] == "exp":
        rf_sched = get_expon_lr_func(
            cfg["rf_feature_lr"], cfg["rf_feature_lr_final"],
            int(0.01 * cfg["iterations"]), 0.01, int(cfg["iterations"]))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(scene.model_path, exist_ok=True)
    with open(os.path.join(scene.model_path, "corf_experiment.json"), "w") as f:
        json.dump({**cfg, "P": P, "first_iter": first_iter,
                   "spatial_lr_scale": spatial_lr_scale,
                   "cameras_extent": scene.cameras_extent}, f, indent=2)

    log(f"[corf] P={P}  mode={cfg['mode']}  共享几何 + 双模态外观  "
        f"gamma={opt.lambda_dssim}  lambda_rf={cfg['lambda_rf']}  "
        f"迭代 {first_iter} -> {cfg['iterations']}  warmup={cfg['warmup_iters']}")
    log(f"[corf] 视觉 {dataset.source_path}  分辨率缩放 --resolution {dataset.resolution}")
    log(f"[corf] 射频 {cfg['rf_source_path']}  {len(scene.getRfTrainCameras())} 训练相机")

    opt_stack, rf_stack = None, None
    ema = 0.0
    cur_phase = None
    pbar = tqdm(range(first_iter + 1, int(cfg["iterations"]) + 1),
                desc="CORF training")
    for iteration in range(first_iter + 1, int(cfg["iterations"]) + 1):
        ph = phase_at(iteration, cfg["mode"], int(cfg["warmup_iters"]),
                      cfg["lambda_rf"])
        # 相位切换时才动 requires_grad（每次都设也便宜，但切换点要记一笔）
        if ph != cur_phase:
            set_grad(gaussians, GEO, ph["geo"])
            set_grad(gaussians, OPT_APP, ph["opt_app"])
            set_grad(gaussians, RF_APP, ph["rf_app"])
            log(f"[corf] iter {iteration} 相位切换 -> geo={ph['geo']} "
                f"opt_app={ph['opt_app']} rf_app={ph['rf_app']} "
                f"w_opt={ph['w_opt']} w_rf={ph['w_rf']}")
            cur_phase = ph

        set_group_lr(gaussians.optimizer, "xyz", xyz_sched(iteration))
        if rf_sched is not None and ph["rf_app"]:
            set_group_lr(gaussians.optimizer, "f_rf_dc", rf_sched(iteration))
            set_group_lr(gaussians.optimizer, "f_rf_rest", rf_sched(iteration) / 20.0)

        loss = torch.zeros((), device="cuda")
        l1_opt_v = l1_rf_v = None
        if ph["w_opt"] > 0.0:
            if not opt_stack:
                opt_stack = scene.getTrainCameras().copy()
            cam = opt_stack.pop(randint(0, len(opt_stack) - 1))
            pkg = render(cam, gaussians.view("opt"), pipe, background)
            L, l1v = rgb_loss(pkg["render"], cam.original_image.cuda(),
                              opt.lambda_dssim)
            loss = loss + ph["w_opt"] * L
            l1_opt_v = float(l1v)
        if ph["w_rf"] > 0.0:
            if not rf_stack:
                rf_stack = scene.getRfTrainCameras().copy()
            cam = rf_stack.pop(randint(0, len(rf_stack) - 1))
            pkg = render(cam, gaussians.view("rf"), pipe, background)
            L, l1v = rgb_loss(pkg["render"], cam.original_image.cuda(),
                              opt.lambda_dssim)
            loss = loss + ph["w_rf"] * L
            l1_rf_v = float(l1v)

        loss.backward()
        with torch.no_grad():
            ema = 0.4 * float(loss) + 0.6 * ema
            if iteration % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema:.5f}"})
                pbar.update(10)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration % cfg["log_every"] == 0:
                log(f"[corf] iter {iteration:>6d}  loss {float(loss):.5f}  "
                    f"L1_opt {('%.5f' % l1_opt_v) if l1_opt_v is not None else '--'}  "
                    f"L1_rf {('%.5f' % l1_rf_v) if l1_rf_v is not None else '--'}  "
                    f"P {gaussians.get_xyz.shape[0]}")
            if iteration in saving_iterations:
                log(f"[ITER {iteration}] Saving coupled gaussians")
                save_coupled(scene, gaussians, iteration, cfg)
    pbar.close()
    print("\nCORF training complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="CORF coupled optical-RF training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, default=None,
                        help="起点权重；给了就跳过 MVSplat 点云、载入几何与光学外观。"
                             "--iterations 一律按**绝对迭代**理解（同既有阶段三口径）")
    parser.add_argument("--rf_source_path", type=str, required=True,
                        help="射频数据集目录（COLMAP 格式，含 train_index.txt）")
    parser.add_argument("--rf_images", type=str, default="images")
    parser.add_argument("--rf_resolution", type=int, default=-1,
                        help="射频图像分辨率；-1 = 原生 300×200（与既有阶段三口径一致）")
    parser.add_argument("--rf_data_device", type=str, default="cpu",
                        help="射频图像驻留设备；cpu 省显存（同既有阶段三）")
    parser.add_argument("--mode", type=str, default="joint",
                        choices=["joint", "rf_tuned", "frozen"])
    parser.add_argument("--warmup_iters", type=int, default=12000,
                        help="两阶段臂的光学预训练步数（--mode joint 时忽略）")
    parser.add_argument("--lambda_rf", type=float, default=0.1,
                        help="射频损失权重（CORF 论文取 0.1）")
    parser.add_argument("--rf_opacity_lr", type=float, default=None)
    parser.add_argument("--rf_feature_lr", type=float, default=None)
    parser.add_argument("--rf_feature_lr_final", type=float, default=None)
    parser.add_argument("--rf_lr_schedule", type=str, default="fixed",
                        choices=["fixed", "exp"])
    parser.add_argument("--rf_opacity_init", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.save_iterations is None:
        args.save_iterations = [args.iterations]
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    opt = op.extract(args)
    rf_opacity_lr = (args.rf_opacity_lr if args.rf_opacity_lr is not None
                     else opt.opacity_lr)
    rf_feature_lr = (args.rf_feature_lr if args.rf_feature_lr is not None
                     else opt.feature_lr)
    rf_feature_lr_final = (args.rf_feature_lr_final
                           if args.rf_feature_lr_final is not None
                           else rf_feature_lr * 0.01)
    if args.rf_lr_schedule == "exp" and rf_feature_lr_final >= rf_feature_lr:
        parser.error("--rf_lr_schedule exp 需要 --rf_feature_lr_final < --rf_feature_lr")
    # 两条合法的两阶段口径：
    #   ① `--start_checkpoint` 直接给收敛后的光学模型 ⇒ `--warmup_iters 0`
    #      （三条臂从**同一个起点**出发，唯一差异是「哪些参数可训 / 哪些损失开着」）
    #   ② 不给 seed ⇒ 需要一段纯光学预训练把几何跑到收敛，`--warmup_iters > 0`
    if args.mode != "joint" and args.warmup_iters <= 0 and not args.start_checkpoint:
        parser.error("--mode %s 未给 --start_checkpoint 时必须 --warmup_iters > 0" % args.mode)
    if args.warmup_iters >= args.iterations:
        parser.error("--warmup_iters 必须小于 --iterations")

    cfg = {
        "mode": args.mode, "warmup_iters": args.warmup_iters,
        "start_checkpoint": args.start_checkpoint,
        "lambda_rf": args.lambda_rf, "iterations": args.iterations,
        "rf_source_path": args.rf_source_path, "rf_images": args.rf_images,
        "rf_resolution": args.rf_resolution,
        "rf_data_device": args.rf_data_device,
        "rf_opacity_lr": rf_opacity_lr, "rf_feature_lr": rf_feature_lr,
        "rf_feature_lr_final": rf_feature_lr_final,
        "rf_lr_schedule": args.rf_lr_schedule,
        "rf_opacity_init": args.rf_opacity_init,
        "log_every": args.log_every,
    }

    safe_state(args.quiet)
    prepare_output_and_logger(args)
    training(lp.extract(args), opt, pp.extract(args), cfg,
             set(args.save_iterations), args.start_checkpoint)
    print("\nTraining complete.")
