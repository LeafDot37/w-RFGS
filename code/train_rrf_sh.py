# -*- coding: utf-8 -*-
"""
train_rrf_sh.py —— RRF 阶段二原版（逐点 SH 颜色头）+ LR 协议开关
--------------------------------------------------------------------------------
原版 RRF（engine train.py 的 --start_checkpoint 分支，train.py:42-50）语义逐字复刻：
    restore(chkpnt) 后冻结 xyz/scaling/rotation，opacity 可训，把 _features_dc/_features_rest
    清零后当"逐点 view-dependent SH 颜色"重拟合频谱图；densify=0、active_sh_degree 保持
    视觉期档位（全阶视角相关）、每 1000 步 oneupSHdegree（engine 同款，超过则空转）。
唯一与 train.py 的差异 = **不修改引擎内核**，用独立入口自建 optimizer，从而能给颜色组
(f_dc / f_rest) 挂指数衰减 LR —— 即"原版 SH 颜色头 + A* LR 协议"（Phase1b S_A* 对照），
引擎原版语义 = 本入口缺省（lr_schedule=fixed）跑出的结果（可对拍 rrf40k_sh 验证一致性）。

LR 协议（只衰减 SH 颜色组；opacity 恒为 opacity_lr；几何冻结）：
    --lr_schedule {exp,fixed}    fixed=引擎默认（f_dc@feature_lr, f_rest@feature_lr/20，无衰减）
                                 exp = 颜色 lr 初值 --color_lr_init 起 log-linear 指数衰减到
                                       --color_lr_final（f_rest 恒为其 /20，保留引擎 1:0.05）
    --lr_max_steps               调度总步数（RRF 本地步 = iterations - first_iter；扫档/确认
                                 同曲线 ⇒ 与 MLP 臂一致用 40000）
    --lr_delay_steps/--lr_delay_mult  warmup（缺省 0.01*max_steps 起，同 MLP 臂）

评测链：render.py 类 SH 渲染（本模型 ply 即含 SH 颜色）→ 标准 test/ours_<i> → metrics 复用。
对比对象：code/train_rrf_mlp.py（MLP 颜色头 + 同一 LR 协议）。两臂其余全部一致 ⇒
    S_A* vs M_A* 差 = 同 LR 协议下颜色形式的净收益（Phase1b 判定 LR 好处还是形式好处）。

用法（服务器 code/ 目录，gs3d env）：
    <PY> train_rrf_sh.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out> \
        --start_checkpoint <视觉chkpnt30000.pth> --iterations 40000 \
        --save_iterations 32000 40000 --densify_until_iter 0 --data_device cpu \
        --lr_schedule exp --color_lr_init 8e-3 --color_lr_final 8e-5 --lr_max_steps 40000
"""
import os
import json
import torch
from torch import nn
from random import randint
from splat.utils.loss_utils import l1_loss, ssim
from splat.gaussian_renderer import render
from splat.scene import Scene, GaussianModel
from splat.utils.general_utils import safe_state, get_expon_lr_func
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from splat.arguments import ModelParams, PipelineParams, OptimizationParams
import uuid


def load_visual_geometry_rrf(chkpnt_path, gaussians, sh_degree):
    """同 train.py 的 restore + RF reset：取几何/外观参数，冻结几何、清零 SH 颜色可训，
    opacity 可训。返回绝对起始迭代 first_iter。参数顺序须与 capture() 一致：
    (active_sh_degree, xyz, f_dc, f_rest, scaling, rotation, opacity,
     max_radii2D, xyz_grad_acc, denom, opt_dict, spatial_lr_scale)"""
    ckpt = torch.load(chkpnt_path, map_location="cuda")
    model_params, first_iter = ckpt
    (gaussians.active_sh_degree, xyz, f_dc, f_rest,
     scaling, rotation, opacity, _max_radii2D,
     _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params

    gaussians.max_sh_degree = sh_degree
    # 冻结几何（保留为 param 供 save_ply）
    gaussians._xyz = nn.Parameter(xyz.contiguous().requires_grad_(False))
    gaussians._scaling = nn.Parameter(scaling.contiguous().requires_grad_(False))
    gaussians._rotation = nn.Parameter(rotation.contiguous().requires_grad_(False))
    # SH 颜色：可训 + 清零（RF reset，同 train.py:49-50）
    gaussians._features_dc = nn.Parameter(f_dc.contiguous().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(f_rest.contiguous().requires_grad_(True))
    with torch.no_grad():
        gaussians._features_dc.data.fill_(0.0)
        gaussians._features_rest.data.fill_(0.0)
    gaussians._opacity = nn.Parameter(opacity.contiguous().requires_grad_(True))
    gaussians.spatial_lr_scale = _spatial_lr_scale if _spatial_lr_scale is not None else 0
    return first_iter


def _set_group_lr(optimizer, name: str, lr: float):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            group["lr"] = float(lr)
            return


def training(dataset, opt, pipe, sc, start_checkpoint,
             saving_iterations, iterations):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    if start_checkpoint is None:
        raise RuntimeError("train_rrf_sh 必须从阶段一视觉 chkpnt 出发（--start_checkpoint）")

    first_iter = load_visual_geometry_rrf(start_checkpoint, gaussians, dataset.sh_degree)
    P = gaussians.get_xyz.shape[0]

    color_lr_init = sc["color_lr_init"]
    color_lr_final = sc["color_lr_final"]
    lr_schedule = sc["lr_schedule"]
    lr_max_steps = sc["lr_max_steps"]
    lr_delay_steps = sc["lr_delay_steps"]
    lr_delay_mult = sc["lr_delay_mult"]
    spatial_lr_scale = float(gaussians.spatial_lr_scale or 0.0)

    # 独立 optimizer：只含真正可训项（opacity / f_dc / f_rest）。几何冻结不入（省显存）。
    # f_dc:f_rest = 1:0.05（引擎 training_setup 同比例）；eps 同引擎 1e-15。
    gaussians.optimizer = torch.optim.Adam([
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [gaussians._features_dc], "lr": color_lr_init, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": color_lr_init / 20.0, "name": "f_rest"},
    ], lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = None

    # 指数衰减调度（仅颜色组；opacity 固定）。本地步 = iteration - first_iter。
    color_sched = None
    if lr_schedule == "exp":
        if color_lr_final >= color_lr_init:
            raise ValueError(f"exp: color_lr_final({color_lr_final}) 须 < color_lr_init({color_lr_init})")
        delay = lr_delay_steps if lr_delay_steps is not None else int(0.01 * lr_max_steps)
        color_sched = get_expon_lr_func(color_lr_init, color_lr_final, delay,
                                        lr_delay_mult, lr_max_steps)

    # 实验溯源
    exp_json = os.path.join(scene.model_path, "experiment.json")
    with open(exp_json, "w") as f:
        json.dump({
            "engine": "SH原版(逐点SH颜色)+LR协议",
            "sh_degree": dataset.sh_degree,
            "active_sh_degree": gaussians.active_sh_degree,
            "color_lr_init": color_lr_init, "color_lr_final": color_lr_final,
            "lr_schedule": lr_schedule, "lr_max_steps": lr_max_steps,
            "lr_delay_steps": lr_delay_steps, "lr_delay_mult": lr_delay_mult,
            "opacity_lr": opt.opacity_lr,
            "spatial_lr_scale": spatial_lr_scale, "start_iter": first_iter,
            "iterations": iterations, "P": P,
        }, f, indent=2)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[sh-rrf] P={P}  start_iter={first_iter}  ->  {iterations}  data={dataset.source_path}")
    print(f"[scenario] SH颜色(清零重学) | color_lr {color_lr_init:g}->{color_lr_final:g} "
          f"schedule={lr_schedule} max_steps={lr_max_steps} delay_steps={lr_delay_steps} "
          f"delay_mult={lr_delay_mult} | spatial_lr_scale={spatial_lr_scale}")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    pbar = tqdm(range(first_iter, iterations), desc="SH-RRF training")
    for iteration in range(first_iter + 1, iterations + 1):
        # SH 阶升档（engine 同款；active 已达 max 时空转）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)   # SH 颜色原生路径
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}"})
                pbar.update(10)

            if iteration < iterations:
                if color_sched is not None:
                    p = color_sched(iteration - first_iter)
                    _set_group_lr(gaussians.optimizer, "f_dc", p)
                    _set_group_lr(gaussians.optimizer, "f_rest", p / 20.0)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)      # 标准 ply（SH 颜色在其中），无 sidecar
    pbar.close()
    print("\nSH-RRF training complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="SH-RRF training (原版 SH 颜色头 + LR 协议开关)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # —— LR 协议（作用于 SH 颜色组 f_dc/f_rest）——
    parser.add_argument("--color_lr_init", type=float, default=None,
                        help="SH 颜色 lr 初值（缺省 = opt.feature_lr = 2.5e-3 = 引擎默认）")
    parser.add_argument("--color_lr_final", type=float, default=None,
                        help="SH 颜色 lr 终值（exp 须 < init；缺省 exp=init/100 / fixed=init）")
    parser.add_argument("--lr_schedule", type=str, choices=["fixed", "exp"],
                        default="fixed",
                        help="fixed=引擎默认（f_dc@feature_lr、f_rest@feature_lr/20 无衰减）；"
                             "exp=颜色组 log-linear 指数衰减")
    parser.add_argument("--lr_max_steps", type=int, default=40000,
                        help="exp 总步数（RRF 本地步 = iterations - first_iter）")
    parser.add_argument("--lr_delay_steps", type=int, default=None,
                        help="exp warmup 步数（缺省 = 0.01*lr_max_steps）")
    parser.add_argument("--lr_delay_mult", type=float, default=0.01,
                        help="exp warmup 起始倍率")
    # —— 通用 ——
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    print("Optimizing (SH-RRF) " + args.model_path)

    opt = op.extract(args)
    color_lr_init = args.color_lr_init if args.color_lr_init is not None else opt.feature_lr
    if args.lr_schedule == "exp":
        color_lr_final = args.color_lr_final if args.color_lr_final is not None else color_lr_init * 0.01
    else:
        color_lr_final = args.color_lr_final if args.color_lr_final is not None else color_lr_init
    if args.lr_schedule == "exp" and color_lr_final >= color_lr_init:
        parser.error(f"--lr_schedule exp 需要 color_lr_final({color_lr_final}) < "
                     f"color_lr_init({color_lr_init})")
    if args.lr_max_steps <= 0:
        parser.error("--lr_max_steps 须 > 0")

    scenario = {
        "color_lr_init": color_lr_init, "color_lr_final": color_lr_final,
        "lr_schedule": args.lr_schedule, "lr_max_steps": args.lr_max_steps,
        "lr_delay_steps": args.lr_delay_steps, "lr_delay_mult": args.lr_delay_mult,
    }

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    prepare_output_and_logger(args)
    training(lp.extract(args), opt, pp.extract(args), scenario,
             args.start_checkpoint, args.save_iterations, args.iterations)
    print("\nTraining complete.")
