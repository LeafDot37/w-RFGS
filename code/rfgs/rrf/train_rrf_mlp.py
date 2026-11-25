# -*- coding: utf-8 -*-
"""
train_rrf_mlp.py —— RRF 阶段二变体：SH 颜色头 → 逐点 embedding + 共享 MLP 颜色头
--------------------------------------------------------------------------------
原版 RRF 训练入口 code/rfgs/engine/train.py（引擎内核，不改）的做法：从阶段一视觉 chkpnt 出发，
冻结 xyz/scaling/rotation、清零 _features_dc/_features_rest 后用逐点 SH 颜色重拟合
频谱图。本入口是"train_resume.py 式"变体（不动引擎内核），把颜色表达换成
embedding+共享 MLP（见 splat/color_mlp.py），其余语义与 RRF 保持一致，从而与
原版形成**单变量（颜色表达形式）**对比：

  - 起点：阶段一视觉 chkpnt（本实验固定用 原版 RF-3DGS 的视觉 chkpnt30000）
  - 冻结几何 xyz/scaling/rotation（requires_grad=False，同原 RRF）
  - opacity 可训（同原 RRF）；SH 系数保留但不参与训练/渲染
  - 新增可训：每高斯 embedding (P,D) + 共享浅层 MLP(e_p, dir) → RGB
  - densify=0（同 RRF 冻结协议）；损失 L1+SSIM(λ=0.2) 与 train.py 一致
  - 每 save_iteration 档写标准 point_cloud.ply + sidecar color_mlp_<i>.pth
    （embedding + MLP 权重，供评测重建颜色），其余文件约定与 train.py 相同

§6 实验扩展（缺省 = 严格复现旧 run，全部可选）
--------------------------------------------------------------------------------
LR 调度与位置编码两组变量（设计见 docs/RRF-MLP改进方案_对照Scaffold-GS.md §6）：

  位置编码（NeuralColorHead 侧，见 color_mlp.py）：
    --pe_xyz_L <L>     γ(xyz) 谐波频带数，0=不喂 xyz（现状）   [A1/X 系列]
    --pe_dir_L <L>     dir 高频化谐波频带数，0=仅原始 dir      [A2/D 系列]
    --use_dist         输入追加 1 维归一化距离 r                [A2/R 系列]

  LR 调度（本入口侧；embedding/opacity 恒为固定 lr）：
    --lr_schedule {fixed,exp}    fixed=现状；exp=MLP 组指数衰减（log-linear）
    --mlp_lr_init <f>            MLP lr 初值   （缺省 = feature_lr/20 = 1.25e-4）
    --mlp_lr_final <f>           MLP lr 终值   （exp 须 < init；缺省 init/100）
    --lr_max_steps <n>           exp 总步数（**RRF 本地步**；缺省 40000。扫档
                                 abs40000 与确认 abs70000 走同一曲线 ⇒ max_steps
                                 须 = RRF 步数 = iterations - first_iter）
    --lr_delay_steps <n>         warmup 步数（缺省 0.01×max_steps，即 init×0.01 起）
    --lr_delay_mult <f>          warmup 起始倍率（缺省 0.01）
    --embed_lr <f>               embedding lr（缺省 = feature_lr = 2.5e-3）

调度函数复用引擎只读工具 splat/utils/general_utils.get_expon_lr_func（纯 import，
不改内核）。本地步 step = iteration - first_iter，故 abs 语义下训练窗始终从 init 衰减。

评测链：render_rrf_mlp.py（读 ply 几何 + sidecar → MLP 颜色 → 标准 renders）
      + 现有 metrics.py（读 test/ours_<i> → results.json 的 ours_<i> 键）。

用法（服务器 code/ 目录，gs3d env；本地 WSL 同构）：
    <PY> train_rrf_mlp.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out_model_dir> \
        --start_checkpoint <视觉chkpnt30000.pth> --iterations 40000 \
        --save_iterations 40000 --densify_until_iter 0 \
        --data_device cpu \
        [--pe_xyz_L 5 --pe_dir_L 3 --use_dist \
         --lr_schedule exp --mlp_lr_init 4e-3 --mlp_lr_final 4e-5 \
         --lr_max_steps 40000 --embed_lr 5e-3]
  绝对迭代语义与 train.py 相同：first_iter 来自 chkpnt，新训 = iterations - first_iter。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import json
import torch
from torch import nn
from random import randint
from splat.utils.loss_utils import l1_loss, ssim
from splat.gaussian_renderer import render
from splat.scene import Scene, GaussianModel
from splat.utils.general_utils import safe_state, get_expon_lr_func
from splat.color_mlp import (NeuralColorHead, compute_colors, save_color_state,
                             set_scene_norm)
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from splat.arguments import ModelParams, PipelineParams, OptimizationParams
import uuid


def load_visual_geometry(chkpnt_path: str, gaussians: GaussianModel,
                         sh_degree: int):
    """轻量 restore：仅取几何/外观参数（与 train.py restore 同源解包），
    不重建 6 组 Adam（冻结组的 Adam 状态在 8GB 本地卡上是纯开销），
    返回绝对起始迭代 first_iter。参数顺序必须与 GaussianModel.capture() 一致：
    (active_sh_degree, _xyz, _features_dc, _features_rest, _scaling, _rotation,
     _opacity, max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale)
    """
    ckpt = torch.load(chkpnt_path, map_location="cuda")
    model_params, first_iter = ckpt
    (gaussians.active_sh_degree, xyz, f_dc, f_rest,
     scaling, rotation, opacity, _max_radii2D,
     _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params

    gaussians.max_sh_degree = sh_degree
    # 冻结几何：仍保留为 param（save_ply 需要），但 requires_grad=False
    gaussians._xyz = nn.Parameter(xyz.contiguous().requires_grad_(False))
    gaussians._scaling = nn.Parameter(scaling.contiguous().requires_grad_(False))
    gaussians._rotation = nn.Parameter(rotation.contiguous().requires_grad_(False))
    # SH 外观：本变体不用，保留占位但不参与训练（省梯度），也不再清零重学
    gaussians._features_dc = nn.Parameter(f_dc.contiguous().requires_grad_(False))
    gaussians._features_rest = nn.Parameter(f_rest.contiguous().requires_grad_(False))
    # opacity：同原 RRF 可训
    gaussians._opacity = nn.Parameter(opacity.contiguous().requires_grad_(True))
    gaussians.spatial_lr_scale = _spatial_lr_scale if _spatial_lr_scale is not None else 0
    return first_iter


def _set_group_lr(optimizer, name: str, lr: float):
    """按 name 定位参数组并赋 lr（不触发建组，引擎惯例）。"""
    for group in optimizer.param_groups:
        if group.get("name") == name:
            group["lr"] = float(lr)
            return


def training(dataset, opt, pipe, sc, start_checkpoint,
             saving_iterations, iterations):
    """sc = scenario dict（含全部实验开关，见 __main__ 解析）。"""
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    if start_checkpoint is None:
        raise RuntimeError("train_rrf_mlp 必须从阶段一视觉 chkpnt 出发（--start_checkpoint）")

    first_iter = load_visual_geometry(start_checkpoint, gaussians, dataset.sh_degree)
    P = gaussians.get_xyz.shape[0]

    embed_dim = sc["embed_dim"]
    mlp_hidden = sc["mlp_hidden"]
    mlp_depth = sc["mlp_depth"]
    pe_xyz_L = sc["pe_xyz_L"]
    pe_dir_L = sc["pe_dir_L"]
    use_dist = sc["use_dist"]
    embed_lr = sc["embed_lr"]
    mlp_lr_init = sc["mlp_lr_init"]
    mlp_lr_final = sc["mlp_lr_final"]
    lr_schedule = sc["lr_schedule"]
    lr_max_steps = sc["lr_max_steps"]
    lr_delay_steps = sc["lr_delay_steps"]
    lr_delay_mult = sc["lr_delay_mult"]
    spatial_lr_scale = float(gaussians.spatial_lr_scale or 0.0)

    # 新增颜色状态：每高斯 embedding + 共享 MLP（进同一 optimizer 的参数组）
    embed = nn.Parameter(torch.zeros((P, embed_dim), device="cuda").normal_(0, 0.02),
                         requires_grad=True)
    head = NeuralColorHead(embed_dim=embed_dim, hidden=mlp_hidden, depth=mlp_depth,
                           pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L,
                           use_dist=use_dist).cuda()
    # xyz-PE 归一化（中心/半径，冻结几何恒定）：即使本轮不开 X 也存好，便于复现
    if P > 0:
        set_scene_norm(head, gaussians.get_xyz)

    # 独立 optimizer：只含真正可训项（opacity / embedding / MLP）。冻结参数不入。
    gaussians.optimizer = torch.optim.Adam([
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [embed], "lr": embed_lr, "name": "embed"},
        {"params": list(head.parameters()), "lr": mlp_lr_init, "name": "mlp"},
    ], lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = None

    # 指数衰减调度（仅 MLP 组；embedding/opacity 固定）。本地步 = iteration - first_iter
    # ⇒ abs 语义下训练窗始终映射到 t∈(0,1]，扫档/确认跑用同一 lr_max_steps 同曲线。
    mlp_sched = None
    if lr_schedule == "exp":
        if mlp_lr_final >= mlp_lr_init:
            raise ValueError(f"exp: mlp_lr_final({mlp_lr_final}) 须 < mlp_lr_init({mlp_lr_init})")
        delay = lr_delay_steps if lr_delay_steps is not None else int(0.01 * lr_max_steps)
        mlp_sched = get_expon_lr_func(mlp_lr_init, mlp_lr_final, delay,
                                      lr_delay_mult, lr_max_steps)

    # 复现场景配置（实验溯源：把每档的 5 个变量原样落盘）
    exp_json = os.path.join(scene.model_path, "experiment.json")
    with open(exp_json, "w") as f:
        json.dump({
            "pe_xyz_L": pe_xyz_L, "pe_dir_L": pe_dir_L, "use_dist": use_dist,
            "embed_lr": embed_lr, "mlp_lr_init": mlp_lr_init,
            "mlp_lr_final": mlp_lr_final, "lr_schedule": lr_schedule,
            "lr_max_steps": lr_max_steps, "lr_delay_steps": lr_delay_steps,
            "lr_delay_mult": lr_delay_mult,
            "spatial_lr_scale": spatial_lr_scale, "start_iter": first_iter,
            "iterations": iterations, "P": P,
        }, f, indent=2)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[mlp-rrf] P={P}  embed_dim={embed_dim}  mlp_hidden={mlp_hidden}  "
          f"depth={mlp_depth}  start_iter={first_iter}  ->  {iterations}  "
          f"data={dataset.source_path}")
    print(f"[scenario] pe_xyz_L={pe_xyz_L} pe_dir_L={pe_dir_L} "
          f"use_dist={use_dist} | embed_lr={embed_lr:g} "
          f"mlp_lr {mlp_lr_init:g}->{mlp_lr_final:g} schedule={lr_schedule} "
          f"max_steps={lr_max_steps} delay_steps={lr_delay_steps} "
          f"delay_mult={lr_delay_mult} | spatial_lr_scale={spatial_lr_scale}")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    pbar = tqdm(range(first_iter, iterations), desc="MLP-RRF training")
    for iteration in range(first_iter + 1, iterations + 1):
        # 随机取一个训练相机（同 train.py）
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        colors = compute_colors(head, embed, gaussians.get_xyz,
                                viewpoint_cam.camera_center, spatial_lr_scale)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            override_color=colors)
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
                # exp：每步先按本地步设 MLP 组 lr，再 step 使用该 lr
                if mlp_sched is not None:
                    _set_group_lr(gaussians.optimizer, "mlp",
                                  mlp_sched(iteration - first_iter))
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians + color sidecar")
                scene.save(iteration)      # 写标准 point_cloud/iteration_<i>/point_cloud.ply
                sidecar = os.path.join(
                    scene.model_path, "point_cloud", f"iteration_{iteration}",
                    f"color_mlp_{iteration}.pth")
                save_color_state(sidecar, head, embed, iteration, spatial_lr_scale)
    pbar.close()
    print("\nMLP-RRF training complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="MLP-RRF training (embedding+MLP color head)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    # —— §6 位置编码（X / D / R 系列）——
    parser.add_argument("--pe_xyz_L", type=int, default=0,
                        help="γ(xyz) 谐波频带数（+6L 维）；0=不喂 xyz（现状）")
    parser.add_argument("--pe_dir_L", type=int, default=0,
                        help="dir 高频化谐波频带数（+6L 维）；0=仅原始 dir（现状）")
    parser.add_argument("--use_dist", action="store_true",
                        help="输入追加 1 维场景尺度归一化距离 r")
    # —— §6 LR 调度（embedding/opacity 固定，仅 MLP 组可指数衰减）——
    parser.add_argument("--embed_lr", type=float, default=None,
                        help="embedding lr（缺省 = opt.feature_lr = 2.5e-3）")
    parser.add_argument("--mlp_lr_init", type=float, default=None,
                        help="MLP lr 初值（缺省 = opt.feature_lr/20 = 1.25e-4）")
    parser.add_argument("--mlp_lr_final", type=float, default=None,
                        help="MLP lr 终值（exp 须 < init；缺省 exp=init/100 / fixed=init）")
    parser.add_argument("--lr_schedule", type=str, choices=["fixed", "exp"],
                        default="fixed",
                        help="fixed=现状固定；exp=MLP 组 log-linear 指数衰减")
    parser.add_argument("--lr_max_steps", type=int, default=40000,
                        help="exp 总步数（RRF 本地步，= iterations - first_iter）")
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
    print("Optimizing (MLP-RRF) " + args.model_path)

    # 解析默认值依赖 opt.feature_lr（须先 extract op）
    opt = op.extract(args)
    embed_lr = args.embed_lr if args.embed_lr is not None else opt.feature_lr
    mlp_lr_init = args.mlp_lr_init if args.mlp_lr_init is not None else opt.feature_lr / 20.0
    if args.lr_schedule == "exp":
        mlp_lr_final = args.mlp_lr_final if args.mlp_lr_final is not None else mlp_lr_init * 0.01
    else:
        mlp_lr_final = args.mlp_lr_final if args.mlp_lr_final is not None else mlp_lr_init
    if args.lr_schedule == "exp" and mlp_lr_final >= mlp_lr_init:
        parser.error(f"--lr_schedule exp 需要 mlp_lr_final({mlp_lr_final}) < "
                     f"mlp_lr_init({mlp_lr_init})")
    if args.lr_max_steps <= 0:
        parser.error("--lr_max_steps 须 > 0")

    scenario = {
        "embed_dim": args.embed_dim, "mlp_hidden": args.mlp_hidden,
        "mlp_depth": args.mlp_depth,
        "pe_xyz_L": args.pe_xyz_L, "pe_dir_L": args.pe_dir_L,
        "use_dist": args.use_dist,
        "embed_lr": embed_lr, "mlp_lr_init": mlp_lr_init,
        "mlp_lr_final": mlp_lr_final, "lr_schedule": args.lr_schedule,
        "lr_max_steps": args.lr_max_steps,
        "lr_delay_steps": args.lr_delay_steps, "lr_delay_mult": args.lr_delay_mult,
    }

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    prepare_output_and_logger(args)
    training(lp.extract(args), opt, pp.extract(args), scenario,
             args.start_checkpoint, args.save_iterations, args.iterations)
    print("\nTraining complete.")
