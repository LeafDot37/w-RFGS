import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from splat.utils.loss_utils import l1_loss, ssim
from splat.gaussian_renderer import render, network_gui
import sys
from splat.scene import Scene, GaussianModel
from splat.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from splat.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from splat.arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def apply_cov_init(gaussians):
    """把 MVSplat 的**世界系协方差**（各向异性尺度 + 局部表面朝向）灌进刚建好的高斯。

    为什么需要这个变体
    ------------------
    `create_from_pcd`（splat/scene/gaussian_model.py，引擎核心，不许改）**只吃 xyz 与 colors**：
        scales = log(sqrt(distCUDA2))[...,None].repeat(1,3)   ← 各向同性球
        rots   = 单位四元数                                     ← 朝向为零
    ⇒ MVSplat 已经预测出来的协方差被整段丢掉。这不是「种下去被洗掉」，是**从没种下去**。

    时机
    ----
    必须在 `gaussians.training_setup(opt)` **之前**：那一步才建优化器；晚于它就会
    拿到旧参数的引用（`.data.copy_` 其实仍然生效，但日志会撒谎）。

    行为
    ----
    开关 `RF_COV_INIT=1` + `RF_COV_PLY=<3DGS 格式 ply>`。缺文件 / 缺字段 / 点数不匹配
    一律**报错退出**，绝不静默降级 —— 静默降级会让「处理臂」实际上等于对照臂，
    而且跑完看不出任何异常。位置不一致只告警并打印差值（那意味着单变量对照破了，
    结论要打折，但不该直接崩）。

    来源 ply 的字段约定：`scale_0..2` 是 **log 半径**（3DGS 标准），`rot_0..3` 是
    **wxyz**（3DGS 标准）。本仓 `train_cov` 配套的导出脚本 `init_cloud_cov.py`
    已经把四元数从 MVSplat 的相机系搬到了世界系（`R_world = R_c2w @ R_cam`，
    依据 adapter 的 `cov = R diag(s²) Rᵀ` ⇒ R 的第 i 列是标准差 s_i 的轴）。
    """
    if os.environ.get("RF_COV_INIT", "0") != "1":
        print("[cov] RF_COV_INIT != 1 ⇒ 按原样继续（各向同性球 + 单位旋转）")
        return
    ply = (os.environ.get("RF_COV_PLY") or "").strip()
    if not ply:
        raise SystemExit("[cov] RF_COV_INIT=1 但 RF_COV_PLY 未设置")
    if not os.path.exists(ply):
        raise SystemExit(f"[cov] RF_COV_INIT=1 但 RF_COV_PLY 不存在：{ply}")

    from plyfile import PlyData
    import numpy as np
    el = PlyData.read(ply)["vertex"]
    have = [p.name for p in el.properties]
    need = ("x", "y", "z", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3")
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(f"[cov] {ply} 缺字段 {missing}（现有 {have}）")

    n_cov, n_g = el.count, gaussians.get_xyz.shape[0]
    if n_cov != n_g:
        raise SystemExit(
            f"[cov] 点数不匹配：ply {n_cov} vs 高斯 {n_g}。"
            f"注意 create_from_pcd 的点数必须与 ply 逐行对应。"
        )

    def col(*names):
        return torch.from_numpy(
            np.stack([np.asarray(el[c], dtype=np.float32) for c in names], 1)
        ).cuda()

    xyz = col("x", "y", "z")
    scale_log = col("scale_0", "scale_1", "scale_2")
    rot = torch.nn.functional.normalize(col("rot_0", "rot_1", "rot_2", "rot_3"), dim=1)

    with torch.no_grad():
        d = (xyz - gaussians._xyz).abs().max().item()
        if d == 0.0:
            print("[cov] 位置与 ply 逐位相同 ✓（单变量对照成立：只有协方差在变）")
        else:
            print(f"[cov] ⚠ 位置最大绝对差 {d:.3e} ≠ 0 ⇒ 位置也变了，"
                  f"协方差效应将与位置效应混淆，结论要按混杂处理")
        s_ref = gaussians._scaling.detach()
        ratio_ref = float(torch.exp(s_ref.max(1).values - s_ref.min(1).values).median())
        ratio_new = float(torch.exp(scale_log.max(1).values - scale_log.min(1).values).median())
        gaussians._scaling.data.copy_(scale_log)
        gaussians._rotation.data.copy_(rot)
    print(f"[cov] 尺度（米）中位：原 {float(s_ref.exp().median()):.5f} → 新 "
          f"{float(scale_log.exp().median()):.5f}")
    print(f"[cov] 各向异性比 p50：原 {ratio_ref:.3f}（各向同性球恒为 1.000）→ 新 {ratio_new:.3f}")
    print(f"[cov] 来源 {ply}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    apply_cov_init(gaussians)          # ← 本变体唯一的新增：协方差初始化
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
