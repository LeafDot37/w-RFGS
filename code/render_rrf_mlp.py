# -*- coding: utf-8 -*-
"""
render_rrf_mlp.py —— MLP-RRF 评测渲染（读 ply 几何 + color_mlp_<i>.pth sidecar）
--------------------------------------------------------------------------------
render.py（引擎内核）的变体：Scene 载入与输出布局完全一致（test/ours_<i>/{renders,gt}），
唯一差异是颜色来自 sidecar（embedding + 共享 MLP）而非 point_cloud.ply 里的 SH 系数。
产出的 test/ours_<i>/... 由现有 metrics.py 直接评测 → results.json 写 ours_<i> 键。

合成链收在模块级 `synthesize()`：评测路径与漫游路径（tools/render_rrf_walk.py）
**共用同一个函数**（与 render_rrf_scalar.py 同纪律），改一处两边同时变。

用法（同 render.py；-m 模型目录里须有训练时存的 sidecar）：
    <PY> render_rrf_mlp.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out_model_dir> \
        --iteration 40000 --eval --skip_train [--quiet]
"""
import torch
import torchvision
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from splat.scene import Scene
from splat.gaussian_renderer import render, GaussianModel
from splat.arguments import ModelParams, PipelineParams, get_combined_args
from splat.utils.general_utils import safe_state
from splat.color_mlp import load_color_state, compute_colors
import os


def synthesize(view, gaussians, head, embedding, spatial_lr_scale, pipeline,
               background):
    """单视角全链：MLP 色头 → 打包 → 光栅化，返回 (3,H,W) 的 RGB。

    **评测路径（render_set）与漫游路径（tools/render_rrf_walk.py）共用本函数。**
    两条路径必须算的是同一个函数，否则漫游视频展示的就不是被评测的那个模型；
    把合成链收在这里，改一处两边同时变。（与 render_rrf_scalar.py 的同名助手同构，
    差别只在颜色头与是否有 jet 查表。）
    """
    colors = compute_colors(head, embedding, gaussians.get_xyz, view.camera_center,
                            spatial_lr_scale)                        # (P,3) RGB
    return render(view, gaussians, pipeline, background,
                  override_color=colors)["render"]                   # (3,H,W)


def render_set(model_path, name, iteration, views, gaussians, head, embedding,
               spatial_lr_scale, pipeline, background, subset=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # subset>0：确定性抽帧渲染（步长 = len//subset，帧名=全局索引），复刻
    # metrics_fast 对全量 640 的采样（names[::step][:subset]）⇒ 与全量渲染后抽帧同口径。
    n = len(views)
    step = max(1, n // subset) if (subset and subset > 0 and subset < n) else 0
    cnt = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if step:
            if idx % step != 0:
                continue
            if cnt >= subset:
                break
            cnt += 1
        rendering = synthesize(view, gaussians, head, embedding, spatial_lr_scale,
                               pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering,
                                     os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt,
                                     os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, subset: int = 0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        sidecar = os.path.join(dataset.model_path, "point_cloud",
                               "iteration_{}".format(iteration),
                               "color_mlp_{}.pth".format(iteration))
        if not os.path.isfile(sidecar):
            raise FileNotFoundError(
                "MLP-RRF sidecar 缺失: {}（需 train_rrf_mlp.py 在该档写过）".format(sidecar))
        # load_color_state 返回 (head, embedding, spatial_lr_scale)；
        # 旧 sidecar（无 cfg）→ scale=0、use_dist=False，行为与原版一致。
        head, embedding, spatial_lr_scale = load_color_state(sidecar, device="cuda")
        print("Loaded color sidecar iteration {} (embedding {}, "
              "use_dist={}, spatial_lr_scale={})".format(
                  iteration, tuple(embedding.shape), head.use_dist,
                  spatial_lr_scale))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, head, embedding,
                       spatial_lr_scale, pipeline, background, subset)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, head, embedding,
                       spatial_lr_scale, pipeline, background, subset)


if __name__ == "__main__":
    parser = ArgumentParser(description="MLP-RRF testing script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--subset", type=int, default=0,
                        help="每档渲染帧数上限（0=全部；>0 确定性均匀抽帧，帧名=全局索引）")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering (MLP-RRF) " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.subset)
