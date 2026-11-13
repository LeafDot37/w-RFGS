# -*- coding: utf-8 -*-
"""
render_sh_rrf.py —— SH-RRF 评测渲染（读 ply 内 SH 系数；引擎 render.py 变体）
--------------------------------------------------------------------------------
engine render.py（内核不改）的变体：Scene 载入与输出布局完全一致
（test/ours_<i>/{renders,gt}），差异仅加 --subset 抽帧渲染（中间档画收敛曲线用）：
    subset=0  全量 640（末档 results.json 口径，同引擎 render.py）
    subset>0  确定性均匀抽帧（步长=len//subset，帧名=全局索引），与全量渲染后
             metrics_fast 抽帧同口径 ⇒ 曲线点可直接与 rrf40k 固定-LR 曲线叠加。
颜色走原生 SH 路径（无需 sidecar）——用于 train_rrf_sh.py 产出的模型目录。

用法（同 render.py）：
    <PY> render_sh_rrf.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out> --iteration 40000 \
        --eval --skip_train [--subset 0|48] [--quiet]
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
import os


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, subset=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

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
        rendering = render(view, gaussians, pipeline, background)["render"]
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

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline, background, subset)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline, background, subset)


if __name__ == "__main__":
    parser = ArgumentParser(description="SH-RRF testing script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--subset", type=int, default=0,
                        help="每档渲染帧数上限（0=全部；>0 确定性均匀抽帧，帧名=全局索引）")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering (SH-RRF) " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.subset)
