# -*- coding: utf-8 -*-
"""render_corf.py —— 耦合模型的评测渲染（读 ply 光学半边 + `coupled_rf_<i>.pth` 射频半边）

`render.py`（引擎内核）的变体。与内核**唯一的**差异是第二个模态：
耦合模型有**两套外观**（`train_corf.py` 只把光学那一半写进标准 ply，射频那一半写 sidecar），
所以本脚本要按 `--modality` 决定渲染哪一边、读哪一半外观。

为什么渲染输出要分目录
--------------------------------------------------------------------------------
`metrics_fast.py` 的口径是「`<模型目录>/test/ours_<i>/{renders,gt}`」，
光学与射频若都写进 `<模型目录>/test/`，两边的 640 张会互相覆盖。
所以射频侧走 `--out_root <模型目录>/rf_eval` ⇒ 落成
`<模型目录>/rf_eval/test/ours_<i>/`，`metrics_fast.py -m <模型目录>/rf_eval` 直接可用，
**评测脚本一行不改**（本仓铁律：既有工具链尽量复用，不改口径）。

用法
--------------------------------------------------------------------------------
    # 射频三指标（640 测试视角）
    <PY> render_corf.py -s <blender_visual_dataset> -m <CORF模型目录> \
        --rf_source_path <.../3dgs_MVDR_100> --modality rf --out_root <CORF模型目录>/rf_eval \
        --iteration 70000 --eval --skip_train --quiet

    # 光学三指标（复用引擎口径，落 <CORF模型目录>/test/）
    <PY> render_corf.py -s <blender_visual_dataset> -m <CORF模型目录> \
        --rf_source_path <.../3dgs_MVDR_100> --modality opt --eval --skip_train --quiet
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from corf.coupled_model import CoupledGaussianModel
from corf.dual_scene import load_rf_cameras
from splat.arguments import ModelParams, PipelineParams, get_combined_args
from splat.gaussian_renderer import render
from splat.scene import Scene
from splat.utils.general_utils import safe_state


def load_rf_appearance(model_path, iteration, gaussians):
    """把 sidecar 里的射频外观装回模型（三个张量，顺序与 `save_coupled` 一致）。"""
    p = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration),
                     "coupled_rf_{}.pth".format(iteration))
    if not os.path.isfile(p):
        raise FileNotFoundError("耦合 sidecar 缺失：{}（需 train_corf.py 在该档写过）".format(p))
    d = torch.load(p, map_location="cuda")
    with torch.no_grad():
        gaussians._opacity_rf.data = d["opacity_rf"].to("cuda")
        gaussians._features_rf_dc.data = d["features_rf_dc"].to("cuda")
        gaussians._features_rf_rest.data = d["features_rf_rest"].to("cuda")
    print("Loaded coupled RF sidecar {} (P={}, cfg.mode={})".format(
        iteration, int(gaussians._opacity_rf.shape[0]),
        d.get("config", {}).get("mode", "?")))
    return d.get("config", {})


def render_set(out_root, name, iteration, views, gaussians, modality, pipeline,
               background, subset=0):
    render_path = os.path.join(out_root, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(out_root, name, "ours_{}".format(iteration), "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    # subset>0：确定性抽帧渲染（步长 = len//subset，帧名=全局索引），与既有臂同口径。
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
        rendering = render(view, gaussians.view(modality), pipeline,
                           background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}.png".format(idx)))
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}.png".format(idx)))


def render_sets(dataset, rf_source_path, rf_images, rf_resolution, rf_data_device,
                modality, out_root, iteration, pipeline, skip_train, skip_test,
                subset):
    with torch.no_grad():
        gaussians = CoupledGaussianModel(dataset.sh_degree)
        # 视觉半边照旧走引擎的 Scene（它按 load_iteration 载入标准 ply）。
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        it = scene.loaded_iter

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 只有射频侧才需要 sidecar 与射频相机 —— 光学渲染**完全不碰**射频那一半
        # （CoupledGaussianModel 的射频参数此时是空张量，光学路径也不读它们）。
        # 顺带省掉 2560+640 张 300×200 的加载与 199 MB 的 sidecar 上传。
        if modality == "opt":
            pairs = [("train", scene.getTrainCameras()),
                     ("test", scene.getTestCameras())]
        else:
            load_rf_appearance(dataset.model_path, it, gaussians)
            rf_train, rf_test = load_rf_cameras(rf_source_path, rf_images,
                                                rf_resolution, rf_data_device)
            pairs = [("train", rf_train), ("test", rf_test)]
        print("[render-corf] modality={}  out_root={}  iteration={}".format(
            modality, out_root, it))
        for name, views in pairs:
            if (name == "train" and skip_train) or (name == "test" and skip_test):
                continue
            print("[render-corf] {}: {} 视角".format(name, len(views)))
            render_set(out_root, name, it, views, gaussians, modality, pipeline,
                       background, subset)


if __name__ == "__main__":
    parser = ArgumentParser(description="Coupled optical-RF testing script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--rf_source_path", type=str, default=None)
    parser.add_argument("--rf_images", type=str, default="images")
    parser.add_argument("--rf_resolution", type=int, default=-1)
    parser.add_argument("--rf_data_device", type=str, default="cpu")
    parser.add_argument("--modality", type=str, default="rf",
                        choices=["opt", "rf"])
    parser.add_argument("--out_root", type=str, default=None,
                        help="渲染落盘根目录；缺省 = 模型目录（射频侧务必另给，"
                             "否则会覆盖光学那 640 张）")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--subset", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 走 cfg_args 合并：-s / --resolution / --rf_source_path / --rf_resolution
    # 这些都取自训练时写下的 cfg_args，保证渲染口径与训练逐字一致（不加 --resolution
    # 就会被命令行默认值 -1 顶掉，静默换一套分辨率）。
    args = get_combined_args(parser)
    print("Rendering (Coupled) " + str(args.model_path))

    # `get_combined_args` 只把**非 None** 的命令行值并进结果 ⇒ 缺省为 None 的开关
    # （--out_root）在没给时**根本不会出现在 Namespace 里**，直接取属性会 AttributeError。
    # 这里一律用 getattr 兜底。
    out_root = getattr(args, "out_root", None) or args.model_path
    if args.modality == "rf" and out_root == args.model_path:
        parser.error("--modality rf 必须给 --out_root（否则射频渲染会覆盖光学渲染）")

    safe_state(getattr(args, "quiet", False))
    if args.modality == "rf" and not getattr(args, "rf_source_path", None):
        parser.error("--modality rf 需要 --rf_source_path（或模型目录 cfg_args 里有）")
    render_sets(model.extract(args), getattr(args, "rf_source_path", None),
                getattr(args, "rf_images", "images"),
                getattr(args, "rf_resolution", -1),
                getattr(args, "rf_data_device", "cpu"), args.modality,
                out_root, args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.subset)
