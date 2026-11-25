# -*- coding: utf-8 -*-
"""
render_rrf_scalar.py —— 形态 A' 评测渲染（读 ply 几何 + color_scalar_<i>.pth sidecar）
--------------------------------------------------------------------------------
render.py（引擎内核）的变体：Scene 载入与输出布局完全一致（test/ours_<i>/{renders,gt}），
唯一差异是颜色来自 sidecar 的**标量头 + 固定 jet 查表**而非 ply 里的 SH 系数。

与 train_rrf_scalar.py 的后处理必须**逐字一致**，否则评测数字与训练目标不是同一个函数。
2026-09-12 起这一致性由**共享助手**保证：两侧都调 `to_override_color()` / `combine_bands()`，
合成方式**按 sidecar 里 head 自己的 n_bands / band_mix 分派**（形态 A' 与 A× 共用本脚本），
不再有任何硬编码的通道数。产出的 PNG 是普通 RGB，
现有 metrics.py / metrics_fast.py 直接可用 → results.json 写 ours_<i> 键。

用法（同 render.py；-m 模型目录里须有训练时存的 sidecar）：
    <PY> render_rrf_scalar.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out_model_dir> \
        --iteration 70000 --eval --skip_train [--subset N] [--quiet]
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import torch
import torchvision
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from splat.scene import Scene
from splat.gaussian_renderer import render, GaussianModel
from splat.arguments import ModelParams, PipelineParams, get_combined_args
from splat.utils.general_utils import safe_state
from splat.color_scalar import (load_scalar_state, compute_scalar, apply_jet,
                                make_jet_lut, to_override_color, combine_bands,
                                check_knn_xyz)
import os


def synthesize(view, gaussians, head, embedding, spatial_lr_scale, lut, pipeline,
               background, sh_coeff=None):
    """单视角全链：标量头 → 打包 → 光栅化 → 合成 → jet 查表，返回 (3,H,W) 的调色板 RGB。

    **评测路径（render_set）与漫游路径（tools/render_rrf_walk.py）共用本函数。**
    两条路径必须算的是同一个函数，否则漫游视频展示的就不是被评测的那个模型；
    把合成链收在这里，改一处两边同时变。

    打包/合成走与 train_rrf_scalar.py 同一对共享助手（to_override_color /
    combine_bands），合成方式按 head 自己的 band_mix 分派，绝不硬编码通道数。

    `sh_coeff` = (P,3,16) 的 SH 系数，**尾随参数**：`render_rrf_walk.py:224-226` 按位置
    调本函数，加在最后才不会打乱它。`head.use_sh != "none"` 时必传（缺了 compute_scalar
    会具名报错）。
    """
    s = compute_scalar(head, embedding, gaussians.get_xyz, view.camera_center,
                       spatial_lr_scale, sh_coeff)               # (P,K)，K = head.n_bands
    mixed = render(view, gaussians, pipeline, background,
                   override_color=to_override_color(s))["render"]   # (3,H,W) 各 band 混合场
    return apply_jet(combine_bands(mixed, head), lut)           # 合成 → 查表 → 调色板上的 RGB



def render_set(model_path, name, iteration, views, gaussians, head, embedding,
               spatial_lr_scale, lut, pipeline, background, subset=0, sh_coeff=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

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
        rendering = synthesize(view, gaussians, head, embedding, spatial_lr_scale,
                               lut, pipeline, background, sh_coeff)
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
                               "color_scalar_{}.pth".format(iteration))
        if not os.path.isfile(sidecar):
            raise FileNotFoundError(
                "scalar-RRF sidecar 缺失: {}（需 train_rrf_scalar.py 在该档写过）".format(sidecar))
        head, embedding, spatial_lr_scale = load_scalar_state(sidecar, device="cuda")
        # 合成方式**按 head 自己的配置分派，绝不硬编码**：若对旧 A' 臂（K=1）套用
        # mult 合成，会静默算出 ū_0(1+λ(ū_0−init_u)) 并污染旧评测。
        # combine_bands() 已按 head.band_mix 分派；这里只做一致性自检（真实矛盾才报错）。
        if head.band_mix == "mult" and head.n_bands < 2:
            raise ValueError(f"sidecar 配置自相矛盾：band_mix=mult 但 n_bands={head.n_bands}")
        if head.band_mix == "single" and head.n_bands != 1:
            print("[warn] sidecar: band_mix=single 但 n_bands={} —— band 1+ 会被渲染却"
                  "不参与合成（消融臂语义）".format(head.n_bands))
        lut_n = head.lut_n              # load_scalar_state 从 cfg 挂上来的，无需再读一遍
        lut = make_jet_lut(lut_n).cuda()
        # —— SH 输入块：提升到循环之外（synthesize 每个视角都要用同一份）——
        # degree 一律取 sidecar 的 head.sh_degree：load_ply 会把 active_sh_degree 强设成
        # max_sh_degree（gaussian_model.py:256），渲染侧从 ply 拿不到训练时的真值。
        sh_coeff = None
        if head.use_sh != "none":
            sh_coeff = gaussians.get_features.transpose(1, 2).contiguous()
            check_knn_xyz(head, gaussians.get_xyz)   # 图（若有）与这份 xyz 是否同源
        print("Loaded scalar sidecar iteration {} (embedding {}, lut_n={}, "
              "use_dist={}, n_bands={}, band_mix={}, λ={}, spatial_lr_scale={}, "
              "use_sh={}, sh_degree={}, ctx_knn={}, ctx_dim={}, P={})".format(
                  iteration, tuple(embedding.shape), lut_n, head.use_dist,
                  head.n_bands, head.band_mix, head.band_lambda(), spatial_lr_scale,
                  head.use_sh, head.sh_degree, head.ctx_knn, head.ctx_dim,
                  int(embedding.shape[0])))
        if head.ctx_knn > 0:
            # 渲染侧只读近邻图（load_scalar_state 已经读过并挂上），这里只回显溯源信息
            print("  ctx: {} 形状={} xyz_sha1={}…".format(
                head.knn_file, tuple(head.knn.shape), str(head.xyz_sha1)[:12]))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, head, embedding,
                       spatial_lr_scale, lut, pipeline, background, subset, sh_coeff)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, head, embedding,
                       spatial_lr_scale, lut, pipeline, background, subset, sh_coeff)


if __name__ == "__main__":
    parser = ArgumentParser(description="Scalar-RRF testing script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--subset", type=int, default=0,
                        help="每档渲染帧数上限（0=全部；>0 确定性均匀抽帧，帧名=全局索引）")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering (Scalar-RRF) " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.subset)
