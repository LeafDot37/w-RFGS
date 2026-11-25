# -*- coding: utf-8 -*-
"""阶段 2 · 前馈学生的评测渲染器。

学生不产出新的点云（几何/光学钉死取自种子），只产出一个**头** `head_<N>.pth`。
本脚本重建同一个种子 + 同一个头，把逐高斯射频外观算出来再渲染。

输出布局**与 `render_corf.py` 逐字一致**，好让 `metrics_fast.py` 直接吃：

    <out_root>/test/ours_<i>/{renders,gt}       ← 频谱测试集（640 个视角）

用法：
    python render_student.py -m <学生模型目录> --head head_30000.pth \\
        --feature_cache <vis_opt_all_r2.pt> --rf_source_path <3dgs_MVDR_100> \\
        --out_root <学生模型目录>/rf_eval --iteration 30000 --subset 0

⚠ `--out_root` 必须与 `--rf_source_path` 区分开；也不要用 `-m` 当 out_root 的默认值
（会把频谱渲染写进学生目录根，与实验元数据混在一起）。

⚠ **本文件与 `render_corf.py` 是两份**。原因是阶段 1 的驱动正在跑、
稍后还要调 `render_corf.py`，动它会改变一个在跑管线的行为。两份的渲染口径
（`render(view, view_obj, pipeline, background)`、`ours_<i>/{renders,gt}`、
`{0:05d}.png`、subset 步长）**逐字相同**，改一处要同步另一处。
"""

from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
import sys
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corf.dual_scene import load_rf_cameras      # noqa: E402
from corf.rf_head import RFHead, xyz_norm_stats  # noqa: E402
from splat.gaussian_renderer import render       # noqa: E402
from splat.utils.general_utils import safe_state  # noqa: E402
from rfgs.coupled.train_student import StudentView, load_seed, _Pipe   # noqa: E402


def render_set(out_root, name, iteration, views, view_obj, pipeline, background, subset=0):
    render_path = os.path.join(out_root, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(out_root, name, "ours_{}".format(iteration), "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    n = len(views)
    step = max(1, n // subset) if (subset and subset > 0 and subset < n) else 0
    cnt = 0
    # ⚠ 必须包 no_grad：`render()` 内部自建 `screenspace_points(requires_grad=True)`
    # （`gaussian_renderer/__init__.py:26`），不包的话 640 张图每张都白建一张计算图
    # 并驻留显存。阶段 1 的 `render_corf.py` 是包了的，这里与它对齐。
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering student")):
            if step:
                if idx % step != 0:
                    continue
                if cnt >= subset:
                    break
                cnt += 1
            rendering = render(view, view_obj, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, "{0:05d}.png".format(idx)))
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, "{0:05d}.png".format(idx)))


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--head", required=True, help="头权重（head_<N>.pth，也可给绝对路径）")
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--out_root", default=None, help="缺省 = <model_path>/rf_eval")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="0 = 全部 640 个测试视角")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--white_background", action="store_true",
                    help="必须与训练该头时一致，也与阶段 1 一致（现为 False）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    safe_state(args.quiet)
    out_root = args.out_root or os.path.join(args.model_path, "rf_eval")

    head_path = args.head if os.path.isabs(args.head) else os.path.join(args.model_path, args.head)
    ck = torch.load(head_path, map_location="cpu")
    cfg = ck.get("config", {})
    it = args.iteration or int(ck.get("iteration", 0))

    cache = torch.load(args.feature_cache, map_location="cpu")
    xyz = cache["xyz"].cuda().contiguous()
    use_visual = bool(cfg.get("use_visual", True))
    vis = cache["vis"].float().cuda().contiguous() if use_visual else None
    if cfg.get("shuffle_vis", False):
        g = torch.Generator(device="cpu").manual_seed(12345)
        vis = vis[torch.randperm(vis.shape[0], generator=g).to(vis.device)].contiguous()

    from splat.scene.gaussian_model import GaussianModel
    gaussians = GaussianModel(3)
    load_seed(cache["meta"]["ckpt"], gaussians, "cuda")

    # 位置编码的频带数**必须从 cfg 读**，不能取默认值 —— 与训练时必须一致。
    # 归一化统计量从同一个 xyz 重算，并与 cfg 里落盘的明文**对拍**：
    # 不一致说明两颗种子/缓存不是同一份，位置编码的尺度两边就不同了，宁可中止。
    pe_bands = int(cfg.get("pe_bands", 0))
    pe_mean = pe_std = None
    if pe_bands > 0:
        pe_mean, pe_std = xyz_norm_stats(xyz)
        if "pe_mean" in cfg:
            d = max(abs(a - b) for a, b in zip(cfg["pe_mean"], [float(v) for v in pe_mean]))
            if d > 1e-6:
                raise SystemExit(
                    f"位置编码统计量对不上：cfg 里记的与本次从缓存重算的差 {d:.3e}。"
                    f"缓存可能不是训练这个头时用的那一份 —— 先查 --feature_cache。")
    head = RFHead(use_visual=use_visual, hidden=int(cfg.get("hidden", 256)),
                  n_layers=int(cfg.get("n_layers", 3)), pe_bands=pe_bands,
                  xyz_mean=pe_mean, xyz_std=pe_std).cuda()
    head.load_state_dict(ck["head"])
    head.eval()

    with torch.no_grad():
        op, sh = head(xyz, vis)
    view_obj = StudentView(gaussians)
    view_obj.set_rf(op, sh)
    print("[render-student] 头={}  迭代 {}  use_visual={}  P={}".format(
        os.path.basename(head_path), it, use_visual, xyz.shape[0]))

    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    train_cams, test_cams = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                                            args.rf_data_device, verbose=not args.quiet)
    if not args.skip_test:
        render_set(out_root, "test", it, test_cams, view_obj, _Pipe(), background, args.subset)
    if not args.skip_train:
        render_set(out_root, "train", it, train_cams, view_obj, _Pipe(), background,
                   args.subset if args.subset else 24)
    print("[render-student] 完成 -> {}".format(out_root))


if __name__ == "__main__":
    main()
