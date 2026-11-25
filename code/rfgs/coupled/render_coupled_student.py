# -*- coding: utf-8 -*-
"""耦合学生的评测渲染器：同一套权重，按指定的**输入组成**渲两个模态。

输出布局与 `render_student.py` / `render_corf.py` 逐字一致，好让
`tools/metrics_fast.py` 直接吃：

    <out_root>/rf/test/ours_<i>/{renders,gt}    射频测试集（640 视角 / 160 位置）
    <out_root>/opt/test/ours_<i>/{renders,gt}   光学测试集（161 视角，**是训练图的重复引用**）

模态选择
--------------------------------------------------------------------------------
`--use_opt/--use_rf` 决定推理时喂哪几支证据：
    1 0  只给光学      （「只输入光学」）
    1 1  光学 + 射频
    0 1  只给射频      （「只输入射频」）
被关掉的支路喂 0 向量 + 标志位 0，与训练时模态丢弃的口径逐字相同。

⚠ 本脚本**不重建几何**：几何就是 `--feature_cache` 里那份 MVSplat 点云。
  「只输入射频」时几何仍然来自这份点云 —— 射频测量本身不含空间几何信息，
  这是信息论边界，不是实现偷懒（见 `docs/CORF复现与前馈蒸馏_实施记录.md` §2）。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
import sys
from argparse import ArgumentParser, Namespace

import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corf.coupled_head import build                     # noqa: E402
from corf.dual_scene import load_rf_cameras             # noqa: E402
from corf.rf_head import xyz_norm_stats                 # noqa: E402
from splat.gaussian_renderer import render              # noqa: E402
from splat.scene.gaussian_model import GaussianModel    # noqa: E402
from splat.utils.general_utils import safe_state        # noqa: E402
from rfgs.coupled.train_student import StudentView, _Pipe, load_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_optical_eval_cams(source_path, resolution):
    """光学 test 相机（161 个）。

    走引擎 `Scene(eval=True)`：读数与阶段 1 的 161 视角口径对齐。
    ⚠ 这 161 帧的 `file_path` 指回 `train/`（`make_student_features.py:58-68`），
    **不是真正的留出视角** —— 光学指标只能叫「重建」。
    """
    from splat.scene import Scene

    os.environ.setdefault("RF_POINTCLOUD", os.path.join(ROOT, "results/mvsplat_clouds/init.ply"))
    a = Namespace(source_path=source_path, images="images", eval=True,
                  resolution=resolution, data_device="cpu", sh_degree=3,
                  model_path=os.path.join(ROOT, "results/_student_feat"),
                  white_background=False)
    os.makedirs(a.model_path, exist_ok=True)
    scene = Scene(a, GaussianModel(3), shuffle=False)
    return scene.getTestCameras()


def render_set(out_root, name, iteration, views, view_obj, pipeline, background, subset=0):
    d = os.path.join(out_root, name, f"ours_{iteration}")
    rp, gp = os.path.join(d, "renders"), os.path.join(d, "gt")
    os.makedirs(rp, exist_ok=True)
    os.makedirs(gp, exist_ok=True)
    n = len(views)
    step = max(1, n // subset) if (subset and 0 < subset < n) else 0
    cnt = 0
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc=f"render {name}")):
            if step:
                if idx % step != 0:
                    continue
                if cnt >= subset:
                    break
                cnt += 1
            img = render(view, view_obj, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(img, os.path.join(rp, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(gp, f"{idx:05d}.png"))


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--rf_evidence", default="")
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--opt_source_path", required=True)
    ap.add_argument("--opt_resolution", type=int, default=2)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--use_opt", type=int, default=1)
    ap.add_argument("--use_rf", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_opt", action="store_true")
    ap.add_argument("--skip_rf", action="store_true")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    safe_state(args.quiet)
    out_root = args.out_root or os.path.join(args.model_path, "eval")
    hp = args.head if os.path.isabs(args.head) else os.path.join(args.model_path, args.head)
    ck = torch.load(hp, map_location="cpu")
    cfg = ck["config"]
    it = args.iteration or int(ck.get("iteration", 0))
    evidence = cfg["evidence"]
    use_opt, use_rf = bool(args.use_opt), bool(args.use_rf)
    if use_opt and not (evidence in ("opt", "opt_rf")):
        raise SystemExit(f"该头 evidence={evidence} 没有光学支路，不能 --use_opt 1")
    if use_rf and not (evidence in ("rf", "opt_rf")):
        raise SystemExit(f"该头 evidence={evidence} 没有射频支路，不能 --use_rf 1")
    if not (use_opt or use_rf) and evidence != "pos":
        raise SystemExit("--use_opt 0 --use_rf 0：至少留一支（pos 臂除外）")

    cache = torch.load(args.feature_cache, map_location="cpu")
    xyz = cache["xyz"].cuda().contiguous()
    vis = cache["vis"].float().cuda().contiguous()
    rf = None
    rf_dim = int(cfg.get("rf_dim", 0))
    if evidence in ("rf", "opt_rf"):
        if not args.rf_evidence:
            raise SystemExit(f"evidence={evidence} 需要 --rf_evidence")
        ev = torch.load(args.rf_evidence, map_location="cpu")
        rf = ev["rf_emb"].float().cuda().contiguous()
        if rf.shape[1] != rf_dim:
            raise SystemExit(f"射频证据维度 {rf.shape[1]} != 头配置里的 {rf_dim}")

    gaussians = GaussianModel(3)
    load_seed(cache["meta"]["ckpt"], gaussians, "cuda")

    pe_bands = int(cfg.get("pe_bands", 0))
    pe_mean = pe_std = None
    if pe_bands > 0:
        pe_mean, pe_std = xyz_norm_stats(xyz)
        if "pe_mean" in cfg:
            d = max(abs(a - b) for a, b in zip(cfg["pe_mean"], [float(v) for v in pe_mean]))
            if d > 1e-6:
                raise SystemExit(f"位置编码统计量对不上（差 {d:.3e}）—— 缓存不是训练那一份")
    head = build(evidence, cache, rf_dim=rf_dim, hidden=int(cfg.get("hidden", 64)),
                 n_layers=int(cfg.get("n_layers", 3)), pe_bands=pe_bands).cuda()
    head.load_state_dict(ck["head"])
    head.eval()

    with torch.no_grad():
        oo, osh, ro, rsh = head(xyz, vis, rf, use_opt=use_opt, use_rf=use_rf)
    print(f"[render-cpl] {os.path.basename(hp)} it={it} evidence={evidence} "
          f"输入={'O' if use_opt else '-'}{'R' if use_rf else '-'} P={xyz.shape[0]}")

    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    optv, rfv = StudentView(gaussians), StudentView(gaussians)
    optv.set_rf(oo, osh)
    rfv.set_rf(ro, rsh)

    if not args.skip_rf:
        train_cams, test_cams = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                                                args.rf_data_device, verbose=not args.quiet)
        render_set(os.path.join(out_root, "rf"), "test", it, test_cams, rfv,
                   _Pipe(), background, args.subset)
        if not args.skip_train:
            render_set(os.path.join(out_root, "rf"), "train", it, train_cams, rfv,
                       _Pipe(), background, args.subset if args.subset else 24)
    if not args.skip_opt:
        test_cams = load_optical_eval_cams(args.opt_source_path, args.opt_resolution)
        render_set(os.path.join(out_root, "opt"), "test", it, test_cams, optv,
                   _Pipe(), background, args.subset)
    print(f"[render-cpl] 完成 -> {out_root}")


if __name__ == "__main__":
    main()
