# -*- coding: utf-8 -*-
"""「光学点云 → 射频点云」前馈模型的评测渲染器。

输出布局与 `render_coupled_student.py` / `render_corf.py` 逐字一致，好让
`tools/metrics_fast.py` 直接吃：

    <out_root>/rf/test/ours_<i>/{renders,gt}    射频测试集（640 视角 / 160 留出位置）

⚠ 几何与光学外观**原样来自 `--cloud_ply`**，不重建、不优化。
  推理路径 = 读点云 → 一次前向 → 逐点射频属性 → 光栅化。
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

from corf.dual_scene import load_rf_cameras                    # noqa: E402
from corf.geo_head import GeoToRfHead, app_stats, build_app    # noqa: E402
from corf.rf_head import xyz_norm_stats                        # noqa: E402
from splat.color_mlp import _dist, make_dir                    # noqa: E402
from splat.color_scalar import (apply_jet, combine_bands,     # noqa: E402
                                make_jet_lut, to_override_color)
from splat.gaussian_renderer import render                     # noqa: E402
from splat.scene.gaussian_model import GaussianModel           # noqa: E402
from splat.utils.general_utils import safe_state               # noqa: E402
from rfgs.coupled.train_student import StudentView, _Pipe                   # noqa: E402


def render_set(out_root, name, iteration, views, view_obj, pipeline, background, subset=0,
               override=None, head=None, lut=None, scalar_fn=None):
    """三条路径：
      scalar_fn 非空 ⇒ 每个视角现算标量（视线条件化），再混合查表；
      override 非空 ⇒ 固定标量场（仅位置），混合查表；
      两者皆空   ⇒ SH 路径（原行为）。"""
    d = os.path.join(out_root, "rf", name, f"ours_{iteration}")
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
            if scalar_fn is not None:
                ov = scalar_fn(view)
                mixed = render(view, view_obj, pipeline, background,
                               override_color=ov)["render"]
                img = apply_jet(combine_bands(mixed, head), lut)
            elif override is not None:
                mixed = render(view, view_obj, pipeline, background,
                               override_color=override)["render"]
                img = apply_jet(combine_bands(mixed, head), lut)
            else:
                img = render(view, view_obj, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(img, os.path.join(rp, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(gp, f"{idx:05d}.png"))


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--cloud_ply", required=True)
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    safe_state(args.quiet)
    out_root = args.out_root or os.path.join(args.model_path, "eval")
    hp = args.head if os.path.isabs(args.head) else os.path.join(args.model_path, args.head)
    ck = torch.load(hp, map_location="cpu")
    cfg = ck["config"]
    it = args.iteration or int(ck.get("iteration", 0))

    gm = GaussianModel(3)
    gm.load_ply(args.cloud_ply)
    for n in ("_xyz", "_scaling", "_rotation", "_features_dc", "_features_rest", "_opacity"):
        getattr(gm, n).requires_grad = False
    p = gm.get_xyz.shape[0]
    if p != int(cfg["n_points"]):
        raise SystemExit(f"点云点数 {p} != 训练时的 {cfg['n_points']} —— 不是同一份点云")
    xyz = gm.get_xyz.detach().float().contiguous()

    app_dim = int(cfg.get("app_dim", 0))
    app = build_app(gm) if app_dim > 0 else None
    if app is not None:
        # 统计量必须与训练时逐字相同：不重新估计，只用来校验点云没换
        a_mean, a_std = app_stats(app)
        d = max(abs(float(x) - float(y)) for x, y in zip(a_mean, cfg["app_mean"]))
        if d > 1e-4:
            raise SystemExit(f"光学属性均值对不上（差 {d:.3e}）—— 点云不是训练那一份")

    pe_mean, pe_std = xyz_norm_stats(xyz)
    d = max(abs(float(x) - float(y)) for x, y in zip(pe_mean, cfg["pe_mean"]))
    if d > 1e-6:
        raise SystemExit(f"位置统计量对不上（差 {d:.3e}）—— 点云不是训练那一份")

    out_kind = cfg.get("out_kind", "sh")
    use_dir = bool(cfg.get("use_dir", 0))
    pe_dir_bands = int(cfg.get("pe_dir_bands", 0))
    use_dist = bool(cfg.get("use_dist", 0))
    scene_scale = float(cfg.get("scene_scale", 0.0))
    head = GeoToRfHead(app_dim=app_dim, hidden=int(cfg.get("hidden", 64)),
                       n_layers=int(cfg.get("n_layers", 3)),
                       pe_bands=int(cfg.get("pe_bands", 6)),
                       pe_mean=cfg["pe_mean"], pe_std=cfg["pe_std"],
                       app_mean=cfg.get("app_mean"), app_std=cfg.get("app_std"),
                       sh_bound=float(cfg.get("sh_bound", 2.0)),
                       out_kind=out_kind, n_bands=int(cfg.get("n_bands", 1)),
                       init_u=float(cfg.get("init_u", 0.5)),
                       use_dir=use_dir, pe_dir_bands=pe_dir_bands,
                       use_dist=use_dist).cuda()
    head.load_state_dict(ck["head"])
    head.eval()
    print(f"[render-g2r] {os.path.basename(hp)} it={it} app_dim={app_dim} P={p} "
          f"out_kind={out_kind} use_dir={use_dir}")

    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    view = StudentView(gm)
    override, lut, scalar_fn = None, None, None
    if out_kind == "sh":
        with torch.no_grad():
            op, attr = head(xyz, app)
        view.set_rf(op, attr)
    else:
        lut = make_jet_lut(256).cuda()
        if use_dir:
            # 视线条件化：标量随相机变 ⇒ 每渲一个视角都要重跑一次头
            def scalar_fn(v):
                dv = make_dir(xyz, v.camera_center)
                dd = (_dist(xyz, v.camera_center, scene_scale)
                      if use_dist else None)
                with torch.no_grad():
                    o, s = head(xyz, app, dv, dd)
                view.set_rf_scalar(o)
                return to_override_color(s)
        else:
            with torch.no_grad():
                op, attr = head(xyz, app)
            view.set_rf_scalar(op)
            override = to_override_color(attr)

    train_cams, test_cams = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                                            args.rf_data_device, verbose=not args.quiet)
    render_set(out_root, "test", it, test_cams, view, _Pipe(), background, args.subset,
               override=override, head=head, lut=lut, scalar_fn=scalar_fn)
    if not args.skip_train:
        render_set(out_root, "train", it, train_cams, view, _Pipe(), background,
                   args.subset if args.subset else 24,
                   override=override, head=head, lut=lut, scalar_fn=scalar_fn)
    print(f"[render-g2r] 完成 -> {out_root}")


if __name__ == "__main__":
    main()
