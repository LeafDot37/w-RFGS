# -*- coding: utf-8 -*-
"""阶段 2 · 光学↔射频**耦合蒸馏**的前馈学生。

相对 `train_student.py` 的三处扩展
--------------------------------------------------------------------------------
1. **两个外观头**：光学球谐 + 射频球谐，都从共享主干出来
   （`corf/coupled_head.py`）。`train_student.py` 只有射频头，光学一半钉死。
2. **证据可变**：`--evidence {pos,opt,rf,opt_rf}`，配合 `--modality_dropout`
   让同一个权重覆盖 {只光学}、{只射频}、{光学+射频} 三种输入。
3. **两种渲染损失**：光学视角 + 射频视角各渲一张，各自对自己模态的 GT。

几何全程冻结（MVSplat 前馈点云，1 277 062 点，与阶段 1 逐点同序）。
「前馈」在这里指：**推理时没有任何逐场景优化**，一次前向出全部逐高斯属性。

蒸馏信号
--------------------------------------------------------------------------------
    L = λ_opt·L_opt + λ_rf·L_rf + λ_attr·L_attr

- `L_opt` / `L_rf`：对该模态 GT 图像的 L1 + DSSIM（主信号）
- `L_attr`：对阶段 1「冻结几何」老师逐点属性的均方归一回归（可关，`--teacher_ckpt`）
  —— 老师的几何与点序都和学生**逐点相同**，所以能直接同序号回归，不需要最近邻。

⚠ 光学数据集的 161 个 test 帧是 train/ 图像的重复引用（`docs` 与
`tools/make_student_features.py:58-68` 都记了）⇒ **光学没有真留出**，
`L_opt` 与光学评测都只是「重建」口径。真正按位置留出的是射频（test 160 位置）。

用法（主臂：光学证据 + 射频证据 + 模态丢弃）：
    python train_coupled_student.py -m <out> \
        --feature_cache results/student_feats/vis_mvsplat_full_r2.pt \
        --rf_evidence results/student_feats/rf_ev_mvsplat.pt \
        --rf_source_path datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 \
        --opt_source_path datasets/RF-3DGS_dataset/blender_visual_dataset \
        --evidence opt_rf --modality_dropout \
        --teacher_ckpt <冻结几何老师>/point_cloud/iteration_40000/coupled_rf_40000.pth
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corf.coupled_head import EVIDENCE_KINDS, build          # noqa: E402
from corf.dual_scene import load_rf_cameras                  # noqa: E402
from corf.rf_head import xyz_norm_stats                      # noqa: E402
from splat.gaussian_renderer import render                   # noqa: E402
from splat.scene.gaussian_model import GaussianModel         # noqa: E402
from splat.utils.general_utils import safe_state             # noqa: E402
from splat.utils.loss_utils import l1_loss, ssim             # noqa: E402
from rfgs.coupled.train_student import StudentView, _Pipe, load_seed, rgb_loss   # noqa: E402
from tools.make_student_features import load_optical_cameras  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pick_evidence(has_opt: bool, has_rf: bool, dropout: bool,
                  rng: random.Random) -> tuple[bool, bool]:
    """本步用哪几支证据。

    模态丢弃只在**本臂可用的子集**里抽，且不抽空集。
    `pos` 臂两支都没有 ⇒ 恒 (False, False)，此时 head 会拒绝，所以单独处理。
    """
    if not has_opt and not has_rf:
        return False, False
    if not dropout:
        return has_opt, has_rf
    choices = []
    if has_opt:
        choices.append((True, False))
    if has_rf:
        choices.append((False, True))
    if has_opt and has_rf:
        choices.append((True, True))
    return rng.choice(choices)


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--rf_evidence", default="",
                    help="make_rf_evidence.py 的产物；evidence 含 rf 时必填")
    ap.add_argument("--rf_source_path", required=True)
    ap.add_argument("--rf_images", default="images")
    ap.add_argument("--rf_data_device", default="cpu")
    ap.add_argument("--opt_source_path", required=True)
    ap.add_argument("--opt_resolution", type=int, default=2)
    ap.add_argument("--teacher_ckpt", default="", help="阶段 1 冻结几何老师 sidecar")
    ap.add_argument("--evidence", choices=EVIDENCE_KINDS, default="opt_rf")
    ap.add_argument("--modality_dropout", action="store_true")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--pe_bands", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_opt", type=float, default=1.0)
    ap.add_argument("--lambda_rf", type=float, default=1.0)
    ap.add_argument("--lambda_attr", type=float, default=0.1)
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--opt_every", type=int, default=1, help="每几步渲一次光学（省算力用）")
    ap.add_argument("--rf_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_iterations", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations 必须 > 0（本仓明令防「成功退出、什么都没训」）")
    if args.evidence in ("rf", "opt_rf") and not args.rf_evidence:
        raise SystemExit(f"--evidence {args.evidence} 需要 --rf_evidence")
    if args.modality_dropout and args.evidence not in ("opt_rf",):
        raise SystemExit("--modality_dropout 只对 --evidence opt_rf 有意义；"
                         "单支证据的臂本来就没有可丢的模态")

    safe_state(args.quiet)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.model_path, exist_ok=True)

    # ---------------------------------------------------------------- 缓存
    cache = torch.load(args.feature_cache, map_location="cpu")
    xyz = cache["xyz"].cuda().contiguous()
    p = xyz.shape[0]
    vis = cache["vis"].float().cuda().contiguous()
    rf = None
    rf_dim = 0
    if args.evidence in ("rf", "opt_rf"):
        ev = torch.load(args.rf_evidence, map_location="cpu")
        if ev["xyz"].shape[0] != p or (ev["xyz"] - cache["xyz"]).abs().max() > 1e-6:
            raise SystemExit(
                "射频证据与特征缓存的点云不是同一份/不同序 —— "
                f"P {ev['xyz'].shape[0]} vs {p}。逐点证据必须逐点对齐，不能硬拼。")
        rf = ev["rf_emb"].float().cuda().contiguous()
        rf_dim = rf.shape[1]
        log(f"[cpl] 射频证据 {tuple(rf.shape)}  "
            f"（{ev['meta']['n_positions_train']} 训练位置 × "
            f"{ev['meta']['n_orient']} 朝向，PCA {ev['meta']['pca_dim']}）")

    # ---------------------------------------------------------------- 几何
    gaussians = GaussianModel(3)
    first_iter = load_seed(cache["meta"]["ckpt"], gaussians, "cuda")
    if gaussians.get_xyz.shape[0] != p:
        raise SystemExit(f"点数不一致：缓存 {p} vs 种子 {gaussians.get_xyz.shape[0]}")

    # ---------------------------------------------------------------- 老师
    teacher = None
    t_scale = None
    if args.teacher_ckpt:
        t = torch.load(args.teacher_ckpt, map_location="cpu")
        tcfg = t.get("config", {})
        seed_here = os.path.abspath(cache["meta"]["ckpt"])
        seed_there = os.path.abspath(str(tcfg.get("start_checkpoint", "")))
        if seed_there and seed_here != seed_there:
            raise SystemExit(
                "老师的起点与特征缓存的种子不是同一个 —— 逐点属性蒸馏会错位。\n"
                f"  cache: {seed_here}\n  teacher: {seed_there}")
        if t["opacity_rf"].shape[0] != p:
            raise SystemExit(f"老师点数 {t['opacity_rf'].shape[0]} != 学生点数 {p}")
        teacher = {k: t[k].cuda().float() for k in
                   ("opacity_rf", "features_rf_dc", "features_rf_rest")}
        # 均方归一：零预测下每项 = 1，λ_attr 才是可解释的系数
        t_scale = {k: v.pow(2).mean().clamp_min(1e-12) for k, v in teacher.items()}
        log("[cpl] 属性蒸馏尺度（零预测基线=1）："
            + "  ".join(f"{k}={float(v):.4g}" for k, v in t_scale.items()))
    elif args.lambda_attr > 0:
        log("[cpl] 未给 --teacher_ckpt ⇒ 属性蒸馏项关闭（只用渲染损失）")

    # ---------------------------------------------------------------- 相机
    rf_train, rf_test = load_rf_cameras(args.rf_source_path, args.rf_images, -1,
                                        args.rf_data_device, verbose=not args.quiet)
    opt_cams = load_optical_cameras(args.opt_source_path, args.opt_resolution, False)
    log(f"[cpl] 光学视角 {len(opt_cams)}（去重后）  射频训练视角 {len(rf_train)}")
    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # ---------------------------------------------------------------- 头
    pe_mean, pe_std = xyz_norm_stats(xyz)
    head = build(args.evidence, cache, rf_dim=rf_dim, hidden=args.hidden,
                 n_layers=args.n_layers, pe_bands=args.pe_bands,
                 chunk=args.chunk).cuda()
    n_param = sum(q.numel() for q in head.parameters())
    opt_view, rf_view = StudentView(gaussians), StudentView(gaussians)
    optm = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-15)

    cfg = {**vars(args), "n_points": p, "first_iter_seed": first_iter,
           "n_opt_cams": len(opt_cams), "n_rf_train_cams": len(rf_train),
           "n_rf_test_cams": len(rf_test), "head_params": n_param,
           "rf_dim": rf_dim,
           "pe_mean": [float(v) for v in pe_mean],
           "pe_std": [float(v) for v in pe_std]}
    with open(os.path.join(args.model_path, "student_experiment.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    log(f"[cpl] P={p}  头参数量={n_param}  evidence={args.evidence}  "
        f"dropout={args.modality_dropout}  c_in={head.trunk[0].in_features}")

    # ---------------------------------------------------------------- 训练
    t0 = time.time()
    ema_opt = ema_rf = None
    n_opt_render = n_rf_render = 0
    for it in range(1, args.iterations + 1):
        use_opt, use_rf = pick_evidence(head.has_opt, head.has_rf,
                                        args.modality_dropout, rng)
        oo, osh, ro, rsh = head(xyz, vis, rf, use_opt=use_opt, use_rf=use_rf)

        L = xyz.new_zeros(())
        l_opt = l_rf = 0.0
        if it % args.opt_every == 0:
            cam = opt_cams[it % len(opt_cams)]
            opt_view.set_rf(oo, osh)
            pkg = render(cam, opt_view, _Pipe(), background)
            lo, l1o = rgb_loss(pkg["render"], cam.original_image[:3].cuda(),
                               args.lambda_dssim)
            L = L + args.lambda_opt * lo
            l_opt = float(l1o.item())
            n_opt_render += 1
            ema_opt = l_opt if ema_opt is None else 0.99 * ema_opt + 0.01 * l_opt
        if it % args.rf_every == 0:
            cam = rf_train[it % len(rf_train)]
            rf_view.set_rf(ro, rsh)
            pkg = render(cam, rf_view, _Pipe(), background)
            lr_, l1r = rgb_loss(pkg["render"], cam.original_image[:3].cuda(),
                                args.lambda_dssim)
            L = L + args.lambda_rf * lr_
            l_rf = float(l1r.item())
            n_rf_render += 1
            ema_rf = l_rf if ema_rf is None else 0.99 * ema_rf + 0.01 * l_rf

        attr_v = 0.0
        if teacher is not None and args.lambda_attr > 0:
            dc, rest = rsh[:, :1, :], rsh[:, 1:, :]
            pairs = ((ro, "opacity_rf"), (dc, "features_rf_dc"), (rest, "features_rf_rest"))
            la = xyz.new_zeros(())
            for pred, key in pairs:
                la = la + F.mse_loss(pred, teacher[key]) / t_scale[key]
            L = L + args.lambda_attr * la
            attr_v = float(la.item())

        if not L.requires_grad:
            raise SystemExit("损失没有梯度 —— 检查渲染/头是否真的接上了")
        L.backward()
        with torch.no_grad():
            optm.step()
        optm.zero_grad(set_to_none=False)

        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            msg = (f"[cpl] {it:6d}/{args.iterations}  "
                   f"L1opt={l_opt:.5f}(ema {ema_opt if ema_opt is None else round(ema_opt,5)})  "
                   f"L1rf={l_rf:.5f}(ema {ema_rf if ema_rf is None else round(ema_rf,5)})")
            if teacher is not None:
                msg += f"  attr={attr_v:.5f}"
            msg += f"  ev={'O' if use_opt else '-'}{'R' if use_rf else '-'}"
            msg += f"  {el / it:.2f}s/it  剩 {(args.iterations - it) * el / it / 60:.0f} 分"
            log(msg)
            if it <= 5 * args.log_every:
                log(f"[cpl]   mem {torch.cuda.memory_allocated() / 2**30:.2f} GB "
                    f"alloc / {torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak")

        if it in args.save_iterations:
            torch.save({"iteration": it, "head": head.state_dict(), "config": cfg},
                       os.path.join(args.model_path, f"head_{it}.pth"))
            log(f"[cpl] 存头 -> head_{it}.pth")

    torch.save({"iteration": args.iterations, "head": head.state_dict(), "config": cfg},
               os.path.join(args.model_path, f"head_{args.iterations}.pth"))
    log(f"[cpl] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟  "
        f"渲染次数 opt={n_opt_render} rf={n_rf_render}")


if __name__ == "__main__":
    main()
