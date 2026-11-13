# -*- coding: utf-8 -*-
"""
train_rrf_scalar.py —— RRF 阶段二变体：颜色头 → 「标量头 + 固定 jet 查表」（形态 A'）
--------------------------------------------------------------------------------
与 train_rrf_mlp.py **逐行同构**，唯一差异是颜色的产生方式：

    MLP 臂（对照）：colors = MLP(e, dir, …) → (P,3) RGB
                    image  = render(override_color=colors)          ← RGB 域混合
    A' 臂（本档）：  s      = MLP(e, dir, …) → (P,1) 标量
                    image0 = render(override_color=s.repeat(1,3))   ← 只借它做标量混合
                    image  = apply_jet(image0, lut)                 ← 再查表

即把"先查表再混合"换成"**先混合再查表**"。查表在 `render()` **之外**做，故
gaussian_renderer / gaussian_model / train.py 一行不改（本档为入口变体）。

形态 A×（2026-09-12 加，`--n_bands 2 --band_mix mult`）
--------------------------------------------------------------------------------
A' 在 CBF 上渲染通道 std 37.05 远低于 GT 51.08（过平滑），MVDR 则几乎持平。本档给标量头
**多开一个 band** 并在标量域做**乘法调制**，直击该症状：

    s      = head(e, dir, …) → (P,2)
    mixed  = render(override_color=cat[s_0, s_1, s_1])   # 借光栅化逐通道线性，一次出两个场
    u      = ū_0 · (1 + λ·(ū_1 − init_u))                # λ = sigmoid(lam_raw) ∈ (0,1)
    image  = apply_jet(u, lut)                           # 只查一次表 ⇒ 输出仍在 jet 上

**λ=0 时精确退化成 A'** ⇒ A× 严格包含 A'，对照是嵌套的。默认 `--n_bands 1 --band_mix single`
与旧 A' 行为**逐字相同**（旧 sidecar 也照旧读得进来）。设计与不可塌缩性证明见
`splat/color_scalar.py` 模块头部。

为什么这不是同一函数类的换参数化（见 splat/color_scalar.py 头部）
    Σ wᵢ·LUT(sᵢ)  ≠  LUT(Σ wᵢ·sᵢ)      （jet 非线性，与线性混合不可交换）
GT 像素 100% 在调色板上，故 A' 的输出**按构造永远在板上**，原版则不然。

其余语义与 train_rrf_mlp.py / 原 RRF 完全一致：
  - 起点：阶段一视觉 chkpnt（--start_checkpoint）
  - 冻结几何 xyz/scaling/rotation；opacity 可训；densify=0；L1+SSIM(λ=0.2)
  - 每 save_iteration 档写 point_cloud.ply + sidecar color_scalar_<i>.pth
  - PE 三开关与 LR 调度参数**与 MLP 臂同名同义**（复用 color_mlp 的 PE 实现），
    保证是单变量对比

用法（服务器 code/ 目录，gs3d env）：
    <PY> train_rrf_scalar.py -s <P_SPECTRUM>/3dgs_<M>_100 -m <out_model_dir> \
        --start_checkpoint <视觉chkpnt30000.pth> --iterations 70000 \
        --save_iterations 32000 35000 ... 65000 --densify_until_iter 0 \
        --data_device cpu --quiet \
        --lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 \
        --lr_max_steps 40000 --pe_xyz_L 5 --pe_dir_L 3 --use_dist
评测链：render_rrf_scalar.py（同款后处理）+ 现有 metrics_fast.py / metrics.py，口径不变。
"""
import os
import sys
import json
import torch
from torch import nn
from random import randint
from splat.utils.loss_utils import l1_loss, ssim
from splat.gaussian_renderer import render
from splat.scene import Scene, GaussianModel
from splat.utils.general_utils import safe_state, get_expon_lr_func
from splat.color_scalar import (ScalarColorHead, compute_scalar, apply_jet,
                                make_jet_lut, save_scalar_state, set_scene_norm,
                                to_override_color, combine_bands,
                                build_or_load_knn, check_knn_xyz)
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from splat.arguments import ModelParams, PipelineParams, OptimizationParams
import uuid


def load_visual_geometry(chkpnt_path: str, gaussians: GaussianModel,
                         sh_degree: int):
    """轻量 restore：仅取几何/外观参数（与 train.py restore 同源解包），不重建 6 组 Adam。
    参数顺序必须与 GaussianModel.capture() 一致。返回绝对起始迭代 first_iter。"""
    ckpt = torch.load(chkpnt_path, map_location="cuda")
    model_params, first_iter = ckpt
    (gaussians.active_sh_degree, xyz, f_dc, f_rest,
     scaling, rotation, opacity, _max_radii2D,
     _xyz_grad_acc, _denom, _opt_dict, _spatial_lr_scale) = model_params

    gaussians.max_sh_degree = sh_degree
    # 一律 `.detach()` 在前。原写法是 `x.contiguous().requires_grad_(False)`，它**依赖
    # x 恰好连续**才不炸：`.contiguous()` 对已连续的张量返回自身（叶子，改 requires_grad 合法），
    # 对非连续的张量则**产生一次 copy**，而 copy 是 autograd 图里的非叶子节点 ⇒
    # 原张量 requires_grad=True 时 `.requires_grad_(False)` 抛
    # "you can only change requires_grad flags of leaf variables"，整个 RRF 节点秒退。
    # 实测（.diag/probe_ckpt_xyz.py）：chkpnt500.pth 的 _xyz 非连续（prune 用布尔掩码切片
    # `self._xyz[mask]` 留下的是 view），其余档连续（densify 的 torch.cat 产物）⇒ **崩不崩
    # 与步数无关，是运气**。detach 不改数值/dtype/device，也不改 `.contiguous()` 是否 copy
    # ⇒ 对本就能过的档**逐位不变**，不构成与历史臂的口径差异。
    gaussians._xyz = nn.Parameter(xyz.detach().contiguous(), requires_grad=False)
    gaussians._scaling = nn.Parameter(scaling.detach().contiguous(), requires_grad=False)
    gaussians._rotation = nn.Parameter(rotation.detach().contiguous(), requires_grad=False)
    # SH 外观：本变体不用（颜色全部由标量头+查表给出），保留占位但不参与训练
    gaussians._features_dc = nn.Parameter(f_dc.detach().contiguous(), requires_grad=False)
    gaussians._features_rest = nn.Parameter(f_rest.detach().contiguous(), requires_grad=False)
    gaussians._opacity = nn.Parameter(opacity.detach().contiguous(), requires_grad=True)
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
        raise RuntimeError("train_rrf_scalar 必须从阶段一视觉 chkpnt 出发（--start_checkpoint）")

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
    init_u = sc["init_u"]
    lut_n = sc["lut_n"]
    n_bands = sc["n_bands"]
    band_mix = sc["band_mix"]
    use_sh = sc["use_sh"]
    ctx_knn = int(sc["ctx_knn"])
    ctx_dim = int(sc["ctx_dim"])
    # SH 求值阶数。**钉进 sidecar cfg 当唯一真源**：load_ply 会把 active_sh_degree 强设成
    # max_sh_degree，而渲染路径从 ply 走 ⇒ 不钉死就会出现训练/渲染 degree 静默不一致。
    sh_degree = (int(gaussians.active_sh_degree) if sc["sh_eval_degree"] < 0
                 else int(sc["sh_eval_degree"]))
    spatial_lr_scale = float(gaussians.spatial_lr_scale or 0.0)

    # 固定 jet 查表（不可训；随 make_jet_lut 构造，不进 optimizer 也不进 sidecar）
    lut = make_jet_lut(lut_n).cuda()

    embed = nn.Parameter(torch.zeros((P, embed_dim), device="cuda").normal_(0, 0.02),
                         requires_grad=True)
    head = ScalarColorHead(embed_dim=embed_dim, hidden=mlp_hidden, depth=mlp_depth,
                           pe_xyz_L=pe_xyz_L, pe_dir_L=pe_dir_L,
                           use_dist=use_dist, init_u=init_u,
                           n_bands=n_bands, band_mix=band_mix,
                           use_sh=use_sh, sh_degree=sh_degree,
                           ctx_knn=ctx_knn, ctx_dim=ctx_dim).cuda()
    if P > 0:
        set_scene_norm(head, gaussians.get_xyz)

    # —— SH 输入块 / 空间上下文块：**全部在训练循环之外**准备好 ——
    sh_coeff = None
    if use_sh != "none":
        # get_features 是 torch.cat（gaussian_model.py:108-111），每次调用新分配一个
        # (N,16,3) ≈ 194 MB；循环里调 3 万次是纯浪费 ⇒ 提升一次。
        # eval_sh 要 [..., C, coeff] ⇒ transpose 到 (N,3,16)。
        sh_coeff = gaussians.get_features.transpose(1, 2).contiguous()
        if sh_coeff.shape[0] != P:
            raise RuntimeError(f"SH 行数 {sh_coeff.shape[0]} != P {P} ⇒ 行序不对齐")
    if ctx_knn > 0:
        if ctx_knn >= P:
            raise RuntimeError(f"--ctx_knn {ctx_knn} >= P {P}：邻居数不能超过点数")
        # 建/读**固定**近邻图。几何冻结（_xyz requires_grad_(False) + densify_until_iter 0）
        # ⇒ 数量与顺序恒定 ⇒ 图只建一次、落 model_path，续跑直接读。
        head.knn, head.xyz_sha1, head.knn_file = build_or_load_knn(
            scene.model_path, gaussians.get_xyz, ctx_knn)
        check_knn_xyz(head, gaussians.get_xyz)

    # 独立 optimizer：只含真正可训项（opacity / embedding / 标量头）
    # λ（lam_raw）落在 "mlp" 组内（list(head.parameters())）⇒ 自动跟随 8e-3→8e-5 的 exp 调度
    gaussians.optimizer = torch.optim.Adam([
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [embed], "lr": embed_lr, "name": "embed"},
        {"params": list(head.parameters()), "lr": mlp_lr_init, "name": "mlp"},
    ], lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = None

    mlp_sched = None
    if lr_schedule == "exp":
        if mlp_lr_final >= mlp_lr_init:
            raise ValueError(f"exp: mlp_lr_final({mlp_lr_final}) 须 < mlp_lr_init({mlp_lr_init})")
        delay = lr_delay_steps if lr_delay_steps is not None else int(0.01 * lr_max_steps)
        mlp_sched = get_expon_lr_func(mlp_lr_init, mlp_lr_final, delay,
                                      lr_delay_mult, lr_max_steps)

    exp_json = os.path.join(scene.model_path, "experiment.json")
    with open(exp_json, "w") as f:
        json.dump({
            "head_type": "scalar_jet",
            "n_bands": n_bands, "band_mix": band_mix,
            "pe_xyz_L": pe_xyz_L, "pe_dir_L": pe_dir_L, "use_dist": use_dist,
            "init_u": init_u, "lut_n": lut_n,
            "embed_lr": embed_lr, "mlp_lr_init": mlp_lr_init,
            "mlp_lr_final": mlp_lr_final, "lr_schedule": lr_schedule,
            "lr_max_steps": lr_max_steps, "lr_delay_steps": lr_delay_steps,
            "lr_delay_mult": lr_delay_mult,
            "spatial_lr_scale": spatial_lr_scale, "start_iter": first_iter,
            "iterations": iterations, "P": P,
            # —— SH 输入块 / 空间上下文块 ——
            "use_sh": use_sh, "sh_degree": sh_degree,
            "ctx_knn": ctx_knn, "ctx_dim": ctx_dim,
            "knn_file": head.knn_file, "xyz_sha1": head.xyz_sha1,
        }, f, indent=2)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # 标量域里 bg=0 ⇒ 未覆盖像素 u=0 ⇒ jet(0) 深蓝（**在调色板上**）；
    # 而 RGB 域里同样的 bg=0 给出纯黑（调色板外）。GT 纯黑占比实测 0.00%。
    if dataset.white_background:
        print("[scalar-rrf] 警告：white_background=True，标量域 bg=1 会把背景推到 jet(1) 红端")

    def log(msg):   # --quiet 时 safe_state 会把 sys.stdout 变成 no-op ⇒ 关键读数一律走 stderr
        print(msg, file=sys.stderr, flush=True)

    log(f"[scalar-rrf] P={P}  embed_dim={embed_dim}  mlp_hidden={mlp_hidden}  "
        f"depth={mlp_depth}  start_iter={first_iter}  ->  {iterations}  "
        f"lut_n={lut_n}  init_u={init_u}  n_bands={n_bands}  band_mix={band_mix}  "
        f"head_params={sum(p.numel() for p in head.parameters())}  "
        f"data={dataset.source_path}")
    if n_bands > 1:
        log(f"[scalar-rrf] 形态 A×：λ 初值 = {head.band_lambda():.4f}（lam_raw 初值 0）"
            f"，λ=0 时精确退化成 A'")
    if use_sh != "none" or ctx_knn > 0:
        # 新块的总览：**每一条都进 stderr**（--quiet 下 stdout 是 no-op）
        log(f"[scalar-rrf] 新输入块：use_sh={use_sh} sh_degree={sh_degree} "
            f"sh_dims={head.sh_dims} | ctx_knn={ctx_knn} ctx_dim={ctx_dim} "
            f"ctx_dims={head.ctx_dims} | in_dim={head.in_dim}"
            f"（A2 基线 = {head.in_dim - head.sh_dims - head.ctx_dims}）")
        if ctx_knn > 0:
            log(f"[scalar-rrf] 近邻图：{head.knn_file} k={ctx_knn} "
                f"xyz_sha1={head.xyz_sha1[:12]}… 形状={tuple(head.knn.shape)}")
        log(f"[scalar-rrf] 新增通路在 net[0] 的权重列**零初始化** ⇒ 给定 A2 的权重，"
            f"新块贡献恰好 0（A4/A5 在初始化处是 A2 的严格嵌套，不是\"碰巧第 0 步相同\"）")
    print(f"[scenario] pe_xyz_L={pe_xyz_L} pe_dir_L={pe_dir_L} "
          f"use_dist={use_dist} | embed_lr={embed_lr:g} "
          f"mlp_lr {mlp_lr_init:g}->{mlp_lr_final:g} schedule={lr_schedule} "
          f"max_steps={lr_max_steps} delay_steps={lr_delay_steps} "
          f"delay_mult={lr_delay_mult} | spatial_lr_scale={spatial_lr_scale}")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 复位显存峰值统计：包裹脚本（tools/_prealloc.py）的预占发生在**本进程**里，
    # 不复位的话 torch.cuda.max_memory_allocated() 会把那 10 GiB 也算进来 ⇒ 假报警。
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _peak_warn_gb = float(os.environ.get("RF_PEAK_WARN_GB", "5.0"))
    _peak_max_gb = float(os.environ.get("RF_PEAK_MAX_GB", "8.0"))
    pbar = tqdm(range(first_iter, iterations), desc="Scalar-RRF training")
    for iteration in range(first_iter + 1, iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        s = compute_scalar(head, embed, gaussians.get_xyz,
                           viewpoint_cam.camera_center, spatial_lr_scale,
                           sh_coeff)                                          # (P,K)
        # **必须打包成 (P,3)**：(P,K) 直传 override_color 不报错但静默产出错误图像。
        # 打包/合成走共享助手，保证与 render_rrf_scalar.py **结构性**逐字一致。
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            override_color=to_override_color(s))
        # 查表在 render 之外：先在各 band 的混合场上做标量域非线性合成，再查一次表
        image = apply_jet(combine_bands(render_pkg["render"], head), lut)

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
                if mlp_sched is not None:
                    _set_group_lr(gaussians.optimizer, "mlp",
                                  mlp_sched(iteration - first_iter))
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 第一步之后量一次显存峰值。上下文块的真实代价不在前向的瞬态 gather 上，
            # 而在 autograd 为 backward 留住的每个 chunk 的 K_g/V_g（约 1.2 GB，见
            # color_scalar.py 模块头）⇒ 只有量一次才知道放不放得下。
            if iteration == first_iter + 1 and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
                log(f"[scalar-rrf] 首步后显存峰值 {peak:.2f} GiB"
                    f"（告警阈 {_peak_warn_gb:g} / 硬阈 {_peak_max_gb:g} GiB）")
                if peak > _peak_max_gb:
                    raise RuntimeError(
                        f"首步显存峰值 {peak:.2f} GiB > RF_PEAK_MAX_GB={_peak_max_gb:g}。"
                        f"先降 --ctx_knn（线性降 K_g/V_g，当前 {ctx_knn}）再重跑；"
                        f"确要放行就调高 RF_PEAK_MAX_GB（但那会顶到显存包裹的上限）。")
                if peak > _peak_warn_gb:
                    log(f"[scalar-rrf] ⚠ 峰值超过告警阈 {_peak_warn_gb:g} GiB，"
                        f"留意显存包裹（RF_VRAM_GB）是否还兜得住")

            if iteration in saving_iterations:
                lam = head.band_lambda()
                lam_s = "" if lam is None else f"  λ={lam:.4f}（初值 0.5000）"
                log(f"[ITER {iteration}] Saving Gaussians + scalar sidecar{lam_s}")
                scene.save(iteration)
                sidecar = os.path.join(
                    scene.model_path, "point_cloud", f"iteration_{iteration}",
                    f"color_scalar_{iteration}.pth")
                save_scalar_state(sidecar, head, embed, iteration,
                                  spatial_lr_scale, lut_n)
    pbar.close()
    print("\nScalar-RRF training complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="Scalar-RRF training (scalar head + fixed jet LUT)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    # —— 与 train_rrf_mlp 同名同义（单变量对比）——
    parser.add_argument("--pe_xyz_L", type=int, default=0)
    parser.add_argument("--pe_dir_L", type=int, default=0)
    parser.add_argument("--use_dist", action="store_true")
    parser.add_argument("--embed_lr", type=float, default=None)
    parser.add_argument("--mlp_lr_init", type=float, default=None)
    parser.add_argument("--mlp_lr_final", type=float, default=None)
    parser.add_argument("--lr_schedule", type=str, choices=["fixed", "exp"],
                        default="fixed")
    parser.add_argument("--lr_max_steps", type=int, default=40000)
    parser.add_argument("--lr_delay_steps", type=int, default=None)
    parser.add_argument("--lr_delay_mult", type=float, default=0.01)
    # —— 形态 A' 专有 ——
    parser.add_argument("--init_u", type=float, default=0.5,
                        help="标量头初始 u（jet 上的初始位置；缺省 0.5 = 亮绿）")
    parser.add_argument("--lut_n", type=int, default=256,
                        help="jet 查表级数（缺省 256，与 matplotlib 默认 LUT 一致）")
    # —— 形态 A× 专有（默认值 = 形态 A'，逐字不变）——
    parser.add_argument("--n_bands", type=int, default=1, choices=[1, 2],
                        help="标量头输出带数：1=形态 A'（缺省）；2=形态 A× 多频段")
    parser.add_argument("--band_mix", type=str, default="single",
                        choices=["single", "mult"],
                        help="多带合成方式：single=取 ū_0（A'）；mult=ū_0·(1+λ(ū_1−init_u))")
    # —— SH 输入块 / 空间上下文块（2026-09-13；默认关 = 与改动前逐位一致）——
    # ⚠ 名字**不能**叫 --sh_degree：ModelParams 已注册同名 flag（splat/arguments/__init__.py:49
    #   自动把每个属性注册成 CLI 项）⇒ 重名会直接 argparse.ArgumentError。
    parser.add_argument("--use_sh", type=str, default="none",
                        choices=["none", "dc", "eval"],
                        help="把物理阶段 SH 当颜色头输入：none=关（缺省）；"
                             "dc=只用基色（deg0，视角无关）；"
                             "eval=在视线方向上求值（逐字照抄渲染器口径）")
    parser.add_argument("--sh_eval_degree", type=int, default=-1,
                        help="use_sh 求值用的 SH 阶数；-1（缺省）= 取 gaussians.active_sh_degree。"
                             "该值会被钉进 sidecar cfg 成为训练/渲染的**唯一真源**")
    parser.add_argument("--ctx_knn", type=int, default=0,
                        help="空间上下文块：每点的邻居数（0=关，缺省）。>0 时在固定 kNN 图上"
                             "做单层单头注意力，聚合邻居的 (embed, SH)")
    parser.add_argument("--ctx_dim", type=int, default=16,
                        help="上下文块的通道数（ctx_knn>0 时生效）")
    # —— 通用 ——
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    print("Optimizing (Scalar-RRF) " + args.model_path)

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
    if not (0.0 < args.init_u < 1.0):
        parser.error("--init_u 须在 (0,1) 内")
    if args.lut_n < 2:
        parser.error("--lut_n 须 >= 2")
    if args.n_bands == 1 and args.band_mix != "single":
        parser.error("--n_bands 1 只能配 --band_mix single")
    if args.n_bands > 1 and args.band_mix == "single":
        print("[warn] --n_bands>1 但 --band_mix single：band 1 会被渲染出来却不参与合成，"
              "等价于 y 轴上的一个死分支（仅作消融用）")
    if not (-1 <= args.sh_eval_degree <= 3):
        parser.error("--sh_eval_degree 须 ∈ [-1, 3]（-1 = 取 active_sh_degree）")
    if args.ctx_knn < 0:
        parser.error("--ctx_knn 须 >= 0（0 = 关）")
    if args.ctx_dim <= 0:
        parser.error("--ctx_dim 须 > 0")
    if args.ctx_knn > 0:
        # 建图要 scipy.spatial.cKDTree。**在花钱之前**就把缺依赖变成一条立刻可见的错，
        # 而不是训到一半才炸（图片环境的 bare except 会把它拖成一次静默的空转）。
        try:
            from scipy.spatial import cKDTree      # noqa: F401
        except ImportError as e:
            parser.error(f"--ctx_knn>0 需要 scipy：{e}。"
                         "服务器 gs3d 环境自带（实测 scipy 1.17.1）；"
                         "本机 mvsplat 环境亦有。若确实没有就装 scipy 或把 --ctx_knn 设为 0。")

    scenario = {
        "embed_dim": args.embed_dim, "mlp_hidden": args.mlp_hidden,
        "mlp_depth": args.mlp_depth,
        "n_bands": args.n_bands, "band_mix": args.band_mix,
        "pe_xyz_L": args.pe_xyz_L, "pe_dir_L": args.pe_dir_L,
        "use_dist": args.use_dist,
        "init_u": args.init_u, "lut_n": args.lut_n,
        "embed_lr": embed_lr, "mlp_lr_init": mlp_lr_init,
        "mlp_lr_final": mlp_lr_final, "lr_schedule": args.lr_schedule,
        "lr_max_steps": args.lr_max_steps,
        "lr_delay_steps": args.lr_delay_steps, "lr_delay_mult": args.lr_delay_mult,
        "use_sh": args.use_sh, "sh_eval_degree": args.sh_eval_degree,
        "ctx_knn": args.ctx_knn, "ctx_dim": args.ctx_dim,
    }

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    prepare_output_and_logger(args)
    training(lp.extract(args), opt, pp.extract(args), scenario,
             args.start_checkpoint, args.save_iterations, args.iterations)
    print("\nTraining complete.")
