# -*- coding: utf-8 -*-
"""角域 RRF 的**跨发射源前馈**训练：一个头 + 一份逐点 embedding → 任意发射源的谱图。

与 `train_rrf_scalar.py`（逐场景基线）的关系
--------------------------------------------------------------------------------
完全相同的是**渲染与损失口径**（这是与原 RF-3DGS 可比的前提）：
    u = head(…) ∈ (0,1)  →  override_color  →  光栅化（标量域线性混合）
      →  combine_bands  →  apply_jet(256 级表)  →  L1 + 0.2·(1−SSIM)
    起点 = 作者发布的光学 chkpnt30000；几何冻结；densify=0；opacity 可训
不同的是**训练样本**与**头的输入**：
    每个迭代先抽一个训练发射源，再从该发射源抽若干相机；GT 图按
    (发射源, 图号) 现取（`angular_data.SpectrumStore`）。
    头额外吃发射源位置（相对 Δ + 绝对 t̂x），见 `splat/color_scalar_tx.py`。

为什么每个迭代只用一个发射源
--------------------------------------------------------------------------------
GT 图依赖发射源，而 `Camera.original_image` 是位姿的一部分。为每个 (TX, 视角) 造一个
Camera 需要 37×2560 个对象与 CUDA 张量；这里保持 3200 个相机不变，GT 在损失处替换。
代价是单步只更新一个发射源 —— 所以训练步数要按「每个发射源的有效视角更新数」换算：
    iterations × views_per_step / n_tx   ≈ 原逐场景臂的 10000 步
本仓默认 120000 × 2 / 36 ≈ 6700，再用 `--lr_max_steps` 拉长退火区间。

用法（WSL，code/ 目录）：
    PY=python3.10
    $PY train_rrf_angular.py \
        -s /path/to/w-RFGS/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100 \
        -m /path/to/w-RFGS/results/<run> \
        --spectra_root /path/to/w-RFGS/datasets/RF-3DGS_cbf_multi \
        --meta /path/to/w-RFGS/datasets/NIST_lobby_spatial_full/meta.json \
        --start_checkpoint /path/to/w-RFGS/datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
        --iterations 120000 --views_per_step 2
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from random import randint

import numpy as np
import torch
from torch import nn

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from angular_data import (SpectrumStore, load_cameras, load_instances,   # noqa: E402
                          load_tx_table)
from splat.arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from splat.color_mlp import set_scene_norm                               # noqa: E402
from splat.color_scalar import (apply_jet, combine_bands, make_jet_lut,  # noqa: E402
                                to_override_color)
from splat.color_scalar_tx import (TxScalarColorHead, compute_scalar_tx,  # noqa: E402
                                   load_tx_state, save_tx_state, tx_head_cfg)
from splat.color_scalar_tx_cplx import TxComplexColorHead               # noqa: E402
from splat.color_scalar_tx_fold import TxFoldColorHead                  # noqa: E402
from splat.color_scalar_af import TxArrayFactorColorHead                 # noqa: E402
from splat.color_scalar_coh import TxCoherentColorHead                   # noqa: E402
from splat.gaussian_renderer import render                               # noqa: E402
from splat.scene import GaussianModel                                    # noqa: E402
from splat.utils.general_utils import get_expon_lr_func, safe_state      # noqa: E402
from splat.utils.loss_utils import l1_loss, ssim                         # noqa: E402
from tqdm import tqdm                                                    # noqa: E402

from train_rrf_scalar import _set_group_lr, load_visual_geometry         # noqa: E402


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def training(dataset, opt, pipe, sc, iterations, saving_iterations,
             start_checkpoint):
    torch.manual_seed(sc["seed"])
    random.seed(sc["seed"])
    np.random.seed(sc["seed"])

    # ---------------- 数据面 ----------------
    cams, train_names, test_names = load_cameras(dataset.source_path)
    # 训练用哪一批视角。默认作者 2560 个训练视角；w 轴数据集只生成了 640 个测试
    # 视角（理由见 `make_w_instances.py`：w 轴要回答的是「见过的权能否外推到没见过的
    # 方向」，泛化轴是 w 不是视角），这时必须显式 `--train_views test`，否则完整性闸
    # 会按 2560 张判而直接报「谱图不全」。
    view_names = {"train": train_names, "test": test_names,
                  "all": train_names + test_names}[sc["train_views"]]
    # 随机抽 N 个（测试时适应的**无偏**抽样）。
    # 为什么不用「前 N 个」也不用「步长抽样」：交付数据的 640 个 test 视角是
    # 「每位置 4 个朝向连续排」，前 N 个只覆盖 N/4 个位置；步长 s 抽到的是
    # 固定朝向（s=4 → 每个位置的第 0 个朝向，s=2 → 第 0/2 个），评测集则全是
    # 另外几个朝向 ⇒ 适应集与评测集在**朝向上系统性错配**，测到的是错配不是泛化。
    # 只有随机抽样同时无偏于位置与朝向。评测端 `.tmp/_tta_eval.py --random/--seed`
    # 必须用同一对参数（同一个 numpy 生成器、同一份 test_index 顺序）。
    if int(sc.get("train_view_random", 0)) > 0:
        import numpy as _np
        _n = int(sc["train_view_random"])
        _idx = _np.random.default_rng(int(sc.get("train_view_seed", 0))).choice(
            len(view_names), size=min(_n, len(view_names)), replace=False)
        _idx.sort()
        view_names = [view_names[i] for i in _idx]
    # 先按步长抽样（测试时适应的覆盖控制），再截前 N 个。
    if int(sc.get("train_view_stride", 1)) > 1:
        view_names = view_names[:: int(sc["train_view_stride"])]
    if sc["train_view_limit"] and sc["train_view_limit"] < len(view_names):
        view_names = view_names[: int(sc["train_view_limit"])]
    # 两种数据口径：旧表（只有位置，每个 TX 一份 GT）/ 参数表（每实例一份 GT）。
    # meta 里没有 `instances` 就走旧路径，逐字保持历史实验可复现。
    inst = load_instances(sc["meta"])
    tx_ids, rows = [], []
    pair_of: dict[int, int] = {}     # 成对耦合：row → 同配置、不同 w 的另一个 row
    if inst is None:
        tx_world, tx_train, tx_test, orig = load_tx_table(sc["meta"])
        tx_ids = [int(i) for i in (sc["tx_list"] or tx_train)]
        if orig in tx_ids:
            raise RuntimeError(f"训练集里出现了原发射源 {orig} ⇒ 与「原发射源留作测试」冲突")
        store = SpectrumStore(sc["spectra_root"])
        ids_for_check = tx_ids
        log(f"[ang] 口径=位置 发射源 训 {len(tx_ids)}（测试 {tx_test}，原发射源 {orig}）")
    else:
        roles = sc["train_roles"] or ["train"]
        rows = np.concatenate([inst.roles[r] for r in roles if r in inst.roles]).astype(int)
        if rows.size == 0:
            raise RuntimeError(f"实例表里没有角色 {roles}（有 {list(inst.roles)}）")
        if sc["max_train_instances"]:
            rows = rows[: sc["max_train_instances"]]
            log(f"[ang] 冒烟：只取前 {rows.size} 个训练实例")
        if sc.get("train_instance_ids"):
            _keep = {int(i) for i in sc["train_instance_ids"]}
            rows = np.array([r for r in rows if int(inst.ids[r]) in _keep],
                            dtype=int)
            if rows.size == 0:
                raise RuntimeError(f"--train_instance_ids {sorted(_keep)} "
                                   f"在角色 {roles} 里一个都没有")
            log(f"[ang] 指定实例：{rows.size} 个（id={[int(inst.ids[r]) for r in rows]}）")
        store = SpectrumStore(sc["spectra_root"], key_fmt=inst.key_fmt)
        ids_for_check = [int(inst.ids[r]) for r in rows]
        log(f"[ang] 口径=参数 实例 {len(rows)}（角色 {roles}；"
            f"总实例 {len(inst)}，目录 {inst.key_fmt}）")
        if sc["use_tx_w"] or sc["w_head"] in ("cplx", "fold", "af", "cohaf"):
            # 波束块只有 5 维常量输入，若训练集里的 w 全同（例如 meta 里根本没有
            # w_label，全部退化成 uniform），这一块就是死列 —— 早报错，别白跑。
            # 复值头同理：w 只走闭式收缩，码本取值不散就等于没有这条轴。
            # 相干头 coh（use_af=0）不看 w，不在此列。
            labs = {str(inst.w_label[int(r)]) for r in rows}
            if len(labs) < 2:
                raise RuntimeError(
                    f"波束条件打开（w_head={sc['w_head']}, use_tx_w={sc['use_tx_w']}），"
                    f"但训练实例的 w_label 只有 {len(labs)} 种 "
                    f"（{sorted(labs)[:3]}）⇒ 波束轴是常量，检查 meta 的 w_label")
            log(f"[ang] 波束 训练集 {len(labs)} 种 w（uniform "
                f"{sum(bool(inst.w_uniform[int(r)]) for r in rows)} 个实例）")
        elif inst.w_label is not None:
            log(f"[ang] 波束 关（w_head={sc['w_head']}, use_tx_w={sc['use_tx_w']}）")

        # 成对耦合：把「同配置、不同 w」的两个实例配成对。必须**同时**在训练里，
        # 否则 pair_of 指向的实例不在 rows 里，配对静默失效。
        if sc["pair_coupled"]:
            id2row = {int(inst.ids[r]): r for r in rows}
            for it in inst.meta.get("instances", []):
                b = int(it.get("pair_of", -1))
                if b < 0:
                    continue
                rb, rc = id2row.get(b, -1), id2row.get(int(it["id"]), -1)
                if rb >= 0 and rc >= 0:
                    pair_of[rb] = rc
                    pair_of[rc] = rb
            if not pair_of:
                raise RuntimeError(
                    "--pair_coupled 打开，但 meta 里没有任何一对同时落在训练集里"
                    "（检查 pair_of 字段与 --train_roles）")
            log(f"[ang] 成对耦合：{len(pair_of)//2} 对，同一视角同步计算"
                + (f"，差分损失 λ={sc['pair_diff_w']}" if sc["pair_diff_w"] > 0 else ""))

    # 完整性闸：每个训练源/实例的训练视角都要在盘上。按计数判（比逐文件 stat 快），
    # 再用前 8 张做点检 —— 目录存在但图是上一轮半截产物时，计数会露馅。
    bad = []
    for i in ids_for_check:
        d = store.dir(i)
        n = len(list(d.glob("*.png"))) if d.is_dir() else 0
        if n < len(view_names) or any(not store.has(i, x) for x in view_names[:8]):
            bad.append((i, n))
    if bad:
        raise RuntimeError(f"谱图不全（(源/实例, 张数)）：{bad[:5]}，"
                           f"要求 >= {len(view_names)} 张；先生成完再训")

    train_cams = [cams[n] for n in view_names]
    log(f"[ang] 相机 训 {len(train_cams)} / 全部 {len(cams)}"
        f"（--train_views {sc['train_views']}）")

    # ---------------- 几何（冻结）----------------
    gaussians = GaussianModel(dataset.sh_degree)
    first_iter = load_visual_geometry(start_checkpoint, gaussians, dataset.sh_degree)
    P = gaussians.get_xyz.shape[0]
    spatial_lr_scale = float(gaussians.spatial_lr_scale or 0.0)
    if spatial_lr_scale <= 0:
        raise RuntimeError("空间尺度 spatial_lr_scale 缺失/为 0，距离项会失去归一化")

    # ---------------- 头 ----------------
    # 两种头共用同一套输入特征，只在「w 怎么进图」上分开：
    #   none —— w 当 5 维条件喂 MLP（历史口径，已判死在表示层）
    #   cplx —— w 不进 MLP，逐高斯输出复系数、在输出侧做闭式二次型收缩
    n_pos_emb = 0
    if sc["use_pos_embed"]:
        if sc["w_head"] != "none":
            raise SystemExit(f"--use_pos_embed 只接在基础标量头上（--w_head none），"
                             f"当前 {sc['w_head']}")
        if inst is None:
            raise SystemExit("--use_pos_embed 需要参数口径：meta 里要有 instances/"
                             "position_id")
        train_pos = np.unique(inst.position_id[rows])
        n_pos_emb = int(sc["pos_max"]) or (int(train_pos.max()) + 1)
        log(f"[ang] 位置条件化：码表 {n_pos_emb} 行 + 1 回退行 × {sc['pos_dim']} 维，"
            f"训练涉及 {len(train_pos)} 个位置，dropout={sc['pos_dropout']}")
    common = dict(
        embed_dim=sc["embed_dim"], hidden=sc["mlp_hidden"], depth=sc["mlp_depth"],
        pe_xyz_L=sc["pe_xyz_L"], pe_dir_L=sc["pe_dir_L"],
        pe_rel_L=sc["pe_rel_L"], pe_tx_L=sc["pe_tx_L"],
        use_dist=sc["use_dist"], use_rel=sc["use_rel"], use_tx=sc["use_tx"],
        use_yaw=sc["use_yaw"], init_u=sc["init_u"],
        use_tx_orient=sc["use_tx_orient"], use_tx_local_dir=sc["use_tx_local_dir"],
        use_tx_freq=sc["use_tx_freq"], use_geom=sc["use_geom"],
        freq_lo_hz=sc["freq_lo_hz"], freq_hi_hz=sc["freq_hi_hz"])
    if sc["w_head"] == "cplx":
        head = TxComplexColorHead(
            n_ant=sc["n_ant"], n_pos=sc["n_pos"], n_neg=sc["n_neg"],
            c_init=sc["c_init"], **common).cuda()
        log(f"[ang] 头=复值二次型  n_ant={head.n_ant}  n_pos={head.n_pos}  "
            f"n_neg={head.n_neg}  c_init={head.c_init}")
    elif sc["w_head"] == "fold":
        head = TxFoldColorHead(
            n_ant=sc["n_ant"], c_init=sc["c_init"], **common).cuda()
        log(f"[ang] 头=相干折叠  n_ant={head.n_ant}  c_init={head.c_init}  "
            f"输出={2*head.n_ant} 维/点  norm={head.norm}")
    elif sc["w_head"] == "af":
        head = TxArrayFactorColorHead(
            n_ant=sc["n_ant"], norm=sc["af_norm"],
            post_norm=sc["af_post_norm"], **common).cuda()
        log(f"[ang] 头=解析阵列因子  n_ant={head.n_ant}  输出=1 维/点  "
            f"norm={head.norm}  post_norm={head.post_norm}")
    elif sc["w_head"] in ("coh", "cohaf"):
        head = TxCoherentColorHead(
            n_ant=sc["n_ant"], use_af=(sc["w_head"] == "cohaf"), **common).cuda()
        log(f"[ang] 头=相干双通道  n_ant={head.n_ant}  输出=2 维/点（Re/Im）  "
            f"w={'解析 AF 复数乘' if head.use_af else '不进图'}")
    else:
        head = TxScalarColorHead(
            use_tx_w=sc["use_tx_w"], use_tx_w_diff=sc["use_tx_w_diff"],
            use_rx_dir=sc["use_rx_dir"],
            n_bands=sc["n_bands"],
            band_mix=sc["band_mix"],
            use_pos_embed=sc["use_pos_embed"], n_pos=n_pos_emb,
            pos_dim=sc["pos_dim"], pos_dropout=sc["pos_dropout"],
            film=sc["film"], film_pos=sc["film_pos"], **common).cuda()
    set_scene_norm(head, gaussians.get_xyz)
    if getattr(head, "film", False):
        log(f"[ang] FiLM 开启：实例级参数 {head.film_in} 维"
            f"（film_pos={'开' if getattr(head, 'film_pos', False) else '关'}）→ "
            f"{head.mlp_depth} 组 (scale,shift) / "
            f"{head.mlp_depth * 2 * head.mlp_hidden} 个数；"
            f"零初始化 ⇒ 第 0 步与原头逐位相同")
    # 逐高斯几何（法向/各向异性）：几何冻结，只算一次。旧头不开 use_geom 时不算。
    geom_n = geom_a = None
    if head.use_geom:
        from splat.color_scalar_tx import gaussian_geom
        geom_n, geom_a = gaussian_geom(gaussians)
        log(f"[ang] 几何块开启：法向 3 + 两夹角 2 + 镜面对齐 1 + 各向异性 1；"
            f"aniso 中位 {float(geom_a.median()):.2f}  in_dim={head.in_dim}")
    embed = nn.Parameter(torch.zeros((P, sc["embed_dim"]), device="cuda").normal_(0, 0.02),
                         requires_grad=True)
    lut = make_jet_lut(sc["lut_n"]).cuda()
    # 感知损失（可选）：默认用与评测同一个 **VGG**-LPIPS（`eval_dir_pairs.py` 也是 vgg）。
    # 历史 run 的 lpips_w 全是 0，所以换默认值不影响任何旧检查点的复现。
    lpips_fn = None
    if sc["lpips_w"] > 0.0:
        from splat.lpipsPyTorch.modules.lpips import LPIPS
        lpips_fn = LPIPS(net_type=sc["lpips_net"], version="0.1").cuda().eval()
        for _p in lpips_fn.parameters():
            _p.requires_grad_(False)
        log(f"[ang] 感知项开启：λ_lpips={sc['lpips_w']}（{sc['lpips_net']} v0.1，jet 域，输入 [0,1]）")

    # 续训：从本脚本自己的存档接着训（头/embed/opacity 全部恢复）。
    # 只做结构兼容检查，不猜任何超参——学习率仍由本次命令行决定。
    if sc.get("resume_angular"):
        _h, _e, _op, _cfg_old, _it0 = load_tx_state(sc["resume_angular"], device="cuda")
        if _h.in_dim != head.in_dim or int(_h.n_bands) != int(head.n_bands):
            raise RuntimeError(
                f"续训存档结构不兼容：in_dim {_h.in_dim} vs {head.in_dim}，"
                f"n_bands {_h.n_bands} vs {head.n_bands}")
        # 语义也要对齐：同 in_dim 也可能是「朝向开、频率关」与「朝向关、频率开」
        # use_pos_embed 也必须在列：它改变 in_dim，但同宽度也可能是「pos 开、别的块关」，
        # 语义必须逐项对齐。
        for k in ("use_tx_orient", "use_tx_local_dir", "use_tx_freq", "use_tx_w",
                  "use_tx_w_diff", "use_geom", "use_pos_embed", "film"):
            if bool(_cfg_old.get(k, False)) != bool(getattr(head, k)):
                raise RuntimeError(f"续训存档的 {k}={_cfg_old.get(k)} 与本次 "
                                   f"{getattr(head, k)} 不一致 ⇒ 同宽度不同语义")
        head.load_state_dict(_h.state_dict())
        with torch.no_grad():
            embed.copy_(_e.detach())
            if _op is not None:
                gaussians._opacity.data.copy_(_op)
        log(f"[ang] 续训：载入 {sc['resume_angular']}（原档 iter={_it0}）")

    # 输入扩维的暖启动：载入一个**输入更窄**的旧存档，把第一层的新列**置零**，
    # 其余权重原样搬过来。置零保证第 0 步的函数与旧存档**逐位相同**，
    # 之后任何变化只可能来自新输入块——这是「加一路输入到底有没有用」的干净单变量。
    # 用法：--use_geom 1 --init_from_other <旧 pth>（旧档的 use_geom 必须为假）。
    if sc.get("init_from_other"):
        _h, _e, _op, _cfg_old, _it0 = load_tx_state(sc["init_from_other"], device="cuda")
        if int(_h.n_bands) != int(head.n_bands):
            raise RuntimeError(f"暖启动 n_bands {_h.n_bands} vs {head.n_bands}")
        for k in ("use_tx_orient", "use_tx_local_dir", "use_tx_freq", "use_tx_w",
                  "use_tx_w_diff", "use_pos_embed"):
            if bool(_cfg_old.get(k, False)) != bool(getattr(head, k)):
                raise RuntimeError(f"暖启动的 {k}={_cfg_old.get(k)} 与本次 "
                                   f"{getattr(head, k)} 不一致 ⇒ 语义不同")
        _grew_input = head.in_dim > _h.in_dim
        _added_film = (bool(getattr(head, "film", False))
                       and not bool(_cfg_old.get("film", False)))
        # 只有旧档本来就开了 FiLM 时才比对 film_pos；旧档没开 film 属于「本次新加
        # 调制器」，零初始化恒等，film_pos 开不开都不影响第 0 步。
        if (bool(_cfg_old.get("film", False))
                and bool(_cfg_old.get("film_pos", False))
                != bool(getattr(head, "film_pos", False))):
            raise RuntimeError("暖启动的 film_pos 不一致 ⇒ 调制器语义不同")
        if not (_grew_input or _added_film):
            raise RuntimeError(f"暖启动要求本次 in_dim > 旧 in_dim（或新开 film），"
                               f"收到 {head.in_dim} vs {_h.in_dim}")
        old_sd, new_sd = _h.state_dict(), head.state_dict()
        grown = []
        for k, v in new_sd.items():
            src = old_sd.get(k)
            if src is None:
                # 本次新加的模块（FiLM 调制器）：旧档里没有，保持初始化值。
                # 零初始化的 FiLM 在逻辑上等价于「恒等调制」，故第 0 步逐位相同。
                if k.startswith("film_mlp."):
                    continue
                raise RuntimeError(f"暖启动缺参数 {k}")
            if tuple(src.shape) == tuple(v.shape):
                v.copy_(src)
            elif k == "net.0.weight":
                # 新列置零：多出来的输入在第 0 步完全不起作用
                v.zero_()
                v[:, : src.shape[1]].copy_(src)
                grown.append(v.shape[1] - src.shape[1])
            else:
                raise RuntimeError(f"暖启动形状不兼容 {k}: {tuple(src.shape)} → "
                                   f"{tuple(v.shape)}")
        with torch.no_grad():
            embed.copy_(_e.detach())
            if _op is not None:
                gaussians._opacity.data.copy_(_op)
        log(f"[ang] 暖启动：载入 {sc['init_from_other']}（原档 iter={_it0}，"
            f"in_dim {_h.in_dim}→{head.in_dim}，新增 {grown} 列置零）")

    # 只放开头的最后一层 / 最后一块（测试时标定用）。
    # 动机：小样本适应时整个头（36.7k 参数）在 16–64 个视角上过拟合，留出视角反而变差。
    # 把可训参数压到 129（out）/ 16.5k（net.2+out），可以在同一批视角上量出
    # 「摊销特征图里到底还有多少可被逐实例线性读出的信息」——
    # 若极小容量就能追平，缺的是标定；若不能，缺的是特征本身。
    # 冻结靠 requires_grad：Adam 对 grad=None 的参数天然跳过，不必再分参数组。
    if sc.get("train_only", "all") != "all":
        _keep = ("out.",) if sc["train_only"] == "out" else ("net.2.", "out.")
        n_tr = 0
        for _n, _p in head.named_parameters():
            _p.requires_grad_(any(_n.startswith(k) for k in _keep))
            if _p.requires_grad:
                n_tr += _p.numel()
        log(f"[ang] train_only={sc['train_only']}：头只放开 {_keep}，可训 {n_tr} 个参数")

    gaussians.optimizer = torch.optim.Adam([
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [embed], "lr": sc["embed_lr"], "name": "embed"},
        {"params": list(head.parameters()), "lr": sc["mlp_lr_init"], "name": "mlp"},
    ], lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = None
    if sc["lr_schedule"] == "cosine":
        # 余弦退火：前半程维持较高学习率，后半程才压到底。
        # 指数退火在 8e-3→8e-5 的跨度下，走完一半就已经掉到 8e-4 量级，
        # 60k 的基准训练因此在中段就进了假平台（对照臂重启后 +0.91 dB 可证）。
        import math
        _delay = max(1, int(sc["lr_delay_steps"] or max(1, sc["lr_max_steps"] // 100)))
        _mult = float(sc["lr_delay_mult"])
        _maxs = max(_delay + 1, int(sc["lr_max_steps"]))

        def _cos_sched(init, final):
            def f(s):
                if s < _delay:
                    return init * (_mult + (1.0 - _mult) * s / _delay)
                t = min(1.0, (s - _delay) / float(_maxs - _delay))
                return final + 0.5 * (init - final) * (1.0 + math.cos(math.pi * t))
            return f

        mlp_sched = _cos_sched(sc["mlp_lr_init"], sc["mlp_lr_final"])
    else:
        mlp_sched = get_expon_lr_func(sc["mlp_lr_init"], sc["mlp_lr_final"],
                                      sc["lr_delay_steps"], sc["lr_delay_mult"],
                                      sc["lr_max_steps"])
        _cos_sched = None

    # 逐高斯 embedding 的调度（默认与旧行为等价：final == init ⇒ 恒定）。
    # 历史 run 只有 mlp 组退火，embed 组一直固定在 embed_lr（≈2.5e-3）——
    # 在 sign-Adam 下这是每步 ±2.5e-3 的持续抖动，恰好落在编码高频空间结构的
    # 参数上。加这个开关是为了验证「embed 不收敛 ⇒ 峰定不住」。
    _emb_final = sc.get("embed_lr_final")
    if _emb_final is None or abs(_emb_final - sc["embed_lr"]) < 1e-12:
        embed_sched = None
    elif _cos_sched is not None:
        embed_sched = _cos_sched(sc["embed_lr"], _emb_final)
    else:
        embed_sched = get_expon_lr_func(sc["embed_lr"], _emb_final,
                                        sc["lr_delay_steps"], sc["lr_delay_mult"],
                                        sc["lr_max_steps"])

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    if dataset.white_background:
        raise RuntimeError("角域谱图的背景必须是标量 0（jet 深蓝端）；white_background=True "
                           "会把未覆盖像素推到 u=1（红端），与 GT 口径不符")

    cfg = {"head_type": getattr(head, "head_type", "scalar_jet_tx"),
           "w_head": sc["w_head"], "lpips_w": sc["lpips_w"],
           "lpips_net": sc["lpips_net"],
           "bright_w": sc["bright_w"],
           "n_ant": getattr(head, "n_ant", None),
           "n_pos": getattr(head, "n_pos", None),
           "n_neg": getattr(head, "n_neg", None),
           "start_iter": first_iter,
           "iterations": iterations, "P": P,
           "data_mode": "tx" if inst is None else "inst",
           "n_tx": len(tx_ids) if inst is None else len(rows),
           "tx_ids": tx_ids, "train_roles": sc["train_roles"],
           "instance_ids": ([int(x) for x in inst.ids] if inst is not None else None),
           "key_fmt": (store.key_fmt),
           "use_tx_orient": bool(head.use_tx_orient),
           "use_tx_local_dir": bool(head.use_tx_local_dir),
           "use_tx_freq": bool(head.use_tx_freq),
           "use_tx_w": bool(head.use_tx_w),
           "use_tx_w_diff": bool(head.use_tx_w_diff),
           "train_pos_ids": ([int(x) for x in np.unique(inst.position_id[rows])]
                             if (inst is not None and sc["use_pos_embed"]) else None),
           "train_views": sc["train_views"],
           "train_view_limit": int(sc["train_view_limit"]),
           "freq_lo_hz": head.freq_lo_hz, "freq_hi_hz": head.freq_hi_hz,
           "param_spec": tx_head_cfg(head)["param_spec"],
           "views_per_step": sc["views_per_step"],
           "spatial_lr_scale": spatial_lr_scale, "lut_n": sc["lut_n"],
           "spectra_root": sc["spectra_root"], "meta": sc["meta"],
           "source_path": dataset.source_path,
           "start_checkpoint": start_checkpoint, "seed": sc["seed"],
           "embed_lr": sc["embed_lr"], "embed_lr_final": sc["embed_lr_final"],
           "mlp_lr_init": sc["mlp_lr_init"],
           "mlp_lr_final": sc["mlp_lr_final"], "lr_max_steps": sc["lr_max_steps"],
           "lr_delay_steps": sc["lr_delay_steps"], "lr_delay_mult": sc["lr_delay_mult"],
           "lr_schedule": sc["lr_schedule"]}
    with open(os.path.join(dataset.model_path, "angular_experiment.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    save_tx_state(os.path.join(dataset.model_path, "angular_init.pth"), head, embed,
                  first_iter, extra=cfg, opacity=gaussians._opacity)

    log(f"[ang] P={P}  in_dim={head.in_dim}  头参数={sum(p.numel() for p in head.parameters())}  "
        f"embed={sc['embed_dim']}  hidden={sc['mlp_hidden']}×{sc['mlp_depth']}  "
        f"scale={spatial_lr_scale:.3f}  init_u={sc['init_u']}  "
        f"yaw={sc['use_yaw']}  rx_dir={sc['use_rx_dir']}  "
        f"rel={sc['use_rel']}  tx={sc['use_tx']}  "
        f"w_head={sc['w_head']}  lpips_w={sc['lpips_w']}  "
        f"views/step={sc['views_per_step']}")

    # ⚠ `--iterations` 在本脚本里是**训练步数**，不是全局迭代号。
    #   起点检查点自带 first_iter=30000（视觉阶段的终点），若照搬 train_rrf_scalar 的
    #   `range(first_iter+1, iterations+1)` 语义，`--iterations 200` 会得到空循环
    #   —— 冒烟时实测「训练成功、一步没跑、渲染出纯初始绿图」。故此处一律按步计数，
    #   全局迭代号只记进 cfg 供溯源。
    pbar = tqdm(range(1, iterations + 1), desc="Angular-RRF")
    ema = 0.0
    t0 = time.time()
    _diag = {"done": False}

    def _grad_diag(li_base, lp_term):
        """首步一次性诊断：分别求「任务项」与「LPIPS 项」对**参数**的梯度范数。

        动机：sign-Adam 下步长恒为 lr，梯度只定方向 ⇒ λ_lpips 不是平滑加权，
        而是一个**阈值**：λ·|g_lp| 超过 |g_task| 的地方符号被翻转。
        不知道两个范数的比就没法选 λ，只能瞎猜。
        只跑一次，settle 后把 .grad 清空，不影响后续任何一步的优化。
        """
        ps = [p for g in gaussians.optimizer.param_groups
              for p in g["params"] if p.requires_grad]
        def _norm(f):
            for p in ps:
                p.grad = None
            gs = torch.autograd.grad(f, ps, retain_graph=True, allow_unused=True)
            tot = torch.zeros((), device="cuda")
            for g in gs:
                if g is not None:
                    tot = tot + (g.detach() ** 2).sum()
            return float(tot.sqrt()), [None if g is None else g.detach()
                                       for g in gs]
        n_task, _ = _norm(li_base)
        n_lp, _ = _norm(lp_term)
        for p in ps:
            p.grad = None
        _diag["done"] = True
        lam_eq = n_task / max(n_lp, 1e-30)
        log(f"[diag] |g_task|={n_task:.3e}  |g_lpips|={n_lp:.3e}  "
            f"比值={n_lp / max(n_task, 1e-30):.2f}×  "
            f"两项等量需 λ={lam_eq:.4g}；当前 λ={sc['lpips_w']:.4g} "
            f"⇒ {'LPIPS 主导方向' if sc['lpips_w'] > lam_eq else '任务项主导方向（λ 低于阈值，等于没开）'}")
    def _img_loss(image, gt, jj, cam):
        """单张渲染图的损失。抽出来供成对耦合复用（两条支路必须逐字一致）。"""
        if sc["bright_w"] > 0.0:
            # 亮区加权：按 GT 的归一化谱值 u 加权（暗处 1，u=1 处 1+BW）。
            wgt = 1.0 + sc["bright_w"] * store.unit(jj, cam.image_name)
            if sc["loss_l2"]:
                Ll1 = (((image - gt) ** 2) * wgt).mean() / wgt.mean()
            else:
                Ll1 = ((image - gt).abs() * wgt).mean() / wgt.mean()
        elif sc["loss_l2"]:
            # L2：把最优解从条件**中位数**推到条件**均值**。GT 严重右偏
            # （均值 0.0925 / 中位数 0.0433），L1 训练出来的场均值恰好贴着中位数
            # （实测 0.0466），量程只有 GT 的 1/4 —— 见 `_chk_u_range.py`。
            Ll1 = ((image - gt) ** 2).mean()
        else:
            Ll1 = l1_loss(image, gt)
        li_base = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt))
        li = li_base
        if lpips_fn is not None:
            lp = lpips_fn(image, gt).mean()
            if sc["grad_diag"] and not _diag["done"]:
                _grad_diag(li_base, lp)
            li = li + sc["lpips_w"] * lp
        return li

    def _sample(rr, cam):
        """(实例行, 视角) → (损失, jet 图, GT)。成对耦合时同一 cam 调两次。"""
        jj = int(inst.ids[rr])
        txp = torch.as_tensor(inst.xyz[rr], device="cuda", dtype=torch.float32)
        pms = dict(
            tx_yaw=float(inst.yaw_off[rr]),
            tx_pitch=float(inst.pitch_off[rr]),
            tx_freq=float(inst.freq_hz[rr]),
            tx_rot=torch.as_tensor(inst.rot[rr], device="cuda", dtype=torch.float32))
        if sc["use_tx_w"] or sc["w_head"] in ("cplx", "fold", "af", "cohaf"):
            pms["tx_w"] = [float(inst.w_theta[rr]), float(inst.w_phi[rr]),
                           1.0 if bool(inst.w_uniform[rr]) else 0.0]
        if head.use_pos_embed:
            pms["pos_id"] = int(inst.position_id[rr])
        yaw = None
        if sc["use_yaw"]:
            yaw = torch.full((P, 1), float(cam_yaw(cam)), device="cuda")
        s = compute_scalar_tx(head, embed, gaussians.get_xyz, cam.camera_center,
                              spatial_lr_scale, txp, yaw=yaw,
                              normal=geom_n, aniso=geom_a, **pms)
        rp = render(cam, gaussians, pipe, bg,
                    override_color=to_override_color(s))
        im = apply_jet(combine_bands(rp["render"], head), lut)
        gt = store(jj, cam.image_name)
        return _img_loss(im, gt, jj, cam), im, gt

    for step in range(1, iterations + 1):
        loss_sum = 0.0
        n_done = 0
        # inst_per_step > 1：一次 optimizer.step() 里累加 K 个**不同配置**的梯度。
        # 优化器实际是 sign-SGD（Adam lr=0, eps=1e-15），单样本梯度里逐配置的差异会
        # 把符号翻来翻去；跨配置平均后噪声按 √K 降，所有配置共享的那一小部分才可能
        # 被看见。K=1 时与历史行为逐字一致。
        for _ in range(sc["inst_per_step"]):
            if inst is None:
                j = int(tx_ids[randint(0, len(tx_ids) - 1)])
                txpos = torch.as_tensor(tx_world[j], device="cuda", dtype=torch.float32)
                params = {}
                row = -1
            else:
                row = int(rows[randint(0, len(rows) - 1)])
                j = int(inst.ids[row])                      # GT 目录键与日志用
                txpos = torch.as_tensor(inst.xyz[row], device="cuda", dtype=torch.float32)
                params = dict(
                    tx_yaw=float(inst.yaw_off[row]),
                    tx_pitch=float(inst.pitch_off[row]),
                    tx_freq=float(inst.freq_hz[row]),
                    tx_rot=torch.as_tensor(inst.rot[row], device="cuda", dtype=torch.float32),
                )
                if sc["use_tx_w"] or sc["w_head"] in ("cplx", "fold", "af", "cohaf"):
                    # 两条路都从同一个码本标签取值：喂 MLP 时是 5 维条件，复值头时是
                    # 收缩用的 w 向量。uniform 与否由第三位标志带出。
                    params["tx_w"] = [float(inst.w_theta[row]), float(inst.w_phi[row]),
                                      1.0 if bool(inst.w_uniform[row]) else 0.0]
                if head.use_pos_embed:
                    params["pos_id"] = int(inst.position_id[row])
            for _ in range(sc["views_per_step"]):
                cam = train_cams[randint(0, len(train_cams) - 1)]
                if inst is None:
                    yaw = None
                    if sc["use_yaw"]:
                        yaw = torch.full((P, 1), float(cam_yaw(cam)), device="cuda")
                    s = compute_scalar_tx(head, embed, gaussians.get_xyz,
                                          cam.camera_center, spatial_lr_scale, txpos,
                                          yaw=yaw, normal=geom_n, aniso=geom_a,
                                          **params)
                    render_pkg = render(cam, gaussians, pipe, bg,
                                        override_color=to_override_color(s))
                    image = apply_jet(combine_bands(render_pkg["render"], head), lut)
                    gt = store(j, cam.image_name)
                    li = _img_loss(image, gt, j, cam)
                    li.backward()              # 逐视角 backward，峰值显存只按一个视角算
                    loss_sum += float(li.item())
                    n_done += 1
                    continue

                # 实例口径：成对耦合时，本步的两个视角都落在**同一对**的两个 w 上。
                # 同一相机意味着两张图的共有误差完全相同，梯度里只剩「换 w 带来的差异」。
                prow = pair_of.get(row)
                if prow is None:
                    li, _, _ = _sample(row, cam)
                    li.backward()
                    loss_sum += float(li.item())
                    n_done += 1
                elif sc["pair_diff_w"] > 0.0:
                    # 差分监督：两张图之差要对上 GT 之差。两条计算图同时在显存里
                    # （约 2×），只在显存够时开。
                    lia, ima, gta = _sample(row, cam)
                    lib, imb, gtb = _sample(prow, cam)
                    ld = l1_loss(ima - imb, gta - gtb)
                    (lia + lib + sc["pair_diff_w"] * ld).backward()
                    loss_sum += 0.5 * (float(lia.item()) + float(lib.item()))
                    n_done += 2
                else:
                    # 顺序 backward：不保留第一张的图，显存与单视角相同。
                    lia, _, _ = _sample(row, cam)
                    lia.backward()
                    lib, _, _ = _sample(prow, cam)
                    lib.backward()
                    loss_sum += 0.5 * (float(lia.item()) + float(lib.item()))
                    n_done += 2

        with torch.no_grad():
            _set_group_lr(gaussians.optimizer, "mlp", mlp_sched(step))
            if embed_sched is not None:
                _set_group_lr(gaussians.optimizer, "embed", embed_sched(step))
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            loss_sum /= max(1, n_done)
            ema = 0.4 * loss_sum + 0.6 * ema
            if step % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema:.5f}"})
                pbar.update(10)
            if step % sc["log_every"] == 0 or step == 1:
                if step == 1:
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    msg = f"  首步显存峰值 {peak:.2f} GiB"
                else:
                    msg = ""
                el = time.time() - t0
                log(f"[ang] {step:7d}/{iterations}  loss={loss_sum:.5f}(ema {ema:.5f})  "
                    f"tx={j}  {el/step*1000:.0f} ms/step  剩 {(iterations-step)*el/step/3600:.1f} h"
                    f"{msg}")

        if step in saving_iterations:
            p = os.path.join(dataset.model_path, f"angular_{step}.pth")
            save_tx_state(p, head, embed, step,
                          extra=dict(cfg, band_lambda=head.band_lambda()),
                          opacity=gaussians._opacity)
            log(f"[ITER {step}] 存 -> {p}")
    pbar.close()
    p = os.path.join(dataset.model_path, f"angular_{iterations}.pth")
    save_tx_state(p, head, embed, iterations,
                  extra=dict(cfg, band_lambda=head.band_lambda()),
                  opacity=gaussians._opacity)
    log(f"[ang] 完成，末档 {p}，用时 {(time.time()-t0)/60:.1f} 分钟")


_YAW_CACHE: dict[int, float] = {}


def cam_yaw(cam) -> float:
    """阵列朝向（弧度）。作者数据里 4 个 yaw 由图号决定：`ANGLES[(id−1)%4]`。

    从位姿反解也行，但从图号读是**同一个事实的更短写法**，且与
    `sim/gen_cbf_spectra.py` 生成 GT 时用的角度逐图一致（不会因四元数符号/数值
    误差在 90° 边界上取错分支）。
    """
    uid = int(cam.image_name)
    if uid in _YAW_CACHE:
        return _YAW_CACHE[uid]
    from sim.spectrum_gen import ANGLES
    a = float(ANGLES[(uid - 1) % 4])
    _YAW_CACHE[uid] = a
    return a


def main():
    parser = ArgumentParser(description="Angular-domain feed-forward RRF (TX-conditioned)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--spectra_root", required=True,
                        help="gen_cbf_spectra.py 的输出根（tx<idx:03d>/<图号>.png）")
    parser.add_argument("--meta", required=True,
                        help="NIST_lobby_spatial_full/meta.json（发射源表与划分）")
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--tx_list", type=int, nargs="*", default=None,
                        help="覆盖训练发射源列表（缺省用 meta 的 train 划分）")
    # ⚠ `--iterations` 已由 OptimizationParams 注册（默认 30000），这里**不能**再定义。
    #   本任务的步数靠驱动脚本显式传 `--iterations 120000`，别依赖那个默认值。
    parser.add_argument("--views_per_step", type=int, default=2)
    parser.add_argument("--max_train_instances", type=int, default=0,
                        help="只取前 K 个训练实例（0=全部）。仅用于冒烟/调试，"
                             "正式训练不要传")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_iterations", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    # 头结构
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--pe_xyz_L", type=int, default=5)
    parser.add_argument("--pe_dir_L", type=int, default=3)
    parser.add_argument("--pe_rel_L", type=int, default=4)
    parser.add_argument("--pe_tx_L", type=int, default=4)
    parser.add_argument("--use_dist", type=int, default=1)
    parser.add_argument("--use_rel", type=int, default=1)
    parser.add_argument("--use_tx", type=int, default=1)
    parser.add_argument("--use_yaw", type=int, default=0)
    parser.add_argument("--use_rx_dir", type=int, default=0,
                        help="把逐点方向反旋进接收阵本体系（需 --use_yaw 1）")
    # 发射源参数块（默认全关 ⇒ in_dim 与旧检查点一致）
    parser.add_argument("--use_tx_orient", type=int, default=0)
    parser.add_argument("--use_tx_local_dir", type=int, default=0)
    parser.add_argument("--use_tx_freq", type=int, default=0)
    parser.add_argument("--use_tx_w", type=int, default=0,
                        help="波束指向块 5 维（需同时开 --use_tx_local_dir）；"
                             "按实例的 w_label 取值")
    parser.add_argument("--use_tx_w_diff", type=int, default=0,
                        help="再追加 2 维波束差向量 Δv/Δw（阵元在 y–z 平面 ⇒ 增益只依赖"
                             "这个差）。只在 --use_tx_w 与 --use_tx_local_dir 同时打开时可用；"
                             "默认关，开则 in_dim +2、旧检查点不可续")
    parser.add_argument("--use_geom", type=int, default=0,
                        help="给头加 7 维高斯基元几何：最小轴法向 3、法向与 TX/RX 射线"
                             "夹角余弦 2、与平分线的镜面对齐度 1、log(smax/smin) 1。"
                             "max_depth=1 且 diffuse=off ⇒ 峰位由 (TX,RX,法向) 唯一决定，"
                             "而法向是现有输入推不出来的那一个。默认关，开则 in_dim +7")
    parser.add_argument("--film", type=int, default=0,
                        help="1=给基础标量头加 FiLM：由**非位置**的实例级参数"
                             "（朝向/频率/波束/RX yaw）预测每个隐层的 scale/shift。"
                             "零初始化 ⇒ 第 0 步与原头等价；配 --init_from_other 用")
    parser.add_argument("--film_pos", type=int, default=0,
                        help="1=FiLM 调制器再吃发射源位置（γ 编码）。位置是这批数据里"
                             "最强的变量，也可能是 144 实例互相干扰的来源；不加它，"
                             "FiLM 无法区分不同位置。需同时 --film 1")
    # 位置条件化（需要参数口径的实例表；表大小由 position_id 的最大值推出）
    parser.add_argument("--use_pos_embed", type=int, default=0,
                        help="给头加逐位置可学隐向量（n_pos+1 行码表，末行=未知位置回退）。"
                             "只在 w_head=none 的基础标量头上可用；默认关，开则 in_dim +pos_dim")
    parser.add_argument("--pos_dim", type=int, default=8, help="位置隐向量维度")
    parser.add_argument("--pos_dropout", type=float, default=0.0,
                        help="训练时整批丢弃位置码、改走回退行的概率。必须 >0 才有"
                             "位置留出组的可读读数（否则回退行停在随机初值）")
    parser.add_argument("--pos_max", type=int, default=0,
                        help="位置码表大小覆盖（0=按训练实例的 position_id 最大值+1）。"
                             "位置留出实例的 id 更大时用它撑开表，保证渲染端不越界")
    # w 的进入方式（与 --use_tx_w 互斥）
    parser.add_argument("--w_head", type=str, default="none",
                        choices=["none", "cplx", "fold", "af", "coh", "cohaf"],
                        help="none=w 当 5 维条件喂 MLP（--use_tx_w 控制）；"
                             "cplx=逐高斯复系数 + 输出侧闭式二次型收缩（MVDR 拓扑）；"
                             "fold=逐高斯复**阵元场** + 输出侧相干折叠取模（CBF 拓扑）；"
                             "af=每点一个标量 × **解析阵列因子**（阵元轴由几何给出，不学）；"
                             "coh=逐高斯复场渲成 Re/Im 两通道、像素上取模（w 不进图，基础保真度臂）；"
                             "cohaf=同 coh，再把解析阵列因子以**复数**乘进来（z·conj(AF)）")
    parser.add_argument("--af_norm", type=str, default="mean",
                        choices=["mean", "minmax", "none"],
                        help="解析阵列因子头的逐点尺度锚定。默认 mean：A/mean(A)（detach）。"
                             "不要用 none 配 sigmoid：渲染后 min-max 让损失对全局尺度不变，"
                             "logit 无锚会跑到负无穷、σ 饱和到 0 后梯度恒零（实测过）")
    parser.add_argument("--af_post_norm", type=str, default="minmax",
                        choices=["none", "minmax"],
                        help="解析阵列因子头的**渲染后**逐图 min-max。默认开，与 GT 的"
                             " per_image_minmax 同口径")
    parser.add_argument("--n_ant", type=int, default=16, help="发射阵元数（4×4=16）")
    parser.add_argument("--n_pos", type=int, default=1, help="加号项个数 Σ|wᴴc|²")
    parser.add_argument("--n_neg", type=int, default=1, help="减号项个数 −Σ|wᴴc|²")
    parser.add_argument("--c_init", type=float, default=0.3,
                        help="复系数初始化方差尺度（0 会让 |wᴴc|² 卡在鞍点）")
    parser.add_argument("--lpips_w", type=float, default=0.0,
                        help="感知损失权重（jet 域 LPIPS）；0=关")
    parser.add_argument("--lpips_net", type=str, default="vgg",
                        choices=["alex", "vgg"], help="感知项的骨干；默认 vgg = 评测口径")
    parser.add_argument("--grad_diag", type=int, default=0,
                        help="1=首步分别量任务项与 LPIPS 项对参数的梯度范数（只跑一次）")
    parser.add_argument("--train_views", type=str, default="train",
                        choices=["train", "test", "all"],
                        help="训练抽视角用的集合；w 数据集只有 640 个 test 视角 ⇒ 传 test")
    parser.add_argument("--inst_per_step", type=int, default=1,
                        help="一次 optimizer.step() 累加几个**不同配置**的样本（默认 1，"
                             "与历史行为一致）。优化器实际是 sign-SGD，跨配置平均能把"
                             "逐配置的梯度噪声按 √K 压下去，共享成分才有机会被看见。"
                             "显存不涨（逐样本 backward）")
    parser.add_argument("--train_view_limit", type=int, default=0,
                        help="只取该集合的前 N 个视角（0=全部）。过拟合闸用："
                             "640 视角 × 16 实例 = 10240 张，3000 步连一轮都过不完，"
                             "那不是过拟合。限到 4 个视角才能真正压训练误差")
    parser.add_argument("--train_view_stride", type=int, default=1,
                        help="先按步长抽样再取前 N 个（适应集 = view_names[::stride][:limit]）。"
                             "测试时适应要用它：交付数据的 test_index 是「每位置 4 个朝向连续排」，"
                             "直接取前 N 个只会覆盖 N/4 个位置（N=64 只覆盖 16/160），"
                             "那样测到的是覆盖不足，不是泛化能力。评测端 .tmp/_tta_eval.py "
                             "的 --stride/--skip 必须传同一对数")
    parser.add_argument("--train_instance_ids", type=int, nargs="*", default=None,
                        help="只在这些实例 id 上训练（默认 None=角色里的全部）。"
                             "测试时适应/逐场景训练用它指定单个实例")
    parser.add_argument("--train_view_random", type=int, default=0,
                        help="随机抽 N 个视角当适应集（0=关）。位置与朝向**同时**无偏；"
                             "评测端 `.tmp/_tta_eval.py --random/--seed` 必须传同一对数")
    parser.add_argument("--train_view_seed", type=int, default=0,
                        help="随机抽样的种子（训练端与评测端必须一致）")
    parser.add_argument("--freq_lo_hz", type=float, default=2.0e9)
    parser.add_argument("--freq_hi_hz", type=float, default=4.5e9)
    parser.add_argument("--train_roles", nargs="*", default=None,
                        help="实例口径下参与训练的角色（缺省 train）")
    parser.add_argument("--init_u", type=float, default=0.5)
    parser.add_argument("--n_bands", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--band_mix", type=str, default="single",
                        choices=["single", "mult"])
    parser.add_argument("--lut_n", type=int, default=256)
    # 优化
    parser.add_argument("--embed_lr", type=float, default=None)
    parser.add_argument("--embed_lr_final", type=float, default=None,
                        help="逐高斯 embedding 的末档学习率；不给则与 --embed_lr 相同"
                             "（= 旧行为，恒定不退化）。给值后与 MLP 组同一条调度")
    parser.add_argument("--init_from_other", type=str, default="",
                        help="输入扩维的暖启动：载入一个 in_dim 更小的旧存档，"
                             "第一层新增列置零（第 0 步函数与旧档逐位相同）。"
                             "用于「加一路输入有没有用」的单变量测试")
    parser.add_argument("--train_only", type=str, default="all",
                        choices=["all", "last", "out"],
                        help="头只放开哪些参数：all=全部（默认，旧行为）；"
                             "last=net.2+out（约 16.5k）；out=最后一层（约 129）。"
                             "用于测试时标定：区分「缺标定」与「缺特征」")
    parser.add_argument("--mlp_lr_init", type=float, default=8e-3)
    parser.add_argument("--mlp_lr_final", type=float, default=8e-5)
    parser.add_argument("--lr_schedule", type=str, choices=["exp", "cosine"],
                        default="exp",
                        help="exp=原指数退火；cosine=余弦退火（前半程高位平缓）")
    parser.add_argument("--lr_max_steps", type=int, default=60000)
    parser.add_argument("--lr_delay_steps", type=int, default=0)
    parser.add_argument("--lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--resume_angular", type=str, default="",
                        help="从本脚本自己的 angular_*.pth 续训（空=从头）")
    parser.add_argument("--bright_w", type=float, default=0.0,
                        help="亮区 L1 加权系数，0=原损失；w = 1 + bw·u（u 为 GT 归一化谱值）")
    parser.add_argument("--loss_l2", type=int, default=0,
                        help="0=L1（历史口径，最优解是条件**中位数**）；"
                             "1=L2（最优解是条件**均值**）。GT 右偏严重，L1 会把场压到"
                             "近零（实测场均值 0.0466 ≈ GT 中位数 0.0433，量程只有 1/4）")
    parser.add_argument("--pair_coupled", type=int, default=0,
                        help="成对耦合同步训练：同一配置的两个 w 在**同一个视角**上"
                             "同时算 loss，共有误差相消、梯度里只剩 w 成分。"
                             "需要 meta 带 pair_of（配对实例表）")
    parser.add_argument("--pair_diff_w", type=float, default=0.0,
                        help="差分损失权重：|(图a−图b) − (GTa−GTb)|，直接监督 w 响应；"
                             "0=只做成对耦合。>0 时两张图的计算图要同时保留，显存约翻倍")
    args = parser.parse_args()

    if args.iterations <= 0 or args.views_per_step < 1:
        raise SystemExit("--iterations 须 > 0，--views_per_step 须 >= 1")
    if args.n_bands == 1 and args.band_mix != "single":
        raise SystemExit("--n_bands 1 只能配 --band_mix single")
    if args.n_bands > 1 and args.band_mix == "single":
        raise SystemExit("--n_bands>1 需 --band_mix mult（否则多出的 band 是死分支）")
    if args.w_head != "none":
        if args.use_tx_w:
            raise SystemExit("--w_head 与 --use_tx_w 互斥：一个把 w 放在输出侧收缩，"
                             "一个把 w 当 MLP 输入，混在一起就归因不了")
        if args.n_bands != 1:
            raise SystemExit(f"--w_head {args.w_head} 只支持单 band（输出 1 个标量）")
        if args.n_pos + args.n_neg < 1:
            raise SystemExit("--n_pos + --n_neg 须 >= 1")

    safe_state(args.quiet)
    os.makedirs(args.model_path, exist_ok=True)
    opt = op.extract(args)
    embed_lr = args.embed_lr if args.embed_lr is not None else opt.feature_lr
    sc = {
        "spectra_root": args.spectra_root, "meta": args.meta,
        "tx_list": args.tx_list, "views_per_step": args.views_per_step,
        "embed_dim": args.embed_dim, "mlp_hidden": args.mlp_hidden,
        "mlp_depth": args.mlp_depth, "pe_xyz_L": args.pe_xyz_L,
        "pe_dir_L": args.pe_dir_L, "pe_rel_L": args.pe_rel_L, "pe_tx_L": args.pe_tx_L,
        "use_dist": bool(args.use_dist), "use_rel": bool(args.use_rel),
        "use_tx": bool(args.use_tx), "use_yaw": bool(args.use_yaw),
        "use_rx_dir": bool(getattr(args, "use_rx_dir", 0)),
        "use_tx_orient": bool(args.use_tx_orient),
        "use_tx_local_dir": bool(args.use_tx_local_dir),
        "use_tx_freq": bool(args.use_tx_freq),
        "use_tx_w": bool(args.use_tx_w),
        "use_tx_w_diff": bool(args.use_tx_w_diff),
        "use_geom": bool(args.use_geom),
        "film": bool(args.film),
        "film_pos": bool(args.film_pos),
        "use_pos_embed": bool(args.use_pos_embed),
        "pos_dim": int(args.pos_dim), "pos_dropout": float(args.pos_dropout),
        "pos_max": int(args.pos_max),
        "train_views": args.train_views,
        "train_view_limit": int(args.train_view_limit),
        "train_view_stride": int(args.train_view_stride),
        "train_instance_ids": args.train_instance_ids,
        "train_view_random": int(args.train_view_random),
        "train_view_seed": int(args.train_view_seed),
        "inst_per_step": int(args.inst_per_step),
        "freq_lo_hz": args.freq_lo_hz, "freq_hi_hz": args.freq_hi_hz,
        "train_roles": args.train_roles,
        "max_train_instances": args.max_train_instances,
        "init_u": args.init_u, "n_bands": args.n_bands, "band_mix": args.band_mix,
        "lut_n": args.lut_n, "embed_lr": embed_lr,
        "embed_lr_final": args.embed_lr_final,
        "train_only": args.train_only,
        "init_from_other": args.init_from_other,
        "mlp_lr_init": args.mlp_lr_init, "mlp_lr_final": args.mlp_lr_final,
        "lr_max_steps": args.lr_max_steps, "lr_delay_steps": args.lr_delay_steps,
        "lr_delay_mult": args.lr_delay_mult, "log_every": args.log_every,
        "resume_angular": args.resume_angular, "bright_w": args.bright_w,
        "loss_l2": bool(args.loss_l2),
        "pair_coupled": bool(args.pair_coupled), "pair_diff_w": args.pair_diff_w,
        "lr_schedule": args.lr_schedule,
        "w_head": args.w_head, "n_ant": args.n_ant,
        "n_pos": args.n_pos, "n_neg": args.n_neg, "c_init": args.c_init,
        "af_norm": args.af_norm, "af_post_norm": args.af_post_norm,
        "lpips_w": args.lpips_w, "lpips_net": args.lpips_net,
        "grad_diag": int(args.grad_diag),
        "seed": args.seed,
    }
    saving = sorted(set(args.save_iterations + [args.iterations]))
    log(f"[ang] 训练开始 model_path={args.model_path}")
    training(lp.extract(args), opt, pp.extract(args), sc, args.iterations,
             saving, args.start_checkpoint)


if __name__ == "__main__":
    main()
