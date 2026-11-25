# -*- coding: utf-8 -*-
"""角域前馈 RRF 的渲染/导出：给定一个训练好的头，输出**任意发射源**在指定视角下的谱图。

三条输出通路（同一个头、同一份几何，差别只在 GT 从哪来）：
    --split test   作者的 640 个测试视角（与 RF-3DGS 发布评测同集合）
    --split train  作者剩下的 2560 个视角
    --export_gt    把 `--spectra_root/tx<idx>/` 的灰度谱图查 jet 表落成 PNG
                   （评测时渲染图与 GT 图必须都是 RGB；灰度直接比会口径错位）

渲染口径与训练**逐字一致**：
    s = head(embed, γ(xyz), dir, γ(dir), r, tx, yaw?) → override_color → 光栅化
      → combine_bands → apply_jet → 8bit RGB PNG
背景取标量 0（未覆盖像素 = jet 深蓝端），与训练一致。

用法（WSL）：
    $PY render_rrf_angular.py -s .../3dgs_CBF_100 -m .../results/<run> \
        --checkpoint .../angular_120000.pth --tx 48 --split test \
        --out_root .../results/<run>/tx048_test/renders
    $PY render_rrf_angular.py ... --tx 48 --split test --export_gt \
        --out_root .../results/<run>/tx048_test/gt_repro
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from rfgs.core.angular_data import (SpectrumStore, load_cameras, load_instances,  # noqa: E402
                          load_tx_table)
from splat.arguments import ModelParams, PipelineParams                # noqa: E402
from splat.color_scalar import (apply_jet, combine_bands, make_jet_lut,  # noqa: E402
                                to_override_color)
from splat.color_scalar_tx import compute_scalar_tx, load_tx_state     # noqa: E402
from splat.gaussian_renderer import render                             # noqa: E402
from splat.scene import GaussianModel                                  # noqa: E402

from rfgs.rrf.train_rrf_angular import cam_yaw                                  # noqa: E402
from rfgs.rrf.train_rrf_scalar import load_visual_geometry                      # noqa: E402


def to_u8(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) ∈ [0,1] → (H,W,3) uint8。"""
    a = img.detach().clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return np.transpose(a, (1, 2, 0))


def main():
    ap = ArgumentParser(description="Render angular RRF from a TX-conditioned head")
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--spectra_root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tx", type=int, default=None,
                    help="旧口径：发射源下标（与 --instance 二选一）")
    ap.add_argument("--instance", type=int, default=None,
                    help="参数口径：实例 id（读 meta 的 instances）")
    ap.add_argument("--param_source", type=int, default=None,
                    help="只换条件不换几何：用该实例的 yaw/pitch/freq 喂头，"
                         "但仍渲染 --instance 的 GT（G5 置换对照）")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--role", default=None,
                    help="批量口径：按 meta 里的角色名渲染全部实例（与 --tx/--instance 互斥），"
                         "输出 <out_root>/inst<id>/{renders,gt}，视角由实例的 views 字段决定")
    ap.add_argument("--param_mode", choices=["as_is", "cycle", "sibling_w"],
                    default="as_is",
                    help="cycle：把参数换成同角色下一个实例的（G5 置换对照），几何不动；"
                         "sibling_w：只把 tx_w 换成该实例 base_instance 的（训练域 w），"
                         "其余条件不动 —— 用来测「头是否真的在用 w」")
    ap.add_argument("--max_instances", type=int, default=0)
    ap.add_argument("--save_pmax", action="store_true",
                    help="把每张图查表**前**的标量场峰值写进 <out_root>/pmax.jsonl"
                         "（A0：模型自己的峰值跨 w 排序，即「模型版 EBA」）")
    ap.add_argument("--save_gray", action="store_true",
                    help="额外把查表前的标量场存成 <out_root>/gray/<view>.png（8 位），"
                         "供「整张功率图在导向字典上投影」这类分析用，避免从 jet 反查 u")
    ap.add_argument("--renorm_pred", action="store_true",
                    help="查表前把标量场**逐图** min–max 归一再上色：跨图共同尺度臂的"
                         "形状公平对照（否则亮度域不同，PSNR 会被亮度差主导）")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--export_gt", action="store_true",
                    help="不渲染，只把该发射源的灰度谱图查 jet 表导出成 RGB")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default=None,
                    help="逗号分隔的视角号（如 01503,00442），只渲染/导出这些视角；"
                         "与 --limit 叠加时先取 --views")
    ap.add_argument("--start_checkpoint", default=None,
                    help="覆盖 cfg 里记的几何起点。检查点是在服务器上训的时，"
                         "cfg 里的 start_checkpoint 是绝对路径（/path/to/xxx.pth），"
                         "本机不可达 —— 必须用本机的 blender_visual_trained/chkpnt30000.pth")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    by_role = args.role is not None
    if by_role:
        if args.tx is not None or args.instance is not None:
            raise SystemExit("--role 与 --tx/--instance 互斥")
    elif (args.tx is None) == (args.instance is None):
        raise SystemExit("--tx 与 --instance 必须二选一")

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    cams, train_names, test_names = load_cameras(dataset.source_path)
    names = train_names if args.split == "train" else test_names
    if args.limit:
        names = names[: args.limit]
    want = None
    if args.views:
        want = [v.strip().zfill(5) for v in args.views.split(",") if v.strip()]
        miss = [v for v in want if v not in (train_names + test_names)]
        if miss:
            raise SystemExit(f"--views 里有不存在的视角：{miss}")
    os.makedirs(args.out_root, exist_ok=True)

    inst = load_instances(args.meta) if (args.instance is not None or by_role) else None
    if (args.instance is not None or by_role) and inst is None:
        raise SystemExit("--instance/--role 需要 meta 里有 instances")

    def _row(x):
        r = int(np.searchsorted(inst.ids, int(x)))
        if r >= len(inst.ids) or int(inst.ids[r]) != int(x):
            raise SystemExit(f"实例 {x} 不在表里")
        return r

    def _params(row):
        # tx_w 一律带上：头没开 use_tx_w 时 `_param_block` 不读它，传了也无害；
        # 开着时这就是波束条件。`--param_mode cycle` 会连 w 一起换掉，正好是
        # 「同几何、换条件」的置换对照。
        return dict(tx_yaw=float(inst.yaw_off[row]),
                    tx_pitch=float(inst.pitch_off[row]),
                    tx_freq=float(inst.freq_hz[row]),
                    tx_rot=torch.as_tensor(inst.rot[row], dtype=torch.float32),
                    tx_w=[float(inst.w_theta[row]), float(inst.w_phi[row]),
                          1.0 if bool(inst.w_uniform[row]) else 0.0])

    # jobs: 每个元素是一次「几何位置 + 条件参数 + 输出目录」
    jobs = []
    if by_role:
        if args.role not in inst.roles:
            raise SystemExit(f"角色 {args.role} 不在表里（有 {sorted(inst.roles)}）")
        rows = [int(r) for r in inst.roles[args.role]]
        if args.max_instances:
            rows = rows[: args.max_instances]
        if args.param_mode == "cycle" and len(rows) < 2:
            raise SystemExit("--param_mode cycle 至少需要 2 个实例")
        store = SpectrumStore(args.spectra_root, key_fmt=inst.key_fmt)
        n_wswap = 0
        for n, row in enumerate(rows):
            if args.param_mode == "cycle":
                src = rows[(n + 1) % len(rows)]
            elif args.param_mode == "sibling_w":
                bid = int(inst.base_instance[row])
                if bid < 0:
                    raise SystemExit(f"实例 {int(inst.ids[row])} 没有 base_instance，"
                                     f"不能用 --param_mode sibling_w")
                src = _row(bid)
            else:
                src = row
            iid = int(inst.ids[row])
            params = _params(row)
            if args.param_mode == "sibling_w":
                params["tx_w"] = _params(src)["tx_w"]   # 只换 w，其余条件保持原样
                n_wswap += int(params["tx_w"] != _params(row)["tx_w"])
            base = os.path.join(args.out_root, f"inst{iid:03d}")
            jobs.append(dict(iid=iid, row=row, src=src, txpos_np=inst.xyz[row],
                             params=params,
                             views=str(inst.views[row]),
                             renders=os.path.join(base, "renders"),
                             gt=os.path.join(base, "gt")))
        if args.param_mode == "cycle":
            print(f"[render] G5 置换：{len(jobs)} 个实例的条件参数全部换成同角色"
                  f"下一个实例的（几何不动）", flush=True)
        elif args.param_mode == "sibling_w":
            print(f"[render] w 置换：{len(jobs)} 个实例中 {n_wswap} 个的被换成"
                  f"base_instance 的训练域 w（其余条件不动）", flush=True)
    elif inst is None:
        tx_world, _, _, _ = load_tx_table(args.meta)
        iid = int(args.tx)
        store = SpectrumStore(args.spectra_root)
        jobs.append(dict(iid=iid, row=None, src=None, txpos_np=tx_world[iid],
                         params={}, views=None,
                         renders=args.out_root, gt=args.out_root))
    else:
        row = _row(args.instance)
        if args.param_source is not None:
            src = _row(args.param_source)
        elif args.param_mode == "sibling_w":
            # 单实例口径也要能置换 w，否则「同事换 w」的对照会静默退化成自我对照
            # （2026-11-02 踩过：三个臂全读到 RMS 0.000，实际两批是同一批图）。
            bid = int(inst.base_instance[row])
            if bid < 0:
                raise SystemExit(f"实例 {int(inst.ids[row])} 没有 base_instance，"
                                 f"不能用 --param_mode sibling_w")
            src = _row(bid)
        elif args.param_mode == "cycle":
            raise SystemExit("--param_mode cycle 需要 --role（或显式 --param_source）")
        else:
            src = row
        iid = int(inst.ids[row])
        if args.param_mode == "sibling_w" and args.param_source is None:
            print(f"[render] w 置换：渲染实例 {iid}，tx_w 取自实例 "
                  f"{int(inst.ids[src])}（几何不动）", flush=True)
        store = SpectrumStore(args.spectra_root, key_fmt=inst.key_fmt)
        jobs.append(dict(iid=iid, row=row, src=src,
                         txpos_np=inst.xyz[row],          # 几何位置永远取被渲染实例的
                         params=_params(src), views=None,
                         renders=args.out_root, gt=args.out_root))
        if args.param_source is not None:
            print(f"[render] G5 置换：渲染实例 {iid}，条件取自实例 "
                  f"{int(inst.ids[src])}", flush=True)

    if args.export_gt:
        store = SpectrumStore(args.spectra_root, device="cpu",
                              key_fmt=store.key_fmt)
        n = 0
        for job in jobs:
            nn = (train_names + test_names) if job["views"] == "all" else test_names
            if args.limit:
                nn = nn[: args.limit]
            if want:
                nn = [n for n in nn if n in want]
            os.makedirs(job["gt"], exist_ok=True)
            for nm in nn:
                gt = store(job["iid"], nm)
                Image.fromarray(to_u8(gt)).save(os.path.join(job["gt"], f"{nm}.png"))
                n += 1
        print(f"[gt] 导出 {n} 张 / {len(jobs)} 个实例 → {args.out_root}", flush=True)
        return

    head, embed, opacity, cfg, it = load_tx_state(args.checkpoint, device=args.device)
    # eval 必须显式调：位置码的 dropout 只在 training=True 时生效，渲染期若还留着
    # train 模式，同一实例每次前向会随机换成回退行（指标变成随机量的平均）。
    head.eval()
    use_pos = bool(getattr(head, "use_pos_embed", False))
    seen_pos = set(int(x) for x in (cfg.get("train_pos_ids") or []))
    print(f"[render] checkpoint iter={it}  P={embed.shape[0]}  "
          f"jobs={len(jobs)}  split={args.split}  n_views={len(names)}  "
          f"in_dim={head.in_dim}", flush=True)
    gaussians = GaussianModel(dataset.sh_degree)
    load_visual_geometry(dataset_start(args), gaussians, dataset.sh_degree)
    if opacity is not None:
        if opacity.numel() != gaussians._opacity.numel():
            raise RuntimeError(f"checkpoint opacity {opacity.numel()} != 几何 "
                               f"{gaussians._opacity.numel()} ⇒ 不是同一份点云")
        gaussians._opacity = torch.nn.Parameter(opacity.reshape(-1, 1).contiguous(),
                                                requires_grad=False)
    else:
        print("[render] ⚠ checkpoint 没有 opacity，沿用种子检查点的值", flush=True)

    lut = make_jet_lut(int(cfg.get("lut_n", 256))).to(args.device)
    geom_n = geom_a = None
    if getattr(head, "use_geom", False):
        from splat.color_scalar_tx import gaussian_geom
        geom_n, geom_a = gaussian_geom(gaussians)
        print(f"[render] 几何块：法向+镜面对齐 7 维  in_dim={head.in_dim}", flush=True)
    bg = torch.zeros(3, device=args.device)
    pmax_rows = []
    P = gaussians.get_xyz.shape[0]
    scale = float(cfg.get("spatial_lr_scale", 0.0) or 0.0)
    if scale <= 0:
        raise RuntimeError("cfg 里 spatial_lr_scale 无效")

    with torch.no_grad():
        for ji, job in enumerate(jobs):
            if job["views"] == "all":
                nn = train_names + test_names
            elif job["views"] == "test":
                nn = test_names
            else:
                nn = names
            if args.limit:
                nn = nn[: args.limit]
            if want:
                nn = [n for n in nn if n in want]
            os.makedirs(job["renders"], exist_ok=True)
            txpos = torch.as_tensor(job["txpos_np"], device=args.device,
                                    dtype=torch.float32)
            # 位置码：只有「训练时见过该位置」才查表；位置留出组一律走回退行，
            # 否则读到的是一行从没被训练过的随机向量。
            pk = {}
            if use_pos:
                pid = None if job["row"] is None else int(
                    inst.position_id[job["row"]])
                if pid is not None and pid not in seen_pos:
                    print(f"[render] inst{job['iid']:03d} 位置 {pid} 未参与训练 ⇒ 回退行",
                          flush=True)
                    pid = None
                pk["pos_id"] = pid
            for k, nm in enumerate(nn):
                cam = cams[nm]
                yaw = None
                if cfg.get("use_yaw", False):
                    yaw = torch.full((P, 1), float(cam_yaw(cam)), device=args.device)
                s = compute_scalar_tx(head, embed, gaussians.get_xyz, cam.camera_center,
                                      scale, txpos, yaw=yaw, normal=geom_n,
                                      aniso=geom_a, **job["params"], **pk)
                pkg = render(cam, gaussians, pipe, bg, override_color=to_override_color(s))
                band = combine_bands(pkg["render"], head)
                if args.renorm_pred:
                    bmin, bmax = band.min(), band.max()
                    band = (band - bmin) / torch.clamp(bmax - bmin, min=1e-12)
                img = apply_jet(band, lut)
                Image.fromarray(to_u8(img)).save(
                    os.path.join(job["renders"], f"{nm}.png"))
                if args.save_pmax:
                    # 查表**前**的标量峰值：jet 前的量程才是「这个 w 有多强」的
                    # 载体；归一化（若有）发生在这里之前/之后由训练目标决定。
                    pmax_rows.append({"id": int(job["iid"]), "view": int(nm),
                                      "pmax": float(band.detach().max()),
                                      "pmean": float(band.detach().mean())})
                if args.save_gray:
                    # 统一放 <out_root>/gray：单实例模式 out_root 就是渲染目录，
                    # 按 dirname(renders) 会写到上一层并把 3 个实例互相覆盖（已踩）。
                    gdir = os.path.join(args.out_root, "gray")
                    os.makedirs(gdir, exist_ok=True)
                    g = band.detach().clamp(0, 1)
                    if g.dim() == 3:
                        g = g[0]
                    Image.fromarray(
                        (g.mul(255.0).round().to(torch.uint8).cpu().numpy())).save(
                            os.path.join(gdir, f"{nm}.png"))
                if (k + 1) % 100 == 0:
                    print(f"[render] {ji+1}/{len(jobs)} inst{job['iid']:03d} "
                          f"{k+1}/{len(nn)}", flush=True)
            print(f"[render] inst{job['iid']:03d} 完成 {len(nn)} 张  "
                  f"({ji+1}/{len(jobs)})", flush=True)
    if args.save_pmax and pmax_rows:
        out_p = os.path.join(args.out_root, "pmax.jsonl")
        with open(out_p, "w", encoding="utf-8") as fh:
            for r in pmax_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[render] 写出 {out_p}（{len(pmax_rows)} 行）", flush=True)
    print(f"[render] 完成 → {args.out_root}", flush=True)


def dataset_start(args):
    """起点检查点：`angular_*.pth` 的 cfg 里记了（与 -s 无关）。

    `--start_checkpoint` 优先：服务器上训出来的检查点，cfg 里是服务器路径，
    本机直接载会 PermissionError（/root 不可读，报的却是「权限」而不是「不存在」）。
    """
    if getattr(args, "start_checkpoint", None):
        return args.start_checkpoint
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sc = ck["cfg"].get("start_checkpoint")
    if not sc:
        raise RuntimeError("checkpoint cfg 缺 start_checkpoint，无法恢复几何")
    return sc


if __name__ == "__main__":
    main()
