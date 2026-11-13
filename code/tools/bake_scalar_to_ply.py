# -*- coding: utf-8 -*-
"""
bake_scalar_to_ply.py —— 把「标量头 + jet 查表」的颜色烘进点云 PLY（供通用 3D 查看器）
================================================================================
为什么需要它：形态 A' / A× 的颜色**不在 PLY 里**。PLY 的 `f_dc`/`f_rest` 是从种子
chkpnt30000 冻结下来的旧 SH 系数（两方法逐位相同，已实测），真正的颜色由 sidecar
`color_scalar_<it>.pth`（标量头 MLP + 固定 jet LUT）在渲染时算出。所以直接用查看器
打开 PLY 看到的是**种子的旧颜色，不是 RRF 频谱伪彩图**。

本脚本离线跑一遍标量头，把每个高斯的颜色写进 `f_dc`，得到查看器能直接打开的副本。

⚠️ 口径警告（必读，说明.md 会原样复述）
--------------------------------------------------------------------------------
模型渲染的是      image = jet( Σᵢ wᵢ·sᵢ )      —— **先混合标量，再查表**
本脚本烘的是      color_i = jet( sᵢ )          —— **先查表，再让查看器线性混合**
两者不等价：查看器渲出的是 Σᵢ wᵢ·LUT(sᵢ)，这**正是原版 RRF 的语义**，也正是形态 A'
所要取代的那个东西（见 splat/color_scalar.py 模块头）。因为 jet 不是仿射映射，
所以 `Σwᵢ·LUT(sᵢ) ≠ LUT(Σwᵢ·sᵢ)`。

⇒ 产出物是**标量场的可视化（模型的中间量）**，**不是方法的外观**。它必然比模型渲染
   更"花"，且颜色会偏离调色板。想看方法真正的输出，请看 `test/ours_70000/renders/`。

参考视角：标量头只吃 camera_center，不吃朝向
--------------------------------------------------------------------------------
`compute_scalar(head, embed, xyz, cam_center, scale)` 只经 `dir = normalize(xyz−cam)`
与 `r = ‖xyz−cam‖/scale` 用到相机中心（splat/color_scalar.py:294-307），**与相机朝向无关**。
故同一位置的多台相机给出完全相同的颜色，**参考点就是唯一自由度**。
默认取 `cameras.json` 里去重机位的中位数；`--sens` 会另算两个替代中心报敏感性。

为什么不用现成的 PLY 读写
--------------------------------------------------------------------------------
- `GaussianModel.load_ply` 硬编码 `.cuda()`，且断言 f_rest 列数 —— 本脚本要能纯 CPU 跑。
- `init_cloud.export_ply_3dgs` 会对 scale 重取 log、对四元数做 np.roll —— 那是给
  **未变换的原始数组**用的；本脚本是**就地复制**一份已就位的 PLY，套它会毁掉这两列。
- `feedforward/model/ply_export.py` 会重定心/缩放并丢掉 f_rest。
故本脚本自解析表头、按结构化 dtype 读写，**未改动的列按原始字节原样带回**。

用法（仓库根，本地 WSL env）：
    ~/miniforge3/envs/mvsplat/bin/python code/tools/bake_scalar_to_ply.py \
        --model-dir results/run_20260912_多频段标量分支/MVDR
    # 灰度中间量（= 模型的精确中间场，查看器线性混合后即 ū，后续再套 jet 即得模型输出）
    ... --mode scalar
    ... --sens          # 另报参考视角敏感性
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # code/ → splat.*
from splat.color_scalar import (load_scalar_state, compute_scalar,   # noqa: E402
                                apply_jet, make_jet_lut)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SH_C0 = 0.28209479177387814          # 同 splat/utils/sh_utils.py:114 的 RGB2SH 常数
PLY_TO_NP = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply(path: Path):
    """→ (header_bytes, props[(np_type, name)], structured ndarray)。原样保留表头字节。"""
    with open(path, "rb") as f:
        head = []
        n = None
        props = []
        while True:
            line = f.readline()
            head.append(line)
            s = line.decode("ascii", "replace").strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                parts = s.split()
                if len(parts) == 3:
                    props.append((parts[1], parts[2]))
            elif s == "end_header":
                break
        body = f.read()
    if n is None or not props:
        raise ValueError(f"PLY 表头缺 element vertex / property: {path}")
    dt = np.dtype([(name, PLY_TO_NP[t]) for t, name in props])
    need = n * dt.itemsize
    if len(body) < need:
        raise ValueError(f"PLY 体长度不足: {len(body)} < {need}（{path}）")
    # .copy()：np.frombuffer 返回只读视图，后续要就地改 f_dc/f_rest 两列
    arr = np.frombuffer(body[:need], dtype=dt, count=n).copy()
    return b"".join(head), props, arr


def write_ply(path: Path, header: bytes, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def load_ref_center(model_dir: Path, spec: str):
    """参考相机中心。auto/median → cameras.json 去重机位中位数（稳健）；cloud → 点云中心。"""
    cam_json = model_dir / "cameras.json"
    pos = None
    if cam_json.is_file():
        cams = json.load(open(cam_json, encoding="utf-8"))
        pos = np.array([c["position"] for c in cams], dtype=np.float64)
        pos = np.unique(pos.round(9), axis=0)
    if spec in ("auto", "median"):
        if pos is None:
            raise FileNotFoundError(f"缺 {cam_json}，无法自动定参考机位；请用 --ref-cam x,y,z")
        return np.median(pos, axis=0), f"cameras.json 去重机位**中位数**（{len(pos)} 个）", pos
    if spec == "mean":
        return pos.mean(0), f"cameras.json 去重机位**均值**（{len(pos)} 个）", pos
    return np.array([float(v) for v in spec.split(",")], dtype=np.float64), "命令行指定", pos


def compute_u(head, embedding, xyz_t, ref_center, scale):
    """逐高斯标量 → 合成单一标量 u（与 combine_bands 的 mult 公式同形，但在**查表之前**
    逐高斯做，故与模型输出不等价 —— 见模块头警告）。"""
    # SH 输入块 / 空间上下文块：本脚本**不做**这两条通路，且刻意不静默零填 ——
    # 零填出来的 ply 会被下游当成"这一臂的颜色"，但它是另一个函数的值。
    # 要烘这种臂得先想清楚「参考机位」对视角相关的 SH 块意味着什么（见模块头警告：
    # 逐高斯 u 本来就是参考机位的函数，与模型输出不等价）。
    _bad = []
    if getattr(head, "use_sh", "none") != "none":
        _bad.append(f"use_sh={head.use_sh}（SH 块随视线变化 ⇒ 没有唯一的逐高斯 u）")
    if int(getattr(head, "ctx_knn", 0)) > 0:
        _bad.append(f"ctx_knn={head.ctx_knn}（上下文块需要近邻图，本脚本不建）")
    if _bad:
        raise NotImplementedError(
            "bake_scalar_to_ply 不支持 SH/上下文臂：" + "；".join(_bad) +
            "。本脚本的逐高斯 u 是**参考机位的函数**（模块头已警告与模型输出不等价），"
            "再叠一层视角相关输入只会让「烘出来的东西」更难对齐 —— 请改用 "
            "render_rrf_scalar.py 出图，或先把这两块关掉（use_sh=none, ctx_knn=0）。")
    s = compute_scalar(head, embedding, xyz_t,
                       torch.as_tensor(ref_center, dtype=torch.float32), scale)  # (P,K)
    if getattr(head, "band_mix", "single") == "mult":
        lam = float(head.band_lambda())
        u = s[:, 0] * (1.0 + lam * (s[:, 1] - float(head.init_u)))
    else:
        u = s[:, 0]
    return u, s


def main():
    ap = argparse.ArgumentParser(description="把标量头颜色烘进点云 PLY")
    ap.add_argument("--model-dir", required=True,
                    help="训练产物目录（含 point_cloud/iteration_<it>/ 与 cameras.json）")
    ap.add_argument("--iteration", type=int, default=70000)
    ap.add_argument("--out", default=None,
                    help="输出 PLY；默认 <model_dir>/../阶段2_多频段_三维可视化/<方法>/point_cloud_colored.ply")
    ap.add_argument("--ref-cam", default="auto",
                    help="auto/median | mean | cloud 由 --ref-cloud 触发 | 'x,y,z'")
    ap.add_argument("--ref-cloud", action="store_true", help="参考点改用点云自身中位中心")
    ap.add_argument("--mode", choices=("jet", "scalar"), default="jet",
                    help="jet=烤 jet(u) 的 RGB（即插即用）；scalar=烤灰度 u（模型的精确中间场）")
    ap.add_argument("--device", default="cpu", help="cpu | cuda（1M 点 CPU 约几十秒）")
    ap.add_argument("--sens", action="store_true", help="另报参考视角敏感性（3 个中心）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写盘")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    M = model_dir.name
    it = args.iteration
    ply_in = model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    sidecar = model_dir / "point_cloud" / f"iteration_{it}" / f"color_scalar_{it}.pth"
    for p in (ply_in, sidecar):
        if not p.is_file():
            raise FileNotFoundError(f"缺文件: {p}")

    print(f"[1/5] 读 PLY {ply_in}")
    header, props, arr = read_ply(ply_in)
    names = [n for _, n in props]
    if not {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"} <= set(names):
        raise ValueError(f"PLY 缺 x/y/z/f_dc_*: {names[:12]}")
    frest_cols = [n for n in names if n.startswith("f_rest_")]
    n_frest = len(frest_cols)
    print(f"      {len(arr)} 高斯，{len(names)} 属性（f_rest {n_frest} 列）")

    print(f"[2/5] 读 sidecar {sidecar.name}")
    head, embedding, scale = load_scalar_state(str(sidecar), device=args.device)
    lut = make_jet_lut(head.lut_n).to(args.device)
    print(f"      head_type=scalar_jet  n_bands={head.n_bands}  band_mix={head.band_mix}  "
          f"λ={head.band_lambda()}  use_dist={head.use_dist}  lut_n={head.lut_n}  "
          f"spatial_lr_scale={scale}")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    xyz_t = torch.as_tensor(xyz, dtype=torch.float32, device=args.device)

    print("[3/5] 定参考机位")
    if args.ref_cloud:
        ref, ref_desc, _ = np.median(xyz.astype(np.float64), axis=0), "点云自身中位中心", None
    else:
        ref, ref_desc, pos = load_ref_center(model_dir, args.ref_cam)
    print(f"      参考点 {np.round(ref, 4).tolist()}  ← {ref_desc}")

    print("[4/5] 跑标量头")
    with torch.no_grad():
        u, s = compute_u(head, embedding, xyz_t, ref, scale)
        u = u.float()
        lo, hi = float(u.min()), float(u.max())
        n_clamp = int(((u < 0) | (u > 1)).sum())
        print(f"      u: min {lo:.4f}  max {hi:.4f}  均值 {float(u.mean()):.4f}  "
              f"越界 {n_clamp} ({100*n_clamp/len(u):.3f}%) → 与 apply_jet 一致地 clamp")
        if args.mode == "jet":
            rgb = apply_jet(u.view(1, 1, -1), lut)[:, 0, :].T.contiguous()    # (P,3)
        else:
            rgb = u.clamp(0, 1).unsqueeze(1).repeat(1, 3)                     # (P,3) 灰度
        f_dc = ((rgb - 0.5) / SH_C0).cpu().numpy().astype(np.float32)         # (P,3)

    if args.sens:
        print("      参考视角敏感性（平均 |ΔRGB|，对 max 1.0 归一）:")
        base = rgb.cpu().numpy()
        cands = []
        if pos is not None and len(pos):
            mid = np.median(pos, axis=0)
            cands.append(("机位中位数", mid))
            cands.append(("机位均值", pos.mean(0)))
            d = np.linalg.norm(pos - mid, axis=1)
            cands.append(("最远机位", pos[int(d.argmax())]))
            cands.append(("最近机位", pos[int(d.argmin())]))
        cands.append(("点云中位中心", np.median(xyz.astype(np.float64), axis=0)))
        seen = set()
        with torch.no_grad():
            for tag, c in cands:
                key = tuple(np.round(c, 6).tolist())
                if key in seen:
                    continue
                seen.add(key)
                u2, _ = compute_u(head, embedding, xyz_t, c, scale)
                r2 = (apply_jet(u2.view(1, 1, -1), lut)[:, 0, :].T
                      if args.mode == "jet" else u2.clamp(0, 1).unsqueeze(1).repeat(1, 3))
                print(f"        {tag:<12s} {np.round(c,3).tolist()}  "
                      f"平均|ΔRGB| {np.abs(r2.cpu().numpy()-base).mean():.4f}")

    if args.dry_run:
        print("[5/5] --dry-run，不写盘。")
        return

    print("[5/5] 写 PLY（f_rest 清零以防查看器叠加视角相关项）")
    out = Path(args.out) if args.out else (
        model_dir.parent / "阶段2_多频段_三维可视化" / M / "point_cloud_colored.ply")
    nz_before = int(np.count_nonzero(arr["f_rest_0"])) if n_frest else 0
    for i, cn in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        arr[cn] = f_dc[:, i]
    for cn in frest_cols:
        arr[cn] = 0.0
    write_ply(out, header, arr)
    print(f"      → {out}  ({out.stat().st_size/2**20:.1f} MB)")
    assert frest_cols and nz_before >= 0
    print(f"      写入前 f_rest_0 非零数 {nz_before}；写入后已全部清零")
    print(f"      还原公式（查看器行为）：rgb = clip(0.5 + {SH_C0}·f_dc) = "
          f"{'jet(u)' if args.mode == 'jet' else 'u 的灰度'}")
    print("完成。⚠️ 这是标量场的可视化，不是方法渲染的输出（见脚本头警告）。")


if __name__ == "__main__":
    main()
