# -*- coding: utf-8 -*-
"""
render_rrf_walk.py —— RRF 各颜色头的**新视角漫游**渲染（构造数据集中不存在的机位）
================================================================================
支持三种颜色头（`--head`），**渲染链一律与评测路径共用同一个函数**：

| `--head` | 颜色来源 | 共用的合成函数 |
|---|---|---|
| `scalar` | sidecar `color_scalar_<i>.pth`（标量头 + jet 查表，形态 A'/A×） | `render_rrf_scalar.synthesize()` |
| `mlp`    | sidecar `color_mlp_<i>.pth`（embedding + 共享 MLP → RGB） | `render_rrf_mlp.synthesize()` |
| `sh`     | ply 里的逐步 SH 系数（原版引擎 `train.py`，**无 sidecar**） | `gaussian_renderer.render()` |
| `auto`   | 按目录里存在的 sidecar 自动判定（都没有 ⇒ sh） | — |

两条路径必须算同一个函数，否则漫游视频展示的就不是被评测的那个模型；故这里绝不
自己重写合成链，只做「取相机 → 调对应头的共享合成函数 → 存图」。

相机位姿族（实测，本脚本据此设计轨迹，不是拍脑袋）
--------------------------------------------------------------------------------
对 3200 台相机全量复核（`.diag` 侧的 numpy 验证，误差 0.0 / 5.3e-15）：
  - 800 个去重位置 × **4 个真正不同的方位角**（{0,90,180,270}°，90° 间隔）
  - 天顶角 **恒 90°**（光轴水平）、up 恒 (0,0,1)（**零 roll**）、det(R)=1
  - `rotation` 是 **camera-to-world**（`camera_utils.py:62-70` 的 `W2C` 变量名是误导的，
    它存的是 inv(Rt)）⇒ 光轴 = R 的第 3 **列**，up = −(R 的第 2 列)
  - 相机中心到水平质心 (−0.775,−5.86) 的半径：min 2.08 / 中位 6.00 / max 8.90 m

⇒ **绕垂直轴的水平环绕正是数据集自己的位姿族**，属**方位角内插**，不是外推。
  只有**俯仰**（光轴偏离水平，数据 0/3200 采样）与 roll 才是真外推，
  故 `orbit_pitch` 单独作为**标注清楚的外推对照臂**，默认不跑。

轨迹
--------------------------------------------------------------------------------
  orbit        绕过水平质心的垂直轴 360°，半径 r（默认 6.0 m，= 真实机位半径中位数），
               视线指向轴心，up=(0,0,1)。**方位角内插。**
  dolly        在两个真实机位之间直线推进（端点均为实测机位 ⇒ 首末帧是内插端点），
               光轴 = 行进方向的**水平投影**（保持天顶角 90°，不白送外推）。**位置内插。**
  orbit_pitch  orbit + 光轴俯仰 `--pitch-deg`。**真外推，默认不跑。**

诚实性证据：逐帧记录相机中心到**最近真实机位**的距离，以及方位角/天顶角，写进
`轨迹.json`。内插还是外推，看数不看嘴。

用法（服务器 code/ 下；`-m` 指向训练产物目录）：
    <PY> tools/render_rrf_walk.py -m <model_dir> --iteration 70000 \
         --traj orbit --frames 120 --out <out_dir>
    <PY> tools/render_rrf_walk.py -s <dataset> -m <model_dir> --iteration 70000 --selfcheck

多臂并排（把不同臂渲进同一 out 的不同子目录，`--arm-out` 一给就多一层目录）：
    <PY> tools/render_rrf_walk.py -m <A1_dir> --iteration 60000 --arm-out A1 --out <VIS>
    <PY> tools/render_rrf_walk.py -m <A2_dir> --iteration 60000 --arm-out A2 --out <VIS>
    <PY> tools/render_rrf_walk.py -m <A3_dir> --iteration 60000 --arm-out A3 --out <VIS>
    → <VIS>/{A1,A2,A3}/<traj>/{00000.png..,轨迹.json}
不给 `--arm-out` 时输出仍是 <out>/<traj>/（与既有交付物逐字兼容）。
"""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # code/ → splat.*
from splat.scene.cameras import Camera                                # noqa: E402
from splat.scene.gaussian_model import GaussianModel                  # noqa: E402
from splat.arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from splat.utils.general_utils import safe_state                      # noqa: E402
from splat.color_scalar import (load_scalar_state, make_jet_lut,      # noqa: E402
                                check_knn_xyz)
from splat.color_mlp import load_color_state                          # noqa: E402
from splat.gaussian_renderer import render                            # noqa: E402
from rfgs.rrf.render_rrf_scalar import synthesize as synthesize_scalar         # noqa: E402
from rfgs.rrf.render_rrf_mlp import synthesize as synthesize_mlp               # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

E3 = np.array([0.0, 0.0, 1.0])


def log(*a):
    """关键读数一律走 stderr：--quiet 会把 sys.stdout 换成 no-op（general_utils.py:112），
    但 stderr 不受影响。"""
    print("[walk]", *a, file=sys.stderr)


# ── 相机构造 ─────────────────────────────────────────────────────────────────
def build_R(fwd, up_hint):
    """(fwd, up) → camera-to-world R。

    约定（实测定标，见模块头）：+X_cam = right，+Y_cam = down，+Z_cam = fwd，
    且 right × up = −fwd。故 right = fwd × up_hint，up = right × fwd，
    R = [right, −up, fwd] **按列堆**（axis=-1；按行堆误差达 2.0）。
    """
    fwd = np.asarray(fwd, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    right = np.cross(fwd, up_hint)
    n = np.linalg.norm(right)
    if n < 1e-9:
        raise ValueError("fwd 与 up 平行，无法定出 right")
    right = right / n
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd], axis=-1)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, np.linalg.det(R)
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-9
    return R, right, up, fwd


def make_camera(C, fwd, up_hint, W, H, fovx, fovy, data_device="cpu"):
    """由 (中心, 光轴, up) 造一台 Camera。T = −Rᵀ·C（不是 −R·C）。"""
    C = np.asarray(C, dtype=np.float64)
    R, right, up, fwd = build_R(fwd, up_hint)
    T = -R.T @ C
    img = torch.zeros(3, H, W, dtype=torch.float32)
    cam = Camera(colmap_id=-1, R=R, T=T, FoVx=fovx, FoVy=fovy, image=img,
                 gt_alpha_mask=None, image_name="walk", uid=-1,
                 data_device=data_device)
    # 自洽自检：构造出的 camera_center 必须回到 C。
    # **必须用相对容差**：`getWorld2View2` 末行是 `return np.float32(Rt)`
    # （splat/utils/graphics_utils.py:49），故 `camera_center` 是 **float32** ——
    # 真实评测路径里的 Camera 同样是 float32，不是本脚本的特例。
    # C2W 在 float64 里 inv→inv 再落 float32，实测 |ΔC| 随量级走（6 m 处 1.8e-7，
    # 10.5 m 处 1.4e-6 ≈ 1.5 ulp），是纯舍入。相对误差 ~1e-7，对渲染无影响。
    # 立场若写错（T=−RC 而非 −RᵀC）会差到米级，这个断言抓得住。
    cc = cam.camera_center.detach().cpu().numpy()
    d_cc = float(np.abs(cc - C).max())
    rel = float((np.abs(cc - C) / np.maximum(np.abs(C), 1.0)).max())
    assert np.all(np.abs(cc - C) <= 1e-6 + 1e-6 * np.abs(C)), (cc, C)
    cam._walk = {"C": C, "fwd": fwd, "up": up, "d_center_f32": d_cc, "rel_center_f32": rel}
    return cam


def rot_about(v, k, ang):
    """v 绕单位轴 k 转 ang 弧度（Rodrigues）。k ⟂ v 时退化成 v·cos + (k×v)·sin。"""
    k = np.asarray(k, dtype=np.float64)
    return (v * np.cos(ang) + np.cross(k, v) * np.sin(ang)
            + k * np.dot(k, v) * (1.0 - np.cos(ang)))


# ── 轨迹 ─────────────────────────────────────────────────────────────────────
def traj_orbit(P, W, H, fovx, fovy, frames, radius, pitch_deg):
    cx, cy = np.median(P[:, 0]), np.median(P[:, 1])
    z0 = float(np.median(P[:, 2]))
    log(f"环绕轴: 水平 ({cx:.3f}, {cy:.3f})  z={z0:.3f}  半径 {radius:.3f} m"
        + (f"  俯仰 {pitch_deg:+.1f}°（外推）" if pitch_deg else "  （光轴水平 = 数据集位姿族）"))
    p = np.radians(pitch_deg)
    out = []
    for i in range(frames):
        a = 2 * np.pi * i / frames
        C = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z0])
        fwd = np.array([-np.cos(a), -np.sin(a), 0.0])
        up_h = E3.copy()
        if p:
            R0, right0, up0, fwd0 = build_R(fwd, up_h)
            fwd = rot_about(fwd0, right0, p)
            up_h = rot_about(up0, right0, p)
        out.append(make_camera(C, fwd, up_h, W, H, fovx, fovy))
    return out


def traj_dolly(P, W, H, fovx, fovy, frames, pair):
    if pair == "auto":
        # 取相距最远的一对真实机位（两端都落在实测位置上 ⇒ 首末帧是内插的端点）
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        i, j = np.unravel_index(int(d.argmax()), d.shape)
    else:
        i, j = (int(v) for v in pair.split(","))
    A, B = P[i], P[j]
    step = B - A
    fwd = np.array([step[0], step[1], 0.0])
    # 光轴**投影到水平面**：两端机位的 z 不同（实测差 0.487 m ⇒ 行进方向带 1.8° 俯角），
    # 若直接拿行进方向当光轴，天顶角就不是 90° 了，而这 1.8° 是白送的**真外推** ——
    # 数据集 3200 台天顶角恒为 90.0000°。路径本身不变（端点仍是实测机位）。
    log(f"推轨: 机位 {i} {A.round(3).tolist()} → {j} {B.round(3).tolist()}  "
        f"长度 {np.linalg.norm(step):.3f} m（水平投影 {np.linalg.norm(fwd):.3f} m）  "
        f"光轴=行进方向的水平投影")
    out = []
    for k in range(frames):
        t = k / max(frames - 1, 1)
        out.append(make_camera(A + t * step, fwd, E3.copy(), W, H, fovx, fovy))
    return out


# ── 主流程 ───────────────────────────────────────────────────────────────────
def sidecar_paths(model_path, iteration):
    """三种头各自可能存在的 sidecar 路径（judge 顺序：scalar → mlp → sh）。"""
    it = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    return (it / f"color_scalar_{iteration}.pth",
            it / f"color_mlp_{iteration}.pth")


def resolve_head(model_path, iteration, device, head="auto"):
    """定出这一档用的是哪种颜色头，并把渲染所需的全部状态 load 出来。

    返回 (kind, state)：
      kind="scalar" → state=(head, embedding, scale, lut)
      kind="mlp"    → state=(head, embedding, scale)
      kind="sh"     → state=()          # 颜色在 ply 的 SH 系数里，无需额外状态
    """
    sc, ml = sidecar_paths(model_path, iteration)
    if head == "auto":
        # 一个档只该有一种 sidecar；两个都在说明训练侧写错了，宁可报错不要猜。
        if sc.is_file() and ml.is_file():
            raise RuntimeError(f"两种 sidecar 同时存在，无法自动判定颜色头: {sc} / {ml}")
        head = "scalar" if sc.is_file() else ("mlp" if ml.is_file() else "sh")
    if head == "scalar":
        if not sc.is_file():
            raise FileNotFoundError(f"缺 scalar sidecar: {sc}（需 train_rrf_scalar.py 在该档写过）")
        h, e, s = load_scalar_state(str(sc), device=device)
        return "scalar", (h, e, s, make_jet_lut(h.lut_n).to(device))
    if head == "mlp":
        if not ml.is_file():
            raise FileNotFoundError(f"缺 mlp sidecar: {ml}（需 train_rrf_mlp.py 在该档写过）")
        h, e, s = load_color_state(str(ml), device=device)
        return "mlp", (h, e, s)
    return "sh", ()


def make_synth(kind, state, g, pipeline, background):
    """把「合成一条射线成像」这件事收敛成一个 view → (3,H,W) 的可调用对象。

    三个分支各自调**对应头的共享合成函数**，本脚本不重写任何合成链
    （render_rrf_scalar.synthesize / render_rrf_mlp.synthesize / 引擎原生 render）。
    """
    if kind == "scalar":
        h, e, s, lut = state
        # SH 输入块：几何冻结 ⇒ 整条轨迹共用一份。用尾随位置参数传给 synthesize_scalar
        # （它的 sh_coeff 就加在最后，正是为了不打乱这一行的既有位置参数）。
        sh_coeff = (g.get_features.transpose(1, 2).contiguous()
                    if h.use_sh != "none" else None)
        if h.ctx_knn > 0:
            check_knn_xyz(h, g.get_xyz)      # 近邻图与这份 ply 是否同源
        return lambda view: synthesize_scalar(view, g, h, e, s, lut, pipeline,
                                              background, sh_coeff)
    if kind == "mlp":
        h, e, s = state
        # 必须把 s（= spatial_lr_scale）传进去：render_rrf_mlp.synthesize 的签名是
        #   (view, gaussians, head, embedding, spatial_lr_scale, pipeline, background)
        # 漏掉它 → TypeError: synthesize() missing 1 required positional argument。
        # **闸② 抓到的就是这个**，而且是它唯一能抓到的地方：scalar 分支一直是对的，
        # 所以在此之前「相机等价性」只被验过标量链，"只有标量链有证据"这句掩盖了
        # 「MLP 链根本没跑通过」—— 这正是闸② 要按头各跑一次的理由。
        return lambda view: synthesize_mlp(view, g, h, e, s, pipeline, background)
    return lambda view: render(view, g, pipeline, background)["render"]


def head_info(kind, state):
    """该头的可 JSON 化描述，写进 轨迹.json（哪一臂、哪个头，事后得能对上）。"""
    if kind == "scalar":
        h = state[0]
        # 新块必须进 轨迹.json：A2/A4/A5 的**唯一**差别就在这四个键上，
        # 事后要靠它把视频对上哪一臂（selfcheck 也因此免费覆盖新头）。
        return {"kind": "scalar", "n_bands": h.n_bands, "band_mix": h.band_mix,
                "lambda": h.band_lambda(), "lut_n": h.lut_n,
                "use_dist": bool(h.use_dist), "spatial_lr_scale": state[2],
                "use_sh": h.use_sh, "sh_degree": int(h.sh_degree),
                "ctx_knn": int(h.ctx_knn), "ctx_dim": int(h.ctx_dim),
                "knn_file": h.knn_file}
    if kind == "mlp":
        h = state[0]
        return {"kind": "mlp", "embed_dim": h.embed_dim, "mlp_hidden": h.mlp_hidden,
                "mlp_depth": h.mlp_depth, "pe_xyz_L": h.pe_xyz_L,
                "pe_dir_L": h.pe_dir_L, "use_dist": bool(h.use_dist),
                "spatial_lr_scale": state[2]}
    return {"kind": "sh",
            "note": "原版引擎：颜色 = ply 里的逐步 SH 系数，无 sidecar"}


def load_model(model_path, iteration, device, sh_degree, pipeline, background,
               head="auto"):
    ply = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"缺文件: {ply}")
    kind, state = resolve_head(model_path, iteration, device, head)
    g = GaussianModel(sh_degree)          # load_ply 会断言 f_rest 列数 == 3(deg+1)²−3
    g.load_ply(str(ply))
    log(f"载入 {ply.parent.name}: {g.get_xyz.shape[0]} 高斯  颜色头={kind}  "
        f"{ {k: v for k, v in head_info(kind, state).items() if k != 'kind'} }")
    return g, make_synth(kind, state, g, pipeline, background), head_info(kind, state)


def dataset_pose_stats(model_path):
    """从 cameras.json 取真实位姿统计（用于轨迹选参与诚实性对照）。"""
    cams = json.load(open(Path(model_path) / "cameras.json", encoding="utf-8"))
    R = np.array([c["rotation"] for c in cams], dtype=np.float64)
    C = np.array([c["position"] for c in cams], dtype=np.float64)
    fwd = R @ E3                       # camera-to-world 的第 3 列 = 光轴
    up = -(R @ np.array([0.0, 1.0, 0.0]))
    zen = np.degrees(np.arccos(np.clip(fwd[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]))
    return C, np.unique(C.round(9), axis=0), up, az, zen


def selfcheck(args, dataset, pipeline, device):
    """V2：用真实 test 相机的位姿重建一台合成 Camera，两次渲染必须逐元素一致。

    这才是相机构造的正确性判据。**不与已存 PNG 比** —— PNG 是 8 bit 量化的弱判据。
    **三种颜色头各跑一次**（`--head` 决定跑哪个；scalar/mlp/sh 都要有等价性证据，
    相机口径与头无关，但"合成链调对了没有"只有真渲一遍才知道）。
    """
    from splat.scene import Scene
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    views = scene.getTestCameras()
    log(f"selfcheck: test 视角 {len(views)} 台，取第 0 台")
    kind, state = resolve_head(args.model_path, args.iteration, device, args.head)
    log(f"  颜色头 = {kind}  {head_info(kind, state)}")
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    synth = make_synth(kind, state, g, pipeline, bg)

    v = views[0]
    A = synth(v)
    R_real = np.asarray(v.R, dtype=np.float64)
    C_real = v.camera_center.detach().cpu().numpy().astype(np.float64)
    fwd = R_real @ E3
    up = -(R_real @ np.array([0.0, 1.0, 0.0]))
    log(f"  真实: C={C_real.round(6).tolist()}  R[:3,2]={R_real[:,2].round(6).tolist()}")
    cam = make_camera(C_real, fwd, up, v.image_width, v.image_height,
                      v.FoVx, v.FoVy, data_device=getattr(args, "data_device", "cpu"))
    R_syn = np.asarray(cam.R, dtype=np.float64)
    log(f"  max|R_syn − R_real| = {np.abs(R_syn - R_real).max():.3e}")
    log(f"  max|FoV 差| = {abs(cam.FoVx - v.FoVx):.3e}, {abs(cam.FoVy - v.FoVy):.3e}")
    B = synth(cam)
    d = (A - B).abs().max().item()
    log(f"  两次渲染 max|Δ| = {d:.3e}")
    ok = d < 1e-6 and np.abs(R_syn - R_real).max() < 1e-9
    log(f"selfcheck [{kind}] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="标量头 RRF 的新视角漫游渲染")
    mp = ModelParams(ap, sentinel=True)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=70000)
    ap.add_argument("--traj", choices=("orbit", "dolly", "orbit_pitch"), default="orbit")
    ap.add_argument("--frames", type=int, default=120)
    # ⚠ `get_combined_args`（splat/arguments/__init__.py:92-112）只把**非 None** 的
    # 命令行值并进 cfg_args 命名空间；默认值为 None 的参数若不显式传，属性会**整个缺失**
    # → AttributeError。故这里一律用非 None 哨兵（-1 / "" ），不用 None。
    ap.add_argument("--radius", type=float, default=-1.0,
                    help="orbit 半径；<=0 则取真实机位半径中位数")
    ap.add_argument("--pitch-deg", type=float, default=20.0, help="orbit_pitch 的俯仰角（真外推）")
    ap.add_argument("--dolly-pair", default="auto",
                    help="auto | 'i,j'（**去重机位**序号，不是 cameras.json 行号）")
    ap.add_argument("--head", choices=("auto", "scalar", "mlp", "sh"), default="auto",
                    help="颜色头；auto = 按目录里存在的 sidecar 判定（都没有 ⇒ 引擎原版 SH）")
    ap.add_argument("--out", default="", help="输出目录；空则 <model_path>/walk")
    ap.add_argument("--arm-out", default="",
                    help="在 --out 下再套一层臂名（三臂并排用）；空则输出仍是 <out>/<traj>/")
    ap.add_argument("--selfcheck", action="store_true",
                    help="只做相机构造等价性自检（需 -s，且须带 --eval，否则 cfg_args 的 eval=False 会让 test 集为空）")
    # 注意：**不要**再加 --data_device / --quiet —— ModelParams 已经定义了它们，
    # 重复添加会触发 argparse "conflicting option string"。直接用 args 上已有的即可。
    ap.add_argument("--quiet", action="store_true")
    args = get_combined_args(ap)
    safe_state(args.quiet)

    device = "cuda"
    dataset = mp.extract(args)
    pipeline = pp.extract(args)
    if args.selfcheck:
        sys.exit(selfcheck(args, dataset, pipeline, device))

    if not torch.cuda.is_available():
        raise RuntimeError("漫游渲染需要 GPU 光栅化器")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)
    g, synth, h_info = load_model(args.model_path, args.iteration, device,
                                  dataset.sh_degree, pipeline, bg, args.head)

    P_all, P, up, az, zen = dataset_pose_stats(args.model_path)
    log(f"真实位姿: {len(P_all)} 台 / {len(P)} 个去重位置；"
        f"方位角 {np.unique(az.round(3)).tolist()}；天顶角 {np.unique(zen.round(3)).tolist()}；"
        f"up {np.unique(up.round(6), axis=0).tolist()}")
    cx, cy = np.median(P[:, 0]), np.median(P[:, 1])
    r_all = np.linalg.norm(P[:, :2] - np.array([cx, cy]), axis=1)
    log(f"真实机位半径: min {r_all.min():.3f} 中位 {np.median(r_all):.3f} max {r_all.max():.3f}")

    c0 = json.load(open(Path(args.model_path) / "cameras.json", encoding="utf-8"))[0]
    W, H = int(c0["width"]), int(c0["height"])
    fovx = 2 * np.arctan(W / (2 * float(c0["fx"])))
    fovy = 2 * np.arctan(H / (2 * float(c0["fy"])))
    log(f"分辨率 {W}x{H}  fx={c0['fx']:.4f} fy={c0['fy']:.4f}  "
        f"FoVx={np.degrees(fovx):.4f}° FoVy={np.degrees(fovy):.4f}°")

    r = args.radius if args.radius > 0 else float(np.median(r_all))
    if args.traj == "orbit":
        cams = traj_orbit(P, W, H, fovx, fovy, args.frames, r, 0.0)
    elif args.traj == "orbit_pitch":
        cams = traj_orbit(P, W, H, fovx, fovy, args.frames, r, args.pitch_deg)
    else:
        cams = traj_dolly(P, W, H, fovx, fovy, args.frames, args.dolly_pair)

    log(f"相机构造自检: |camera_center − 目标|/|目标| ≤ "
        f"{max(c._walk['rel_center_f32'] for c in cams):.1e}（float32 往返，与评测路径同一精度）")

    out = Path(args.out) if args.out else Path(args.model_path) / "walk"
    if args.arm_out:                      # 三臂并排：每臂一层目录，否则同名 PNG 互相覆盖
        out = out / args.arm_out
    outdir = out / args.traj
    outdir.mkdir(parents=True, exist_ok=True)

    import torchvision
    C_all = np.array([c._walk["C"] for c in cams])
    fwd_all = np.array([c._walk["fwd"] for c in cams])
    az_t = np.degrees(np.arctan2(fwd_all[:, 1], fwd_all[:, 0]))
    zen_t = np.degrees(np.arccos(np.clip(fwd_all[:, 2], -1, 1)))
    dn = np.linalg.norm(C_all[:, None, :] - P[None, :, :], axis=2).min(1)

    t0 = time.time()
    with torch.no_grad():
        for k, cam in enumerate(cams):
            img = synth(cam)
            torchvision.utils.save_image(img, str(outdir / f"{k:05d}.png"))
            if (k + 1) % 20 == 0 or k == 0:
                log(f"  {k+1}/{len(cams)}  最近真实机位 {dn[k]:.3f} m  "
                    f"az {az_t[k]:8.2f}°  zen {zen_t[k]:6.2f}°  "
                    f"{(time.time()-t0)/(k+1):.3f} s/帧")
    log(f"渲染完成 {len(cams)} 帧，用时 {time.time()-t0:.1f} s → {outdir}")

    meta = {
        "method": Path(args.model_path).name,
        "arm_out": args.arm_out or None,
        "iteration": args.iteration,
        "traj": args.traj,
        "frames": len(cams),
        "resolution": [W, H],
        "fovx_deg": float(np.degrees(fovx)), "fovy_deg": float(np.degrees(fovy)),
        "n_gaussians": int(g.get_xyz.shape[0]),
        "head": h_info,
        "orbit": {"axis_xy": [float(cx), float(cy)], "z": float(np.median(P[:, 2])),
                  "radius": float(r), "pitch_deg": args.pitch_deg if args.traj == "orbit_pitch" else 0.0},
        "dataset_pose": {
            "n_cameras": int(len(P_all)), "n_unique_positions": int(len(P)),
            "azimuths_deg": [float(v) for v in np.unique(az.round(3))],
            "zeniths_deg": [float(v) for v in np.unique(zen.round(3))],
            "radius_min": float(r_all.min()), "radius_median": float(np.median(r_all)),
            "radius_max": float(r_all.max()),
        },
        # 内插/外推判据：逐帧到最近真实机位的距离。0 = 与真实机位重合（纯内插）。
        "frames_pose": [
            {"k": k, "C": [float(v) for v in C_all[k]],
             "fwd": [float(v) for v in fwd_all[k]],
             "az_deg": float(az_t[k]), "zen_deg": float(zen_t[k]),
             "dist_nearest_real_cam": float(dn[k])}
            for k in range(len(cams))
        ],
        # 内插/外推的**可判据**证据：位置看距离，朝向看天顶角与方位角是否落在实测集合里。
        # 光轴俯仰（天顶角偏离 90°）与 roll 在数据里 0/3200 采样 ⇒ 只有它们是真外推。
        "honesty": {
            "dist_nearest_real_cam_min": float(dn.min()),
            "dist_nearest_real_cam_median": float(np.median(dn)),
            "dist_nearest_real_cam_max": float(dn.max()),
            "traj_zenith_deg": sorted({float(v) for v in zen_t.round(4)}),
            "dataset_zenith_deg": sorted({float(v) for v in zen.round(4)}),
            "zenith_in_dataset": bool(all(
                np.min(np.abs(zen - z)) < 1e-4 for z in np.unique(zen_t.round(4)))),
            # 方位角**最坏**采样间隔（不是最好）：数据集每位置只有 4 个方位角（间隔 90°），
            # 故连续方位角最远离实测方位角 45° —— 这正是「方位角内插」的诚实刻画。
            "azimuth_gap_deg_max": float(np.max([
                np.min(np.abs((az - a + 180.0) % 360.0 - 180.0))
                for a in np.unique(az_t.round(4))])),
            "dataset_azimuth_spacing_deg": (
                float(np.min(np.diff(np.unique(az)))) if len(np.unique(az)) > 1 else None),
        },
    }
    (outdir / "轨迹.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    log(f"写 {outdir/'轨迹.json'}")
    h = meta["honesty"]
    log(f"诚实性: 到最近真实机位 min {h['dist_nearest_real_cam_min']:.3f} / "
        f"中位 {h['dist_nearest_real_cam_median']:.3f} / max {h['dist_nearest_real_cam_max']:.3f} m")
    log(f"        天顶角 {h['traj_zenith_deg']} vs 数据 {h['dataset_zenith_deg']} → "
        f"落在位姿族内 {h['zenith_in_dataset']}")
    log(f"        方位角最坏采样间隔 {h['azimuth_gap_deg_max']:.2f}°"
        f"（数据集方位角间隔 {h['dataset_azimuth_spacing_deg']}°）")


if __name__ == "__main__":
    main()
