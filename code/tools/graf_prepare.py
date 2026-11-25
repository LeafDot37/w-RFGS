# -*- coding: utf-8 -*-
"""GRaF-sim → 本仓「点云 + 相机 + jet 伪彩图」格式的转换器。

做四件事（全部确定性）：
  1. 选 TX 子集（最远点采样，保证空间铺满），按 FPS 顺序每 4 个取 1 个做测试；
  2. 下载这些 TX 的 360×90 角度谱（带重试/校验，hf-mirror 会间歇限流）；
  3. 把角度谱重采样成 4 个针孔视角（仰角 45°、方位四象限、水平 110°/垂直 92°），
     并用本仓同一条 jet 查表落成 RGB —— 与训练/渲染端逐位同源；
  4. 用房间尺寸生成上表面点云（天花板 + 网关以上的墙带），坐标变换到
     **网关局部归一化系**：p' = R_gwᵀ (p − gw) / norm，norm = 0.5·‖room/2‖
     （取 0.5 是为了躲开光栅化器 `p_view.z <= 0.2` 的硬剔除，见 --norm_mul）。

⚠ 这不是光学重建的点云，是房间表面点云；跨场景实验测的是头的场景迁移能力，
  不是完整光学管线（见输出目录的 说明.md）。

用法：
    python code/tools/graf_prepare.py --root datasets/GRaF-sim \
        --out datasets/GRaF-sim_processed --scenes D2 D8 D14 --n_tx 800
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import torch                                                        # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut              # noqa: E402

MIRROR = "https://hf-mirror.com/datasets/kyang73/GRaF-sim/resolve/main"


# ---------------------------------------------------------------- 几何
def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def dir_local(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1)


def build_view_R(yaw_deg, pitch_deg):
    """网关局部系里的 R_c2w（列 = right, down, forward），与 3DGS/COLMAP 约定一致。"""
    fwd = dir_local(np.array(yaw_deg, dtype=np.float64),
                    np.array(pitch_deg, dtype=np.float64))
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def pixel_dirs_local(R_c2w, W, H, fx, fy):
    """每像素在网关局部系的单位方向 [H,W,3]（y 向下）。"""
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = (u + 0.5 - W / 2.0) / fx
    y = (v + 0.5 - H / 2.0) / fy
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ R_c2w.T


def sample_equirect(img, dirs_local):
    """角度谱 [90,360] 双线性采样。dirs_local 已是网关局部系。"""
    az = np.rad2deg(np.arctan2(dirs_local[..., 1], dirs_local[..., 0])) % 360.0
    el = np.rad2deg(np.arcsin(np.clip(dirs_local[..., 2], -1.0, 1.0)))
    col = az - 1.0
    col = np.where(col < 0, col + 360.0, col)
    row = np.clip(el - 1.0, 0.0, 89.0)
    c0 = np.floor(col).astype(np.int64) % 360
    r0 = np.floor(row).astype(np.int64)
    fc, fr = col - np.floor(col), row - np.floor(row)
    c1 = (c0 + 1) % 360
    r1 = np.minimum(r0 + 1, 89)
    return (img[r0, c0] * (1 - fc) * (1 - fr) + img[r0, c1] * fc * (1 - fr)
            + img[r1, c0] * (1 - fc) * fr + img[r1, c1] * fc * fr)


# ---------------------------------------------------------------- 下载
_LOCAL = __import__("threading").local()


def _session():
    """每线程复用一个 requests.Session：hf-mirror 单次 TLS 握手约 3–5 s，
    反复建连会把 800 张谱图拖到 40 分钟。连接保活后每张 <1 s。"""
    s = getattr(_LOCAL, "s", None)
    if s is None:
        import requests
        from requests.adapters import HTTPAdapter
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=4))
        s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        _LOCAL.s = s
    return s


def fetch_spectrum(root, scene, file_idx, dst, tries=6):
    if os.path.isfile(dst):
        try:
            im = Image.open(dst)
            if im.size == (360, 90):
                return True
        except Exception:
            pass
    url = f"{MIRROR}/{scene}/spectrum/{file_idx:05d}.png"
    last = ""
    for t in range(tries):
        try:
            r = _session().get(url, timeout=(10, 60))
            data = r.content
            if r.status_code != 200 or data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 500:
                last = f"http {r.status_code} {len(data)}B"
                raise ValueError(last)
            with open(dst, "wb") as f:
                f.write(data)
            im = Image.open(dst)
            if im.size != (360, 90):
                raise ValueError(f"尺寸 {im.size}")
            return True
        except Exception as e:                                    # noqa: BLE001
            last = repr(e)
            time.sleep(0.5 * (2 ** t) + 0.2 * np.random.rand())
    print(f"  [warn] 下载失败 {scene}/{file_idx:05d}: {last}", flush=True)
    return False


# ---------------------------------------------------------------- 选点
def fps_select(pts, n, seed=0):
    """最远点采样（确定性：首点取离网关最远者，带固定随机扰动避免并列）。"""
    pts = np.asarray(pts, dtype=np.float64)
    rng = np.random.RandomState(seed)
    d2 = ((pts - pts.mean(0)) ** 2).sum(1) * (1 + 1e-6 * rng.rand(len(pts)))
    sel = [int(d2.argmax())]
    dist = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(1, n):
        i = int(dist.argmax())
        sel.append(i)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[i], axis=1))
    return sel


# ---------------------------------------------------------------- 点云
def build_surface_cloud(room, gw, R_gw, unit, spacing, below=0.2):
    """天花板 + 四墙（网关以下 below 米到天花板）上的规则点阵，转到网关局部归一化系。"""
    W, D, H = [float(v) for v in room]
    gx, gy, gz = [float(v) for v in gw]
    xs = np.arange(-W / 2, W / 2 + 1e-9, spacing)
    ys = np.arange(-D / 2, D / 2 + 1e-9, spacing)
    zs_wall = np.arange(gz - below, H + 1e-9, spacing)
    pts = []
    # 天花板
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, H)], axis=1))
    # 四面墙
    Z, L = np.meshgrid(zs_wall, xs, indexing="ij")
    pts.append(np.stack([np.full(Z.size, W / 2), L.ravel(), Z.ravel()], axis=1))
    pts.append(np.stack([np.full(Z.size, -W / 2), L.ravel(), Z.ravel()], axis=1))
    Z, L = np.meshgrid(zs_wall, ys, indexing="ij")
    pts.append(np.stack([L.ravel(), np.full(Z.size, D / 2), Z.ravel()], axis=1))
    pts.append(np.stack([L.ravel(), np.full(Z.size, -D / 2), Z.ravel()], axis=1))
    p = np.concatenate(pts, axis=0)
    # 世界 → 网关局部归一化
    local = (p - np.asarray(gw, dtype=np.float64)) @ R_gw / unit
    return local.astype(np.float32)


def write_cloud_ply(path, xyz, log_scale):
    n = xyz.shape[0]
    fields = ([("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
               ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
              + [(f"f_rest_{i}", "f4") for i in range(45)]
              + [(f"opacity", "f4")]
              + [(f"scale_{i}", "f4") for i in range(3)]
              + [(f"rot_{i}", "f4") for i in range(4)])
    arr = np.zeros(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["opacity"] = 4.0                       # 头会覆盖；这里只作为几何占位
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = float(log_scale)
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


# ---------------------------------------------------------------- 主流程
def main():
    ap = ArgumentParser()
    ap.add_argument("--root", default="datasets/GRaF-sim")
    ap.add_argument("--out", default="datasets/GRaF-sim_processed")
    ap.add_argument("--scenes", nargs="+", default=["D2", "D8", "D14"])
    ap.add_argument("--n_tx", type=int, default=800)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--fovx", type=float, default=110.0)
    ap.add_argument("--fovy", type=float, default=92.0)
    ap.add_argument("--pitch", type=float, default=45.0)
    # 0.025 m：oracle 表示上限实测 24.4 dB（0.05 m 只有 20.6 dB），
    # 点数为 4 倍但 96×96 渲染仍只需毫秒级。
    ap.add_argument("--spacing", type=float, default=0.025, help="世界坐标点间距（米）")
    # 归一化尺度 = half_diag × norm_mul。**不能取 1.0**：光栅化器 in_frustum 硬编码
    # `p_view.z <= 0.2` 剔除，而网关到天花板的归一化距离只有 1.0/5.22=0.19 ⇒ 仰角
    # >68° 的天花板整片被剔掉（实测四视角顶部 1/4 全空）。0.5 ⇒ 最近距离 0.38。
    ap.add_argument("--norm_mul", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lut = make_jet_lut(256)

    manifest = {}
    for scene in args.scenes:
        sdir = os.path.join(args.root, scene)
        odir = os.path.join(args.out, scene)
        os.makedirs(os.path.join(odir, "gt"), exist_ok=True)
        os.makedirs(os.path.join(odir, "scalar"), exist_ok=True)
        os.makedirs(os.path.join(sdir, "spectrum"), exist_ok=True)

        with open(os.path.join(sdir, "gateway_info.yml")) as f:
            gw_cfg = yaml.safe_load(f)
        room = np.array(gw_cfg["room_size"], dtype=np.float64)
        gw = np.array(gw_cfg["gateway1"]["position"], dtype=np.float64)
        R_gw = quat_to_mat(gw_cfg["gateway1"]["orientation"])
        hd = float(np.linalg.norm(room / 2.0))
        norm = hd * float(args.norm_mul)          # 归一化尺度（= 单位长度对应的米数）

        tx = np.loadtxt(os.path.join(sdir, "tx_pos.csv"), delimiter=",", skiprows=1)
        print(f"== {scene}: {tx.shape[0]} TX, room {room.tolist()}, gw {gw.tolist()}, "
              f"hd {hd:.3f}, norm {norm:.3f}")

        sel = fps_select(tx, args.n_tx, args.seed)
        order = np.argsort(sel)                    # 原文件号顺序，方便追溯
        sel = [sel[i] for i in order]
        test_pos = [i for i in range(len(sel)) if i % 4 == 3]
        train_pos = [i for i in range(len(sel)) if i % 4 != 3]
        file_idx = [int(s) + 1 for s in sel]

        # ---------- 下载 ----------
        if not args.skip_download:
            todo = [(fi, os.path.join(sdir, "spectrum", f"{fi:05d}.png")) for fi in file_idx]
            ok = 0
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(fetch_spectrum, args.root, scene, fi, dst)
                        for fi, dst in todo]
                for k, fu in enumerate(as_completed(futs)):
                    ok += int(fu.result())
                    if (k + 1) % 200 == 0:
                        print(f"  下载 {k+1}/{len(todo)}  ok={ok}  "
                              f"{time.time()-t0:.0f}s", flush=True)
            print(f"  下载完成 ok={ok}/{len(todo)}  {time.time()-t0:.0f}s")
            if ok < len(todo):
                raise SystemExit(f"{scene} 有 {len(todo)-ok} 张谱图没下下来，先解决再继续")

        # ---------- 相机 ----------
        yaws = [k * 360.0 / args.n_views + 180.0 / args.n_views
                for k in range(args.n_views)]
        view_R = np.stack([build_view_R(y, args.pitch) for y in yaws])
        fx = (args.img / 2.0) / math.tan(math.radians(args.fovx) / 2.0)
        fy = (args.img / 2.0) / math.tan(math.radians(args.fovy) / 2.0)
        dirs = [pixel_dirs_local(view_R[k], args.img, args.img, fx, fy)
                for k in range(args.n_views)]

        # ---------- 重采样 GT ----------
        n_ok = 0
        for j, fi in enumerate(file_idx):
            sp = os.path.join(sdir, "spectrum", f"{fi:05d}.png")
            img = np.asarray(Image.open(sp).convert("L"), dtype=np.float64)
            for k in range(args.n_views):
                v = sample_equirect(img, dirs[k])
                u8 = np.clip(np.round(v), 0, 255).astype(np.uint8)
                Image.fromarray(u8).save(os.path.join(odir, "scalar",
                                                      f"{j:05d}_{k}.png"))
                rgb = apply_jet(torch.from_numpy(u8.astype(np.float32) / 255.0)[None],
                                lut).permute(1, 2, 0).numpy()
                Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(odir, "gt", f"{j:05d}_{k}.png"))
                n_ok += 1
        print(f"  GT 视角 {n_ok} 张（{len(file_idx)} TX × {args.n_views}）")

        # ---------- 点云 ----------
        cloud = build_surface_cloud(room, gw, R_gw, norm, args.spacing)
        # 尺度取 1.0×间距：投影后 ≈1 px 标准差、3σ 截断 ⇒ 相邻点必重叠，
        # 不留孔洞；再小会在 96×96 上出现稀疏空洞（0.6× 只有 0.6 px）。
        scale_norm = args.spacing * 1.0 / norm
        write_cloud_ply(os.path.join(odir, "cloud.ply"), cloud, math.log(scale_norm))
        print(f"  点云 P={cloud.shape[0]}  scale_norm={scale_norm:.5f}")

        # ---------- meta ----------
        tx_local = (tx - gw) @ R_gw / norm
        meta = {
            "scene": scene,
            "room_size": room.tolist(),
            "gateway_world": gw.tolist(),
            "R_gw": R_gw.tolist(),
            "half_diag": hd,
            "norm_unit": norm,
            "n_source_tx": int(tx.shape[0]),
            "sel_rows": sel,
            "file_idx": file_idx,
            "tx_world": tx[sel].tolist(),
            "tx_local": tx_local[sel].tolist(),
            "split": {"train": train_pos, "test": test_pos},
            "views": {
                "n": args.n_views, "img": args.img,
                "fovx": args.fovx, "fovy": args.fovy, "pitch": args.pitch,
                "yaw": yaws, "fx": fx, "fy": fy, "R": view_R.tolist(),
            },
            "cloud_ply": os.path.join(scene, "cloud.ply"),
            "gt_pattern": "gt/{j:05d}_{k}.png",
            "scalar_pattern": "scalar/{j:05d}_{k}.png",
        }
        json.dump(meta, open(os.path.join(odir, "meta.json"), "w"),
                  ensure_ascii=False, indent=2)
        manifest[scene] = {"meta": os.path.join(scene, "meta.json"),
                           "n_tx": len(sel), "n_train": len(train_pos),
                           "n_test": len(test_pos), "n_points": int(cloud.shape[0])}
        print(f"  -> {odir}/meta.json")

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[done] {args.out}/manifest.json")


if __name__ == "__main__":
    main()
