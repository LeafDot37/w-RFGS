# -*- coding: utf-8 -*-
"""NIST lobby 多发射源空间射频场生成器（独立 sionna 环境运行）。

与 `gen_sionna_field.py` 同一份产物口径（`spatial_rf_v1`），区别只在场景与点云：
  * 场景 = RF-3DGS 官方教程的 NIST lobby（`prep_nist_scene.py` 整理后的 ASCII 版）；
  * 点云 = 作者发布的光学 3DGS 点云（ROI + 不透明度筛选），不再自己造格点。

产物：
    datasets/NIST_lobby_spatial/
        cloud.ply            光学高斯（世界坐标，米），法向由协方差最小轴给出
        query_xyz.npy        每个高斯的 GT 求值点（沿法向朝相机侧偏移 delta 米）
        field/<tx>.npy       每个求值点的接收功率 dBm（float16，TX 0 dBm）
        meta.json            TX/RX 划分、6 面相机、dB 映射、偏移量
        scene.json           场景与材质、频率、仿真口径

两条必须写进论文口径的约定
--------------------------------------------------------------------------------
1) **表面点自遮挡**：Sionna 的接收机若正好落在网格面上，LoS 会被自身命中。
   实测（TX=[6.905,0,0.287]，5000 点，仅 LoS）：高斯原位置 15.3% 有径，
   沿协方差法向朝最近相机偏移 0.05 m 后 48.2%、0.10 m 后 56.2%；
   同一 x/y 在 z=0.2 的自由空间对照是 93.0% ⇒ 低有径率主要是数值自遮挡，
   不是几何遮挡。GT 因此在 `query_xyz = xyz + delta·n̂` 上取值。
2) **TX 功率是加性轴**：`paths.a` 已含发射功率；这里固定 `power_dbm=0`，
   所以 `10log10(|a|²)` 就是 dBm。换成功率 P dBm 时整场加 P dB，
   不需要重新仿真。

跑法：
    python code/sim/gen_nist_field.py --mode smoke
    python code/sim/gen_nist_field.py --mode gen --n_tx 48 --delta 0.10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

# ------------------------------------------------------------------ 常量
SCENE_XML = Path("_external/nist_lobby_ascii/scene.xml")
OPTICAL_PLY = Path(
    "datasets/RF-3DGS_dataset/blender_visual_trained/point_cloud/"
    "iteration_30000/point_cloud.ply")
CAMERAS_JSON = Path("datasets/RF-3DGS_dataset/blender_visual_trained/cameras.json")

# 光学点云真正稠密的区域：相机轨迹只有 x[-3.5,7]、y[-9.3,0.5]，但可见面
# 一直延伸到墙；按 Sionna 网格的楼面范围与点数直方图取下面这个盒子。
ROI = {"x": (-6.00, 10.80), "y": (-12.50, 4.50), "z": (-1.85, 1.45)}
FREQ_HZ = 60e9
C_LIGHT = 299792458.0

# 材质表与教程一致；电导率用 GHz（教程写成 scene_freq/1e-9 会得到 ~1e16 S/m）
MATERIALS = {
    "custom_plastic": (2.3, 0.0, 0.05, 6),
    "custom_leather": (1.8, 0.0, 0.1, 3),
    "custom_cloth": (1.8, 0.0, 0.2, 3),
    "custom_plasterboard": (2.73, 0.0085, 0.1, 5),
    "custom_wood": (1.99, 0.0047, 0.2, 3),
    "custom_concrete": (5.24, 0.0462, 0.1, 5),
    "custom_metal": (1.0, 1e7, 0.025, 10),
    "custom_glass": (6.31, 0.0036, 0.025, 10),
    "custom_ceiling_board": (1.48, 0.0011, 0.1, 5),
    "custom_chipboard": (2.58, 0.0217, 0.1, 5),
    "custom_marble": (7.074, 0.0055, 0.05, 10),
}
SCAT_MULT = 4.0
# 电导率的频率指数（教程 cell 9）；不在表里的材质电导率取常数
SIG_EXP = {
    "custom_plasterboard": 0.9395, "custom_wood": 1.0718, "custom_concrete": 1.0718,
    "custom_glass": 1.3394, "custom_ceiling_board": 1.0750,
    "custom_chipboard": 0.7800, "custom_marble": 0.9262,
}


# ------------------------------------------------------------------ 场景
def load_scene(freq_hz: float | None = None, rewrite_materials: bool = True,
               material_freq_hz: float | None = None, sig_literal: bool = False):
    """加载 NIST lobby。

    材质口径以教程 cell 9 为准：11 个 custom_* 材质按 60 GHz 构造（电导率用
    `scene_freq=60e9` 代入），散射系数乘 `global_scattering_coeff=4`，方向图为
    DirectivePattern(alpha_r)。之后 cell 107 才把 `scene.frequency` 改成 2.4 GHz
    生成 CBF 谱图 —— **custom 材质的电导率不会跟着变**。

    参数
    ----
    rewrite_materials : 是否套用教程材质；False 保持 XML 里的占位材质（无散射，
        只用于对照，不能当原口径）。
    material_freq_hz : 材质构造时用的频率，默认 = freq_hz。
    sig_literal : True 时照抄教程字面 `base*(f/1e-9)**exp`（f 为 Hz，会得到
        ~1e17 S/m，等效理想导体）。False 用修正后的 GHz 版 `base*(f/1e9)**exp`。
    """
    freq_hz = FREQ_HZ if freq_hz is None else float(freq_hz)
    mf_hz = freq_hz if material_freq_hz is None else float(material_freq_hz)
    import mitsuba as mi
    import sionna.rt as rt
    from sionna.rt import load_scene_from_string

    xml = SCENE_XML.resolve()
    fres = mi.FileResolver(mi.file_resolver())
    fres.append(str(xml.parent))
    old = mi.file_resolver()
    mi.set_file_resolver(fres)
    try:
        sc = load_scene_from_string(xml.read_text(encoding="utf-8"))
    finally:
        mi.set_file_resolver(old)

    sc.frequency = freq_hz
    for k in ("tx_array", "rx_array"):
        setattr(sc, k, rt.PlanarArray(
            num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"))
    sc.synthetic_array = True

    if not rewrite_materials:
        print(f"[scene] 原样材质（不改写），频率 {freq_hz / 1e9:.3g} GHz")
        return sc

    # 教程 cell 9 的电导率：base * (f/1e-9)**exp，f = scene_freq = 60 GHz
    fscale = mf_hz / 1e-9 if sig_literal else mf_hz / 1e9
    n = 0
    for m in sc.radio_materials.values():
        base = re.sub(r"\.\d{3}$", "", m.name).replace("itu_", "custom_")
        if base not in MATERIALS:
            continue
        eps, sig, scat, alpha = MATERIALS[base]
        exp = SIG_EXP.get(base)
        if exp is not None:
            sig = sig * fscale ** exp
        m.relative_permittivity = eps
        m.conductivity = sig
        m.scattering_coefficient = min(scat * SCAT_MULT, 1.0)
        m.scattering_pattern = rt.DirectivePattern(alpha_r=alpha)
        n += 1
    print(f"[scene] 材质改写 {n} 个：材质频率 {mf_hz/1e9:.3g} GHz"
          f"{'（字面 1e-9 口径）' if sig_literal else ''}，仿真频率 {freq_hz/1e9:.3g} GHz")
    return sc


# ------------------------------------------------------------------ 点云
def _quat_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)


def load_optical_points(opacity_thr: float):
    v = PlyData.read(str(OPTICAL_PLY))["vertex"]
    names = set(v.data.dtype.names)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    mask = ((xyz[:, 0] > ROI["x"][0]) & (xyz[:, 0] < ROI["x"][1])
            & (xyz[:, 1] > ROI["y"][0]) & (xyz[:, 1] < ROI["y"][1])
            & (xyz[:, 2] > ROI["z"][0]) & (xyz[:, 2] < ROI["z"][1])
            & (op > opacity_thr))
    idx = np.where(mask)[0]
    print(f"[pts] 全量 {len(xyz)}，ROI+op>{opacity_thr:g} 后 {len(idx)}")

    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    R = _quat_to_R(rot[idx])
    amin = np.argmin(scale[idx], 1)
    nrm = R[np.arange(len(idx)), :, amin]          # 世界系最小轴 = 表面法向
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9

    cams = np.asarray([c["position"] for c in
                       json.load(open(CAMERAS_JSON, encoding="utf-8"))], np.float32)
    _, ci = cKDTree(np.unique(cams, axis=0)).query(xyz[idx])
    tocam = np.unique(cams, axis=0)[ci] - xyz[idx]
    sgn = np.sign((nrm * tocam).sum(1))
    sgn[sgn == 0] = 1.0
    nrm *= sgn[:, None]

    fields = {}
    for k in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
              "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
        if k in names:
            fields[k] = np.asarray(v[k], dtype=np.float32)[idx]
    for k in range(45):
        fields[f"f_rest_{k}"] = np.asarray(v[f"f_rest_{k}"], dtype=np.float32)[idx]
    return xyz[idx], nrm, fields


# ------------------------------------------------------------------ 采样
def fps_select(pts, n, seed=0):
    """最远点采样；返回索引（n 大于点数时返回全部）。"""
    rng = np.random.default_rng(seed)
    n = min(n, len(pts))
    sel = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(1, n):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return sel


def sample_free_space(anchors, lo, hi, n, clear, seed=0, pool=400_000):
    """在盒子内采样离 anchors（表面点）至少 clear 米的点，再 FPS 铺开。"""
    rng = np.random.default_rng(seed)
    p = rng.uniform(lo, hi, size=(pool, 3)).astype(np.float32)
    d, _ = cKDTree(anchors).query(p)
    p = p[d > clear]
    if len(p) < n:
        raise SystemExit(f"自由空间点不足：{len(p)} < {n}（放宽 clear 或盒子）")
    return p[fps_select(p, n, seed)]


# ------------------------------------------------------------------ 相机
def face_cameras(fov_deg=95.0, img=96):
    """6 个正交面；R 是 c2w，列 = [right, down, fwd]（与 3DGS 一致）。"""
    faces = {
        "px": ((1, 0, 0), (0, 0, -1)), "nx": ((-1, 0, 0), (0, 0, -1)),
        "py": ((0, 1, 0), (0, 0, -1)), "ny": ((0, -1, 0), (0, 0, -1)),
        "pz": ((0, 0, 1), (0, 1, 0)), "nz": ((0, 0, -1), (0, -1, 0)),
    }
    f = math.radians(fov_deg)
    fx = (img / 2.0) / math.tan(f / 2.0)
    out = []
    for name, (fw, dn) in faces.items():
        fw = np.asarray(fw, float)
        dn = np.asarray(dn, float)
        rt = np.cross(dn, fw)
        R = np.stack([rt, dn, fw], axis=1)
        out.append({"name": name, "R": R.tolist(), "fovx": fov_deg,
                    "fovy": fov_deg, "fx": fx, "fy": fx})
    return out


# ------------------------------------------------------------------ 场
def path_gain_multi(sc, rt, query, tx_pos, chunk=8192, max_depth=3):
    """query [P,3] 上、发射源 tx_pos 处的接收功率 dBm（TX 0 dBm）。"""
    from sionna.rt import Receiver, Transmitter

    sc.add(Transmitter("tx", position=np.asarray(tx_pos, np.float32), power_dbm=0.0))
    dbm = np.empty(len(query), np.float32)
    try:
        for s in range(0, len(query), chunk):
            q = query[s:s + chunk]
            rxs = [Receiver(f"r{i}", position=q[i]) for i in range(len(q))]
            sc.add(rxs)
            paths = rt.PathSolver()(
                sc, max_depth=max_depth, los=True, specular_reflection=True,
                diffuse_reflection=False, refraction=False, diffraction=False,
                synthetic_array=True, seed=0)
            ar, ai = paths.a
            a = (np.asarray(ar) + 1j * np.asarray(ai)).reshape(len(q), -1)
            g = (np.abs(a) ** 2).sum(1)
            dbm[s:s + chunk] = (10.0 * np.log10(np.maximum(g, 1e-30))).astype(np.float32)
            sc.remove([r.name for r in rxs])
    finally:
        sc.remove("tx")
    return dbm


def los_fraction(sc, rt, rx_pos, tx_pos):
    from sionna.rt import Receiver, Transmitter

    sc.add(Transmitter("tx", position=np.asarray(tx_pos, np.float32), power_dbm=0.0))
    rxs = [Receiver(f"r{i}", position=rx_pos[i]) for i in range(len(rx_pos))]
    sc.add(rxs)
    paths = rt.PathSolver()(sc, max_depth=0, los=True, specular_reflection=False,
                            diffuse_reflection=False, refraction=False,
                            diffraction=False, synthetic_array=True, seed=0)
    ar, ai = paths.a
    g = (np.abs(np.asarray(ar) + 1j * np.asarray(ai)).reshape(len(rx_pos), -1) ** 2).sum(1)
    sc.remove([r.name for r in rxs])
    sc.remove("tx")
    return float(np.mean(g > 0))


# ------------------------------------------------------------------ 写盘
def write_cloud(path, xyz, nrm, fields):
    n = len(xyz)
    spec = ([("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
            + [(f"f_rest_{k}", "f4") for k in range(45)]
            + [("opacity", "f4")]
            + [(f"scale_{k}", "f4") for k in range(3)]
            + [(f"rot_{k}", "f4") for k in range(4)])
    arr = np.zeros(n, dtype=spec)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    for k, v in fields.items():
        arr[k] = v
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "gen"), default="smoke")
    ap.add_argument("--out", default="datasets/NIST_lobby_spatial")
    ap.add_argument("--n_tx", type=int, default=48)
    ap.add_argument("--n_rx", type=int, default=24)
    ap.add_argument("--delta", type=float, default=0.10, help="沿法向朝相机侧的 GT 偏移（米）")
    ap.add_argument("--opacity", type=float, default=0.5)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--fov", type=float, default=95.0)
    # 60 GHz、0 dBm 发射时，1 m 自由空间约 −68 dBm、10 m 约 −88 dBm；
    # 固定区间保留绝对功率口径（换发射功率 = 整场平移），不被弱径长尾压扁。
    ap.add_argument("--lo_dbm", type=float, default=-125.0,
                    help="u 映射下界；给 0 以外的 None 不合法，保持固定口径")
    ap.add_argument("--hi_dbm", type=float, default=-55.0)
    ap.add_argument("--limit_points", type=int, default=0, help="调试：只留前 N 个点")
    ap.add_argument("--n_tx_cand", type=int, default=0,
                    help="调试：只验证前 N 个候选 TX")
    args = ap.parse_args()

    import sionna.rt as rt

    t_all = time.time()
    sc = load_scene()
    xyz, nrm, fields = load_optical_points(args.opacity)
    if args.limit_points > 0:
        xyz, nrm = xyz[:args.limit_points], nrm[:args.limit_points]
        fields = {k: v[:args.limit_points] for k, v in fields.items()}
    query = (xyz + args.delta * nrm).astype(np.float32)

    # ---- 自由空间：TX 高 1.3–2.6 m（本帧地面 z≈−1.75，相机 z=0）----
    tx_lo = np.array([ROI["x"][0] + 0.3, ROI["y"][0] + 0.3, -0.45], np.float32)
    tx_hi = np.array([ROI["x"][1] - 0.3, ROI["y"][1] - 0.3, 0.85], np.float32)
    n_cand = max(args.n_tx * 4, 32)
    if args.n_tx_cand > 0:
        n_cand = args.n_tx_cand
    tx_cand = sample_free_space(xyz, tx_lo, tx_hi, n_cand, clear=0.70,
                                seed=args.seed + 1)

    # ---- 接收位姿：同样在自由空间，留更大的余量 ----
    rx_lo = np.array([ROI["x"][0] + 0.5, ROI["y"][0] + 0.5, -0.70], np.float32)
    rx_hi = np.array([ROI["x"][1] - 0.5, ROI["y"][1] - 0.5, 0.50], np.float32)
    rx = sample_free_space(xyz, rx_lo, rx_hi, args.n_rx, clear=0.80,
                           seed=args.seed + 2)
    print(f"[sample] TX 候选 {len(tx_cand)}，RX {len(rx)}")

    # ---- TX 可用性：LoS 到接收位姿的比例 ----
    keep = []
    for i, t in enumerate(tx_cand):
        f = los_fraction(sc, rt, rx, t)
        if f >= 0.5:
            keep.append(i)
        if len(keep) >= args.n_tx:
            break
    if len(keep) < args.n_tx:
        raise SystemExit(f"可用 TX 只有 {len(keep)} < {args.n_tx}")
    txs = tx_cand[keep[:args.n_tx]]
    order = fps_select(txs, len(txs), seed=args.seed + 3)
    txs = txs[order]
    test = [i for i in range(len(txs)) if i % 4 == 3]
    train = [i for i in range(len(txs)) if i % 4 != 3]
    print(f"[tx] {len(txs)} 个（train {len(train)} / test {len(test)}）")

    out = Path(args.out)
    (out / "field").mkdir(parents=True, exist_ok=True)

    # ---- 逐 TX 场 ----
    t0 = time.time()
    all_db = []
    for i, t in enumerate(txs):
        dbm = path_gain_multi(sc, rt, query, t, args.chunk, args.max_depth)
        np.save(out / "field" / f"{i:05d}.npy", dbm.astype(np.float16))
        all_db.append(dbm)
        nz = dbm > -299.0
        print(f"  [{i+1}/{len(txs)}] tx={np.round(t, 2)}  "
              f"有径 {nz.mean():.3f}  dBm p5/50/95 "
              f"{np.percentile(dbm[nz], [5, 50, 95]).round(1) if nz.any() else '-'}  "
              f"累计 {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- dB 映射：只用有径点定上下界，无径点自然落到 0 ----
    nz_all = np.concatenate([d[d > -299.0] for d in all_db])
    lo = args.lo_dbm
    hi = args.hi_dbm
    if lo is None:
        # 用 p2 而不是 p0.5：长尾弱径会把下界拖到 −160 dB，压扁有径点的主区间
        lo = float(np.floor(np.percentile(nz_all, 2.0)))
    if hi is None:
        hi = float(np.ceil(np.percentile(nz_all, 99.5)))
    if hi - lo < 20:
        hi = lo + 20
    u_all = np.clip((nz_all - lo) / (hi - lo), 0, 1)
    print(f"[db] lo={lo:.1f} hi={hi:.1f}  有径点 u p5/50/95 "
          f"{np.percentile(u_all, [5, 50, 95]).round(3)}  "
          f"u=0 占比 {np.mean(np.concatenate(all_db) <= lo):.3f}")

    # ---- 写点云（世界坐标；norm_unit=1 便于与光学点云同一口径）----
    write_cloud(out / "cloud.ply", xyz, nrm, fields)
    np.save(out / "query_xyz.npy", query)

    cams = face_cameras(args.fov, args.img)
    meta = {
        "format": "spatial_rf_v1",
        "scene": "NIST_lobby",
        "source_cloud": str(OPTICAL_PLY).replace("\\", "/"),
        "source_scene": str(SCENE_XML).replace("\\", "/"),
        "freq_hz": FREQ_HZ,
        "tx_power_dbm": 0.0,
        "norm_unit": 1.0,
        "center": [0.0, 0.0, 0.0],
        "n_points": int(len(xyz)),
        "roi": {k: list(v) for k, v in ROI.items()},
        "opacity_thr": args.opacity,
        "query_offset_delta": args.delta,
        "query_xyz_file": "query_xyz.npy",
        "tx_world": txs.tolist(),
        "tx_local": txs.tolist(),
        "split": {"train": train, "test": test},
        "rx_world": rx.tolist(),
        "rx_local": rx.tolist(),
        "rx_test": [i for i in range(len(rx)) if i % 3 == 2],
        "rx_train": [i for i in range(len(rx)) if i % 3 != 2],
        "views": {"n": 6, "img": args.img, "fov": args.fov, "cameras": cams},
        "db_map": {"lo_dbm": lo, "hi_dbm": hi},
        "gt_pattern": "field/{j:05d}.npy",
        "cloud_ply": "cloud.ply",
    }
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({
        "scene": "NIST_lobby", "freq_hz": FREQ_HZ, "tx_power_dbm": 0.0,
        "simulator": "sionna-rt 2.2.0", "max_depth": args.max_depth,
        "reflection": "specular only", "diffuse": False, "diffraction": False,
        "materials": {k: list(v) for k, v in MATERIALS.items()},
        "scattering_multiplier": SCAT_MULT,
        "delta_m": args.delta, "opacity_thr": args.opacity,
        "db_map": meta["db_map"],
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }, open(out / "scene.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    print(f"[done] {out}  点 {len(xyz)}  TX {len(txs)}  "
          f"总耗时 {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
