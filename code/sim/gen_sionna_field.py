# -*- coding: utf-8 -*-
"""Sionna RT 空间射频场生成器（独立 sionna 环境运行）。

产物（目标：RF-3DGS 的结果类型 —— 每个发射源一张「整个空间的接收场」）：
    datasets/SionnaRoom/
        cloud.ply            房间全部表面点（与仿真几何同一份）
        field/<tx>.npy       每个表面点的标量 u ∈ [0,1]（float16，全局 dB 映射）
        meta.json            TX 列表 / 划分 / 6 面相机 / 归一化 / dB 映射
        scene.json           房间与材质、频率、TX 功率等可追溯参数

关键口径：
  * GT 是**空间场**（接收机放在每个表面点时的接收功率），不是网关方向谱；
  * u 用**全局** dB 区间映射，绝对功率信息保留 ⇒ 发射功率轴是物理的，不是合成增益；
  * 点云 = 仿真几何本身（同一份坐标），没有「空盒子未知」的问题。

跑法：
    python code/sim/gen_sionna_field.py --mode smoke
    python code/sim/gen_sionna_field.py --mode calib
    python code/sim/gen_sionna_field.py --mode gen --out /d/GS3D/mvsplat/datasets/SionnaRoom
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# ------------------------------------------------------------------ 场景
ROOM = [8.0, 6.0, 3.0]                # 与 D2 同尺寸，便于与 GRaF-sim 结果对齐
FREQ = 2.412e9
TX_POWER_DBM = 30.0                   # 1 W
C_LIGHT = 299792458.0

# ITU-R P.2040 材料（名字由 Sionna 内置参数表给）
MAT_WALL = "concrete"
MAT_CEIL = "plasterboard"
MAT_FLOOR = "wood"

# 家具：轴对齐盒子 (cx, cy, cz, sx, sy, sz, 材质)。默认关闭，--furniture 1 打开。
FURNITURE = [
    (1.0, -0.6, 0.375, 1.6, 0.9, 0.75, "wood"),      # 桌子
    (-2.4, 1.8, 1.0, 0.8, 0.6, 2.0, "plasterboard"),  # 柜子
    (2.6, 2.0, 0.45, 0.7, 0.7, 0.9, "wood"),          # 边柜
]


def _box(xml_lines, cx, cy, cz, sx, sy, sz, mat):
    xml_lines.append(f"""
  <shape type="cube">
    <transform name="to_world">
      <scale x="{sx/2:.6f}" y="{sy/2:.6f}" z="{sz/2:.6f}"/>
      <translate x="{cx:.6f}" y="{cy:.6f}" z="{cz:.6f}"/>
    </transform>
    <ref id="{mat}" name="bsdf"/>
  </shape>""")


def room_xml(room, furniture=True):
    W, L, H = room
    W2, L2, H2 = W / 2, L / 2, H / 2
    mats = [MAT_WALL, MAT_CEIL, MAT_FLOOR] + ([f[6] for f in FURNITURE] if furniture else [])
    lines = ['<scene version="2.1.0">']
    for m in dict.fromkeys(mats):
        lines.append(f'  <bsdf type="itu-radio-material" id="{m}">'
                     f'<string name="type" value="{m}"/>'
                     f'<float name="thickness" value="0.1"/></bsdf>')
    # floor (z=0, 法向 +z) / ceiling (z=H, 法向 -z)
    lines.append(f"""
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{L2}" z="1"/><translate x="0" y="0" z="0"/>
    </transform><ref id="{MAT_FLOOR}" name="bsdf"/>
  </shape>
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{L2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="180"/><translate x="0" y="0" z="{H}"/>
    </transform><ref id="{MAT_CEIL}" name="bsdf"/>
  </shape>
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{H2}" y="{L2}" z="1"/>
      <rotate x="0" y="1" z="0" angle="90"/><translate x="{W2}" y="0" z="{H2}"/>
    </transform><ref id="{MAT_WALL}" name="bsdf"/>
  </shape>
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{H2}" y="{L2}" z="1"/>
      <rotate x="0" y="1" z="0" angle="-90"/><translate x="{-W2}" y="0" z="{H2}"/>
    </transform><ref id="{MAT_WALL}" name="bsdf"/>
  </shape>
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{H2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="90"/><translate x="0" y="{L2}" z="{H2}"/>
    </transform><ref id="{MAT_CEIL}" name="bsdf"/>
  </shape>
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{H2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="-90"/><translate x="0" y="{-L2}" z="{H2}"/>
    </transform><ref id="{MAT_CEIL}" name="bsdf"/>
  </shape>""")
    if furniture:
        for f in FURNITURE:
            _box(lines, *f)
    lines.append("</scene>")
    return "\n".join(lines)


# ------------------------------------------------------------------ 点云
def surface_points(room, spacing=0.05, inset=0.01, furniture=False):
    """房间 6 面 + 家具表面的规则点阵（世界坐标）。"""
    W, L, H = room
    pts = []

    def grid(a0, a1, b0, b1, c, axis):
        na = max(2, int(round((a1 - a0) / spacing)) + 1)
        nb = max(2, int(round((b1 - b0) / spacing)) + 1)
        A, B = np.meshgrid(np.linspace(a0, a1, na), np.linspace(b0, b1, nb),
                           indexing="ij")
        A, B = A.ravel(), B.ravel()
        if axis == "xy":
            return np.stack([A, B, np.full_like(A, c)], -1)
        if axis == "xz":
            return np.stack([A, np.full_like(A, c), B], -1)
        return np.stack([np.full_like(A, c), A, B], -1)

    e = inset
    pts.append(grid(-W / 2 + e, W / 2 - e, -L / 2 + e, L / 2 - e, e, "xy"))
    pts.append(grid(-W / 2 + e, W / 2 - e, -L / 2 + e, L / 2 - e, H - e, "xy"))
    pts.append(grid(-L / 2 + e, L / 2 - e, e, H - e, W / 2 - e, "yz"))
    pts.append(grid(-L / 2 + e, L / 2 - e, e, H - e, -W / 2 + e, "yz"))
    pts.append(grid(-W / 2 + e, W / 2 - e, e, H - e, L / 2 - e, "xz"))
    pts.append(grid(-W / 2 + e, W / 2 - e, e, H - e, -L / 2 + e, "xz"))

    if furniture:
        for cx, cy, cz, sx, sy, sz, _ in FURNITURE:
            x0, x1 = cx - sx / 2, cx + sx / 2
            y0, y1 = cy - sy / 2, cy + sy / 2
            z0, z1 = cz - sz / 2, cz + sz / 2
            pts.append(grid(x0 + e, x1 - e, y0 + e, y1 - e, z1 - e, "xy"))  # top
            pts.append(grid(x0 + e, x1 - e, y0 + e, y1 - e, z0 + e, "xy"))  # bottom
            for xx in (x0 + e, x1 - e):
                pts.append(grid(y0 + e, y1 - e, z0 + e, z1 - e, xx, "yz"))
            for yy in (y0 + e, y1 - e):
                pts.append(grid(x0 + e, x1 - e, z0 + e, z1 - e, yy, "xz"))

    p = np.concatenate(pts, 0)
    # 去重（角/边重合点），保留首次出现
    _, idx = np.unique(np.round(p, 4), axis=0, return_index=True)
    return p[np.sort(idx)]


# ------------------------------------------------------------------ 相机（6 面）
def face_cameras(fov_deg=95.0, img=96):
    """6 个正交面：fwd / down 列出，right = down × fwd（与 3DGS 的 [right,down,fwd] 一致）。"""
    faces = {
        "px": ((1, 0, 0), (0, 0, -1)),
        "nx": ((-1, 0, 0), (0, 0, -1)),
        "py": ((0, 1, 0), (0, 0, -1)),
        "ny": ((0, -1, 0), (0, 0, -1)),
        "pz": ((0, 0, 1), (0, 1, 0)),
        "nz": ((0, 0, -1), (0, -1, 0)),
    }
    f = math.radians(fov_deg)
    fx = (img / 2.0) / math.tan(f / 2.0)
    out = []
    for name, (fw, dn) in faces.items():
        fw = np.array(fw, float)
        dn = np.array(dn, float)
        rt = np.cross(dn, fw)          # right = down × fwd
        R = np.stack([rt, dn, fw], axis=1)   # c2w，列 = right/down/fwd
        out.append({"name": name, "R": R.tolist(), "fovx": fov_deg, "fovy": fov_deg,
                    "fx": fx, "fy": fx})
    return out


def pixel_dirs(R_c2w, W, H, fx, fy):
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = (u + 0.5 - W / 2.0) / fx
    y = (v + 0.5 - H / 2.0) / fy
    d = np.stack([x, y, np.ones_like(x)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ np.asarray(R_c2w, float).T


# ------------------------------------------------------------------ 采样
def fps_select(pts, n, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n, len(pts))
    sel = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(1, n):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return sel


def sample_tx(room, n, margin=0.5, seed=0):
    rng = np.random.default_rng(seed)
    W, L, H = room
    lo = np.array([-W / 2 + margin, -L / 2 + margin, margin])
    hi = np.array([W / 2 - margin, L / 2 - margin, H - margin])
    p = rng.uniform(lo, hi, size=(n * 8, 3))
    # 最远点采样铺开（与 GRaF-sim 的处理一致）
    sel = fps_select(p, n, seed)
    return p[sel]


def sample_rx(room, nx=4, ny=3, nz=3, margin=0.9):
    """房间内的接收机位姿位置网格；每个位置配 6 个面相机。"""
    W, L, H = room
    xs = np.linspace(-W / 2 + margin, W / 2 - margin, nx)
    ys = np.linspace(-L / 2 + margin, L / 2 - margin, ny)
    zs = np.linspace(0.6, H - 0.6, nz)
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)


# ------------------------------------------------------------------ Sionna
def make_scene(rt, xml):
    from sionna.rt.scene import load_scene_from_string
    from sionna.rt import PlanarArray
    scene = load_scene_from_string(xml)
    scene.frequency = FREQ
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso",
                                 polarization="V")
    return scene


def plane_specs(room, inset=0.01):
    """6 个测量面：每个面的 (名字, 中心, orientation[弧度], size)。

    注意 RadioMapSolver 的 orientation 单位是**弧度**（设备 orientation 才是度）。
    (0,0,0) = XY 平面；β=π/2 把平面转到 YZ（局部 x→−z, y→y），
    γ=π/2 把平面转到 XZ（局部 x→x, y→−z）。
    """
    W, L, H = room
    return [
        ("floor", [0, 0, inset], [0, 0, 0], [W, L]),
        ("ceil", [0, 0, H - inset], [0, 0, 0], [W, L]),
        ("x+", [W / 2 - inset, 0, H / 2], [0, np.pi / 2, 0], [H, L]),
        ("x-", [-W / 2 + inset, 0, H / 2], [0, np.pi / 2, 0], [H, L]),
        ("y+", [0, L / 2 - inset, H / 2], [0, 0, np.pi / 2], [W, H]),
        ("y-", [0, -L / 2 + inset, H / 2], [0, 0, np.pi / 2], [W, H]),
    ]


def _rmap(rt, scene, spec, cell, samples, max_depth):
    _, c, o, sz = spec
    return rt.RadioMapSolver()(
        scene, center=np.array(c, np.float32),
        orientation=np.array(o, np.float32), size=np.array(sz, np.float32),
        cell_size=cell, samples_per_tx=samples, max_depth=max_depth,
        los=True, specular_reflection=True, diffuse_reflection=False,
        refraction=False, diffraction=False, seed=0)


def build_cloud(rt, scene, room, cell, samples, max_depth):
    """点云 = 6 个测量面的 cell 中心（与 GT 场逐点一一对应）。

    先放一个占位 TX（之后复用它，只改 position），返回 (pts, specs, tx)。
    """
    from sionna.rt import Transmitter
    tx = Transmitter("tx", position=np.zeros(3, np.float32),
                     power_dbm=TX_POWER_DBM)
    scene.add(tx)
    specs = plane_specs(room)
    grids = []
    for spec in specs:
        rm = _rmap(rt, scene, spec, cell, samples, max_depth)
        cc = np.array(rm.cell_centers).reshape(-1, 3)
        grids.append(cc)
        print(f"  面 {spec[0]:5s} cells {len(cc):6d} "
              f"bbox {cc.min(0).round(2)} .. {cc.max(0).round(2)}")
    return np.concatenate(grids, 0), specs, tx


def field_for_tx(rt, scene, tx, specs, tx_pos, cell, samples, max_depth):
    """一个 TX 的整屋表面场：6 个面的 path_gain 拼接，返回 dBm [P]。"""
    tx.position = np.asarray(tx_pos, np.float32)
    t0 = time.perf_counter()
    gains = []
    for spec in specs:
        rm = _rmap(rt, scene, spec, cell, samples, max_depth)
        gains.append(np.array(rm.path_gain).reshape(-1))
    g = np.concatenate(gains)
    # path_gain 是 unitless 的 P_rx/P_tx（不含发射功率）
    dbm = 10.0 * np.log10(np.maximum(g, 1e-30)) + TX_POWER_DBM
    return dbm.astype(np.float32), time.perf_counter() - t0


def dbm_to_u(dbm, lo_dbm, hi_dbm):
    return np.clip((dbm - lo_dbm) / (hi_dbm - lo_dbm), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ 模式
def mode_calib(rt):
    """两条链各自与 Friis 对拍。

    1) PathSolver.cfr：|h|² = (λ/4πd)² · P_tx  [瓦]
    2) RadioMapSolver.path_gain：g = (λ/4πd)²      [unitless]
    """
    xml = ('<scene version="2.1.0">'
           '<bsdf type="itu-radio-material" id="concrete">'
           '<string name="type" value="concrete"/></bsdf></scene>')
    d = 3.0
    friis = (C_LIGHT / (4 * math.pi * FREQ * d)) ** 2
    ptx_w = 10 ** ((TX_POWER_DBM - 30.0) / 10.0)

    # --- PathSolver ---
    from sionna.rt import Receiver, Transmitter
    scene = make_scene(rt, xml)
    tx = Transmitter("tx", position=np.zeros(3, np.float32), power_dbm=TX_POWER_DBM)
    rx = Receiver("rx", position=np.array([d, 0, 0], np.float32))
    scene.add([tx, rx])
    paths = rt.PathSolver()(scene, max_depth=3, los=True,
                            specular_reflection=True, diffuse_reflection=False,
                            refraction=False, diffraction=False,
                            synthetic_array=True)
    c = paths.cfr(frequencies=np.array([FREQ], np.float32), out_type="numpy")
    if isinstance(c, tuple):
        c = np.asarray(c[0]) + 1j * np.asarray(c[1])
    p = float(np.abs(np.asarray(c).reshape(-1))[0] ** 2)
    print(f"[calib] PathSolver  d={d} m  |h|²={p:.6e} W  "
          f"Friis×Ptx={friis*ptx_w:.6e} W  比值={p/(friis*ptx_w):.6f}")

    # --- RadioMapSolver（空房间里直接放在 TX 下方的平面）---
    scene2 = make_scene(rt, xml)
    tx2 = Transmitter("tx", position=np.zeros(3, np.float32), power_dbm=TX_POWER_DBM)
    scene2.add(tx2)
    rm = rt.RadioMapSolver()(scene2, center=np.array([0, 0, -d], np.float32),
                             orientation=np.zeros(3, np.float32),
                             size=np.array([0.1, 0.1], np.float32),
                             cell_size=0.05, samples_per_tx=10_000_000,
                             max_depth=0, los=True, specular_reflection=False,
                             diffuse_reflection=False, refraction=False,
                             diffraction=False, seed=0)
    g = np.array(rm.path_gain).reshape(-1)
    print(f"[calib] RadioMapSolver  g={g.mean():.6e}  Friis={friis:.6e}  "
          f"比值={g.mean()/friis:.6f}  (应≈1，path_gain 不含发射功率)")


def mode_smoke(rt, room, cell=0.05, n_tx=2, samples=50_000_000, max_depth=3):
    xml = room_xml(room, furniture=False)
    scene = make_scene(rt, xml)
    t0 = time.time()
    pts, specs, tx = build_cloud(rt, scene, room, cell, samples, max_depth)
    print(f"[smoke] 点云 {pts.shape[0]}  建面 {time.time()-t0:.2f}s")
    txs = sample_tx(room, n_tx, seed=1)
    for i, t in enumerate(txs):
        dbm, dt = field_for_tx(rt, scene, tx, specs, t, cell, samples, max_depth)
        print(f"[smoke] tx{i} {np.round(t,2)}  {dt*1000:.0f} ms  "
              f"dBm p1/p50/p99 {np.percentile(dbm,1):.1f}/"
              f"{np.percentile(dbm,50):.1f}/{np.percentile(dbm,99):.1f}  "
              f"min {dbm.min():.1f} max {dbm.max():.1f}")


def mode_gen(rt, room, out, n_tx, spacing, furniture, max_depth, rng_seed,
             lo_dbm, hi_dbm, n_rx, samples):
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "field"), exist_ok=True)
    if furniture:
        raise SystemExit("家具表面尚未接入测量面（当前 6 面只覆盖房间壳体）")

    # 归一化：房间中心为原点，norm = 0.5·半对角线（与 GRaF 处理同一动机）
    hd = float(np.linalg.norm(np.array(room) / 2))
    norm = 0.5 * hd
    center = np.zeros(3)

    xml = room_xml(room, furniture=False)
    scene = make_scene(rt, xml)

    pts, specs, tx = build_cloud(rt, scene, room, spacing, samples, max_depth)
    print(f"[gen] 表面点 {pts.shape[0]}  cell={spacing}  norm={norm:.4f}")

    txs = sample_tx(room, n_tx, seed=rng_seed)
    order = fps_select(txs, len(txs), seed=rng_seed + 1)
    txs = txs[order]
    test = [i for i in range(len(txs)) if i % 4 == 3]
    train = [i for i in range(len(txs)) if i % 4 != 3]
    print(f"[gen] TX {len(txs)}（train {len(train)} / test {len(test)}）")

    rx = sample_rx(room, n_rx, n_rx - 1 if n_rx > 1 else 1, 3)
    print(f"[gen] RX 位姿 {len(rx)}（×6 面）")

    # 依此归一化
    pts_n = (pts - center) / norm
    txs_n = (txs - center) / norm
    rx_n = (rx - center) / norm

    from plyfile import PlyData, PlyElement
    t_all = time.time()
    for i, t in enumerate(txs):
        dbm, dt = field_for_tx(rt, scene, tx, specs, t, spacing, samples, max_depth)
        # 存原始 dBm（float16）：显示映射（lo/hi）留到加载时再做，
        # 这样以后调色阶 / 调功率轴都不用重跑仿真。
        np.save(os.path.join(out, "field", f"{i:05d}.npy"), dbm.astype(np.float16))
        if i % 10 == 0 or i == len(txs) - 1:
            u = dbm_to_u(dbm, lo_dbm, hi_dbm)
            print(f"  [{i+1}/{len(txs)}] {dt*1000:.0f} ms  "
                  f"dBm p1/p50/p99 {np.percentile(dbm,1):.1f}/"
                  f"{np.percentile(dbm,50):.1f}/{np.percentile(dbm,99):.1f}  "
                  f"u均值 {u.mean():.3f}  u=0/1 占比 "
                  f"{(u<=0).mean():.3f}/{(u>=1).mean():.3f}  "
                  f"累计 {(time.time()-t_all)/60:.1f} min", flush=True)

    # ---- 点云 PLY（与 graf_prepare 同格式）----
    n = pts_n.shape[0]
    fields_def = ([("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                   ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
                  + [(f"f_rest_{k}", "f4") for k in range(45)]
                  + [("opacity", "f4")]
                  + [(f"scale_{k}", "f4") for k in range(3)]
                  + [(f"rot_{k}", "f4") for k in range(4)])
    arr = np.zeros(n, dtype=fields_def)
    arr["x"], arr["y"], arr["z"] = pts_n[:, 0], pts_n[:, 1], pts_n[:, 2]
    arr["opacity"] = 4.0
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = math.log(spacing / norm)
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(
        os.path.join(out, "cloud.ply"))

    cams = face_cameras(95.0, 96)
    meta = {
        "format": "spatial_rf_v1",
        "scene": "SionnaRoom",
        "room_size": room,
        "freq_hz": FREQ,
        "tx_power_dbm": TX_POWER_DBM,
        "materials": {"wall": MAT_WALL, "ceiling": MAT_CEIL, "floor": MAT_FLOOR},
        "furniture": furniture,
        "norm_unit": norm,
        "center": center.tolist(),
        "n_points": int(n),
        "spacing": spacing,
        "tx_world": txs.tolist(),
        "tx_local": txs_n.tolist(),
        "split": {"train": train, "test": test},
        "rx_world": rx.tolist(),
        "rx_local": rx_n.tolist(),
        "rx_test": [i for i in range(len(rx)) if i % 3 == 2],
        "rx_train": [i for i in range(len(rx)) if i % 3 != 2],
        "views": {"n": 6, "img": 96, "fov": 95.0, "cameras": cams},
        "db_map": {"lo_dbm": lo_dbm, "hi_dbm": hi_dbm},
        "gt_pattern": "field/{j:05d}.npy",
        "cloud_ply": "cloud.ply",
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"room": room, "freq_hz": FREQ, "tx_power_dbm": TX_POWER_DBM,
               "materials": meta["materials"], "furniture_on": furniture,
               "spacing": spacing, "db_map": meta["db_map"],
               "sionna": "2.2.0", "generated": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(out, "scene.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[gen] 完成 -> {out}  总耗时 {(time.time()-t_all)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("calib", "smoke", "gen"), default="smoke")
    ap.add_argument("--out", default="D:/GS3D/mvsplat/datasets/SionnaRoom")
    ap.add_argument("--n_tx", type=int, default=200)
    ap.add_argument("--spacing", type=float, default=0.05)
    ap.add_argument("--furniture", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lo_dbm", type=float, default=-80.0)
    ap.add_argument("--hi_dbm", type=float, default=-30.0)
    ap.add_argument("--n_rx", type=int, default=4)
    ap.add_argument("--samples", type=int, default=50_000_000,
                    help="RadioMapSolver 每 TX 的射线采样数")
    args = ap.parse_args()

    import sionna.rt as rt
    room = ROOM
    if args.mode == "calib":
        mode_calib(rt)
    elif args.mode == "smoke":
        mode_smoke(rt, room, args.spacing, args.n_tx, args.samples, args.max_depth)
    else:
        mode_gen(rt, room, args.out, args.n_tx, args.spacing,
                 bool(args.furniture), args.max_depth, args.seed,
                 args.lo_dbm, args.hi_dbm, args.n_rx, args.samples)


if __name__ == "__main__":
    main()
