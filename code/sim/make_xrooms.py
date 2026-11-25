# -*- coding: utf-8 -*-
"""生成跨场景可行性实验用的合成室内布局（Mitsuba XML + 表面采样点）。

为什么自己造场景
--------------------------------------------------------------------------------
WiSegRT 的 6 个场景原本是最合适的跨场景台，但它的 scene.zip 现在需要登录
SharePoint（实测 403），本机只抓到 meshes 与 scene_01.xml。跨场景可行性实验
不依赖某个特定数据集，只需要「多个布局 + 同一套算子」，因此这里用 6 个可复现的
合成房间：尺寸、隔断、柱子、家具和材质各不相同，其余口径（300×200 AoA 网格、
max_depth=1、镜面反射、4×4 发射阵）与主实验逐项一致。

产物（默认 `_external/xrooms/<scene>/`）：
    scene.xml          Sionna/Mitsuba 场景
    points.npz         xyz(P,3), normal(P,3), mat(P,)  ← 模型用的表面点
    sparse/0/images.txt, cameras.txt
    train_index.txt, test_index.txt
    instances.json     每个 TX 一条（role=train）
    scene.json         布局参数

用法：
    python code/sim/make_xrooms.py --out _external/xrooms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 6 个布局：墙体、地板、天花材质 + 家具。尽量让「反射面几何」差异大。
ROOMS = {
    "room_a": dict(W=6.0, D=5.0, H=3.0, wall="concrete", floor="concrete",
                   ceil="plasterboard", boxes=[]),
    "room_b": dict(W=11.0, D=7.0, H=3.2, wall="concrete", floor="wood",
                   ceil="plasterboard", boxes=[
                       ("box", 4.0, 3.5, 1.6, 0.6, 0.6, 3.2, "concrete"),
                       ("box", 7.0, 3.5, 1.6, 0.6, 0.6, 3.2, "concrete")]),
    "room_c": dict(W=8.0, D=6.0, H=2.8, wall="brick", floor="wood",
                   ceil="ceiling_board", boxes=[
                       ("box", 2.0, 2.0, 0.4, 1.4, 0.9, 0.8, "wood"),
                       ("box", 6.0, 4.2, 0.5, 1.8, 0.6, 1.0, "wood"),
                       ("box", 4.0, 1.2, 0.3, 1.2, 0.5, 0.6, "metal")]),
    "room_d": dict(W=9.0, D=5.0, H=3.0, wall="concrete", floor="wood",
                   ceil="plasterboard", boxes=[
                       ("box", 4.5, 2.5, 1.4, 0.08, 3.0, 2.0, "glass"),
                       ("box", 1.5, 1.2, 0.4, 1.0, 0.8, 0.7, "wood"),
                       ("box", 7.5, 3.8, 0.4, 1.2, 0.9, 0.8, "wood")]),
    "room_e": dict(W=12.0, D=8.0, H=4.0, wall="concrete", floor="concrete",
                   ceil="ceiling_board", boxes=[
                       ("box", 6.0, 5.5, 1.2, 6.0, 0.1, 2.4, "metal"),
                       ("box", 3.0, 4.0, 1.5, 0.7, 0.7, 3.0, "concrete")]),
    "room_f": dict(W=7.0, D=6.0, H=3.0, wall="plasterboard", floor="wood",
                   ceil="plasterboard", boxes=[
                       ("box", 1.6, 1.6, 0.4, 1.3, 0.8, 0.7, "wood"),
                       ("box", 5.4, 1.6, 0.4, 1.3, 0.8, 0.7, "wood"),
                       ("box", 1.6, 4.4, 0.4, 1.3, 0.8, 0.7, "wood"),
                       ("box", 5.4, 4.4, 0.4, 1.3, 0.8, 0.7, "wood"),
                       ("box", 3.5, 3.0, 0.5, 1.0, 1.0, 1.0, "metal"),
                       ("box", 6.6, 3.0, 0.9, 0.3, 2.4, 1.8, "glass")]),
}

MATS = ["concrete", "brick", "plasterboard", "wood", "floorboard",
        "ceiling_board", "glass", "metal"]

FREQ_HZ = 3.6e9
N_POINTS = 16000
N_POS = 12          # RX 位置数（每位置 4 个 yaw）
N_POS_TRAIN = 6     # 前 6 个位置进 train，其余进 test
ANGLES = (-np.pi / 2, 0.0, np.pi / 2, np.pi)


def mat_xml(m: str) -> str:
    return (f'<bsdf type="itu-radio-material" id="{m}">\n'
            f'  <string name="type" value="{m}"/>\n'
            f'  <float name="thickness" value="0.1"/>\n'
            f'</bsdf>')


def rect_xml(sid, m, W, D, H, axis, sign, offset):
    """房间六面。axis 为法向轴；sign=+1 表示法向朝房间内。"""
    if axis == "z" and sign > 0:      # 地板（法向 +z）
        tf = f'<scale x="{W}" y="{D}" z="1"/><translate x="{W/2}" y="{D/2}" z="0"/>'
    elif axis == "z":                 # 天花（法向 -z）
        tf = (f'<scale x="{W}" y="{D}" z="1"/><rotate x="1" angle="180"/>'
              f'<translate x="{W/2}" y="{D/2}" z="{H}"/>')
    elif axis == "y" and sign > 0:    # y=D（法向 -y）
        tf = (f'<scale x="{W}" y="{H}" z="1"/><rotate x="1" angle="90"/>'
              f'<translate x="{W/2}" y="{D}" z="{H/2}"/>')
    elif axis == "y":                 # y=0（法向 +y）
        tf = (f'<scale x="{W}" y="{H}" z="1"/><rotate x="1" angle="-90"/>'
              f'<translate x="{W/2}" y="0" z="{H/2}"/>')
    elif axis == "x" and sign > 0:    # x=0（法向 +x）
        tf = (f'<scale x="{D}" y="{H}" z="1"/><rotate y="1" angle="90"/>'
              f'<translate x="0" y="{D/2}" z="{H/2}"/>')
    else:                             # x=W（法向 -x）
        tf = (f'<scale x="{D}" y="{H}" z="1"/><rotate y="1" angle="-90"/>'
              f'<translate x="{W}" y="{D/2}" z="{H/2}"/>')
    return (f'<shape type="rectangle" id="{sid}">\n'
            f'  <transform name="to_world">{tf}</transform>\n'
            f'  <ref id="{m}" name="bsdf"/>\n</shape>')


def box_xml(sid, m, cx, cy, cz, sx, sy, sz):
    tf = (f'<scale x="{sx}" y="{sy}" z="{sz}"/>'
          f'<translate x="{cx}" y="{cy}" z="{cz}"/>')
    return (f'<shape type="cube" id="{sid}">\n'
            f'  <transform name="to_world">{tf}</transform>\n'
            f'  <ref id="{m}" name="bsdf"/>\n</shape>')


def room_xml(spec) -> str:
    W, D, H = spec["W"], spec["D"], spec["H"]
    parts = [f'<scene version="2.1.0">',
             '<!-- ITU radio materials -->']
    for m in sorted({spec["wall"], spec["floor"], spec["ceil"]}
                    | {b[7] for b in spec["boxes"]}):
        parts.append(mat_xml(m))
    parts.append('<shape type="rectangle" id="floor">'
                 f'<transform name="to_world"><scale x="{W}" y="{D}" z="1"/>'
                 f'<translate x="{W/2}" y="{D/2}" z="0"/></transform>'
                 f'<ref id="{spec["floor"]}" name="bsdf"/></shape>')
    parts.append(rect_xml("ceil", spec["ceil"], W, D, H, "z", -1, None))
    parts.append(rect_xml("wall_n", spec["wall"], W, D, H, "y", 1, None))
    parts.append(rect_xml("wall_s", spec["wall"], W, D, H, "y", -1, None))
    parts.append(rect_xml("wall_w", spec["wall"], W, D, H, "x", 1, None))
    parts.append(rect_xml("wall_e", spec["wall"], W, D, H, "x", -1, None))
    for i, b in enumerate(spec["boxes"]):
        _, cx, cy, cz, sx, sy, sz, m = b
        parts.append(box_xml(f"obj{i}", m, cx, cy, cz, sx, sy, sz))
    parts.append("</scene>")
    return "\n".join(parts)


def sample_rect(rng, n, center, u, v, mat, toward=None):
    """矩形：center 为角点中心，u/v 为两条边向量。

    `toward` 给定时把法向翻到指向该点（房间面朝房间中心；家具面朝外）。
    """
    a = rng.random((n, 2))
    pts = (np.asarray(center)[None, :] + a[:, :1] * np.asarray(u)[None, :]
           + a[:, 1:] * np.asarray(v)[None, :])
    nrm = np.cross(u, v)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
    if toward is not None:
        if np.dot(nrm, np.asarray(toward) - pts[0]) < 0:
            nrm = -nrm
    return pts, np.repeat(nrm[None, :], n, 0)


def sample_surfaces(spec, n_total, rng):
    """按面积分配采样点；返回 xyz, normal, mat_id。"""
    W, D, H = spec["W"], spec["D"], spec["H"]
    room_c = np.array([W / 2, D / 2, H / 2])
    surf = []   # (area, mat, corner, u, v, toward)
    # 六个面（与 XML 同口径）
    surf.append((W * D, spec["floor"], np.array([0, 0, 0.]),
                 np.array([W, 0, 0.]), np.array([0, D, 0.]), room_c))
    surf.append((W * D, spec["ceil"], np.array([0, 0, H]),
                 np.array([W, 0, 0.]), np.array([0, D, 0.]), room_c))
    surf.append((W * H, spec["wall"], np.array([0, 0, 0.]),
                 np.array([W, 0, 0.]), np.array([0, 0, H]), room_c))
    surf.append((W * H, spec["wall"], np.array([0, D, 0.]),
                 np.array([W, 0, 0.]), np.array([0, 0, H]), room_c))
    surf.append((D * H, spec["wall"], np.array([0, 0, 0.]),
                 np.array([0, D, 0.]), np.array([0, 0, H]), room_c))
    surf.append((D * H, spec["wall"], np.array([W, 0, 0.]),
                 np.array([0, D, 0.]), np.array([0, 0, H]), room_c))
    for b in spec["boxes"]:
        _, cx, cy, cz, sx, sy, sz, m = b
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        c = np.array([cx, cy, cz])
        faces = [
            (sx * sy, c + [0, 0, hz], [sx, 0, 0], [0, sy, 0], [0, 0, 1]),
            (sx * sy, c - [0, 0, hz], [sx, 0, 0], [0, sy, 0], [0, 0, -1]),
            (sx * sz, c + [0, hy, 0], [sx, 0, 0], [0, 0, sz], [0, 1, 0]),
            (sx * sz, c - [0, hy, 0], [sx, 0, 0], [0, 0, sz], [0, -1, 0]),
            (sy * sz, c + [hx, 0, 0], [0, sy, 0], [0, 0, sz], [1, 0, 0]),
            (sy * sz, c - [hx, 0, 0], [0, sy, 0], [0, 0, sz], [-1, 0, 0]),
        ]
        for s in faces:
            corner = (np.asarray(s[1], float) - np.asarray(s[2], float) / 2
                      - np.asarray(s[3], float) / 2)
            surf.append((s[0], m, corner,
                         np.asarray(s[2], float), np.asarray(s[3], float),
                         c + np.asarray(s[4], float)))
    area = np.array([s[0] for s in surf])
    alloc = np.maximum(64, np.round(n_total * area / area.sum()).astype(int))
    xs, ns, ms = [], [], []
    for a, s in zip(alloc, surf):
        p, nn = sample_rect(rng, int(a), s[2], s[3], s[4], s[1], toward=s[5])
        xs.append(p); ns.append(nn)
        ms.append(np.full(len(p), MATS.index(s[1]), np.int64))
    return np.concatenate(xs), np.concatenate(ns), np.concatenate(ms)


def make_poses(spec, root: Path, rng):
    W, D, H = spec["W"], spec["D"], spec["H"]
    cx, cy, cz = W / 2, D / 2, min(1.6, 0.55 * H)
    rad = 0.30 * min(W, D)
    lines, cams = [], []
    ids = []
    k = 0
    for i in range(N_POS):
        a = 2 * np.pi * i / N_POS
        c = np.array([cx + rad * np.cos(a), cy + rad * np.sin(a), cz])
        for j in range(4):
            k += 1
            q = (1.0, 0.0, 0.0, 0.0)
            t = -c
            lines.append(f"{k} {q[0]} {q[1]} {q[2]} {q[3]} "
                         f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {k:05d}.png")
            lines.append("")          # COLMAP：每张图后跟一行 2D 点（空行即可）
            ids.append(k)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0" / "images.txt").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    (root / "sparse" / "0" / "cameras.txt").write_text(
        "1 PINHOLE 300 200 150 150 150 100\n", encoding="utf-8")
    n_train = N_POS_TRAIN * 4
    (root / "train_index.txt").write_text(
        " ".join(f"{i:05d}" for i in ids[:n_train]) + "\n", encoding="utf-8")
    (root / "test_index.txt").write_text(
        " ".join(f"{i:05d}" for i in ids[n_train:]) + "\n", encoding="utf-8")
    return ids


def make_instances(spec, name, out: Path):
    W, D, H = spec["W"], spec["D"], spec["H"]
    # 4 个发射源：不同角落/高度/朝向；频率固定 3.6 GHz（跨场景只看布局）
    base = [
        (0.18, 0.18, 1.2, 0.35, 0.15),
        (0.82, 0.22, 1.5, 2.10, -0.20),
        (0.20, 0.82, 1.0, 4.20, 0.25),
        (0.78, 0.80, 1.7, 5.60, -0.10),
    ]
    rows = []
    for k, (fx, fy, z, yaw, pitch) in enumerate(base):
        xyz = [fx * W, fy * D, min(z, H - 0.2)]
        # 避开家具：若落在某个 box 内部，沿 x/y 推 1 m
        for b in spec["boxes"]:
            _, bx, by, bz, sx, sy, sz, _ = b
            if (abs(xyz[0] - bx) < sx / 2 + 0.3
                    and abs(xyz[1] - by) < sy / 2 + 0.3
                    and abs(xyz[2] - bz) < sz / 2 + 0.3):
                xyz[0] = min(max(xyz[0] + sx / 2 + 0.6, 0.4), W - 0.4)
                xyz[1] = min(max(xyz[1] + sy / 2 + 0.6, 0.4), D - 0.4)
        rows.append({"id": k, "role": "train", "scene": name,
                     "xyz": xyz, "yaw_abs_rad": yaw, "pitch_abs_rad": pitch,
                     "freq_hz": FREQ_HZ, "w_label": "uniform",
                     "views": "all", "tx_rows": 4, "tx_cols": 4})
    (out / "instances.json").write_text(
        json.dumps({"instances": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_external/xrooms")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out_root = Path(args.out)
    names = args.only or list(ROOMS)
    for name in names:
        spec = ROOMS[name]
        root = out_root / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "scene.xml").write_text(room_xml(spec), encoding="utf-8")
        rng = np.random.default_rng(args.seed + sum(map(ord, name)))
        xyz, nrm, mat = sample_surfaces(spec, N_POINTS, rng)
        np.savez(root / "points.npz", xyz=xyz.astype(np.float32),
                 normal=nrm.astype(np.float32), mat=mat)
        make_poses(spec, root, rng)
        rows = make_instances(spec, name, root)
        (root / "scene.json").write_text(json.dumps(spec, indent=2),
                                         encoding="utf-8")
        print(f"[xroom] {name}: {spec['W']}x{spec['D']}x{spec['H']} "
              f"pts={len(xyz)} boxes={len(spec['boxes'])} tx={len(rows)}",
              flush=True)


if __name__ == "__main__":
    main()
