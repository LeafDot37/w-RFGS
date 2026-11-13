# -*- coding: utf-8 -*-
"""按 RF-3DGS 原口径，为 NIST lobby 的指定发射源批量生成 CBF 角域谱图。

口径（全部取自教程 notebook，已由 `code/tools/repro_cbf_gate.py` 验证）：
  * 频率 2.4 GHz（教程 cell 107 在 CBF 生成前显式设置）；
  * **教程原样材质**，不做任何改写；
  * RX = 作者 `images.txt` 里的 800 个位置 × 4 个阵列 yaw（−90/0/+90/180°）；
  * 300×200 针孔 AoA 网格，fov 90°；10×10 UPA，tr38901，半波长，synthetic；
  * max_depth=1，反射 + 漫散射，1e6 采样；
  * CIR 0.5 ns 时间格合并 → CBF → 逐图 min-max 归一化。

输出（每张图一个 8bit 灰度 PNG，存的是 min-max 归一化后的谱值；jet 上色在
读取端用 `jet_from_unit` 还原，信息量与作者发布的 jet PNG 等价）：

    <out>/tx<idx:03d>/<img_id:05d>.png
    <out>/meta.json

跑法：
    # 先量吞吐
    D:/anaconda3/envs/sionna_win/python.exe code/sim/gen_cbf_spectra.py \
        --out datasets/RF-3DGS_cbf_multi --tx 0 --views train --bench 20

    # 正式生成（可多进程，按 TX 切）
    ... --tx 0 1 2 3 --views train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_nist_field import load_scene                        # noqa: E402
from sim.spectrum_gen import (CBFMapper, ANGLES,                # noqa: E402
                              compute_angle_matrices, jet_from_unit,
                              normalize_spectrum, read_colmap_poses)

POSE_ROOT = "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"
META_JSON = "datasets/NIST_lobby_spatial_full/meta.json"


def read_ids(path):
    return [int(x) for x in Path(path).read_text().split()]


def group_by_pose(poses, ids):
    """把同一位置（4 个 yaw）的图号归到一组，保持顺序。"""
    groups, seen = [], {}
    for i in ids:
        c, _ = poses[i]
        key = tuple(np.round(c, 4))
        if key not in seen:
            seen[key] = len(groups)
            groups.append([])
        groups[seen[key]].append(i)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--meta", default=META_JSON)
    ap.add_argument("--tx", type=int, nargs="+", required=True, help="空间场数据集里的 TX 下标")
    ap.add_argument("--views", choices=["train", "test", "all"], default="train")
    ap.add_argument("--freq", type=float, default=2.4e9)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--mat_freq", type=float, default=60e9,
                    help="教程 cell 9 构造材质用的频率（CBF 谱图是 60 GHz 建材质、2.4 GHz 仿真）")
    ap.add_argument("--diffuse", action="store_true",
                    help="开漫散射。默认关：Sionna 2.2 的漫散射模型与 1.x 差异大，"
                         "实测开散射会把与发布 GT 的相关从 0.848 压到 0.573（见 scat 扫描）")
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--tx_power_dbm", type=float, default=0.0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="只跑前 N 组位置，不落盘")
    ap.add_argument("--limit", type=int, default=0, help="每个 TX 最多生成 N 张（调试用）")
    args = ap.parse_args()

    import sionna.rt as rt

    meta = json.load(open(args.meta, encoding="utf-8"))
    tx_all = np.asarray(meta["tx_world"], np.float32)
    poses = read_colmap_poses(args.poses_root)
    split_dir = Path(args.poses_root)
    if args.views == "train":
        ids = read_ids(split_dir / "train_index.txt")
    elif args.views == "test":
        ids = read_ids(split_dir / "test_index.txt")
    else:
        ids = sorted(poses)
    groups = group_by_pose(poses, ids)
    if args.bench:
        groups = groups[: args.bench]
    n_imgs = sum(len(g) for g in groups)
    out_root = Path(args.out)

    sc = load_scene(freq_hz=args.freq, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                horizontal_spacing=0.5, pattern="iso", polarization="V")
    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True

    theta, phi = compute_angle_matrices(args.width, args.height, 90.0)
    mapper = CBFMapper(args.M, theta, phi, time_interval_ns=0.5)
    solver = rt.PathSolver()

    print(f"[gen] {len(groups)} 组位置 / {n_imgs} 张  TX={args.tx}  "
          f"freq {args.freq/1e9:.3g} GHz  views={args.views}  out={out_root}", flush=True)

    t_start = time.time()
    n_done = 0
    for k_tx, ti in enumerate(args.tx):
        tx_pos = tx_all[ti]
        sc.add(rt.Transmitter("tx", position=tx_pos, power_dbm=args.tx_power_dbm))
        if not args.bench:
            (out_root / f"tx{ti:03d}").mkdir(parents=True, exist_ok=True)
        t_tx = time.time()
        for gi, ids_g in enumerate(groups):
            todo = ids_g
            if args.skip_existing and not args.bench:
                todo = [i for i in ids_g
                        if not (out_root / f"tx{ti:03d}" / f"{i:05d}.png").exists()]
                if not todo:
                    continue
            # 同一位置的 4 个 yaw 一次性加 4 个 RX，一次求解
            names = []
            for n, i in enumerate(todo):
                c, ang = poses[i]
                nm = f"rx{n}"
                sc.add(rt.Receiver(nm, position=c,
                                   orientation=np.array([ang, 0, 0], np.float32)))
                names.append(nm)
            t0 = time.time()
            paths = solver(sc, max_depth=args.max_depth, los=True,
                           specular_reflection=True, diffuse_reflection=args.diffuse,
                           refraction=False, diffraction=False, edge_diffraction=False,
                           samples_per_src=int(args.samples), synthetic_array=True)
            t1 = time.time()
            for n, i in enumerate(todo):
                u = normalize_spectrum(mapper(paths, rx=n))
                if not args.bench:
                    Image.fromarray(np.round(u * 255).astype(np.uint8)).save(
                        out_root / f"tx{ti:03d}" / f"{i:05d}.png", optimize=False)
            t2 = time.time()
            n_done += len(todo)
            for nm in names:
                sc.remove(nm)
            if args.bench:
                print(f"[bench] group {gi}: {len(todo)} 张  solve {t1-t0:.3f}s  "
                      f"map {t2-t1:.3f}s  paths {int(np.sum(np.asarray(paths.valid)[0]))}",
                      flush=True)
            elif gi % 20 == 0:
                el = time.time() - t_start
                rate = n_done / max(el, 1e-9)
                print(f"[gen] tx{ti:03d} group {gi}/{len(groups)}  "
                      f"{n_done} 张  {el/60:.1f} min  {rate*60:.0f} 张/min  "
                      f"剩 {(n_imgs-n_done)/max(rate,1e-9)/3600:.2f} h", flush=True)
            if args.limit and n_done >= args.limit:
                break
        if not args.bench:
            print(f"[gen] tx{ti:03d} 完成 {time.time()-t_tx:.1f}s", flush=True)
        sc.remove("tx")
        if args.limit and n_done >= args.limit:
            break

    el = time.time() - t_start
    if args.bench:
        print(f"[bench] 合计 {n_done} 张 / {el:.1f}s = {el/max(n_done,1)*1000:.0f} ms/张；"
              f"按此推算 92160 张需 {92160*el/max(n_done,1)/3600:.1f} h（单进程）")
        return

    info = {
        "protocol": {
            "freq_hz": args.freq, "M": args.M, "max_depth": args.max_depth,
            "samples_per_src": args.samples, "diffuse": bool(args.diffuse),
            "material_freq_hz": args.mat_freq, "width": args.width,
            "height": args.height, "fov": 90.0, "time_interval_ns": 0.5,
            "tx_power_dbm": args.tx_power_dbm,
            "materials": "tutorial materials built at mat_freq (cell 9), simulated at freq",
            "storage": "8bit grayscale = per-image min-max normalized spectrum",
        },
        "poses_root": str(args.poses_root),
        "views": args.views,
        "tx_indices": args.tx,
        "tx_world": {str(i): tx_all[i].tolist() for i in args.tx},
        "n_generated_now": n_done,
        "n_images_on_disk": int(sum(len(list(d.glob("*.png")))
                                    for d in out_root.glob("tx*") if d.is_dir())),
        "elapsed_s": el,
        "angles": list(ANGLES),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.json"
    if meta_path.exists():                       # 多次按 TX 分批调用时合并，不互相覆盖
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old_ids = set(old.get("tx_indices", []))
        info["tx_indices"] = sorted(old_ids | set(args.tx))
        merged_world = old.get("tx_world", {})
        merged_world.update(info["tx_world"])
        info["tx_world"] = merged_world
        info["views"] = "mixed" if old.get("views") not in (None, args.views) else args.views
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[gen] 完成 {n_done} 张，用时 {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
