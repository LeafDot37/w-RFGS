# -*- coding: utf-8 -*-
"""可行性探针：MVDR 口径（diffuse=True）下，带发射阵列时 paths.a 到底多大。

只为回答一个问题：本机 8 GB 卡能不能同时持有 16 个发射阵元的逐路径系数。
不求值、不存盘，只打印形状/耗时/显存。

用法::
    D:/anaconda3/envs/sionna_win/python.exe code/sim/_probe_w_mem.py --tx 4 --nrx 1
    D:/anaconda3/envs/sionna_win/python.exe code/sim/_probe_w_mem.py --tx 2 --nrx 4
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids             # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL                     # noqa: E402
from sim.gen_nist_field import load_scene                                     # noqa: E402
from sim.spectrum_gen import read_colmap_poses                                # noqa: E402
import sim.gen_nist_field as gen_nist_field                                   # noqa: E402


def gpu_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0]
    except Exception:                                        # noqa: BLE001
        return "?"


def strip_envmap_patch():
    src = Path(gen_nist_field.SCENE_XML).resolve()
    dst = src.parent / "_probe_scene_noenv.xml"
    txt = src.read_text(encoding="utf-8")
    new, n = re.subn(r'<emitter type="envmap".*?</emitter>', '', txt, flags=re.S)
    if n != 1:
        raise SystemExit(f"envmap 块匹配失败（n={n}）")
    dst.write_text(new, encoding="utf-8")
    gen_nist_field.SCENE_XML = dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", type=int, default=4, help="发射阵 = tx×tx")
    ap.add_argument("--nrx", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--interval_ns", type=float, default=0.1)
    args = ap.parse_args()

    strip_envmap_patch()
    print(f"[mem] GPU 启动 {gpu_mib()}", flush=True)
    import sionna.rt as rt
    from sim.spectrum_gen import paths_complex

    t0 = time.time()
    sc = load_scene(freq_hz=2.4e9, rewrite_materials=True, material_freq_hz=60e9)
    print(f"[mem] 场景载入 {time.time()-t0:.1f}s  GPU {gpu_mib()}", flush=True)

    sc.rx_array = rt.PlanarArray(num_rows=10, num_cols=10,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.tx_array = rt.PlanarArray(num_rows=args.tx, num_cols=args.tx,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True
    sc.frequency = 3.6e9

    poses = read_colmap_poses(POSE_ROOT)
    groups = group_by_pose(poses, read_ids(Path(POSE_ROOT) / "test_index.txt"))
    ids_g = groups[0][: args.nrx]

    sc.add(rt.Transmitter("tx", position=np.array([0.43418, -4.37863, 0.83672], np.float32),
                          orientation=np.array([-0.66153, 0.85055, 0.0], np.float32)))
    for k, i in enumerate(ids_g):
        c, ang = poses[i]
        sc.add(rt.Receiver(f"rx{k}", position=c,
                           orientation=np.array([ang, 0.0, 0.0], np.float32)))

    t1 = time.time()
    paths = rt.PathSolver(deterministic=True)(
        sc, max_depth=1, los=True, specular_reflection=True,
        diffuse_reflection=MVDR_PROTOCOL["diffuse"], refraction=False,
        diffraction=False, edge_diffraction=False,
        samples_per_src=int(args.samples), synthetic_array=True, seed=42)
    print(f"[mem] 求解 {time.time()-t1:.1f}s  GPU {gpu_mib()}", flush=True)

    a = paths_complex(paths)
    print(f"[mem] paths.a 形状 {tuple(a.shape)}  dtype {a.dtype}  "
          f"= {a.nbytes/2**20:.0f} MiB  GPU {gpu_mib()}", flush=True)
    tau = np.asarray(paths.tau)
    valid = np.asarray(paths.valid)
    print(f"[mem] tau {tau.shape}  valid 有效路径 "
          f"{int(valid.sum())} / {valid.size}", flush=True)
    print(f"[mem] 单接收点有效路径数 {int(valid[0].sum())}", flush=True)


if __name__ == "__main__":
    main()
