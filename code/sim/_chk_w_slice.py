# -*- coding: utf-8 -*-
"""单点检查：DrJit 张量的 `ar[:, n, :]` 到底取的是不是第 n 个阵元。

`_chk_w_equiv` 显示 GPU 合成相对误差 1.5（不是舍入）。uniform 权是纯求和，
连「求和」都错，那就只能是**切片取错了轴**。本脚本对同一份 `paths.a` 逐 n 比对
`asarray(ar[:, n, :])` 与 numpy 侧的 `re[:, n, :]`，并试 `ar[:, n:n+1, :]` 的
切片写法作为对照。

用法::
    python code/sim/_chk_w_slice.py --inst_id 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids     # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL             # noqa: E402
from sim.gen_mvdr_params import freeze_materials                      # noqa: E402
from sim.gen_nist_field import load_scene                             # noqa: E402
from sim.spectrum_gen import read_colmap_poses                        # noqa: E402
from sim._probe_w_mvdr import strip_envmap_patch                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="datasets/RF-3DGS_cbf_params/instances.json")
    ap.add_argument("--inst_id", type=int, default=0)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--tx_rows", type=int, default=4)
    ap.add_argument("--tx_cols", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--mat_freq", type=float, default=60e9)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import json
    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)
    poses = read_colmap_poses(args.poses_root)
    ids_g = group_by_pose(poses, read_ids(Path(args.poses_root) / "test_index.txt"))[0]

    strip_envmap_patch()
    import sionna.rt as rt

    sc = load_scene(freq_hz=2.4e9, rewrite_materials=True, material_freq_hz=args.mat_freq)
    freeze_materials(sc)
    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.tx_array = rt.PlanarArray(num_rows=args.tx_rows, num_cols=args.tx_cols,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True
    sc.frequency = float(inst["freq_hz"])
    sc.add(rt.Transmitter("tx", position=np.asarray(inst["xyz"], np.float32),
                          orientation=np.array([float(inst["yaw_abs_rad"]),
                                                float(inst["pitch_abs_rad"]), 0.0],
                                               np.float32), power_dbm=0.0))
    names = []
    for k, i in enumerate(ids_g):
        c, ang = poses[i]
        nm = f"rx{k}"
        sc.add(rt.Receiver(nm, position=c, orientation=np.array([ang, 0, 0], np.float32)))
        names.append(nm)
    path = rt.PathSolver(deterministic=True)(
        sc, max_depth=1, los=True, specular_reflection=True,
        diffuse_reflection=MVDR_PROTOCOL["diffuse"], refraction=False,
        diffraction=False, edge_diffraction=False,
        samples_per_src=int(args.samples), synthetic_array=True, seed=args.seed)

    ar = path.a[0][0, :, 0, :, :]
    ai = path.a[1][0, :, 0, :, :]
    re = np.asarray(ar, np.float32)
    im = np.asarray(ai, np.float32)
    print(f"  ar {type(ar).__name__} {tuple(ar.shape)}   numpy {re.shape}", flush=True)

    # 1) 逐阵元切片
    for n in (0, 1, 5, 15):
        s = np.asarray(ar[:, n, :], np.float32)
        d = float(np.max(np.abs(s - re[:, n, :])))
        s2 = np.asarray(ar[:, n:n + 1, :], np.float32)[:, 0, :]
        d2 = float(np.max(np.abs(s2 - re[:, n, :])))
        print(f"  n={n:2d}  [:,n,:] 偏差 {d:.3e}   [:,n:n+1,:] 偏差 {d2:.3e}", flush=True)

    # 2) 纯求和（uniform 权只差一个常数）
    tot_gpu = np.asarray(sum((ar[:, n, :] for n in range(16)), ar[:, 0, :] * 0.0),
                         np.float32)
    tot_np = re.sum(axis=1)
    print(f"  逐项相加 vs numpy sum 相对偏差 "
          f"{float(np.max(np.abs(tot_gpu - tot_np))) / float(np.max(np.abs(tot_np))):.3e}",
          flush=True)

    # 3) 逐项累加（bench 里实际用的写法）
    acc = ar[:, 0, :] * 1.0
    for n in range(1, 16):
        acc = acc + ar[:, n, :] * 1.0
    acc = np.asarray(acc, np.float32)
    print(f"  链式累加 vs numpy sum 相对偏差 "
          f"{float(np.max(np.abs(acc - tot_np))) / float(np.max(np.abs(tot_np))):.3e}",
          flush=True)

    # 4) 原样复刻 bench 的写法：两个表达式都建（实/虚），再一起取回
    def bench_aw(w):
        wr = w.real
        wi = w.imag
        aw_re = ar[:, 0, :] * float(wr[0])
        aw_im = ai[:, 0, :] * float(wi[0])
        for n in range(1, len(w)):
            if wi[n] == 0.0:
                aw_re = aw_re + ar[:, n, :] * float(wr[n])
                aw_im = aw_im + ai[:, n, :] * float(wr[n])
            else:
                aw_re = aw_re + (ar[:, n, :] * float(wr[n])
                                 - ai[:, n, :] * float(wi[n]))
                aw_im = aw_im + (ar[:, n, :] * float(wi[n])
                                 + ai[:, n, :] * float(wr[n]))
        return (np.asarray(aw_re, np.float32)
                + 1j * np.asarray(aw_im, np.float32))

    def host_aw(w):
        wr = w.real.astype(np.float32)
        wi = w.imag.astype(np.float32)
        return (np.einsum("mnp,n->mp", re, wr) - np.einsum("mnp,n->mp", im, wi)
                + 1j * (np.einsum("mnp,n->mp", re, wi)
                        + np.einsum("mnp,n->mp", im, wr)))

    from sim._bench_w_mvdr import beam_codebook
    _, WB = beam_codebook(3, 4, 4)
    for lab, w in zip(["uniform", "beam0", "beam1"], WB):
        g, h = bench_aw(w), host_aw(w)
        sc_ = float(np.max(np.abs(h)))
        print(f"  [{lab}] 两链同建 vs host einsum 相对偏差 "
              f"{float(np.max(np.abs(g - h))) / sc_:.3e}  "
              f"dtype {g.dtype}  max|g| {np.max(np.abs(g)):.3e} max|h| {sc_:.3e}",
              flush=True)

    # 5) 拆开：先建实部并立刻取回，再建虚部。若这样就对，说明是「两条大表达式
    #    同时存活」把 DrJit 的非连续切片临时数组挤出了变量缓存。
    w = WB[0]
    wr, wi = w.real, w.imag
    acc_r = ar[:, 0, :] * float(wr[0])
    for n in range(1, len(w)):
        acc_r = acc_r + ar[:, n, :] * float(wr[n])
    aw_re = np.asarray(acc_r, np.float32)
    acc_i = ai[:, 0, :] * float(wi[0])
    for n in range(1, len(w)):
        acc_i = acc_i + ai[:, n, :] * float(wi[n])
    aw_im = np.asarray(acc_i, np.float32)
    g2 = aw_re + 1j * aw_im
    h = host_aw(w)
    print(f"  [uniform] 分开建链 vs host einsum 相对偏差 "
          f"{float(np.max(np.abs(g2 - h))) / float(np.max(np.abs(h))):.3e}",
          flush=True)

    # 6) 变量缓存现状（DrJit 有全局上限；非连续切片会临时materialize）
    try:
        import drjit as dr
        print(f"  drjit {dr.__version__}  variable_cache_size "
              f"{dr.variable_cache_size()}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"  drjit 版本/缓存查询不可用: {e}", flush=True)


if __name__ == "__main__":
    main()
