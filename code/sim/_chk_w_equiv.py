# -*- coding: utf-8 -*-
"""核验：GPU 侧合成的 `aw` 与主机 einsum 的 `aw` 是否同一回事。

为什么要单独查：把 einsum 换成 DrJit 逐元乘加后，`_bench_w_mvdr` 的「改图中位」
从 17.94 dB 掉到 16.86 dB、PNG 也从 10.4 KB 变成 11.4 KB。**图上真变了**。
两种可能：
  ① 只是 float32 舍入 —— MVDR 里有 R⁻¹，条件数大，微扰可能被放大；
  ② 合成写错了 —— 那所有基于它的数都不能用。
本脚本对同一个接收点、同一个 w，分别用两条路算 `aw` 与最终图，直接对拍。

判据：`rel_diff_aw` ~1e-7 ⇒ 数值噪声；图像 PSNR 如果仍然很低（< 30 dB），
说明是**映射放大器**而不是合成错 —— 这时要再和 float64 参考比一次，看谁更接近。

用法::
    python code/sim/_chk_w_equiv.py --inst_id 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim._bench_w_mvdr import beam_codebook, png_bytes                  # noqa: E402
from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids      # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL              # noqa: E402
from sim.gen_mvdr_params import freeze_materials                       # noqa: E402
from sim.gen_nist_field import load_scene                              # noqa: E402
from sim.spectrum_gen import (compute_angle_matrices, MVDRMapper,      # noqa: E402
                             normalize_spectrum, read_colmap_poses)
from sim._probe_w_mvdr import strip_envmap_patch                       # noqa: E402

H, W = 200, 300


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
    ap.add_argument("--interval_ns", type=float, default=0.1)
    ap.add_argument("--loading", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)

    poses = read_colmap_poses(args.poses_root)
    groups = group_by_pose(poses, read_ids(Path(args.poses_root) / "test_index.txt"))
    ids_g = groups[0]

    strip_envmap_patch()
    import sionna.rt as rt

    sc = load_scene(freq_hz=2.4e9, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
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
                                               np.float32),
                          power_dbm=0.0))
    solver = rt.PathSolver(deterministic=True)

    names = []
    for k, i in enumerate(ids_g):
        c, ang = poses[i]
        nm = f"rx{k}"
        sc.add(rt.Receiver(nm, position=c, orientation=np.array([ang, 0, 0], np.float32)))
        names.append(nm)
    path = solver(sc, max_depth=1, los=True, specular_reflection=True,
                  diffuse_reflection=MVDR_PROTOCOL["diffuse"],
                  refraction=False, diffraction=False, edge_diffraction=False,
                  samples_per_src=int(args.samples),
                  synthetic_array=True, seed=args.seed)

    theta, phi = compute_angle_matrices(W, H, 90.0)
    mapper = MVDRMapper(args.M, theta, phi, time_interval_ns=args.interval_ns,
                        db=True, loading=args.loading, method="chol")
    mapper.track_cond = False

    labels, WB = beam_codebook(3, args.tx_rows, args.tx_cols)
    r = 0
    ave = path.a
    ar = ave[0][r, :, 0, :, :]
    ai = ave[1][r, :, 0, :, :]
    tau = np.asarray(path.tau)[r, 0, :].astype(np.float64) / 1e-9
    valid = np.ones(len(tau), bool)
    if getattr(path, "valid", None) is not None:
        valid = np.asarray(path.valid)[r, 0, :].astype(bool)

    print(f"  ar dtype={type(ar).__name__} shape={ar.shape}", flush=True)

    # 主机侧只取一次，两条路共用（对拍阶段不在乎这 2 GiB）
    re = np.asarray(ar, np.float32)
    im = np.asarray(ai, np.float32)
    print(f"  |a| 中位 {np.median(np.abs(re + 1j * im)):.4e}", flush=True)

    def to_img(aw):
        prep = mapper.prepare_from_response(aw, tau, valid)
        u = normalize_spectrum(mapper.apply(prep, mapper.merge_matrix(prep)))
        return u

    for k, w in enumerate(WB):
        wr = w.real.astype(np.float32)
        wi = w.imag.astype(np.float32)

        # 路 A：主机 einsum
        aw_host = (np.einsum("mnp,n->mp", re, wr) - np.einsum("mnp,n->mp", im, wi)
                   + 1j * (np.einsum("mnp,n->mp", re, wi)
                           + np.einsum("mnp,n->mp", im, wr)))

        # 路 B：DrJit 逐元乘加（bench 里用的那条）
        aw_re = ar[:, 0, :] * float(w.real[0])
        aw_im = ai[:, 0, :] * float(w.imag[0])
        for n in range(1, len(w)):
            w_r, w_i = float(w.real[n]), float(w.imag[n])
            if w_i == 0.0:
                aw_re = aw_re + ar[:, n, :] * w_r
                aw_im = aw_im + ai[:, n, :] * w_r
            else:
                aw_re = aw_re + (ar[:, n, :] * w_r - ai[:, n, :] * w_i)
                aw_im = aw_im + (ar[:, n, :] * w_i + ai[:, n, :] * w_r)
        aw_gpu = np.asarray(aw_re, np.float32) + 1j * np.asarray(aw_im, np.float32)

        # 路 C：float64 参考（只在这一步用，验证谁更准）
        aw_64 = (np.einsum("mnp,n->mp", re.astype(np.float64), w.real)
                 - np.einsum("mnp,n->mp", im.astype(np.float64), w.imag)
                 + 1j * (np.einsum("mnp,n->mp", re.astype(np.float64), w.imag)
                         + np.einsum("mnp,n->mp", im.astype(np.float64), w.real)))

        scale = float(np.max(np.abs(aw_64)))
        d_host = float(np.max(np.abs(aw_host - aw_64))) / scale
        d_gpu = float(np.max(np.abs(aw_gpu - aw_64))) / scale
        d_hg = float(np.max(np.abs(aw_host - aw_gpu))) / scale

        uA, uB = to_img(aw_host), to_img(aw_gpu)
        mse = float(np.mean((uA - uB) ** 2))
        p = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"  w={labels[k]:16s} 相对误差 host {d_host:.2e}  gpu {d_gpu:.2e}  "
              f"host↔gpu {d_hg:.2e} | 图 PSNR {p:6.2f} dB  "
              f"max|Δu| {np.max(np.abs(uA - uB)):.3e}  "
              f"PNG {png_bytes((uA*255).round().astype(np.uint8))} / "
              f"{png_bytes((uB*255).round().astype(np.uint8))} B", flush=True)


if __name__ == "__main__":
    main()
