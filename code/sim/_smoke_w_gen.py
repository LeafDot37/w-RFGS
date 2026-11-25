# -*- coding: utf-8 -*-
"""w 轴生产路径的三道闸（跑全量前必须先过）。

闸 1 符号：用真实多径的发射角，比「匹配权」与「共轭权」在同一条路径上的
    |Σ a_n w_n|。匹配权相干相加，共轭权互相抵消；差 < 10 dB 说明符号反了或
    阵元序对不上。
闸 2 等价：w = 单阵元 one-hot（第 0 个阵元为 1）。此时 `prepare_from_response`
    必须与旧 `prepare`（它就是把 tx-ant 轴切成 0）**逐位一致**。这条同时验了
    合成公式、阵元序与时间格代码；任何一处错都会在这里露出来。
闸 3 区分度：不同 w 的图必须真的不同（改图中位 dB），否则 w 轴没有信息。

用法::
    python code/sim/_smoke_w_gen.py --inst_id 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids       # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL              # noqa: E402
from sim.gen_mvdr_params import freeze_materials                       # noqa: E402
from sim.gen_nist_field import load_scene                              # noqa: E402
from sim.spectrum_gen import (compute_angle_matrices, MVDRMapper,      # noqa: E402
                             normalize_spectrum, read_colmap_poses)
from sim import w_codebook                                             # noqa: E402
from sim._probe_w_mvdr import strip_envmap_patch                       # noqa: E402

H, W = 200, 300


def synth(paths, k, w):
    """把第 k 台接收机按 w 合成 → aw [M², P]。与生产生成器同一段代码。"""
    ave = paths.a
    if isinstance(ave, (tuple, list)) and len(ave) == 2:
        re = np.asarray(ave[0][k, :, 0, :, :], np.float32)
        im = np.asarray(ave[1][k, :, 0, :, :], np.float32)
    else:
        aa = np.asarray(ave)
        if aa.ndim == 6:
            aa = aa[..., 0]
        ac = aa[k, :, 0, :, :]
        re = np.ascontiguousarray(ac.real, np.float32)
        im = np.ascontiguousarray(ac.imag, np.float32)
    n_path = re.shape[-1]
    n_ant = re.shape[1]
    Rt = np.ascontiguousarray(re.transpose(0, 2, 1)).reshape(-1, n_ant)
    It = np.ascontiguousarray(im.transpose(0, 2, 1)).reshape(-1, n_ant)
    wr = np.ascontiguousarray(w.real, np.float32)
    wi = np.ascontiguousarray(w.imag, np.float32)
    return ((Rt @ wr - It @ wi).reshape(re.shape[0], n_path)
            + 1j * (Rt @ wi + It @ wr).reshape(re.shape[0], n_path))


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
    ap.add_argument("--n_groups", type=int, default=1)
    args = ap.parse_args()

    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)

    poses = read_colmap_poses(args.poses_root)
    groups = group_by_pose(poses, read_ids(Path(args.poses_root) / "test_index.txt"))
    groups = groups[: args.n_groups]

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
    solver = rt.PathSolver(deterministic=True)

    theta, phi = compute_angle_matrices(W, H, 90.0)
    mapper = MVDRMapper(args.M, theta, phi, time_interval_ns=args.interval_ns,
                        db=True, loading=args.loading, method="chol")
    mapper.track_cond = False
    N = args.tx_rows * args.tx_cols

    for gi, ids_g in enumerate(groups):
        names = []
        for k, i in enumerate(ids_g):
            c, ang = poses[i]
            nm = f"rx{k}"
            sc.add(rt.Receiver(nm, position=c, orientation=np.array([ang, 0, 0], np.float32)))
            names.append(nm)
        t0 = time.time()
        paths = solver(sc, max_depth=1, los=True, specular_reflection=True,
                       diffuse_reflection=MVDR_PROTOCOL["diffuse"],
                       refraction=False, diffraction=False, edge_diffraction=False,
                       samples_per_src=int(args.samples),
                       synthetic_array=True, seed=args.seed)
        print(f"[闸] group {gi} 求迹 {time.time()-t0:.2f}s", flush=True)

        # ---- 闸 1：符号（峰位必须落在真实发射方向上） ----
        res = w_codebook.check_sign(paths, rx=0, n_paths=3, step=2.0,
                                    yaw_rad=float(inst["yaw_abs_rad"]),
                                    pitch_rad=float(inst["pitch_abs_rad"]))
        print("[闸1] 波束形成峰位 vs 真实发射方向：", flush=True)
        for th, ph, thp, php, d in res:
            print(f"        真 θ={th:6.2f}° φ={ph:7.2f}°  →  峰 θ={thp:6.1f}° "
                  f"φ={php:7.1f}°  角误差 {d:6.2f}°", flush=True)
        med = float(np.median([r[4] for r in res]))
        print(f"      → 角误差中位 {med:.2f}°  "
              f"{'PASS' if med < 5 else 'FAIL（相位号或阵元序不对）'}", flush=True)

        # ---- 闸 2：one-hot 等价 ----
        w_oh = np.zeros(N, np.complex128)
        w_oh[0] = 1.0
        aw_oh = synth(paths, 0, w_oh)
        prep_old = mapper.prepare(paths, rx=0)
        prep_new = mapper.prepare_from_response(
            aw_oh, np.asarray(paths.tau)[0, 0, :].astype(np.float64) / 1e-9,
            (np.asarray(paths.valid)[0, 0, :].astype(bool)
             if getattr(paths, "valid", None) is not None else None))
        d_re = float(np.max(np.abs(prep_new["re"] - prep_old["re"])))
        d_im = float(np.max(np.abs(prep_new["im"] - prep_old["im"])))
        same_idx = bool(np.array_equal(prep_new["idx"], prep_old["idx"]))
        scale = float(np.max(np.abs(prep_old["re"]))) or 1.0
        print(f"[闸2] one-hot vs prepare: max|Δre| {d_re:.3e} ({d_re/scale:.2e} 相对) "
              f"max|Δim| {d_im:.3e}  idx 相同 {same_idx}", flush=True)
        ok2 = d_re / scale < 1e-6 and d_im / scale < 1e-6 and same_idx
        print(f"      → {'PASS' if ok2 else 'FAIL'}", flush=True)

        # ---- 闸 3：不同 w 的图确有差别，并量单张耗时 ----
        labels = [w_codebook.UNIFORM] + w_codebook.train_labels()[1:4]
        imgs, ts = [], []
        for lab in labels:
            w = w_codebook.vector_for(lab, args.tx_rows, args.tx_cols)
            t1 = time.time()
            aw = synth(paths, 0, w)
            prep = mapper.prepare_from_response(
                aw, np.asarray(paths.tau)[0, 0, :].astype(np.float64) / 1e-9,
                (np.asarray(paths.valid)[0, 0, :].astype(bool)
                 if getattr(paths, "valid", None) is not None else None))
            u = normalize_spectrum(mapper.apply(prep, mapper.merge_matrix(prep)))
            ts.append(time.time() - t1)
            imgs.append(u)
            print(f"       w={lab:12s} 用时 {ts[-1]*1000:.0f} ms  "
                  f"u 范围 [{u.min():.3f}, {u.max():.3f}]", flush=True)
        for k in range(1, len(imgs)):
            mse = float(np.mean((imgs[k] - imgs[0]) ** 2))
            print(f"[闸3] w={labels[k]:12s} vs uniform 改图 "
                  f"{10*np.log10(1/max(mse,1e-12)):.2f} dB", flush=True)

        for nm in names:
            sc.remove(nm)


if __name__ == "__main__":
    main()
