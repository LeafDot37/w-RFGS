# -*- coding: utf-8 -*-
"""w 闸（MVDR 口径）：之前四道闸全在 CBF 口径下，而正在生产的数据集是 MVDR。

为什么要重做
--------------------------------------------------------------------------------
CBF 下图像对 w 是「逐路径增益的线性叠加」，所以能用「逐路径实数增益」代理
逐高斯标量场。**MVDR 不是线性的**：换 w 会改 R = x·xᴴ，自适应权向量跟着变，
图像对 w 的依赖不再可分解成逐路径缩放。因此：

  * 「w 改图 24.60 dB」这个读数不能直接搬到 MVDR；
  * 「逐路径实数增益重构 37.31 dB」在 MVDR 下结构上不成立，需要重新测
    （测的是同一构造在自适应权下的残差，只能当经验读数，不能当表达力证明）。

本探针回答三个前置问题
--------------------------------------------------------------------------------
  A) MVDR 下换 w 改不改图、改多少（对照同输入重跑噪声地板 0.365 dB）；
  B) 不同指向的波束之间分不分得开、随角间隔怎么衰减（可学性的角尺度）；
  C) 「只保留逐路径幅度比」的近似在 MVDR 下还剩多少（表达力的经验读数）。

内存与做法
--------------------------------------------------------------------------------
扩散散射开时单接收点约 16 万条路径；16 阵元的 paths.a 实测 1939 MiB（单接收
点）。所以**一次只放一个接收点**，逐接收点求解、当场算完当场丢。两遍：
  第 1 遍 每个接收点求一次解，造它的匹配波束 w = conj(u)/|u|、算 uniform 图；
  第 2 遍 每个接收点再求一次解，对全部 K 个波束算「精确图」与「只保留幅度图」。
求解 deterministic=True，两遍路径逐位一致。

用法::
    D:/anaconda3/envs/sionna_win/python.exe code/sim/_probe_w_mvdr.py \
        --inst_id 0 --n_groups 8 --out results/_probe_w_mvdr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids            # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL                   # noqa: E402
from sim.gen_mvdr_params import freeze_materials                            # noqa: E402
from sim.gen_nist_field import load_scene                                   # noqa: E402
from sim.spectrum_gen import (MVDRMapper, compute_angle_matrices,           # noqa: E402
                              normalize_spectrum, paths_complex,
                              read_colmap_poses)
import sim.gen_nist_field as gen_nist_field                                 # noqa: E402

H, W = 200, 300


def image_from(mapper, a_eff, tau, valid):
    """a_eff: (M², P) 合成后的逐天线复幅度 → 逐图 min-max 后的 [0,1] 图。"""
    shim = SimpleNamespace(a=a_eff[None, :, None, None, :],
                           tau=tau[None, None, :],
                           valid=valid[None, None, :])
    return normalize_spectrum(mapper(shim))


def strip_envmap_patch():
    src = Path(gen_nist_field.SCENE_XML).resolve()
    dst = src.parent / "_probe_scene_noenv.xml"
    txt = src.read_text(encoding="utf-8")
    new, n = re.subn(r'<emitter type="envmap".*?</emitter>', '', txt, flags=re.S)
    if n != 1:
        raise SystemExit(f"envmap 块匹配失败（n={n}），不要盲目继续")
    dst.write_text(new, encoding="utf-8")
    gen_nist_field.SCENE_XML = dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="datasets/RF-3DGS_cbf_params/instances.json")
    ap.add_argument("--inst_id", type=int, default=0)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--n_groups", type=int, default=8, help="每组 4 个接收点")
    ap.add_argument("--tx_rows", type=int, default=4)
    ap.add_argument("--tx_cols", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--mat_freq", type=float, default=60e9)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--interval_ns", type=float, default=0.1)
    ap.add_argument("--loading", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    strip_envmap_patch()
    import sionna.rt as rt

    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)
    xyz = np.asarray(inst["xyz"], np.float32)
    print(f"[w] 实例 {args.inst_id}: f={inst['freq_hz']/1e9:.4f} GHz", flush=True)

    poses = read_colmap_poses(args.poses_root)
    groups = group_by_pose(poses, read_ids(Path(args.poses_root) / "test_index.txt"))
    ids_all = [i for g in groups[: args.n_groups] for i in g]
    print(f"[w] {min(args.n_groups, len(groups))} 组 / {len(ids_all)} 个接收点", flush=True)

    t0 = time.time()
    sc = load_scene(freq_hz=2.4e9, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    n_frozen = freeze_materials(sc)
    print(f"[w] 场景载入 {time.time()-t0:.1f}s，材质冻结 {n_frozen} 个", flush=True)

    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.tx_array = rt.PlanarArray(num_rows=args.tx_rows, num_cols=args.tx_cols,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True
    sc.frequency = float(inst["freq_hz"])
    sc.add(rt.Transmitter("tx", position=xyz,
                          orientation=np.array([float(inst["yaw_abs_rad"]),
                                                float(inst["pitch_abs_rad"]), 0.0],
                                               np.float32),
                          power_dbm=0.0))
    solver = rt.PathSolver(deterministic=True)
    N_tx = args.tx_rows * args.tx_cols
    u_unif = np.ones(N_tx, np.complex128) / np.sqrt(N_tx)

    theta, phi = compute_angle_matrices(W, H, 90.0)
    mapper = MVDRMapper(args.M, theta, phi, time_interval_ns=args.interval_ns,
                        db=True, loading=args.loading, method="chol")
    mapper.track_cond = False                  # 每图一次 SVD，探针不需要

    def solve_for(c, ang):
        sc.add(rt.Receiver("rx0", position=c,
                           orientation=np.array([ang, 0.0, 0.0], np.float32)))
        try:
            paths = solver(sc, max_depth=1, los=True, specular_reflection=True,
                           diffuse_reflection=MVDR_PROTOCOL["diffuse"],
                           refraction=False, diffraction=False,
                           edge_diffraction=False,
                           samples_per_src=int(args.samples),
                           synthetic_array=True, seed=args.seed)
        finally:
            sc.remove("rx0")
        a = np.asarray(paths_complex(paths)[0, :, 0, :, :], np.complex64)  # (M², N_tx, P)
        tau = np.asarray(paths.tau)[0, 0, :].astype(np.float64)
        valid = np.asarray(paths.valid)[0, 0, :].astype(bool)
        th = np.asarray(paths.theta_t)[0, 0, :].astype(np.float64)
        ph = np.asarray(paths.phi_t)[0, 0, :].astype(np.float64)
        if not valid.all():
            a, tau, th, ph = a[:, :, valid], tau[valid], th[valid], ph[valid]
            valid = np.ones(tau.shape, bool)
        return a, tau, valid, th, ph

    # ---------------- 第 1 遍：造波束 + uniform 图 ----------------
    t1 = time.time()
    beams, bore, n_paths = [], [], np.zeros(len(ids_all), np.int64)
    imgs_unif = np.zeros((len(ids_all), H, W), np.float32)
    for r, i in enumerate(ids_all):
        c, ang = poses[i]
        a, tau, valid, th, ph = solve_for(c, ang)
        n_paths[r] = len(tau)
        g_unif = np.abs(np.einsum("mnp,n->mp", a, u_unif)).max(0)
        i_star = int(np.argmax(g_unif))
        u = a[0, :, i_star].astype(np.complex128)                  # 参考阵元的响应
        beams.append(np.conj(u) / max(np.abs(u).max(), 1e-30))
        bore.append((float(np.rad2deg(th[i_star])), float(np.rad2deg(ph[i_star]))))
        imgs_unif[r] = image_from(mapper, np.einsum("mnp,n->mp", a, u_unif), tau, valid)
        del a
        print(f"[w] 造束 r{r:02d} P={n_paths[r]:7d} 累计 {time.time()-t1:.0f}s", flush=True)

    beams = np.asarray(beams, np.complex64)                        # (K, N_tx)
    bore = np.asarray(bore, np.float64)
    K = beams.shape[0]
    np.save(out / "_beams.npy", beams)
    np.save(out / "_bore.npy", bore)
    np.save(out / "_imgs_unif.npy", imgs_unif)

    # ---------------- 第 2 遍：每个接收点 × 每个波束 ----------------
    t2 = time.time()
    R = len(ids_all)
    imgs = np.zeros((R, K, H, W), np.float16)
    imgs_mag = np.zeros((R, K, H, W), np.float16)
    gain_db = np.zeros((R, K), np.float64)
    cond = np.zeros((R, K), np.float64)
    t_map = 0.0
    for r, i in enumerate(ids_all):
        c, ang = poses[i]
        a, tau, valid, _, _ = solve_for(c, ang)
        a_u = np.einsum("mnp,n->mp", a, u_unif)
        p_u = float(np.sum(np.abs(a_u) ** 2))
        for k in range(K):
            a_w = np.einsum("mnp,n->mp", a, beams[k])
            t_m = time.time()
            imgs[r, k] = image_from(mapper, a_w, tau, valid).astype(np.float16)
            cond[r, k] = mapper.cond
            ratio = np.abs(a_w) / np.maximum(np.abs(a_u), 1e-30)
            imgs_mag[r, k] = image_from(mapper, a_u * ratio, tau, valid).astype(np.float16)
            t_map += time.time() - t_m
            gain_db[r, k] = 10 * np.log10(max(float(np.sum(np.abs(a_w) ** 2)), 1e-30) / p_u)
        del a, a_u
        print(f"[w] 成图 r{r:02d} {K} 束 累计 {time.time()-t2:.0f}s "
              f"(映射 {t_map:.0f}s) P={len(tau)}", flush=True)

    np.savez_compressed(out / "imgs.npz", imgs=imgs, imgs_mag=imgs_mag,
                        gain_db=gain_db, cond=cond, n_paths=n_paths,
                        names=np.asarray(ids_all, np.int64),
                        meta=np.asarray(json.dumps({
                            "inst_id": args.inst_id, "freq_hz": float(inst["freq_hz"]),
                            "tx_rows": args.tx_rows, "tx_cols": args.tx_cols,
                            "M": args.M, "interval_ns": args.interval_ns,
                            "loading": args.loading, "samples": args.samples,
                        }, ensure_ascii=False)))
    print(f"[w] 存 -> {out}  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
