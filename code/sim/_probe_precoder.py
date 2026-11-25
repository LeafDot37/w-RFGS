# -*- coding: utf-8 -*-
"""闸 0/1 探针（GPU 侧）：把发射阵的**逐阵元路径系数**存下来，不落图像。

为什么这样设计
--------------------------------------------------------------------------------
本仓所有生成器都用 `sc.synthetic_array = True`（见 gen_cbf_params.py:225）。
Sionna 的语义是：路径只在阵列参考点追迹一次，阵元之间的差别由**解析阵列响应**
补到 `paths.a` 的 num_tx_ant 轴上。两个直接后果：

  1) 发射阵从 1 元变 16 元，**追迹成本基本不变**（不是 16 倍）；
  2) **路径集合与 w 无关** —— 换波束只是给每条路径乘一个复增益，
     不会新增/消失路径，所以"窄波束把接收视角打成全黑"在这里不会发生。

于是整个波束扫描可以在 numpy 里做完：一次求解 → 任意多个 w。
本脚本只负责求解，产物是一个 npz。

**逐位置组求解**，与 `gen_cbf_params.py` 的调用序列逐字相同（一组 4 个阵列 yaw
一次 solve），这样 1×1 的产物可以与已生成的 inst000 逐像素对拍。

显存：与 MVDR 生成共用一张 8 GB 卡，场景的 envmap 纹理要 1 GiB，会 OOM。
`--no_envmap` 会在加载前从 XML 里删掉 envmap 光源——它只影响渲染，**不参与
路径求解**（求解只从发射机出发，且 diffuse=False），所以对结果无影响；
这一点由「1×1 产物与 inst000 逐像素一致」来证明，不靠断言。

产物 npz 字段
--------------------------------------------------------------------------------
    a       (R, M², N_tx, P) complex64   逐阵元路径系数（已去掉 num_tx 轴，无效补 0）
    tau     (R, P)  float64              时延，秒
    valid   (R, P)  bool
    th_t    (R, P)  float64              离开角 theta（弧度）
    ph_t    (R, P)  float64              离开角 phi（弧度）
    names   (R,)    int                  接收点图号
用法（Windows sionna_win 环境）::

    D:/anaconda3/envs/sionna_win/python.exe code/sim/_probe_precoder.py \
        --out results/_probe_precoder/probe_1x1.npz --tx_rows 1 --tx_cols 1
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim import gen_nist_field                                        # noqa: E402
from sim.gen_cbf_params import (POSE_ROOT, PROTOCOL, freeze_materials,  # noqa: E402
                                group_by_pose, read_ids)
from sim.gen_nist_field import load_scene                             # noqa: E402
from sim.spectrum_gen import paths_complex, read_colmap_poses         # noqa: E402


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, total = [int(x) for x in out.split(",")]
        return total - used, total
    except Exception:
        return -1, -1


def strip_envmap_patch():
    """把 SCENE_XML 换成一份删掉 envmap 光源的副本（同目录，保证相对路径可用）。"""
    src = Path(gen_nist_field.SCENE_XML).resolve()
    dst = src.parent / "_probe_scene_noenv.xml"
    txt = src.read_text(encoding="utf-8")
    new, n = re.subn(r'<emitter type="envmap".*?</emitter>', '', txt, flags=re.S)
    if n != 1:
        raise SystemExit(f"envmap 块匹配失败（n={n}），不要盲目继续")
    dst.write_text(new, encoding="utf-8")
    gen_nist_field.SCENE_XML = dst
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    ap.add_argument("--instances", default="datasets/RF-3DGS_cbf_params/instances.json")
    ap.add_argument("--inst_id", type=int, default=0, help="取哪个实例的位置/朝向/频率")
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--n_groups", type=int, default=4,
                    help="取前几个位置组（每组 4 个阵列 yaw）")
    ap.add_argument("--tx_rows", type=int, default=4)
    ap.add_argument("--tx_cols", type=int, default=4)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--freq", type=float, default=2.4e9)
    ap.add_argument("--mat_freq", type=float, default=60e9)
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--tx_power_dbm", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_envmap", action="store_true",
                    help="加载前删掉 envmap（省 1 GiB 显存；不影响路径求解）")
    args = ap.parse_args()

    free, total = gpu_free_mib()
    print(f"[probe] GPU 空闲 {free} / {total} MiB", flush=True)
    if args.no_envmap:
        p = strip_envmap_patch()
        print(f"[probe] 已用无 envmap 的场景副本 {p.name}", flush=True)

    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)
    xyz = np.asarray(inst["xyz"], np.float32)
    print(f"[probe] 实例 {args.inst_id}: xyz={xyz.tolist()} "
          f"f={inst['freq_hz']/1e9:.4f} GHz", flush=True)

    poses = read_colmap_poses(args.poses_root)
    test_ids = read_ids(Path(args.poses_root) / "test_index.txt")
    groups = group_by_pose(poses, test_ids)[: args.n_groups]
    print(f"[probe] {len(groups)} 个位置组 / {sum(len(g) for g in groups)} 个接收点",
          flush=True)

    import sionna.rt as rt

    t0 = time.time()
    sc = load_scene(freq_hz=args.freq, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    n_frozen = freeze_materials(sc)
    print(f"[probe] 场景载入 {time.time()-t0:.1f}s，材质冻结 {n_frozen} 个", flush=True)

    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.tx_array = rt.PlanarArray(num_rows=args.tx_rows, num_cols=args.tx_cols,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True
    sc.frequency = float(inst["freq_hz"])

    orient = np.array([float(inst["yaw_abs_rad"]),
                       float(inst["pitch_abs_rad"]), 0.0], np.float32)
    sc.add(rt.Transmitter("tx", position=xyz, orientation=orient,
                          power_dbm=args.tx_power_dbm))
    solver = rt.PathSolver(deterministic=True)

    A_list, tau_list, val_list, th_list, ph_list, names = [], [], [], [], [], []
    t1 = time.time()
    for gi, ids_g in enumerate(groups):
        for k, i in enumerate(ids_g):
            c, ang = poses[i]
            sc.add(rt.Receiver(f"rx{k}", position=c,
                               orientation=np.array([ang, 0, 0], np.float32)))
        paths = solver(sc, max_depth=args.max_depth, los=True,
                       specular_reflection=True,
                       diffuse_reflection=PROTOCOL["diffuse"],
                       refraction=False, diffraction=False,
                       edge_diffraction=False,
                       samples_per_src=int(args.samples),
                       synthetic_array=True, seed=args.seed)
        a = paths_complex(paths)                        # (Rg, M², 1, N, P)
        A_list.append(a[:, :, 0, :, :])
        tau_list.append(np.asarray(paths.tau)[:, 0, :])
        val_list.append(np.asarray(paths.valid)[:, 0, :])
        th_list.append(np.asarray(paths.theta_t)[:, 0, :])
        ph_list.append(np.asarray(paths.phi_t)[:, 0, :])
        names.extend(ids_g)
        for k in range(len(ids_g)):
            sc.remove(f"rx{k}")
        print(f"[probe] group {gi}: {len(ids_g)} 接收点  "
              f"P={a.shape[-1]}  累计 {time.time()-t1:.1f}s", flush=True)
    sc.remove("tx")

    Pmax = max(x.shape[-1] for x in A_list)
    N = A_list[0].shape[2]
    M2 = A_list[0].shape[1]
    R = sum(x.shape[0] for x in A_list)

    def pad(arrs, shape_tail, dtype, fill):
        out = np.full((R, *shape_tail), fill, dtype=dtype)
        o = 0
        for x in arrs:
            out[o:o + x.shape[0], ..., :x.shape[-1]] = x if x.ndim == len(shape_tail) + 1 \
                else x[..., :x.shape[-1]]
            o += x.shape[0]
        return out

    A = np.zeros((R, M2, N, Pmax), np.complex64)
    tau = np.zeros((R, Pmax), np.float64)
    valid = np.zeros((R, Pmax), bool)
    th_t = np.zeros((R, Pmax), np.float64)
    ph_t = np.zeros((R, Pmax), np.float64)
    o = 0
    for a_, t_, v_, th_, ph_ in zip(A_list, tau_list, val_list, th_list, ph_list):
        n = a_.shape[0]
        p = a_.shape[-1]
        A[o:o + n, :, :, :p] = a_
        tau[o:o + n, :p] = t_
        valid[o:o + n, :p] = v_
        th_t[o:o + n, :p] = th_
        ph_t[o:o + n, :p] = ph_
        o += n

    print(f"[probe] 汇总 a {A.shape}  num_tx_ant={N}", flush=True)
    n_valid = valid.sum(axis=1)
    print(f"[probe] 每接收点有效路径 min/med/max = "
          f"{n_valid.min()}/{int(np.median(n_valid))}/{n_valid.max()}", flush=True)
    if N == 1:
        print("[probe] N=1：这是一次与 inst000 的对拍运行", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, a=A, tau=tau, valid=valid, th_t=th_t, ph_t=ph_t,
        names=np.asarray(names, np.int64),
        meta=np.asarray(json.dumps({
            "inst_id": args.inst_id, "xyz": xyz.tolist(),
            "freq_hz": float(inst["freq_hz"]),
            "tx_rows": args.tx_rows, "tx_cols": args.tx_cols, "M": args.M,
            "max_depth": args.max_depth, "samples": args.samples,
            "pose_root": args.poses_root, "seed": args.seed,
            "no_envmap": bool(args.no_envmap),
        }, ensure_ascii=False)))
    print(f"[probe] 存 -> {out}  ({out.stat().st_size/2**20:.1f} MiB)  "
          f"总耗时 {time.time()-t0:.0f}s", flush=True)
    free2, _ = gpu_free_mib()
    print(f"[probe] GPU 空闲（收尾）{free2} MiB", flush=True)


if __name__ == "__main__":
    main()
