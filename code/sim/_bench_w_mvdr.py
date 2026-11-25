# -*- coding: utf-8 -*-
"""波束权 w 轴的**开跑前闸**：一次求迹、K 个 w、量时间/体积/改图幅度/显存。

为什么要先量：生产数据集是 1×1 发射（w ≡ 1），要造 w 轴必须换 4×4 阵并在
**同一次求迹里循环 K 个 w**（事后补就得重求迹）。全量是 318 实例 × 640 视角 × K，
K=8 就是 1,694 万张 —— 先把这四件事测出来再决定 K。

码本（**与数据无关、可枚举**，这是「留出 w」有意义的前提）
--------------------------------------------------------------------------------
    w0 = uniform（1/√N）
    w1..w3 = 指向 TX 本地系 (θ=45°, φ=0°/120°/240°) 的匹配波束
`w_n = exp(-j2π(y_n·sinθ·sinφ + z_n·cosθ)) / √N`，y/z 为阵元坐标（波长单位）。
注意本地系 ⇒ 码本不随实例的 yaw/pitch 变，朝向只改整幅图的指向，两者不混。

内存
--------------------------------------------------------------------------------
每个接收点的 `paths.a` 是 [M², N_tx, P] 复数：M²=100、N_tx=16、P≈1.6e5
⇒ 主机侧一份约 2.0 GiB。**逐接收点取回一次**，然后：

    转置 (M²,N_tx,P) → (M²,P,N_tx)（一次，约 2 GiB 拷贝）
    每张图：aw = Rt@wr − It@wi + j(Rt@wi + It@wr)（两次 gemv 量级）

试过的两条岔路，都记在这里免得重走：
  · `np.einsum("mnp,n->mp")` 直接沿中间轴求和 —— 访问模式极差，655 ms/张
    （`optimize=True` 更糟，2962 ms/张）；
  · 把合成搬到 GPU 用 DrJit 逐元乘加 —— 单独测切片与求和都精确（偏差 0），
    但**两条表达式链同时存活时结果错**（相对偏差 1.5），根因未定位，故弃用。

用法（服务器 rfgen 环境）::
    python code/sim/_bench_w_mvdr.py --inst_id 0 --n_groups 16 --K 4 \
        --out results/_bench_w_mvdr
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from PIL import Image                                      # noqa: E402

from sim.gen_cbf_params import POSE_ROOT, group_by_pose, read_ids   # noqa: E402
from sim.gen_mvdr_params import PROTOCOL as MVDR_PROTOCOL          # noqa: E402
from sim.gen_mvdr_params import freeze_materials                   # noqa: E402
from sim.gen_nist_field import load_scene                          # noqa: E402
from sim.spectrum_gen import (_element_grid, MVDRMapper,           # noqa: E402
                              compute_angle_matrices, normalize_spectrum,
                              read_colmap_poses)
from sim._probe_w_mvdr import strip_envmap_patch                   # noqa: E402

H, W = 200, 300


def beam_codebook(K: int, nr: int, nc: int):
    """返回 (标签列表, [K, N_tx] 复权向量)。w0 恒为 uniform。"""
    y, z = _element_grid(nc) if nr == nc else _tx_grid(nr, nc)
    N = nr * nc
    labels = ["uniform"]
    out = [np.ones(N, np.complex128) / np.sqrt(N)]
    dirs = [(45.0, 0.0), (45.0, 120.0), (45.0, 240.0),
            (25.0, 60.0), (25.0, 180.0), (25.0, 300.0), (65.0, 0.0)]
    for th, ph in dirs[: max(0, K - 1)]:
        t, p = np.deg2rad(th), np.deg2rad(ph)
        phase = 2 * np.pi * (np.asarray(y) * np.sin(t) * np.sin(p)
                             + np.asarray(z) * np.cos(t))
        out.append(np.exp(-1j * phase) / np.sqrt(N))
        labels.append(f"theta{th:g}_phi{ph:g}")
    return labels, np.asarray(out, np.complex128)


def _tx_grid(nr: int, nc: int):
    """非方阵时的阵元坐标（波长单位，半波长间距）。"""
    ys = (np.arange(nc) - (nc - 1) / 2) * 0.5
    zs = (np.arange(nr) - (nr - 1) / 2) * 0.5
    Y, Z = np.meshgrid(ys, zs)
    return Y.ravel().astype(np.float64), Z.ravel().astype(np.float64)


def gpu_mem():
    try:
        return int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().split("\n")[0])
    except Exception:
        return -1


def png_bytes(u8: np.ndarray) -> int:
    import io
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG", optimize=False)
    return len(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="datasets/RF-3DGS_cbf_params/instances.json")
    ap.add_argument("--inst_id", type=int, default=0)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--n_groups", type=int, default=16, help="每组 4 个接收点")
    ap.add_argument("--K", type=int, default=4)
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

    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    inst = next(i for i in insts if int(i["id"]) == args.inst_id)
    print(f"[bench] inst{args.inst_id:03d} f={inst['freq_hz']/1e9:.4f} GHz "
          f"K={args.K} TX={args.tx_rows}x{args.tx_cols}", flush=True)

    poses = read_colmap_poses(args.poses_root)
    groups = group_by_pose(poses, read_ids(Path(args.poses_root) / "test_index.txt"))
    groups = groups[: args.n_groups]
    n_rx = sum(len(g) for g in groups)
    print(f"[bench] {len(groups)} 组 / {n_rx} 个接收点 × K={args.K} = "
          f"{n_rx * args.K} 张", flush=True)

    strip_envmap_patch()
    import sionna.rt as rt

    t0 = time.time()
    sc = load_scene(freq_hz=2.4e9, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    freeze_materials(sc)
    print(f"[bench] 场景载入 {time.time()-t0:.1f}s", flush=True)

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

    theta, phi = compute_angle_matrices(W, H, 90.0)
    mapper = MVDRMapper(args.M, theta, phi, time_interval_ns=args.interval_ns,
                        db=True, loading=args.loading, method="chol")
    mapper.track_cond = False

    labels, WB = beam_codebook(args.K, args.tx_rows, args.tx_cols)
    print(f"[bench] 码本: {labels}", flush=True)

    t_solve = t_copy = t_ein = t_map = 0.0
    n_done = 0
    d_psnr = {k: [] for k in range(1, len(WB))}
    sizes = {k: [] for k in range(len(WB))}
    peaks = []

    for gi, ids_g in enumerate(groups):
        names = []
        for k, i in enumerate(ids_g):
            c, ang = poses[i]
            nm = f"rx{k}"
            sc.add(rt.Receiver(nm, position=c,
                               orientation=np.array([ang, 0, 0], np.float32)))
            names.append(nm)
        t1 = time.time()
        path = solver(sc, max_depth=1, los=True, specular_reflection=True,
                      diffuse_reflection=MVDR_PROTOCOL["diffuse"],
                      refraction=False, diffraction=False,
                      edge_diffraction=False,
                      samples_per_src=int(args.samples),
                      synthetic_array=True, seed=args.seed)
        t_solve += time.time() - t1

        for r in range(len(ids_g)):
            t2 = time.time()
            # 取回一次（每接收点 2 GiB），随后**转置成 (M²,P,N_tx)**：
            # 在 (M²,N_tx,P) 上沿中间轴求和是 numpy 最差的访问模式
            # （实测 einsum 655 ms/张，且只在主机上，与 GPU 无关）；
            # 转置一次之后每张图就是一次 gemv，实测 607 ms/张、偏差 1.14e-07。
            ave = path.a
            if isinstance(ave, (tuple, list)) and len(ave) == 2:
                re = np.asarray(ave[0][r, :, 0, :, :], np.float32)   # (M², N_tx, P)
                im = np.asarray(ave[1][r, :, 0, :, :], np.float32)
            else:
                aa = ave
                if getattr(aa, "ndim", 0) == 6:
                    aa = aa[..., 0]
                ac = np.asarray(aa[r, :, 0, :, :])
                re = np.ascontiguousarray(ac.real, np.float32)
                im = np.ascontiguousarray(ac.imag, np.float32)
            n_path = re.shape[-1]
            Rt = np.ascontiguousarray(re.transpose(0, 2, 1)).reshape(-1, re.shape[1])
            It = np.ascontiguousarray(im.transpose(0, 2, 1)).reshape(-1, re.shape[1])
            tau = np.asarray(path.tau)[r, 0, :].astype(np.float64) / 1e-9
            valid = np.ones(len(tau), bool)
            if getattr(path, "valid", None) is not None:
                valid = np.asarray(path.valid)[r, 0, :].astype(bool)
            t_copy += time.time() - t2

            imgs = []
            for k, w in enumerate(WB):
                t3 = time.time()
                wr = np.ascontiguousarray(w.real, np.float32)
                wi = np.ascontiguousarray(w.imag, np.float32)
                aw = ((Rt @ wr - It @ wi).reshape(re.shape[0], n_path)
                      + 1j * (Rt @ wi + It @ wr).reshape(re.shape[0], n_path))
                t_ein += time.time() - t3
                t4 = time.time()
                prep = mapper.prepare_from_response(aw, tau, valid)
                u = normalize_spectrum(mapper.apply(prep, mapper.merge_matrix(prep)))
                u8 = np.round(u * 255).astype(np.uint8)
                t_map += time.time() - t4
                imgs.append(u)
                sizes[k].append(png_bytes(u8))
                n_done += 1
            for k in range(1, len(WB)):
                mse = float(np.mean((imgs[k] - imgs[0]) ** 2))
                d_psnr[k].append(10 * np.log10(1.0 / max(mse, 1e-12)))
            del re, im, Rt, It, aw

        for nm in names:
            sc.remove(nm)
        peaks.append(gpu_mem())
        if gi % 4 == 0 or gi == len(groups) - 1:
            print(f"[bench] group {gi:3d}  solve {t_solve:6.1f}s  copy {t_copy:6.1f}s  "
                  f"einsum {t_ein:5.1f}s  map {t_map:6.1f}s  gpu {peaks[-1]} MiB",
                  flush=True)

    per_img_map = t_map / max(n_done, 1)
    rep = {
        "inst_id": args.inst_id, "K": args.K,
        "tx": [args.tx_rows, args.tx_cols], "n_groups": len(groups),
        "n_receivers": n_rx, "n_images": n_done,
        "labels": labels,
        "t_solve_s": t_solve, "t_copy_s": t_copy, "t_einsum_s": t_ein, "t_map_s": t_map,
        "per_group_solve_s": t_solve / max(len(groups), 1),
        "per_image_map_s": per_img_map,
        "ms_per_image_total": 1000 * (t_solve + t_copy + t_ein + t_map) / max(n_done, 1),
        "png_kb": {labels[k]: float(np.mean(v)) / 1024 for k, v in sizes.items()},
        "delta_psnr_vs_uniform": {labels[k]: float(np.median(v))
                                  for k, v in d_psnr.items()} or {},
        "delta_psnr_p10": {labels[k]: float(np.percentile(v, 10))
                           for k, v in d_psnr.items()} or {},
        "gpu_mem_max_mib": int(max(peaks)) if peaks else -1,
        "gpu_mem_trace": peaks,
    }
    (out / "bench.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print("\n[bench] ==== 结论 ====", flush=True)
    print(f"  总 {n_done} 张 / {t_solve+t_copy+t_ein+t_map:.1f}s = "
          f"{rep['ms_per_image_total']:.0f} ms/张", flush=True)
    print(f"  求迹 {rep['per_group_solve_s']:.2f}s/组（{n_rx/len(groups):.0f} 接收点）  "
          f"取回 {t_copy/n_done*1000:.0f} ms/接收点  "
          f"einsum {t_ein/n_done*1000:.0f} ms/张  映射 {per_img_map*1000:.0f} ms/张",
          flush=True)
    for k in range(1, len(WB)):
        print(f"  w={labels[k]:16s} 改图中位 {np.median(d_psnr[k]):.2f} dB "
              f"(p10 {np.percentile(d_psnr[k],10):.2f})  "
              f"PNG {np.mean(sizes[k])/1024:.1f} KB", flush=True)
    print(f"  显存峰值 {rep['gpu_mem_max_mib']} MiB", flush=True)


if __name__ == "__main__":
    main()
