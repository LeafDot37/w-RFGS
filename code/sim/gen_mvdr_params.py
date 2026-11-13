# -*- coding: utf-8 -*-
"""按「实例表」生成参数化 **MVDR（Capon）** 角域谱图。

与 `gen_cbf_params.py` 的关系
--------------------------------------------------------------------------------
逐行复制它的求解循环（实例表、分片、跳过已完成、材质冻结、原子写盘全部照旧），
只换三个地方：

  1. 后处理 `CBFMapper` → `MVDRMapper`（教程 cell 68 / 110 的 `MVDR_spectrum`）；
  2. **漫散射必须开**（`diffuse_reflection=True`）。MVDR 要先估 R = x·xᴴ，
     教程自己注明「时延格/多径数要 ≥ M²」；本场景镜面路径只有 6 条，R 秩 6，
     `R⁻¹` 会给出一片噪声而不是报错。实测开散射后每接收机约 10 万条路径。
  3. **对角加载**（`loading=0.01`，即 σ² = 1%·tr(R)/M²）。教程正文里的
     `inv(R)` 在本场景数值奇异（cond 1e18~1e20，Cholesky 直接失败），
     加载后收敛（路径数 40 万 vs 80 万，四张图相关 1.000/0.998/1.000/1.000）。

因此本数据集的目标是**正则化 MVDR**，不是教程字面上那个不可求解的 MVDR。

口径（其余与 CBF 参数集逐项相同）
--------------------------------------------------------------------------------
300×200 AoA 网格、fov 90°、10×10 UPA-RX、半波长、`max_depth=1`、
`samples_per_src=1e6`、时延格 **0.1 ns**、`MVDR_spec_dB`、逐图 min-max、jet 8bit RGB。
（教程 cell 110：`time_interval=0.1`，取 dB 谱，`per_view_normalization=True`。）

用法
--------------------------------------------------------------------------------
    # 吞吐/结构自检（不落盘）
    D:/anaconda3/envs/sionna_win/python.exe code/sim/gen_mvdr_params.py \
        --instances datasets/RF-3DGS_cbf_params/instances.json \
        --out datasets/RF-3DGS_mvdr_params --bench 1

    # 正式生成（可续跑；--skip_existing 按图粒度和 _gen.json 双重判断）
    ... --instances datasets/RF-3DGS_cbf_params/instances.json \
        --out datasets/RF-3DGS_mvdr_params --skip_existing

输出
--------------------------------------------------------------------------------
    <out>/inst<id:03d>/<img:05d>.png
    <out>/inst<id:03d>/_gen.json     # 每实例完成后原子写
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim import w_codebook                                       # noqa: E402
from sim.gen_nist_field import load_scene                        # noqa: E402
from sim.spectrum_gen import (ANGLES, MVDRMapper,               # noqa: E402
                              compute_angle_matrices,
                              normalize_spectrum, read_colmap_poses)

POSE_ROOT = "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"

# 整份协议写进 _gen.json 的 protocol_hash，改任何一项都会让旧目录不再被 --skip_existing 认
PROTOCOL = {
    "spectrum": "mvdr",
    "width": 300, "height": 200, "fov": 90.0,
    "max_depth": 1, "diffuse": True,
    "samples_per_src": 1_000_000, "time_interval_ns": 0.1,
    "mvdr_db": True, "mvdr_loading": 0.01, "mvdr_method": "chol",
    "materials": "tutorial @60GHz, callbacks frozen",
    "storage": "8bit grayscale = per-image min-max of 10log10(p)",
    "angles": [float(a) for a in ANGLES],
}


def read_ids(path):
    return [int(x) for x in Path(path).read_text().split()]


def group_by_pose(poses, ids):
    """同一位置（4 个阵列 yaw）的图号归组，保持顺序。"""
    groups, seen = [], {}
    for i in ids:
        c, _ = poses[i]
        key = tuple(np.round(c, 4))
        if key not in seen:
            seen[key] = len(groups)
            groups.append([])
        groups[seen[key]].append(i)
    return groups


def load_instances(path, roles=None, ids=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    insts = raw["instances"] if isinstance(raw, dict) else raw
    if roles:
        want = set(roles)
        insts = [i for i in insts if i.get("role", "train") in want]
    if ids:
        want = set(int(x) for x in ids)
        insts = [i for i in insts if int(i["id"]) in want]
    for i in insts:
        i.setdefault("role", "train")
        i.setdefault("views", "all")
        i.setdefault("yaw_abs_rad", 0.0)
        i.setdefault("pitch_abs_rad", 0.0)
    return sorted(insts, key=lambda i: int(i["id"]))


def _atomic_save_png(arr, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def _atomic_write_json(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def protocol_hash(extra: dict) -> str:
    blob = json.dumps({**PROTOCOL, **extra}, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def freeze_materials(sc):
    n = 0
    for m in sc.radio_materials.values():
        if m.frequency_update_callback is not None:
            m.frequency_update_callback = None
            n += 1
    return n


def image_stats(u8: np.ndarray) -> dict:
    x = u8.astype(np.float32) / 255.0
    g = x.mean(2) if x.ndim == 3 else x
    return {
        "mean_u": float(g.mean()),
        "std_u": float(g.std()),
        "sat0": float((g < 1e-6).mean()),      # 落最小值（jet 深蓝）的比例
        "sat1": float((g > 1 - 1e-6).mean()),  # 落最大值（jet 深红）的比例
    }


def _rss_mb():
    """Windows 进程工作集（MB），只给 MVDR_GPU_LOG 诊断用。"""
    import ctypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in
            ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]

    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 2 ** 20


def _post_one(mapper, prep, G, track_cond, png_path):
    """单张图的后处理：合并 → R → loading → cholesky → 三角解 → 归一化 → 存盘。

    纯 numpy/scipy，不碰 DrJit，所以可以在 worker 线程里跑；与主线程的下一组
    solve+prepare 重叠。返回 (u8, cond, png_path) 交给主线程统计。
    """
    sp_ = mapper.apply(prep, G, track_cond=track_cond)
    u8 = np.round(normalize_spectrum(sp_) * 255).astype(np.uint8)
    if png_path is not None:
        _atomic_save_png(u8, png_path)
    return u8, prep.get("cond"), png_path


def _flush(pending, stats, conds):
    """收掉已经提交的后处理：拿返回值做统计，然后清空队列。"""
    for fut in pending:
        u8, cd, _ = fut.result()
        stats.append(image_stats(u8))
        if cd is not None:
            conds.append(cd)
    pending.clear()


def _prep_with_w(mapper, paths, k: int, w: np.ndarray) -> dict:
    """把第 k 台接收机按波束权 w 合成，再交给 `prepare_from_response`。

    `mapper.prepare` 把 tx-ant 轴切成 0（只取单根发射天线），4×4 阵下必须先在
    这里做 Σ_n a_n·w_n。**沿中间轴求和是 numpy 最差的访问模式**：直接
    `np.einsum("mnp,n->mp")` 实测 655 ms/张；先转置成 (M²,P,N_tx) 再 gemv 是
    607 ms/张、偏差 1e-07。这段与 `_smoke_w_gen.synth` 是同一份代码，改动要同步。

    w = 单阵元 one-hot 时必须与 `prepare` **逐位一致**（已作为闸 2 验证）。
    """
    ave = paths.a
    if isinstance(ave, (tuple, list)) and len(ave) == 2:          # Sionna 2.x：(实, 虚)
        re = np.asarray(ave[0][k, :, 0, :, :], np.float32)        # (M², N_tx, P)
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
    aw = ((Rt @ wr - It @ wi).reshape(re.shape[0], n_path)
          + 1j * (Rt @ wi + It @ wr).reshape(re.shape[0], n_path))
    del re, im, Rt, It

    tau = np.asarray(paths.tau)[k, 0, :].astype(np.float64) / 1e-9
    valid = None
    if getattr(paths, "valid", None) is not None:
        valid = np.asarray(paths.valid)[k, 0, :].astype(bool)
    return mapper.prepare_from_response(aw, tau, valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="实例表 JSON（list 或 {instances:[...]}）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--views", choices=["auto", "train", "test", "all"], default="auto")
    ap.add_argument("--roles", nargs="+", default=None)
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--tx_pattern", default="tr38901")
    ap.add_argument("--tx_rows", type=int, default=1)
    ap.add_argument("--tx_cols", type=int, default=1)
    ap.add_argument("--freq", type=float, default=2.4e9, help="建场景时的频率（材质另按 --mat_freq）")
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--mat_freq", type=float, default=60e9)
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--interval", type=float, default=PROTOCOL["time_interval_ns"],
                    help="CIR 时间格宽（ns）")
    ap.add_argument("--loading", type=float, default=PROTOCOL["mvdr_loading"],
                    help="对角加载系数 ε（σ² = ε·tr(R)/M²）")
    ap.add_argument("--linear", action="store_true", help="输出线性 p 而不是 10log10(p)")
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--tx_power_dbm", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--jobs", type=int, default=0,
                    help="后处理线程数；0=主线程内联（默认，唯一不漏内存的配置）")
    ap.add_argument("--pipeline", type=int, default=1,
                    help="1=上一组后处理与下一组求解重叠；0=每组提交后立刻收（旧行为）")
    ap.add_argument("--gc_every", type=int, default=5,
                    help="每 N 个位置组做一次 gc + 清 Dr.Jit 分配缓存（0=不做）")
    ap.add_argument("--bench", type=int, default=0, help="只跑前 N 个位置组，不落盘")
    ap.add_argument("--limit", type=int, default=0, help="每实例最多 N 张（调试）")
    ap.add_argument("--max_instances", type=int, default=0,
                    help="本次最多处理 N 个实例后退出（供外层定期重启进程用，0=不限）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--w_from_instance", action="store_true",
                    help="按实例的 w_label 用波束权 w 合成（4×4 发射阵）。"
                         "实例缺 w_label 直接报错，不静默回退")
    ap.add_argument("--dry_run", action="store_true", help="只打印实例与视角数，不加载场景")
    args = ap.parse_args()

    insts = load_instances(args.instances, roles=args.roles, ids=args.ids)
    if args.num_shards > 1:
        insts = [i for i in insts if int(i["id"]) % args.num_shards == args.shard]
    if not insts:
        raise SystemExit("实例表为空（检查 --roles/--ids/--shard）")

    poses = read_colmap_poses(args.poses_root)
    split_dir = Path(args.poses_root)
    id_sets = {
        "train": read_ids(split_dir / "train_index.txt"),
        "test": read_ids(split_dir / "test_index.txt"),
        "all": sorted(poses),
    }

    plan = []
    for inst in insts:
        v = inst["views"] if args.views == "auto" else args.views
        ids = id_sets[v]
        groups = group_by_pose(poses, ids)
        if args.bench:
            groups = groups[: args.bench]
        plan.append((inst, v, groups, sum(len(g) for g in groups)))

    n_total = sum(p[3] for p in plan)
    print(f"[mvdr] {len(plan)} 个实例 / {n_total} 张  "
          f"interval={args.interval:g}ns loading={args.loading:g} "
          f"db={not args.linear}  out={args.out}", flush=True)
    for inst, v, groups, n in plan[:6]:
        print(f"  inst{int(inst['id']):03d} {inst['role']:14s} views={v:5s} "
              f"{n:5d} 张  yaw={np.rad2deg(inst['yaw_abs_rad']):7.2f}° "
              f"pitch={np.rad2deg(inst['pitch_abs_rad']):7.2f}° "
              f"f={inst['freq_hz']/1e9:.4f} GHz", flush=True)
    if args.dry_run:
        return

    import drjit as dr
    import sionna.rt as rt

    out_root = Path(args.out)
    sc = load_scene(freq_hz=args.freq, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    n_frozen = freeze_materials(sc)
    print(f"[mvdr] 材质冻结 {n_frozen} 个（60 GHz 的 eps/σ 不再随频率变）", flush=True)

    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True

    theta, phi = compute_angle_matrices(args.width, args.height, 90.0)
    mapper = MVDRMapper(args.M, theta, phi, time_interval_ns=args.interval,
                        db=not args.linear, loading=args.loading, method="chol")
    solver = rt.PathSolver(deterministic=True)
    # 进程级线程池：不每组重建（重建 3200 次本身也有开销），组内最多 4 个接收机
    # **不要**把后处理放进工作线程：实测在线程里跑 numpy/scipy 那一段，进程 RSS 会按
    # 260 MB/组线性涨（主线程内联完全平），56 组就把内存吃光，随后 Dr.Jit 的 pinned
    # 拷贝分配失败、整个进程崩。重建线程池、限 OpenBLAS 线程数都无效。故默认 jobs=0
    # =主线程内联；jobs>0 保留给内存充裕的机器。
    pool = ThreadPoolExecutor(max_workers=args.jobs) if args.jobs > 0 else None
    pending = []          # 已提交、尚未收取的后处理 future

    t_start = time.time()
    n_done = 0
    n_inst_done = 0
    for inst, v, groups, n_expect in plan:
        iid = int(inst["id"])
        idir = out_root / f"inst{iid:03d}"
        gpath = idir / "_gen.json"
        w_label = inst.get("w_label") if args.w_from_instance else None
        if args.w_from_instance and not w_label:
            raise SystemExit(f"inst{iid:03d} 缺 w_label（--w_from_instance 必须逐个实例给）")
        phash = protocol_hash({
            "tx_pattern": inst.get("tx_pattern", args.tx_pattern),
            "M": args.M, "protocol_freq_hz": args.freq, "mat_freq_hz": args.mat_freq,
            "interval": args.interval, "loading": args.loading,
            "linear": bool(args.linear),
            # w 轴：发射阵形状与权都进 hash，否则同一目录会被旧结果误判为已完成
            "tx_array": [int(inst.get("tx_rows", args.tx_rows)),
                         int(inst.get("tx_cols", args.tx_cols))],
            "w_label": w_label,
        })
        if args.skip_existing and not args.bench and gpath.exists():
            old = json.loads(gpath.read_text(encoding="utf-8"))
            if (old.get("complete") and old.get("n_images") == n_expect
                    and old.get("protocol_hash") == phash):
                print(f"[mvdr] inst{iid:03d} 已完成，跳过", flush=True)
                continue

        if not args.bench:
            idir.mkdir(parents=True, exist_ok=True)

        sc.frequency = float(inst["freq_hz"])

        pattern = inst.get("tx_pattern", args.tx_pattern)
        nr = int(inst.get("tx_rows", args.tx_rows))
        nc = int(inst.get("tx_cols", args.tx_cols))
        sc.tx_array = rt.PlanarArray(num_rows=nr, num_cols=nc,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern=pattern, polarization="V")
        orient = np.array([float(inst["yaw_abs_rad"]),
                           float(inst["pitch_abs_rad"]), 0.0], np.float32)
        sc.add(rt.Transmitter("tx", position=np.asarray(inst["xyz"], np.float32),
                              orientation=orient,
                              power_dbm=args.tx_power_dbm))

        w_vec = None
        if w_label:
            w_vec = w_codebook.vector_for(str(w_label), nr, nc)
            if nr * nc == 1:
                raise SystemExit("--w_from_instance 但发射阵是 1×1：w 无处施加")

        t_inst = time.time()
        n_inst = 0
        stats = []
        valid_frac = []
        conds = []
        for gi, ids_g in enumerate(groups):
            todo = ids_g
            if args.skip_existing and not args.bench:
                todo = [i for i in ids_g if not (idir / f"{i:05d}.png").exists()]
                if not todo:
                    continue
            names = []
            for k, i in enumerate(todo):
                c, ang = poses[i]
                nm = f"rx{k}"
                sc.add(rt.Receiver(nm, position=c,
                                   orientation=np.array([ang, 0, 0], np.float32)))
                names.append(nm)
            t0 = time.time()
            paths = solver(sc, max_depth=args.max_depth, los=True,
                           specular_reflection=True,
                           diffuse_reflection=PROTOCOL["diffuse"],
                           refraction=False, diffraction=False,
                           edge_diffraction=False,
                           samples_per_src=int(args.samples),
                           synthetic_array=True, seed=args.seed)
            t1 = time.time()
            if os.environ.get("MVDR_GPU_LOG"):
                import subprocess
                u = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip()
                print(f"[gpu] group {gi} gpu={u}MiB rss={_rss_mb():.0f}MB", flush=True)
            pv = np.asarray(paths.valid)
            if pv.size:
                valid_frac.append(float(pv.mean()))
            # 主线程：把每个接收机的复幅度与时间格索引取回主机（DrJit 拷贝不保证
            # 线程安全），并建一次合并矩阵 G。
            if w_vec is None:
                preps = [mapper.prepare(paths, rx=k) for k in range(len(todo))]
            else:
                preps = [_prep_with_w(mapper, paths, k, w_vec) for k in range(len(todo))]
            base = preps[0]
            Gs = []
            for p in preps:
                same = (p["n_bins"] == base["n_bins"]
                        and np.array_equal(p["idx"], base["idx"]))
                Gs.append(mapper.merge_matrix(base) if same
                          else mapper.merge_matrix(p))

            # 默认 pool=None：后处理在主线程内联（唯一不涨内存的配置，见上方注释）。
            # 若显式给了 --jobs>0，则提交给进程级线程池、此处不等待，让「上一组算图
            # 写盘」与「下一组求解取数」重叠——快约 20%，但会线性涨内存。
            for k in range(len(todo)):
                png = None if args.bench else (idir / f"{todo[k]:05d}.png")
                if pool is None:
                    # 内联：后处理在主线程跑完再进下一组
                    from concurrent.futures import Future
                    f = Future()
                    f.set_result(_post_one(mapper, preps[k], Gs[k], k == 0, png))
                    pending.append(f)
                else:
                    pending.append(pool.submit(_post_one, mapper, preps[k], Gs[k],
                                               k == 0, png))
            del preps, Gs, paths
            # 每组都会留下**引用环**（Sionna 的 paths/receiver 与场景互相引用），
            # 环里的 Dr.Jit 张量只有在 gc 回收该环时才释放；进程跑长以后分代回收
            # 迟迟不触发，显存就按「路径数 × 组数」一直涨——实测 1e6 采样 44 组
            # 就 OOM，降到 2e5 能撑到 290 组（漏的量与路径数成正比）。因此每组定时
            # gc；顺手清一次 Dr.Jit 的分配缓存（它按尺寸留存，尺寸又逐组变化）。
            if args.gc_every and (gi + 1) % args.gc_every == 0:
                gc.collect()
                dr.flush_malloc_cache()
            n_inst += len(todo)
            for nm in names:
                sc.remove(nm)
            if args.bench or not args.pipeline or pool is None:
                # bench 要逐组打印 cond；没有线程池或流水线时也就没有可重叠的东西
                _flush(pending, stats, conds)
            if os.environ.get("MVDR_GPU_LOG") and gi % 5 == 0:
                import subprocess
                u = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip()
                print(f"[gpu] group {gi}  {u} MiB", flush=True)
            if args.bench:
                print(f"[bench] inst{iid:03d} group {gi}: {len(todo)} 张  "
                      f"solve {t1-t0:.3f}s  map {time.time()-t1:.3f}s  "
                      f"paths {int(np.sum(pv)) if pv.size else 0}  "
                      f"cond(R) {conds[-1] if conds else float('nan'):.2e}", flush=True)
            elif gi % 20 == 0:
                el = time.time() - t_start
                print(f"[mvdr] inst{iid:03d} group {gi}/{len(groups)}  "
                      f"{n_done + n_inst} 张  {el/60:.1f} min", flush=True)
            if args.limit and n_inst >= args.limit:
                break

        _flush(pending, stats, conds)     # 本实例最后一组
        sc.remove("tx")
        n_done += n_inst

        if not args.bench:
            n_on_disk = len(list(idir.glob("*.png")))
            agg = {k: float(np.mean([s[k] for s in stats])) for k in
                   ("mean_u", "std_u", "sat0", "sat1")} if stats else {}
            _atomic_write_json({
                "id": iid, "role": inst["role"], "views": v,
                "xyz": list(map(float, inst["xyz"])),
                "yaw_abs_rad": float(inst["yaw_abs_rad"]),
                "pitch_abs_rad": float(inst["pitch_abs_rad"]),
                "freq_hz": float(inst["freq_hz"]),
                "tx_pattern": pattern, "tx_rows": nr, "tx_cols": nc,
                "n_images": n_on_disk, "n_expected": n_expect,
                "complete": n_on_disk == n_expect,
                "image_stats": agg,
                "valid_path_fraction": float(np.mean(valid_frac)) if valid_frac else None,
                "cond_R_median": float(np.median(conds)) if conds else None,
                "cond_R_max": float(np.max(conds)) if conds else None,
                "protocol_hash": phash,
                "elapsed_s": time.time() - t_inst,
            }, gpath)
            print(f"[mvdr] inst{iid:03d} 完成 {n_on_disk}/{n_expect} 张  "
                  f"{time.time()-t_inst:.0f}s", flush=True)
        n_inst_done += 1
        if args.max_instances and n_inst_done >= args.max_instances:
            print(f"[mvdr] 已达 --max_instances {args.max_instances}，退出（外层重启续跑）",
                  flush=True)
            break
        if args.limit and n_inst >= args.limit:
            break

    if pool is not None:
        pool.shutdown()
    el = time.time() - t_start
    if args.bench:
        print(f"[bench] 合计 {n_done} 张 / {el:.1f}s = "
              f"{el/max(n_done,1)*1000:.0f} ms/张；"
              f"按此推算 571520 张需 {571520*el/max(n_done,1)/3600:.1f} h（单进程）",
              flush=True)
    else:
        print(f"[mvdr] 完成 {n_done} 张，用时 {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
