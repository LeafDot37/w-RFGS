# -*- coding: utf-8 -*-
"""按「实例表」生成参数化 CBF 角域谱图（发射源位置 + 朝向 + 频率）。

与 `gen_cbf_spectra.py` 的关系
--------------------------------------------------------------------------------
`gen_cbf_spectra.py` 是历史 iso 口径生成器，**本文件不改它**。这里复制它的
求解循环，只把「发射源」从「一个位置」换成「一条实例」：

    实例 = (位置, 绝对朝向, 频率, 视角集)

口径与旧生成器逐项相同：300×200 AoA 网格、fov 90°、10×10 UPA-RX、半波长、
`max_depth=1`、`diffuse=False`、`samples_per_src=1e6`、0.5 ns CIR 格、
逐图 min-max、8bit 灰度。唯一变化：

  * TX 阵列 `--tx_pattern tr38901`（默认），朝向按实例设置；
  * `sc.frequency` 按实例设置；
  * **材质在 60 GHz 构造后冻结**（见下）。

材质冻结（关键）
--------------------------------------------------------------------------------
`load_scene` 改写出来的材质里，15 个仍带 `frequency_update_callback`。
`Scene.frequency` 的 setter 会对每个材质调 `frequency_update()`，把这些材质的
eps/σ 换回 ITU 模型（实测混凝土电导率 3.719 → 0.1498 @4.5 GHz）。那样频率轴
就同时改了材质和传播，结论不可归因。

因此：`load_scene(...)` 之后、**任何** `sc.frequency = f` 之前，先把所有材质的
回调置 None（setter 会立刻触发一次 `frequency_update()`，但值为 None 时直接
return，不覆写）。此后频率只影响传播相位与 Fresnel 的 ω 依赖。

用法
--------------------------------------------------------------------------------
    # 吞吐/结构自检（不落盘）
    D:/anaconda3/envs/sionna_win/python.exe code/sim/gen_cbf_params.py \
        --instances results/_gate_params/instances.json --out results/_gate_params/data \
        --bench 2

    # 正式生成（可分片并行）
    ... --instances datasets/RF-3DGS_cbf_params/instances.json \
        --out datasets/RF-3DGS_cbf_params --shard 0 --num_shards 4 --skip_existing

输出
--------------------------------------------------------------------------------
    <out>/inst<id:03d>/<img:05d>.png
    <out>/inst<id:03d>/_gen.json     # 每实例完成后原子写
"""
from __future__ import annotations

import argparse
import hashlib
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

from sim import w_codebook                                       # noqa: E402
from sim.gen_nist_field import load_scene                        # noqa: E402
from sim.spectrum_gen import (ANGLES, CBFMapper,                # noqa: E402
                              compute_angle_matrices,
                              normalize_spectrum, read_colmap_poses)

POSE_ROOT = "datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100"

# 与旧生成器一致的固定协议；写进 _gen.json 的 protocol_hash
PROTOCOL = {
    "width": 300, "height": 200, "fov": 90.0,
    "max_depth": 1, "diffuse": False,
    "samples_per_src": 1_000_000, "time_interval_ns": 0.5,
    "materials": "tutorial @60GHz, callbacks frozen",
    "storage": "8bit grayscale = per-image min-max",
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


def _aw_with_w(paths, k: int, w: np.ndarray):
    """把第 k 台接收机的 4×4 发射阵响应按波束权 w 合成 → (aw, tau_ns, valid)。

    `CBFMapper.__call__` 走 `cir_from_paths`，那条路把 tx-ant 轴切成 0（只取单根
    发射天线），4×4 阵下 w 会被**静默丢掉**。所以先在这里做 Σ_n a_n·w_n，再交给
    `CBFMapper.from_response`。

    实现与 `gen_mvdr_params._prep_with_w` 的合成段逐行相同：沿中间轴求和是 numpy
    最差访问模式，直接 `einsum("mnp,n->mp")` 655 ms/张，先转置成 (M²,P,N_tx) 再
    gemv 607 ms/张、偏差 1e-07。两处改动要同步。
    """
    ave = paths.a
    if isinstance(ave, (tuple, list)) and len(ave) == 2:      # Sionna 2.x：(实, 虚)
        re = np.asarray(ave[0][k, :, 0, :, :], np.float32)    # (M², N_tx, P)
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
    tau = np.asarray(paths.tau)[k, 0, :].astype(np.float64) / 1e-9
    valid = None
    if getattr(paths, "valid", None) is not None:
        valid = np.asarray(paths.valid)[k, 0, :].astype(bool)
    return aw, tau, valid


def freeze_materials(sc):
    n = 0
    for m in sc.radio_materials.values():
        if m.frequency_update_callback is not None:
            m.frequency_update_callback = None
            n += 1
    return n


def image_stats(u8: np.ndarray) -> dict:
    x = u8.astype(np.float32) / 255.0
    return {
        "mean_u": float(x.mean()),
        "std_u": float(x.std()),
        "sat0": float((u8 == 0).mean()),       # 落 0 的比例
        "sat1": float((u8 == 255).mean()),     # 落 255 的比例
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="实例表 JSON（list 或 {instances:[...]}）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses_root", default=POSE_ROOT)
    ap.add_argument("--views", choices=["auto", "train", "test", "all"], default="auto")
    ap.add_argument("--roles", nargs="+", default=None)
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--tx_pattern", default="tr38901")
    ap.add_argument("--tx_rows", type=int, default=1, help="TX 阵元行数（1×1=全向宽波束）")
    ap.add_argument("--tx_cols", type=int, default=1, help="TX 阵元列数（4×4≈16° 窄波束）")
    ap.add_argument("--freq", type=float, default=2.4e9, help="建场景时的频率（材质另按 --mat_freq）")
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--mat_freq", type=float, default=60e9)
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--tx_power_dbm", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="只跑前 N 个位置组，不落盘")
    ap.add_argument("--limit", type=int, default=0, help="每实例最多 N 张（调试）")
    ap.add_argument("--views_limit", type=int, default=0,
                    help="每个实例只取视角列表的前 N 个（与 --limit 不同："
                         "它是**整轮**跑完就停，这里是逐实例截断，可跑多实例小闸）")
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

    # 预先把每个实例的视角组算好，dry-run 也能报数
    plan = []
    for inst in insts:
        v = inst["views"] if args.views == "auto" else args.views
        ids = id_sets[v]
        if args.views_limit:
            ids = ids[: int(args.views_limit)]
        groups = group_by_pose(poses, ids)
        if args.bench:
            groups = groups[: args.bench]
        plan.append((inst, v, groups, sum(len(g) for g in groups)))

    n_total = sum(p[3] for p in plan)
    print(f"[params] {len(plan)} 个实例 / {n_total} 张  "
          f"tx_pattern={args.tx_pattern}  out={args.out}", flush=True)
    for inst, v, groups, n in plan[:6]:
        print(f"  inst{int(inst['id']):03d} {inst['role']:14s} views={v:5s} "
              f"{n:5d} 张  yaw={np.rad2deg(inst['yaw_abs_rad']):7.2f}° "
              f"pitch={np.rad2deg(inst['pitch_abs_rad']):7.2f}° "
              f"f={inst['freq_hz']/1e9:.4f} GHz", flush=True)
    if args.dry_run:
        return

    import sionna.rt as rt

    out_root = Path(args.out)
    sc = load_scene(freq_hz=args.freq, rewrite_materials=True,
                    material_freq_hz=args.mat_freq)
    n_frozen = freeze_materials(sc)
    print(f"[params] 材质冻结 {n_frozen} 个（60 GHz 的 eps/σ 不再随频率变）", flush=True)

    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    sc.synthetic_array = True

    theta, phi = compute_angle_matrices(args.width, args.height, 90.0)
    mapper = CBFMapper(args.M, theta, phi, time_interval_ns=PROTOCOL["time_interval_ns"])
    # 确定性：同样输入逐位可复现（跨进程也一样）
    solver = rt.PathSolver(deterministic=True)

    t_start = time.time()
    n_done = 0
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
            # 发射阵形状与 w 都进 hash：否则同一目录会被旧结果误判为已完成
            "tx_array": [int(inst.get("tx_rows", args.tx_rows)),
                         int(inst.get("tx_cols", args.tx_cols))],
            "w_label": w_label,
        })
        if args.skip_existing and not args.bench and gpath.exists():
            old = json.loads(gpath.read_text(encoding="utf-8"))
            if (old.get("complete") and old.get("n_images") == n_expect
                    and old.get("protocol_hash") == phash):
                print(f"[params] inst{iid:03d} 已完成，跳过", flush=True)
                continue

        if not args.bench:
            idir.mkdir(parents=True, exist_ok=True)

        # 频率：可选先设一个「预热」频率，专门验材质冻结（V13）
        pre = inst.get("_pre_freq_hz")
        if pre is not None:
            sc.frequency = float(pre)
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
            pv = np.asarray(paths.valid)
            if pv.size:
                valid_frac.append(float(pv.mean()))
            for k, i in enumerate(todo):
                # w 不能走 `mapper(paths, rx=k)`：那条路把 tx-ant 轴切成 0，
                # 4×4 阵下 w 会被静默丢掉，图与 uniform 逐位相同。
                if w_vec is None:
                    sp_ = mapper(paths, rx=k)
                else:
                    sp_ = mapper.from_response(*_aw_with_w(paths, k, w_vec))
                u = normalize_spectrum(sp_)
                u8 = np.round(u * 255).astype(np.uint8)
                if not args.bench:
                    _atomic_save_png(u8, idir / f"{i:05d}.png")
                stats.append(image_stats(u8))
            n_inst += len(todo)
            for nm in names:
                sc.remove(nm)
            if args.bench:
                print(f"[bench] inst{iid:03d} group {gi}: {len(todo)} 张  "
                      f"solve {t1-t0:.3f}s  map {time.time()-t1:.3f}s  "
                      f"paths {int(np.sum(pv)) if pv.size else 0}", flush=True)
            elif gi % 20 == 0:
                el = time.time() - t_start
                print(f"[params] inst{iid:03d} group {gi}/{len(groups)}  "
                      f"{n_done + n_inst} 张  {el/60:.1f} min", flush=True)
            if args.limit and n_inst >= args.limit:
                break

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
                "pre_freq_hz": (float(pre) if pre is not None else None),
                "tx_pattern": pattern, "tx_rows": nr, "tx_cols": nc,
                "w_label": w_label,
                "n_images": n_on_disk, "n_expected": n_expect,
                "complete": n_on_disk == n_expect,
                "image_stats": agg,
                "valid_path_fraction": float(np.mean(valid_frac)) if valid_frac else None,
                "protocol_hash": phash,
                "elapsed_s": time.time() - t_inst,
            }, gpath)
            print(f"[params] inst{iid:03d} 完成 {n_on_disk}/{n_expect} 张  "
                  f"{time.time()-t_inst:.0f}s", flush=True)
        if args.limit and n_inst >= args.limit:
            break

    el = time.time() - t_start
    if args.bench:
        print(f"[bench] 合计 {n_done} 张 / {el:.1f}s = "
              f"{el/max(n_done,1)*1000:.0f} ms/张；"
              f"按此推算 571520 张需 {571520*el/max(n_done,1)/3600:.1f} h（单进程）",
              flush=True)
    else:
        print(f"[params] 完成 {n_done} 张，用时 {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
