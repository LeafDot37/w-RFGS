# -*- coding: utf-8 -*-
"""跨场景 GT：对任意 Sionna 场景 XML 生成参数化 CBF 角域谱图。

与 `gen_cbf_params.py` 的唯一差别是**场景来源**：那边 `load_scene()` 写死
NIST lobby；这里 `--scene_xml` 指到任意 Mitsuba/Sionna 场景。算子口径逐项相同
（300×200、fov 90、10×10 RX、0.5 ns、max_depth=1、specular only、逐图 min-max），
因此与主实验可比。

用法：
    python code/sim/gen_xscene_params.py --scene_xml _external/xrooms/room_a/scene.xml \
        --poses_root _external/xrooms/room_a --instances .../instances.json \
        --out _external/xrooms/room_a/spectra --views train --tx_rows 4 --tx_cols 4 \
        --w_from_instance --skip_existing
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim.gen_cbf_params import (PROTOCOL, _atomic_save_png,     # noqa: E402
                                _atomic_write_json, _aw_with_w,
                                group_by_pose, image_stats, load_instances,
                                protocol_hash, read_ids)
from sim.spectrum_gen import (CBFMapper, compute_angle_matrices,  # noqa: E402
                              normalize_spectrum, read_colmap_poses)


def load_scene_xml(xml_path: str, freq_hz: float):
    import sionna.rt as rt
    sc = rt.load_scene(str(xml_path))
    sc.frequency = float(freq_hz)
    for k in ("tx_array", "rx_array"):
        setattr(sc, k, rt.PlanarArray(num_rows=1, num_cols=1,
                                      vertical_spacing=0.5,
                                      horizontal_spacing=0.5,
                                      pattern="iso", polarization="V"))
    sc.synthetic_array = True
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_xml", required=True)
    ap.add_argument("--poses_root", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", choices=["train", "test", "all"], default="train")
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--tx_rows", type=int, default=4)
    ap.add_argument("--tx_cols", type=int, default=4)
    ap.add_argument("--w_from_instance", action="store_true")
    ap.add_argument("--save_scale", action="store_true")
    ap.add_argument("--scales_only", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views_limit", type=int, default=0)
    args = ap.parse_args()

    import sionna.rt as rt
    from sim import w_codebook

    sc = load_scene_xml(args.scene_xml, 3.6e9)
    print(f"[xgen] scene={args.scene_xml} objects={len(sc.objects)} "
          f"materials={list(sc.radio_materials)}", flush=True)
    sc.rx_array = rt.PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    theta, phi = compute_angle_matrices(args.width, args.height, args.fov)
    mapper = CBFMapper(args.M, theta, phi,
                       time_interval_ns=PROTOCOL["time_interval_ns"])
    solver = rt.PathSolver(deterministic=True)

    poses = read_colmap_poses(args.poses_root)
    root = Path(args.poses_root)
    ids = read_ids(root / f"{args.views}_index.txt")
    if args.views_limit:
        ids = ids[: args.views_limit]
    groups = group_by_pose(poses, ids)
    insts = load_instances(args.instances, ids=args.ids)
    out_root = Path(args.out)
    print(f"[xgen] {len(insts)} instances, {len(groups)} position groups, "
          f"{sum(len(g) for g in groups)} views", flush=True)

    for inst in insts:
        iid = int(inst["id"])
        idir = out_root / f"inst{iid:03d}"
        if not args.scales_only:
            idir.mkdir(parents=True, exist_ok=True)
        w_label = inst.get("w_label") if args.w_from_instance else None
        if args.w_from_instance and not w_label:
            raise SystemExit(f"inst{iid:03d} 缺 w_label")
        phash = protocol_hash({"scene": str(args.scene_xml),
                               "tx_array": [args.tx_rows, args.tx_cols],
                               "w_label": w_label,
                               "freq_hz": float(inst["freq_hz"]),
                               "split": args.views})
        gpath = idir / "_gen.json"
        n_expect = sum(len(g) for g in groups)
        if (args.skip_existing and not args.scales_only and gpath.exists()):
            old = json.loads(gpath.read_text(encoding="utf-8"))
            if (old.get("complete") and old.get("n_images") == n_expect
                    and old.get("protocol_hash") == phash):
                print(f"[xgen] inst{iid:03d} 已完成，跳过", flush=True)
                continue

        sc.frequency = float(inst["freq_hz"])
        nr = int(inst.get("tx_rows", args.tx_rows))
        nc = int(inst.get("tx_cols", args.tx_cols))
        sc.tx_array = rt.PlanarArray(num_rows=nr, num_cols=nc,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="tr38901", polarization="V")
        orient = np.array([float(inst["yaw_abs_rad"]),
                           float(inst["pitch_abs_rad"]), 0.0], np.float32)
        sc.add(rt.Transmitter("tx", position=np.asarray(inst["xyz"], np.float32),
                              orientation=orient, power_dbm=0.0))
        w_vec = w_codebook.vector_for(str(w_label), nr, nc) if w_label else None

        stats, rows = [], []
        n_inst = 0
        for gi, ids_g in enumerate(groups):
            todo = ids_g
            if args.skip_existing and not args.scales_only:
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
            paths = solver(sc, max_depth=args.max_depth, los=True,
                           specular_reflection=True, diffuse_reflection=False,
                           refraction=False, diffraction=False,
                           edge_diffraction=False,
                           samples_per_src=int(args.samples),
                           synthetic_array=True, seed=args.seed)
            for k, i in enumerate(todo):
                if w_vec is None:
                    sp_ = mapper(paths, rx=k)
                else:
                    sp_ = mapper.from_response(*_aw_with_w(paths, k, w_vec))
                u = normalize_spectrum(sp_)
                u8 = np.round(u * 255).astype(np.uint8)
                if not args.scales_only:
                    _atomic_save_png(u8, idir / f"{i:05d}.png")
                if args.save_scale:
                    rows.append((int(i), float(np.min(sp_)),
                                 float(np.max(sp_)), float(np.mean(sp_))))
                stats.append(image_stats(u8))
            n_inst += len(todo)
            for nm in names:
                sc.remove(nm)
            print(f"[xgen] inst{iid:03d} group {gi}/{len(groups)} "
                  f"{n_inst} 张", flush=True)
        sc.remove("tx")

        if args.save_scale and rows:
            with open(out_root / f"scales_inst{iid:03d}.jsonl", "w",
                      encoding="utf-8") as fh:
                for v_, pmin_, pmax_, pmean_ in rows:
                    fh.write(json.dumps({"view": v_, "pmin": pmin_,
                                         "pmax": pmax_, "pmean": pmean_}) + "\n")
        if not args.scales_only:
            agg = {k: float(np.mean([s[k] for s in stats])) for k in
                   ("mean_u", "std_u", "sat0", "sat1")} if stats else {}
            _atomic_write_json({"id": iid, "complete": True,
                                "n_images": n_inst, "protocol_hash": phash,
                                "stats": agg}, gpath)
        print(f"[xgen] inst{iid:03d} done {n_inst} 张", flush=True)
    print("[xgen] all done", flush=True)


if __name__ == "__main__":
    main()
