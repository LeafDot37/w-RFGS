# -*- coding: utf-8 -*-
"""对比两条算场的路：逐点 Receiver vs RadioMapSolver。
用法: python _dbg_scale.py rx 200 | python _dbg_scale.py rmap
"""
import argparse
import os
import sys
import time

import numpy as np
import sionna.rt as rt
from sionna.rt import Receiver, Transmitter, PlanarArray
from sionna.rt.scene import load_scene_from_string

sys.path.insert(0, os.path.dirname(__file__))
from gen_sionna_field import room_xml, surface_points  # noqa: E402

FREQ = 2.412e9
ROOM = [8.0, 6.0, 3.0]


def mk():
    s = load_scene_from_string(room_xml(ROOM, False))
    s.frequency = FREQ
    for k in ("rx_array", "tx_array"):
        setattr(s, k, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso",
                                  polarization="V"))
    return s


def mode_rx(n):
    scene = mk()
    pts = surface_points(ROOM, 0.05, 0.01, False)
    rng = np.random.default_rng(0)
    sub = pts[rng.choice(len(pts), n, replace=False)]
    tx = Transmitter("tx", position=np.array([0.0, 0.0, 1.5], np.float32),
                     power_dbm=30)
    rxs = [Receiver(f"r{i}", position=sub[i].astype(np.float32)) for i in range(n)]
    t0 = time.time()
    scene.add([tx] + rxs)
    print(f"add {n} receivers: {time.time()-t0:.2f}s", flush=True)
    t0 = time.time()
    paths = rt.PathSolver()(scene, max_depth=3, los=True,
                            specular_reflection=True, diffuse_reflection=False,
                            refraction=False, diffraction=False,
                            synthetic_array=True, max_num_paths_per_src=512)
    t1 = time.time()
    c = paths.cfr(frequencies=np.array([FREQ], np.float32), out_type="numpy")
    p = np.abs(np.asarray(c).reshape(-1)) ** 2
    print(f"num_rx={paths.num_rx}  solve={t1-t0:.2f}s  cfr={time.time()-t1:.2f}s")
    dbm = 10 * np.log10(np.maximum(p, 1e-30)) + 30
    print(f"dBm p1/p50/p99 {np.percentile(dbm,1):.1f}/{np.percentile(dbm,50):.1f}/"
          f"{np.percentile(dbm,99):.1f}  n={len(dbm)}")


def mode_rmap(samples, cell):
    scene = mk()
    tx = Transmitter("tx", position=np.array([0.0, 0.0, 1.5], np.float32),
                     power_dbm=30)
    scene.add(tx)
    t0 = time.time()
    rm = rt.RadioMapSolver()(scene, center=np.array([0, 0, 0.01]),
                             orientation=np.zeros(3), size=np.array([8.0, 6.0]),
                             cell_size=cell, samples_per_tx=samples,
                             max_depth=3, los=True, specular_reflection=True,
                             diffuse_reflection=False, refraction=False,
                             diffraction=False, seed=0)
    print(f"rmap solve={time.time()-t0:.2f}s  cells={rm.cells_count}  "
          f"centers {np.array(rm.cell_centers).shape}", flush=True)
    pg = np.array(rm.path_gain).reshape(-1)
    print(f"path_gain p1/p50/p99 {np.percentile(pg,1):.3e}/{np.percentile(pg,50):.3e}/"
          f"{np.percentile(pg,99):.3e}  min {pg.min():.3e} max {pg.max():.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rx", "rmap"))
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--samples", type=int, default=10_000_000)
    ap.add_argument("--cell", type=float, default=0.05)
    a = ap.parse_args()
    if a.mode == "rx":
        mode_rx(a.n)
    else:
        mode_rmap(a.samples, a.cell)
