# -*- coding: utf-8 -*-
"""RadioMapSolver 收敛性：同一条射线采样数下的地图差异与耗时。"""
import os
import sys
import time

import numpy as np
import sionna.rt as rt
from sionna.rt import Transmitter, PlanarArray
from sionna.rt.scene import load_scene_from_string

sys.path.insert(0, os.path.dirname(__file__))
from gen_sionna_field import room_xml  # noqa: E402

FREQ = 2.412e9
ROOM = [8.0, 6.0, 3.0]


def mk():
    s = load_scene_from_string(room_xml(ROOM, False))
    s.frequency = FREQ
    for k in ("rx_array", "tx_array"):
        setattr(s, k, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso",
                                  polarization="V"))
    tx = Transmitter("tx", position=np.array([0.0, 0.0, 1.5], np.float32),
                     power_dbm=30)
    s.add(tx)
    return s, tx


def run(s, tx, samples, cell=0.05, seed=1):
    t0 = time.time()
    rm = rt.RadioMapSolver()(s, center=np.array([0, 0, 0.01]),
                             orientation=np.zeros(3), size=np.array([8.0, 6.0]),
                             cell_size=cell, samples_per_tx=samples,
                             max_depth=3, los=True, specular_reflection=True,
                             diffuse_reflection=False, refraction=False,
                             diffraction=False, seed=seed)
    dt = time.time() - t0
    pg = np.array(rm.path_gain).reshape(-1)
    return pg, dt


s, tx = mk()
ref, dt = run(s, tx, 200_000_000)
print(f"ref 200M: {dt:.1f}s")
for n in (1_000_000, 10_000_000, 50_000_000):
    pg, dt = run(s, tx, n)
    db = 10 * np.log10(np.maximum(pg, 1e-30))
    dbr = 10 * np.log10(np.maximum(ref, 1e-30))
    print(f"{n/1e6:.0f}M: {dt:.1f}s  corr {np.corrcoef(pg, ref)[0,1]:.5f}  "
          f"median|Δ| {np.median(np.abs(db-dbr)):.2f} dB  "
          f"p95|Δ| {np.percentile(np.abs(db-dbr),95):.2f} dB  "
          f"p50 {np.median(db):.1f} dBm")
# 空间平滑度（相邻格子的差）
for name, arr in (("ref", ref), ("10M", run(s, tx, 10_000_000)[0])):
    a = 10 * np.log10(np.maximum(arr.reshape(60, 80), 1e-30))
    print(name, "相邻差 p50/p95", np.median(np.abs(np.diff(a, axis=0))).round(2),
          np.percentile(np.abs(np.diff(a, axis=0)), 95).round(2))
