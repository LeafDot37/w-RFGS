# -*- coding: utf-8 -*-
"""确定 6 个测量面的 orientation：打印 cell_centers 的实际范围。"""
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
W, L, H = 8.0, 6.0, 3.0

s = load_scene_from_string(room_xml([W, L, H], False))
s.frequency = FREQ
for k in ("rx_array", "tx_array"):
    setattr(s, k, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                              horizontal_spacing=0.5, pattern="iso",
                              polarization="V"))
tx = Transmitter("tx", position=np.array([0.5, -0.5, 1.5], np.float32), power_dbm=30)
s.add(tx)

# 注意：RadioMapSolver 的 orientation 是**弧度**（设备 orientation 才是度）
P = np.pi
cases = [
    ("floor",   [0, 0, 0.01],      [0, 0, 0],   [W, L]),
    ("ceil",    [0, 0, H - 0.01],  [0, 0, 0],   [W, L]),
    ("x+",      [W/2 - .01, 0, H/2], [0, P/2, 0], [H, L]),
    ("x-",      [-W/2 + .01, 0, H/2], [0, P/2, 0], [H, L]),
    ("y+",      [0, L/2 - .01, H/2], [0, 0, P/2], [W, H]),
    ("y-",      [0, -L/2 + .01, H/2], [0, 0, P/2], [W, H]),
]
for name, c, o, sz in cases:
    t0 = time.perf_counter()
    rm = rt.RadioMapSolver()(s, center=np.array(c, np.float32),
                             orientation=np.array(o, np.float32),
                             size=np.array(sz, np.float32), cell_size=0.05,
                             samples_per_tx=10_000_000, max_depth=3, los=True,
                             specular_reflection=True, diffuse_reflection=False,
                             refraction=False, diffraction=False, seed=0)
    dt = (time.perf_counter() - t0) * 1000
    cc = np.array(rm.cell_centers).reshape(-1, 3)
    pg = np.array(rm.path_gain).reshape(-1)
    print(f"{name:6s} {dt:7.1f} ms  cells {rm.cells_count:6d}  "
          f"bbox {cc.min(0).round(2)} .. {cc.max(0).round(2)}  "
          f"pg p50 {np.median(pg):.2e}")
