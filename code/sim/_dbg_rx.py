# -*- coding: utf-8 -*-
"""检查多位置 Receiver 在 PathSolver 里到底是几个接收机。"""
import numpy as np
import sionna.rt as rt
from sionna.rt import Receiver, Transmitter
from sionna.rt.scene import load_scene_from_string

xml = ('<scene version="2.1.0">'
       '<bsdf type="itu-radio-material" id="concrete">'
       '<string name="type" value="concrete"/></bsdf></scene>')
scene = load_scene_from_string(xml)
scene.frequency = 2.412e9
from sionna.rt import PlanarArray
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                             horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                             horizontal_spacing=0.5, pattern="iso", polarization="V")

pts = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]], np.float32)
rx = Receiver("rx", position=pts.T)
tx = Transmitter("tx", position=np.zeros(3, np.float32), power_dbm=30)
scene.add([tx, rx])
print("rx.position", np.array(rx.position).shape)
paths = rt.PathSolver()(scene, max_depth=3, los=True, specular_reflection=True,
                        diffuse_reflection=False, refraction=False,
                        diffraction=False, synthetic_array=True)
print("num_rx", paths.num_rx, "num_tx", paths.num_tx)
a = paths.a
print("a type", type(a), "shape", np.array(a[0]).shape if isinstance(a, tuple) else np.array(a).shape)
c = paths.cfr(frequencies=np.array([2.412e9], np.float32), out_type="numpy")
print("cfr shape", np.array(c).shape)
cc = np.array(c).reshape(-1)
print("|h|^2", (np.abs(cc) ** 2).round(12))
print("dBm", (10 * np.log10(np.abs(cc) ** 2 + 1e-30) + 30).round(2))
