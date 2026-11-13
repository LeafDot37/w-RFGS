# -*- coding: utf-8 -*-
"""Sionna RT 冒烟：一间盒子房间 + 1 个 TX + 若干 RX，算 4x4 阵列波束形成后的
360x90 空间谱（与 GRaF 的定义一致：每个方向 = 一个波束的输出功率）。

跑法（独立环境）：
    python code/sim/sionna_smoke.py
"""
from __future__ import annotations

import time

import numpy as np
import sionna.rt as rt
from sionna.rt import PlanarArray, Receiver, Transmitter
from sionna.rt.scene import load_scene_from_string


ROOM_XML = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="concrete">
    <string name="type" value="concrete"/><float name="thickness" value="0.2"/>
  </bsdf>
  <bsdf type="itu-radio-material" id="plaster">
    <string name="type" value="plasterboard"/><float name="thickness" value="0.05"/>
  </bsdf>
  <bsdf type="itu-radio-material" id="wood">
    <string name="type" value="wood"/><float name="thickness" value="0.05"/>
  </bsdf>

  <!-- floor, z=0 -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{L2}" z="1"/>
      <translate x="0" y="0" z="0"/>
    </transform>
    <ref id="wood" name="bsdf"/>
  </shape>

  <!-- ceiling, z=H -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{L2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="180"/>
      <translate x="0" y="0" z="{H}"/>
    </transform>
    <ref id="plaster" name="bsdf"/>
  </shape>

  <!-- wall x=+W/2, normal -x -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{H2}" y="{L2}" z="1"/>
      <rotate x="0" y="1" z="0" angle="90"/>
      <translate x="{W2}" y="0" z="{H2}"/>
    </transform>
    <ref id="concrete" name="bsdf"/>
  </shape>

  <!-- wall x=-W/2, normal +x -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{H2}" y="{L2}" z="1"/>
      <rotate x="0" y="1" z="0" angle="-90"/>
      <translate x="{nW2}" y="0" z="{H2}"/>
    </transform>
    <ref id="concrete" name="bsdf"/>
  </shape>

  <!-- wall y=+L/2, normal -y -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{H2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="90"/>
      <translate x="0" y="{L2}" z="{H2}"/>
    </transform>
    <ref id="plaster" name="bsdf"/>
  </shape>

  <!-- wall y=-L/2, normal +y -->
  <shape type="rectangle">
    <transform name="to_world">
      <scale x="{W2}" y="{H2}" z="1"/>
      <rotate x="1" y="0" z="0" angle="-90"/>
      <translate x="0" y="{nL2}" z="{H2}"/>
    </transform>
    <ref id="plaster" name="bsdf"/>
  </shape>
</scene>
"""

W, L, H = 8.0, 6.0, 3.0
FREQ = 2.412e9


def beamform(y_ant: np.ndarray, positions: np.ndarray, freq: float,
             n_az: int = 360, n_el: int = 90) -> np.ndarray:
    """y_ant: [n_ant] 复数接收信号；positions: [n_ant,3] 阵元位置（世界系）。
    返回 [n_el, n_az] 的功率谱，行=仰角 1..90°，列=方位 1..360°。"""
    k = 2 * np.pi * freq / 299792458.0
    az = np.deg2rad(np.arange(1, n_az + 1))
    el = np.deg2rad(np.arange(1, n_el + 1))
    AZ, EL = np.meshgrid(az, el)                      # [n_el, n_az]
    d = np.stack([np.cos(EL) * np.cos(AZ),
                  np.cos(EL) * np.sin(AZ),
                  np.sin(EL)], axis=-1)               # [n_el,n_az,3]
    phase = np.einsum("eak,nk->nea", d, positions) * k
    steer = np.exp(-1j * phase)                       # [n_el,n_az,n_ant]
    ss = np.abs(steer @ y_ant) ** 2
    return ss.astype(np.float32)


def main():
    xml = ROOM_XML.format(W2=W / 2, nW2=-W / 2, L2=L / 2, nL2=-L / 2,
                          H=H, H2=H / 2)
    t0 = time.time()
    scene = load_scene_from_string(xml)
    scene.frequency = FREQ
    scene.rx_array = PlanarArray(num_rows=4, num_cols=4,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    tx = Transmitter("tx0", position=[1.2, -0.8, 1.4], orientation=[0, 0, 0])
    rx_pos = [[-2.0, 1.0, 1.5], [2.5, 1.5, 1.2], [0.0, 0.0, 2.0], [1.0, 2.0, 0.8]]
    rxs = [Receiver(f"rx{i}", position=p) for i, p in enumerate(rx_pos)]
    scene.add([tx] + rxs)
    print(f"[smoke] 场景就绪 {time.time()-t0:.1f}s  TX {tx.position}  RX {len(rxs)}")

    t0 = time.time()
    paths = rt.PathSolver()(scene, max_depth=3, los=True, specular_reflection=True,
                            diffuse_reflection=False, diffraction=False,
                            synthetic_array=True, seed=0)
    print(f"[smoke] PathSolver {time.time()-t0:.1f}s")
    print("  a 类型:", type(paths.a))
    try:
        print("  a shape:", paths.a.shape)
    except Exception as e:
        print("  a shape 取不到:", e, "tuple?", [None if not hasattr(x, 'shape') else x.shape for x in paths.a])
    print("  tau shape:", getattr(paths.tau, "shape", None))
    print("  theta_r shape:", getattr(paths.theta_r, "shape", None))
    print("  num_rx/num_tx:", paths.num_rx, paths.num_tx)

    # 4x4 阵列的每个天线在 RX 局部系的位置
    print("  rx_array positions:", np.array(scene.rx_array.positions).shape,
          np.array(scene.rx_array.positions)[:3])

    # 取 RX0 / TX0 的每天线信道响应（单频）
    freqs = np.array([FREQ])
    cfr = paths.cfr(frequencies=freqs, normalize_delays=False, normalize=False,
                    out_type="numpy")
    print("  cfr shape:", np.array(cfr).shape, np.array(cfr).dtype)
    c = np.array(cfr)
    # 约定 [num_rx, num_rx_ant, num_tx, num_tx_ant, num_freq] 或 [num_rx, num_rx_ant, num_tx, 1, nf]
    c = c.reshape(paths.num_rx, -1, paths.num_tx, -1, c.shape[-1])
    y = c[0, :, 0, 0, 0]
    print("  y_ant:", y.shape, "|y|^2 dBm-ish:", 10*np.log10(np.abs(y)**2 + 1e-30).round(1))

    pos = np.array(scene.rx_array.positions)
    ss = beamform(y, pos, FREQ)
    print(f"[smoke] 谱 {ss.shape}  min {ss.min():.3e}  max {ss.max():.3e}  "
          f"峰值位置(行,列) {np.unravel_index(ss.argmax(), ss.shape)}")


if __name__ == "__main__":
    main()
