# -*- coding: utf-8 -*-
"""调试 Mitsuba XML 变换顺序：不同写法下矩形的实际包围盒。"""
import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")


def bbox(xml):
    s = mi.load_string(xml)
    b = s.bbox()
    return np.array(b.min), np.array(b.max)


HEAD = '<scene version="2.1.0"><bsdf type="diffuse" id="d"><rgb name="reflectance" value="0.5"/></bsdf>'
TAIL = "</scene>"

# 顶面：矩形 x∈[-1,1] y∈[-1,1] z=0，想放到 z=3、法向 -z
cases = {
    "A scale,rotate,translate": """
  <shape type="rectangle"><transform name="to_world">
    <scale x="4" y="3" z="1"/><rotate x="1" y="0" z="0" angle="180"/>
    <translate x="0" y="0" z="3"/></transform><ref id="d" name="bsdf"/></shape>""",
    "B translate,rotate,scale": """
  <shape type="rectangle"><transform name="to_world">
    <translate x="0" y="0" z="3"/><rotate x="1" y="0" z="0" angle="180"/>
    <scale x="4" y="3" z="1"/></transform><ref id="d" name="bsdf"/></shape>""",
    "C no rotate": """
  <shape type="rectangle"><transform name="to_world">
    <scale x="4" y="3" z="1"/><translate x="0" y="0" z="3"/></transform>
    <ref id="d" name="bsdf"/></shape>""",
    "D wall scale,rotY,translate": """
  <shape type="rectangle"><transform name="to_world">
    <scale x="1.5" y="3" z="1"/><rotate x="0" y="1" z="0" angle="90"/>
    <translate x="4" y="0" z="1.5"/></transform><ref id="d" name="bsdf"/></shape>""",
}
for name, body in cases.items():
    lo, hi = bbox(HEAD + body + TAIL)
    print(f"{name:28s} min {lo.round(3)} max {hi.round(3)}")
