# -*- coding: utf-8 -*-
"""检查 gen_sionna_field.room_xml 生成的几何：替换成普通材质后看包围盒。"""
import os
import sys

import mitsuba as mi

mi.set_variant("scalar_rgb")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gen_sionna_field import room_xml  # noqa: E402

xml = room_xml([8.0, 6.0, 3.0], furniture=False)
print("shapes:", xml.count("<shape"), "rect:", xml.count('type="rectangle"'),
      "cube:", xml.count('type="cube"'))
xml2 = xml.replace('type="itu-radio-material"', 'type="diffuse"')
# 去掉 itu 参数行，换成 reflectance
import re
xml2 = re.sub(r'<string name="type" value="[^"]+"/>', '', xml2)
xml2 = re.sub(r'<float name="thickness" value="[^"]+"/>',
              '<rgb name="reflectance" value="0.5"/>', xml2)
try:
    s = mi.load_string(xml2)
    b = s.bbox()
    print("bbox", b.min, b.max)
except Exception as e:
    print("load failed:", e)
    open("_room_dbg.xml", "w", encoding="utf-8").write(xml2)
    print("wrote _room_dbg.xml")
