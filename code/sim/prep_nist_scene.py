# -*- coding: utf-8 -*-
"""把 RF-3DGS 官方教程里的 NIST lobby 场景整理成 Sionna RT 2.2 能直接加载的形式。

两处必须处理：
  1) 材质名 `mat-itu_<type>.NNN`：Sionna 2.2 的 `process_xml` 会把后缀当 ITU 类型，
     报 `Invalid ITU material type "plasterboard.001"`。这里给每个这类 BSDF 补一个
     显式的 `<string name="type" value="<type>"/>`，保留唯一 id。
  2) 网格/贴图文件名是中文（平面_001.ply 等）：Mitsuba 在中文 Windows 上按窄字符
     打开路径会失败。这里把引用的资源硬链接成 ASCII 名（m000.ply / t00.jpg），
     并同步改写 XML；原目录不动。

用法：
    python code/sim/prep_nist_scene.py \
        --src _external/rf3dgs_tutorial/out/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0 \
        --out _external/nist_lobby_ascii
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ITU_SUFFIX = re.compile(r"^mat-itu_(.+?)\.\d{3}$")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def ascii_name(prefix: str, idx: int, suffix: str) -> str:
    return f"{prefix}{idx:03d}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 xml/meshes/textures 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default="NIST_lobby_V1.0_material_assigned.xml")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = ET.fromstring((src / args.xml).read_text(encoding="utf-8"))

    # ---- 1) 材质：ITU 后缀 + custom_* 占位 ----
    # Sionna 2.2 在加载时就要求每个 shape 的 BSDF 是 radio material。
    # 教程当年用 Sionna 0.19，diffuse 可以先加载、之后再换成 RadioMaterial；
    # 这里把 mat-custom_* 就地改成 radio-material 占位，真正的散射参数在
    # 加载后用同名的 RadioMaterial 替换（见 nist_scene.py）。
    n_itu = n_custom = 0
    for b in root.findall("./bsdf"):
        mid = b.attrib.get("id", "")
        m = ITU_SUFFIX.match(mid)
        if m:
            ET.SubElement(b, "string", {"name": "type", "value": m.group(1)})
            n_itu += 1
            continue
        if mid.startswith("mat-custom_") and b.attrib.get("type") != "radio-material":
            b.attrib["type"] = "radio-material"
            b.clear()
            b.attrib["type"] = "radio-material"
            b.attrib["id"] = mid
            for k, v in (("thickness", "0.1"), ("relative_permittivity", "1.0"),
                         ("conductivity", "0.0"), ("scattering_coefficient", "0.0")):
                ET.SubElement(b, "float", {"name": k, "value": v})
            n_custom += 1

    # ---- 2) 网格文件名 -> ASCII ----
    shapes = [s for s in root.findall("shape") if s.attrib.get("type") == "ply"]
    refs = []
    for s in shapes:
        f = s.find("string[@name='filename']")
        if f is None:
            raise SystemExit("有 shape 没有 filename")
        refs.append(f.get("value"))

    mapping = {}
    for i, ref in enumerate(dict.fromkeys(refs)):
        rel = Path(ref.replace("\\", "/"))
        dst = Path("meshes") / ascii_name("m", i, ".ply")
        link_or_copy(src / rel, out / dst)
        mapping[ref] = dst.as_posix()

    for s in shapes:
        f = s.find("string[@name='filename']")
        old = f.get("value")
        new = mapping[old]
        f.set("value", new)
        stem = Path(new).stem
        s.set("id", stem)
        s.set("name", stem)

    # ---- 3) 贴图（envmap 等）-> ASCII ----
    n_tex = 0
    for elem in root.iter():
        v = elem.get("value", "")
        if not isinstance(v, str) or not v.lower().startswith("textures/"):
            continue
        rel = Path(v.replace("\\", "/"))
        dst = Path("textures") / ascii_name("t", n_tex, rel.suffix.lower())
        link_or_copy(src / rel, out / dst)
        elem.set("value", dst.as_posix())
        n_tex += 1

    dst_xml = out / "scene.xml"
    dst_xml.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    print(f"ITU 补类型 {n_itu} 个；custom 占位 {n_custom} 个；网格 {len(mapping)} 个 -> {out/'meshes'}")
    print(f"贴图 {n_tex} 个；场景 -> {dst_xml}")


if __name__ == "__main__":
    main()
