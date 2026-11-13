# -*- coding: utf-8 -*-
"""_cnfont.py —— 三个 tools 出图共用：优先按族名匹配已装 CJK 字体，否则从已知
字体文件 addfont（含 Windows /mnt/c 挂载的 msyh.ttc / simhei.ttf，WSL 常用场景）。
若仍找不到 → 返回 False（调用方照常画图，仅中文可能 tofu 并有 glyph 警告）。

用法：
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _cnfont
    _cnfont.configure(plt)
"""
import os
from pathlib import Path

# 常见 CJK 字体族名（matplotlib 已索引时直接匹配）
_FAMILIES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "AR PL UMing CN",
]

# 候选字体文件（Windows 挂载优先，Linux 系统字库兜底）
_FILE_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msyh.ttc",          # 微软雅黑
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",        # 黑体
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def configure(plt):
    """配置 matplotlib 中文字体。返回所用族名或 None。"""
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None
    plt.rcParams["axes.unicode_minus"] = False
    try:
        avail = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        avail = set()
    for name in _FAMILIES:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            return name
    for cand in _FILE_CANDIDATES:
        p = Path(cand)
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except Exception:
            continue
        try:
            name = fm.FontProperties(fname=str(p)).get_name()
        except Exception:
            continue
        if name:
            plt.rcParams["font.sans-serif"] = [name]
            return name
    return None


def pil_font(size=14):
    """PIL ImageFont（PIL 默认位图字库无 CJK，蒙太奇注记需 TrueType）。

    返回 ImageFont 对象；找不到 CJK 字库时返回 None（调用方退化为默认字库，
    仅 ASCII 可显示）。"""
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for cand in _FILE_CANDIDATES:
        p = Path(cand)
        if not p.exists():
            continue
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:
            pass
        try:                                     # .ttc 老版本需显式 index
            return ImageFont.truetype(str(p), size, index=0)
        except Exception:
            continue
    return None
