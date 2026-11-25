# -*- coding: utf-8 -*-
"""把实例 317 全视角面板拼成逐视角对比图，并生成联系表。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\GS3D\mvsplat")
SRC = ROOT / "results/_fig317_all"
RF = ROOT / "final/前馈pipeline/运行/run_20261020_本机4060_角域前馈多发射源_起步点修复/rf3dgs_released_renders"
OUT = SRC / "cmp"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(SRC / "index.tsv"), delimiter="\t"))
font = ImageFont.load_default()


def load(p):
    return Image.open(p).convert("RGB")


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 6, 14], fill=(255, 255, 255))
    d.text((3, 1), text, fill=(0, 0, 0), font=font)
    return im


for r in rows:
    vn = r["view"]
    cg, co = load(SRC / f"v{vn}_cbf_gt.png"), load(SRC / f"v{vn}_cbf_ours.png")
    mg, mo = load(SRC / f"v{vn}_mvdr_gt.png"), load(SRC / f"v{vn}_mvdr_ours.png")
    rf = RF / f"{vn}.png"
    has_rf = rf.exists()
    r["rf3dgs"] = "1" if has_rf else "0"
    W, H = cg.size
    canvas = Image.new("RGB", (3 * W + 8, 2 * H + 26), "white")
    canvas.paste(label(cg.copy(), "CBF GT"), (0, 18))
    if has_rf:
        canvas.paste(label(load(rf).resize((W, H)), "CBF RF-3DGS"), (W + 4, 18))
    else:
        ImageDraw.Draw(canvas).text((W + 6, 20), "CBF RF-3DGS n/a", fill=(120, 120, 120), font=font)
    canvas.paste(label(co.copy(), "CBF Ours"), (2 * W + 8, 18))
    canvas.paste(label(mg.copy(), "MVDR GT"), (0, H + 22))
    canvas.paste(label(mo.copy(), "MVDR Ours"), (W + 4, H + 22))
    canvas.save(OUT / f"cmp_v{vn}.png")
    r["cbf"] = f"v{vn}"

with open(SRC / "index.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["view", "cbf_psnr", "mvdr_psnr", "rf3dgs"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})

# contact sheet: 8 cols x 8 rows of 2x3 strips, downsampled
th = 28
tw = int(th * 1.5)
cols, gap, pad = 8, 6, 24
cell_w, cell_h = 3 * tw + 8, 2 * th + 26
sheet = Image.new("RGB", (cols * (cell_w + gap) + gap, ((len(rows) + cols - 1) // cols) * (cell_h + pad) + gap), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    im = load(OUT / f"cmp_v{r['view']}.png").resize((cell_w, cell_h))
    x = gap + (i % cols) * (cell_w + gap)
    y = gap + (i // cols) * (cell_h + pad)
    sheet.paste(im, (x, y))
    d.text((x, y + cell_h + 1), r["view"], fill=(0, 0, 0), font=font)
sheet.save(SRC / "contact_sheet.png")
print("composed", len(rows), "rf3dgs_avail", sum(r["rf3dgs"] == "1" for r in rows))
