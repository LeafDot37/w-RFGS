# -*- coding: utf-8 -*-
"""把每轴代表实例的全视角 CBF/MVDR 面板拼成选图包。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'D:\GS3D\mvsplat')
SRC = ROOT / 'results/_axis_all'
OPT = ROOT / 'results/_opt64'
DST = ROOT / 'results/fig4_视角选择'
AXES = [('01_position_inst144', 144), ('02_yaw_inst196', 196), ('03_pitch_inst232', 232),
        ('04_carrier_inst268', 268), ('05_weights_inst318', 318)]
font = ImageFont.load_default()
COLS = [('Physical', None), ('CBF GT', 'cbf_gt'), ('CBF ours', 'cbf_ours'),
        ('MVDR GT', 'mvdr_gt'), ('MVDR ours', 'mvdr_ours')]

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)


def load(p):
    return Image.open(p).convert('RGB')


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(text) + 6, 12], fill='white')
    d.text((2, 1), text, fill='black', font=font)
    return im


for name, iid in AXES:
    d = DST / name
    d.mkdir()
    (d / 'raw').mkdir()
    src = SRC / f'inst{iid:03d}'
    rows = list(csv.DictReader(open(src / 'index.tsv'), delimiter='\t'))
    W, H = 300, 200
    for r in rows:
        vn = r['view']
        strip = Image.new('RGB', (5 * W + 8, H + 18), 'white')
        for j, (lab, key) in enumerate(COLS):
            p = OPT / f'v{vn}.png' if key is None else src / f'v{vn}_{key}.png'
            strip.paste(label(load(p).copy(), lab), (j * (W + 2), 16))
        strip.save(d / f'cmp_v{vn}.png')
    # contact sheet 8 x 8
    th, tw = 60, 90
    cw, ch, gap = 5 * tw + 8, th + 18, 6
    sheet = Image.new('RGB', (8 * (cw + gap) + gap, 8 * (ch + 16) + gap), 'white')
    dr = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        im = load(d / f'cmp_v{r["view"]}.png').resize((cw, ch))
        x = gap + (k % 8) * (cw + gap)
        y = gap + (k // 8) * (ch + 16)
        sheet.paste(im, (x, y))
        dr.text((x, y + ch + 1), r['view'], fill='black', font=font)
    sheet.save(d / 'contact_sheet.png')
    shutil.move(str(src / 'index.tsv'), str(d / 'index.tsv'))

print('packed', len(AXES))
