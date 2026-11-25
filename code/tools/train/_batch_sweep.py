# -*- coding: utf-8 -*-
"""一次场景加载、批量跑 3 个位置的 58 个 w 方向（克隆实例），供多位置 w′ 真值。"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/rf")
PYGEN = "/root/miniconda3/envs/rfgen/bin/python"
META = R / "datasets/RF-3DGS_mvdr_w_params/instances.json"
OUT = R / "results/_sweep_clones"
SWEEP = R / "results/_cbf_beam_sweep"
LOG = R / "logs/batch_sweep.log"
IDS = [318, 321, 324]

sys.path.insert(0, str(R / "code"))
from sim import w_codebook  # noqa: E402

meta = json.load(open(META, encoding="utf-8"))
rows = meta["instances"] if isinstance(meta, dict) else meta
by = {int(r["id"]): r for r in rows}
labels = w_codebook.train_labels() + w_codebook.holdout_labels()

clones, mapping = [], {}
for pi, iid in enumerate(IDS):
    base = by[iid]
    for li, lab in enumerate(labels):
        cid = 9000 + pi * 100 + li
        r = dict(base)
        r["id"] = cid
        r["w_label"] = lab
        clones.append(r)
        mapping[cid] = (iid, lab)

OUT.mkdir(parents=True, exist_ok=True)
cpath = OUT / "clones.json"
json.dump({"instances": clones}, open(cpath, "w", encoding="utf-8"),
          ensure_ascii=False)
print("clones", len(clones), flush=True)

cmd = [PYGEN, "code/sim/gen_cbf_params.py", "--instances", str(cpath),
       "--out", str(OUT), "--ids"] + [str(c["id"]) for c in clones] + [
       "--views", "test", "--views_limit", "64",
       "--tx_rows", "4", "--tx_cols", "4",
       "--w_from_instance", "--save_scale", "--skip_existing"]
with open(LOG, "w", encoding="utf-8") as fh:
    rc = subprocess.call(cmd, cwd=str(R), stdout=fh, stderr=subprocess.STDOUT)
print("gen rc", rc, flush=True)

for cid, (iid, lab) in mapping.items():
    p = OUT / f"scales_inst{cid:03d}.jsonl"
    if p.is_file():
        shutil.copy(p, SWEEP / f"scales_{lab}_inst{iid:03d}.jsonl")
print("copied", flush=True)
