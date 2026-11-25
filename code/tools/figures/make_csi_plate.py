# -*- coding: utf-8 -*-
"""定性拼图（三行 × 四视角），给论文 Fig.6 / Fig.7 用。

版式按交付宽度定：画布宽 = 0.747*textwidth = 5.35 in，300 dpi，
行标签用两行横排放在左栏，避免竖排长标签互相压住。

用法（本地仓库根）：
    python code/tools/figures/make_csi_plate.py --set cbf
    python code/tools/figures/make_csi_plate.py --set w
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RFGS = (ROOT / "final/前馈pipeline/运行/"
        "run_20261020_本机4060_角域前馈多发射源_起步点修复/rf3dgs_released_renders")

SETS = {
    # 单发射源 CBF（inst317）：顶四分位里取四张、且相机位姿分散
    "cbf": dict(
        views=["00491", "00543", "00907", "01835"],
        rows=[("Ground\ntruth", lambda v: ROOT / f"results/_csi_cbf/qual/{v}_gt.png"),
              ("RF-3DGS\n(released)", lambda v: RFGS / f"{v}.png"),
              ("Ours\n(CSI, 20k)", lambda v: ROOT / f"results/_csi_cbf/qual/{v}_pred.png")],
        out=ROOT / "results/_csi_cbf/qual/fig6_like_csi_v2.png",
        title_fmt="view {:d}",
    ),
    # 4×4 w（inst318）：留出 w，行 = GT / 盲控 / 带 w
    "w": dict(
        views=["00214", "00213", "00216", "00188"],
        rows=[("Ground\ntruth", lambda v: ROOT / f"results/_csi_cbfw/qual/{v}_gt.png"),
              ("Ours\n(CSI + w)", lambda v: ROOT / f"results/_csi_cbfw/qual/{v}_pred.png")],
        out=ROOT / "results/_csi_cbfw/qual/fig_w_inst318_v2.png",
        title_fmt="view {:d}",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    args = ap.parse_args()
    cfg = SETS[args.set]
    views, rows = cfg["views"], cfg["rows"]

    plt.rcParams.update({"font.size": 6.3, "font.family": "DejaVu Sans"})
    W, H = 5.35, 2.62
    fig, axes = plt.subplots(len(rows), len(views), figsize=(W, H), dpi=300,
                             gridspec_kw=dict(left=0.145, right=0.995, top=0.90,
                                              bottom=0.02, wspace=0.05, hspace=0.07))
    axes = np.atleast_2d(axes)
    for r, (_, fn) in enumerate(rows):
        for c, v in enumerate(views):
            p = fn(v)
            if not p.is_file():
                raise SystemExit(f"[FAIL] 缺图 {p}")
            ax = axes[r, c]
            ax.imshow(np.asarray(Image.open(p).convert("RGB")))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.35); sp.set_color("#666666")
    for c, v in enumerate(views):
        axes[0, c].set_title(cfg["title_fmt"].format(int(v)), fontsize=6.3, pad=2)

    # 行标签：横排两行，放在左栏正中
    for r, (label, _) in enumerate(rows):
        bb = axes[r, 0].get_position()
        fig.text(0.072, (bb.y0 + bb.y1) / 2, label, ha="center", va="center",
                 fontsize=6.3, linespacing=1.05, color="#1a1a1a")

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
