# -*- coding: utf-8 -*-
"""汇总全量 w′：逐视角 / 区域 w′ 的 pmax 相对 58 方向 oracle 的比例。

口径
--------------------------------------------------------------------------------
- oracle：同一实例、同一视角，58 个码本方向里最大的物理 pmax；
- per-view w′ / region w′：模型 H 推出的 w，交给同一个物理生成器后的 pmax；
- codebook：固定用一个码本方向（实例内所有视角相同），取实例内平均比例最好的那个；
- random：58 个方向的平均比例（等价于随机抽一个方向）。

输出 `results/_wprime_full/summary.json` + 屏幕表格。
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/rf")
SWEEP = R / "results/_cbf_beam_sweep"
WP = R / "results/_wprime_full"
IDS = list(range(318, 354))


def load_scales(path: Path) -> dict[int, float]:
    out = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        out[int(r["view"])] = float(r["pmax"])
    return out


def main():
    labels = sorted({p.name.split("scales_")[1].rsplit("_inst", 1)[0]
                     for p in SWEEP.glob("scales_*_inst*.jsonl")})
    print(f"[agg] {len(labels)} labels, {len(IDS)} ids", flush=True)
    rows = []
    for iid in IDS:
        orc_per_label = {}
        for lab in labels:
            p = SWEEP / f"scales_{lab}_inst{iid:03d}.jsonl"
            if p.exists():
                orc_per_label[lab] = load_scales(p)
        if len(orc_per_label) < len(labels):
            print(f"[agg] inst{iid:03d} incomplete labels "
                  f"{len(orc_per_label)}/{len(labels)}", flush=True)
            continue
        pv_p = WP / "perview" / f"scales_inst{iid:03d}.jsonl"
        rg_p = WP / "region" / f"scales_inst{iid:03d}.jsonl"
        if not (pv_p.exists() and rg_p.exists()):
            print(f"[agg] inst{iid:03d} missing wprime scales", flush=True)
            continue
        pv = load_scales(pv_p)
        rg = load_scales(rg_p)
        views = sorted(set(pv) & set(rg))
        views = [v for v in views
                 if all(v in orc_per_label[l] for l in labels)]
        if not views:
            continue
        oracle = np.array([max(orc_per_label[l][v] for l in labels)
                           for v in views])
        pvv = np.array([pv[v] for v in views])
        rgv = np.array([rg[v] for v in views])
        # 丢掉非正 pmax（模型的谱可能整幅为 0），否则比值 log/除零
        ok = (oracle > 0) & (pvv > 0) & (rgv > 0)
        if ok.sum() < 8:
            print(f"[agg] inst{iid:03d} too few positive views {ok.sum()}",
                  flush=True)
            continue
        oracle, pvv, rgv = oracle[ok], pvv[ok], rgv[ok]
        pv_ratio = float(np.mean(pvv / oracle))
        rg_ratio = float(np.mean(rgv / oracle))
        lab_ratio = {}
        for l in labels:
            a = np.array([orc_per_label[l][v] for v in views])[ok]
            lab_ratio[l] = float(np.mean(a / oracle))
        best_lab = max(lab_ratio, key=lab_ratio.get)
        rnd = float(np.mean(list(lab_ratio.values())))
        rows.append({
            "id": iid, "n_views": int(ok.sum()),
            "perview": pv_ratio, "region": rg_ratio,
            "codebook": lab_ratio[best_lab], "codebook_label": best_lab,
            "random": rnd,
        })
        print(f"  inst{iid:03d} n={len(views):3d} perview {pv_ratio:.4f} "
              f"region {rg_ratio:.4f} codebook {lab_ratio[best_lab]:.4f} "
              f"({best_lab}) random {rnd:.4f}", flush=True)

    def ms(key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v)), len(v)

    summary = {"n_instances": len(rows), "labels": labels}
    for k in ("perview", "region", "codebook", "random"):
        m, s, n = ms(k)
        summary[k] = {"mean": m, "std": s, "n": n}
    per_inst = {k: [r[k] for r in rows] for k in
                ("perview", "region", "codebook", "random")}
    summary["per_instance"] = per_inst
    summary["rows"] = rows
    (WP / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY over %d instances ===" % len(rows))
    for k in ("perview", "region", "codebook", "random"):
        m, s, _ = ms(k)
        print(f"{k:9s} {m:.4f} ± {s:.4f}")
    print("wrote", WP / "summary.json")


if __name__ == "__main__":
    main()
