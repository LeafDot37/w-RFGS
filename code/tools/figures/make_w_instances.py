# -*- coding: utf-8 -*-
"""在既有参数集实例表上加**波束权 w** 这一轴，生成新的实例表。

与既有表的关系
--------------------------------------------------------------------------------
`datasets/RF-3DGS_cbf_params/instances.json`（318 实例）已经是「位置 × 朝向 × 频率」
的随机化区组。本脚本只做两件事：

  1. 给每个实例加一个 `w_label`（训练域码本，按实例号定种子 ⇒ 可复现）；
  2. 追加一个 **w 留出组**：36 个训练位置各 1 个实例，w 取留出带。

其余字段（位置、朝向、频率、划分、视角数）**一个都不动**，所以新旧两份表是
逐实例对齐的。

为什么 w 留出组要**克隆该位置的训练实例**
--------------------------------------------------------------------------------
既有 yaw/pitch/freq 留出组用的是「另抽一组参数、只把该轴推到留出带」，
所以留出与训练的差里混着其它轴的差异。w 轴不沿用这个做法：w 留出组直接复制
该位置**训练抽样 0** 的位置、朝向、频率，只把 w 换掉 ⇒ 与实例 (位置, 抽样 0)
构成严格配对，读数差只能来自 w。代价是它不能同时回答「参数也没见过」，
但那不是本组的问题。

口径断层（必须写进文档）
--------------------------------------------------------------------------------
发射阵从 1×1（MVDR/CBF 两臂）换成 **4×4**，「uniform」权下两者也不是同一幅图
（4×4 的 uniform 仍有阵列方向图）。因此本数据集与 MVDR 臂、CBF 臂的数字**不可
直接比较**；能比的是本数据集内部「训练域 w」与「留出 w」。

用法::
    python code/tools/figures/make_w_instances.py \
        --base datasets/RF-3DGS_cbf_params/instances.json \
        --out datasets/RF-3DGS_mvdr_w_params --seed 0
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim import w_codebook                                          # noqa: E402

TX_ROWS = TX_COLS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="datasets/RF-3DGS_cbf_params/instances.json")
    ap.add_argument("--out", default="datasets/RF-3DGS_mvdr_w_params")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orig_w", default="t45_p000",
                    help="主读数（实例 317）用的留出 w")
    ap.add_argument("--views", default="test",
                    help="所有实例统一取的视角集合（默认 test = 作者 640 测试视角）")
    args = ap.parse_args()

    raw = json.loads(Path(args.base).read_text(encoding="utf-8"))
    insts = [dict(i) for i in (raw["instances"] if isinstance(raw, dict) else raw)]

    train_labs = w_codebook.train_labels()
    hold_labs = w_codebook.holdout_labels()
    assert args.orig_w in hold_labs, f"{args.orig_w} 不在留出码本里"

    # ---- 1. 既有实例各配一个训练域 w ----
    for i in insts:
        iid = int(i["id"])
        if i["role"] == "orig":
            i["w_label"] = args.orig_w
            continue
        rng = np.random.default_rng([args.seed, iid])
        i["w_label"] = train_labs[int(rng.integers(0, len(train_labs)))]

    # ---- 2. w 留出组：36 个训练位置，克隆各自抽样 0 的其它参数 ----
    by_pos = {}
    for i in insts:
        if i["role"] == "train":
            by_pos.setdefault(int(i["position_id"]), []).append(i)
    next_id = max(int(i["id"]) for i in insts) + 1
    added = []
    for pos in sorted(by_pos):
        base = min(by_pos[pos], key=lambda x: int(x["id"]))
        rng = np.random.default_rng([args.seed, 991, pos])
        clone = dict(base)
        clone.update({
            "id": next_id,
            "role": "w_holdout",
            "views": "test",
            "w_label": hold_labs[int(rng.integers(0, len(hold_labs)))],
            "base_instance": int(base["id"]),
            "label": f"{base.get('label', '')} | w_holdout",
        })
        added.append(clone)
        next_id += 1

    out_insts = insts + added
    # 视角数统一：4×4 阵让多径数涨到 4 倍（单张 2232 ms、显存峰值 4.4 GB/进程），
    # 3200 视角/训练实例要 3–5 天。w 轴只回答「见过的权能否外推到没见过的方向」，
    # 不需要 3200 的视角密度 ⇒ 全表统一取 test 的 640 视角（每实例 640 张）。
    for i in out_insts:
        i["views"] = args.views
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instances.json").write_text(
        json.dumps({"instances": out_insts}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    n_train = sum(1 for i in out_insts if i["role"] == "train")
    n_test = len(out_insts) - n_train
    n_views = {"test": 640, "all": 3200, "train": 2560}[args.views]
    meta = {
        "format": "mvdr_w_v1",
        "base": str(args.base),
        "protocol": {
            "spectrum": "mvdr", "tx_array": [TX_ROWS, TX_COLS],
            "tx_pattern": "tr38901", "polarization": "V",
            "rx_array_M": 10, "samples_per_src": 1_000_000,
            "time_interval_ns": 0.1, "mvdr_loading": 0.01,
            "material_freq_hz": 6.0e10,
            "w_phase_convention": "-j2*pi*(y*sin(t)*sin(p) + z*cos(t)) / sqrt(N)",
        },
        "codebook": {
            "theta_train_deg": list(w_codebook.THETA_TRAIN),
            "theta_hold_deg": list(w_codebook.THETA_HOLD),
            "phi_train_deg": list(w_codebook.PHI_TRAIN),
            "phi_hold_deg": list(w_codebook.PHI_HOLD),
            "n_train": len(train_labs), "n_holdout": len(hold_labs),
        },
        "counts": {
            "instances": len(out_insts), "train_instances": n_train,
            "test_instances": n_test,
            "views_per_instance": n_views,
            "images": len(out_insts) * n_views,
        },
        "note": "4x4 发射阵与 MVDR/CBF 两臂的 1x1 不同口径，跨臂数字不可直接比；"
                "本表所有实例统一 640 视角（作者 test 划分），与两臂训练实例的 3200 视角不同。",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                   encoding="utf-8")

    import collections
    print(f"[w] 实例 {len(out_insts)} 个（新增 {len(added)}），"
          f"图 {meta['counts']['images']:,} 张")
    print("[w] 角色：", dict(collections.Counter(i["role"] for i in out_insts)))
    print("[w] 训练码本", len(train_labs), "个方向；留出码本", len(hold_labs), "个")
    print("[w] 输出", out)


if __name__ == "__main__":
    main()
