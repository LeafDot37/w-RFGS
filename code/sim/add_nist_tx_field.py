# -*- coding: utf-8 -*-
"""把一个**指定的**发射源追加进已有的空间射频数据集（同一套求值点、同一口径）。

用途：RF-3DGS 原版 NIST lobby 数据集的实测发射源位置
`tx_loc = [6.905, 0, 2-1.713] = [6.905, 0, 0.287]`（见教程 notebook），
它不在我们 48 个随机采样源里（最近的 45 号在 0.97 m 外）。为了让「原发射源」
能作为**测试源**与 RF-3DGS 的结果直接对照，这里用同一份 `query_xyz.npy`、
同一场景/材质/口径把它的场补出来，并强制放进 `split["test"]`。

做法与 `gen_nist_field.py` 完全一致（同一个 `load_scene` / `path_gain_multi`），
只是源位置由命令行给定、求值点从数据集读，因此不会重新采样点云。

跑法（Windows 仿真环境，GPU）：
    D:/anaconda3/envs/sionna_win/python.exe code/sim/add_nist_tx_field.py \
        --data_root datasets/NIST_lobby_spatial_full --tx 6.905 0 0.287
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from sim.gen_nist_field import load_scene, path_gain_multi   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tx", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--index", type=int, default=-1, help="-1 = 追加到末尾")
    # 必须与数据集一致：scene.json 记录 max_depth=3（gen_nist_field.sh 的默认值）
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的场 / 源位置")
    ap.add_argument("--no_meta", action="store_true", help="只写场，不动 meta.json")
    ap.add_argument("--meta_only", action="store_true",
                    help="场已存在，只补 meta（幂等重跑用）")
    args = ap.parse_args()

    root = Path(args.data_root)
    meta_p = root / "meta.json"
    meta = json.load(open(meta_p, encoding="utf-8"))
    P = int(meta["n_points"])
    query = np.load(root / meta.get("query_xyz_file", "query_xyz.npy"))
    if len(query) != P:
        raise SystemExit(f"求值点数不符：query {len(query)} vs meta {P}")

    tx = np.asarray(args.tx, np.float32)
    tw = np.asarray(meta["tx_world"], np.float32)
    if len(tw):
        d = np.linalg.norm(tw - tx, axis=1)
        k = int(np.argmin(d))
        print(f"[add] 与已有源最近距离 {d[k]:.3f} m（{k} 号）")
        if d[k] < 0.2 and not args.force:
            raise SystemExit("[add] 与已有源几乎重合，拒绝重复追加（--force 可覆盖）")

    idx = len(tw) if args.index < 0 else args.index
    out = root / "field" / f"{idx:05d}.npy"
    if out.exists() and not (args.force or args.meta_only):
        raise SystemExit(f"[add] {out} 已存在（--force 覆盖 / --meta_only 只补 meta）")

    if args.meta_only:
        if not out.exists():
            raise SystemExit(f"[add] --meta_only 但场不存在：{out}")
        print(f"[add] 跳过仿真，沿用 {out}")
    else:
        import sionna.rt  # noqa: F401  触发 CUDA/Mitsuba 初始化，失败要早报
        t0 = time.time()
        sc = load_scene()
        dbm = path_gain_multi(sc, sys.modules["sionna.rt"], query, tx,
                              chunk=args.chunk, max_depth=args.max_depth)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, dbm.astype(np.float16))

        lit = dbm > -299.0
        q = np.percentile(dbm[lit], [5, 50, 95]) if lit.any() else [np.nan] * 3
        print(f"[add] tx={np.round(tx, 3).tolist()}  有径 {lit.mean():.3f}  "
              f"dBm p5/50/95 [{q[0]:.1f} {q[1]:.1f} {q[2]:.1f}]  "
              f"{(time.time() - t0) / 60:.1f} min -> {out}")

    if args.no_meta:
        print("[add] 未改 meta.json")
        return

    # ---- meta：补源位置 + 强制进测试集 ----
    if idx == len(meta["tx_world"]):
        meta["tx_world"].append(tx.tolist())
        meta["tx_local"].append(tx.tolist())
    else:
        meta["tx_world"][idx] = tx.tolist()
        meta["tx_local"][idx] = tx.tolist()
    meta["tx_world"] = [list(map(float, t)) for t in meta["tx_world"]]
    meta["tx_local"] = [list(map(float, t)) for t in meta["tx_local"]]
    test = [int(j) for j in meta["split"]["test"] if int(j) != idx]
    train = [int(j) for j in meta["split"]["train"] if int(j) != idx]
    test.append(idx)
    meta["split"] = {"train": sorted(train), "test": sorted(test)}
    meta["orig_tx_index"] = idx
    meta["orig_tx_note"] = "RF-3DGS 教程原版 NIST lobby 发射源 tx_loc=[6.905,0,0.287]"
    json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[add] meta 更新：TX {len(meta['tx_world'])}  "
          f"train {len(train)} / test {len(test)}  原发射源 index={idx}")
    print("[add] 下一步：gen_vis_features.py --tx %d，再重渲 GT 与各臂" % idx)


if __name__ == "__main__":
    main()
