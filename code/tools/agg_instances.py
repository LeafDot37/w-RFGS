# -*- coding: utf-8 -*-
"""按角色汇总两臂读数，并做**逐实例配对**（全参数臂 − 参数盲臂）。

输入是 `eval_instances.py` 写出的 metrics.txt（TSV）。
位置留出组里 52 个实例来自 13 个位置（每位置 4 组），不是 52 个独立样本：
除实例级统计外，另给「先按位置平均再统计」的 13 点版本。

跑法：
    $PY code/tools/_agg_instances.py \
        --a results/<run>/full/test_inst/metrics.txt \
        --b results/<run>/blind/test_inst/metrics.txt \
        --meta datasets/RF-3DGS_cbf_params/meta.json \
        --out results/<run>/agg.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("psnr", "ssim", "lpips")


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        out[int(r["id"])] = dict(role=r["role"], n=int(r["n"]),
                                 **{m: float(r[m]) for m in METRICS})
    if not out:
        raise SystemExit(f"{path} 读空")            # 空文件不许静默通过
    return out


def pos_of(meta_path):
    if not meta_path:
        return {}
    m = json.load(open(meta_path, encoding="utf-8"))
    return {int(i["id"]): int(i.get("position_id", -1)) for i in m.get("instances", [])}


def mean_sd(v):
    return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)


def single_arm(A, pos, src, out):
    """单臂汇总：实例级 mean ± pstdev + 位置级（先按位置平均再统计）。

    位置留出组的 52 个实例只来自 13 个位置，不是 52 个独立样本；
    位置级那 13 个数才是「换一个没见过的发射源位置」的独立读数。
    """
    by_role = defaultdict(list)
    for i, r in A.items():
        by_role[r["role"]].append(i)

    report = {"a": src, "n_instances": len(A), "roles": {}}
    print(f"实例 {len(A)} 个\n")
    print(f"{'角色':14s} {'k':>4s} {'位置数':>6s}  " + "  ".join(
        f"{'实例-' + m:>13s} {'位置-' + m:>14s}" for m in METRICS))
    for role, ids in sorted(by_role.items()):
        cell = {"k": len(ids), "metrics": {}}
        line = f"{role:14s} {len(ids):4d} "
        groups = defaultdict(list)
        for i in ids:
            groups[pos.get(i, i)].append(i)
        line += f"{len(groups):6d}  "
        for m in METRICS:
            im, isd = mean_sd([A[i][m] for i in ids])
            pv = [statistics.mean(A[i][m] for i in g) for g in groups.values()]
            pm, psd = mean_sd(pv)
            cell["metrics"][m] = dict(inst_mean=im, inst_sd=isd,
                                      pos_mean=pm, pos_sd=psd, n_pos=len(pv))
            line += f"  {im:7.4f}±{isd:.4f}  {pm:7.4f}±{psd:.4f}"
        report["roles"][role] = cell
        print(line)

    print("\n位置级 = 先按位置把同一位置的 4 组参数平均，再对位置求 mean ± pstdev")
    if out:
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"写出 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="全参数臂 metrics.txt")
    ap.add_argument("--b", default="", help="参数盲臂 metrics.txt（不给则只报单臂汇总）")
    ap.add_argument("--meta", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A = load(args.a)
    if not args.b:
        single_arm(A, pos_of(args.meta), args.a, args.out)
        return

    B = load(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("两臂没有共同实例 id")
    pos = pos_of(args.meta)
    diff_roles = sorted({A[i]["role"] for i in common} ^ {B[i]["role"] for i in common})
    if diff_roles:
        print(f"[warn] 角色集合不一致：{diff_roles}（只按 A 的角色归类）")
    for i in common:
        if A[i]["role"] != B[i]["role"]:
            print(f"[warn] inst{i:03d} 角色不一致 A={A[i]['role']} B={B[i]['role']}")

    by_role = defaultdict(list)
    for i in common:
        by_role[A[i]["role"]].append(i)

    report = {"a": args.a, "b": args.b, "n_common": len(common), "roles": {}}
    print(f"共同实例 {len(common)} 个\n")
    hdr = f"{'角色':14s} {'k':>4s}  " + "  ".join(
        f"{'A-' + m:>9s} {'B-' + m:>9s} {'Δ' + m:>8s}" for m in METRICS)
    print(hdr)
    for role, ids in sorted(by_role.items()):
        cell = {"k": len(ids)}
        line = f"{role:14s} {len(ids):4d}  "
        for m in METRICS:
            av, bv = [A[i][m] for i in ids], [B[i][m] for i in ids]
            d = [x - y for x, y in zip(av, bv)]
            am, asd = mean_sd(av)
            bm, _ = mean_sd(bv)
            dm = statistics.mean(d)
            dst = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
            wins = sum(1 for x in d if x > 0)
            cell[m] = dict(a=am, a_sd=asd, b=bm, delta=dm, delta_se=dst,
                           wins=wins, k=len(d))
            line += f"  {am:9.4f} {bm:9.4f} {dm:+8.4f}"
        # 位置级口径（先按位置平均再统计）
        groups = defaultdict(list)
        for i in ids:
            groups[pos.get(i, i)].append(i)
        if len(groups) < len(ids):
            line += f"   [位置级 k={len(groups)}: "
            for m in METRICS:
                pv = []
                for _, g in groups.items():
                    pv.append(statistics.mean(A[i][m] for i in g)
                              - statistics.mean(B[i][m] for i in g))
                pm, psd = mean_sd(pv)
                cell[f"{m}_by_pos"] = dict(delta=pm, sd=psd, k=len(pv))
                line += f"Δ{m} {pm:+.4f}±{psd:.4f}  "
            line += "]"
        report["roles"][role] = cell
        print(line)

    print("\nΔ = 全参数臂 − 参数盲臂（PSNR/SSIM 越大越好，LPIPS 越小越好）")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
