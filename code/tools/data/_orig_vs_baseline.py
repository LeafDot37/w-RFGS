# -*- coding: utf-8 -*-
"""主读数：留出发射源（作者 TX48 的位置）上的三指标，与原始 RF-3DGS 对照。

口径
--------------------------------------------------------------------------------
- 视角：作者 640 个测试视角（与 RF-3DGS 发布评测同集合）；
- 位置：作者 TX48 的位置（`meta.orig_tx_index` = 48），落在测试划分里；
- 配置：新协议（tr38901）下最接近原始的设置 —— 朝向偏移 0°/0°（指向场景质心）、
  2.4 GHz。三个参数都不在训练取值里，所以这同时是「未见位置 + 未见参数」。
- 对照：作者发布渲染 vs 作者 GT = 14.13 dB；旧角域前馈（iso 协议）= 11.85 dB。

跨协议为什么还能比
--------------------------------------------------------------------------------
换天线本身造成的图差是 43.0 dB（均方根差约 1.8 个灰阶），而两个方法的误差都在
50–65 个灰阶量级：sqrt(50² + 1.8²) = 50.03 ⇒ 读数只掉 0.005 dB。所以旧读数
仍然可用，但要在文档里注明这是「跨协议参考」。

跑法：
    $PY code/tools/data/_orig_vs_baseline.py \
        --metrics results/<run>/test_inst/metrics.txt \
        --meta datasets/RF-3DGS_cbf_params/meta.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

# 出处：results/_日志/angular_console.log（2026-10-20，同一 GT、同一 640 视角）
REF = [
    ("RF-3DGS 发布渲染", 14.1315, 0.7233, 0.4457, "作者发布包，iso 协议"),
    ("旧角域前馈（单源）", 11.8472, 0.7379, 0.4404, "本仓 iso 协议，TX48"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--meta", default="datasets/RF-3DGS_cbf_params/meta.json")
    ap.add_argument("--iid", type=int, default=None)
    ap.add_argument("--tag", default="本次（tr38901 协议）")
    args = ap.parse_args()

    iid = args.iid
    if iid is None:
        m = json.load(open(args.meta, encoding="utf-8"))
        # 参数化 w 数据集的 meta 没有 orig_instance_id（`orig_tx_index` 是 -1）：
        # 作者原发射源不是这批实例的一部分，作者口径的对照本来就不存在。
        # 这里不能 `m["orig_instance_id"]` —— 会把整条评测流水线（包括已经算好
        # 的 metrics）判成失败。取不到就跳过对照，只报按角色汇总。
        iid = os.environ.get("ORIG_ID") or m.get("orig_instance_id")
        if iid is not None:
            iid = int(iid)
            row = next((i for i in m.get("instances", [])
                        if int(i["id"]) == iid), None)
            pos = row.get("position_id", "?") if row else "?"
            print(f"留出发射源：实例 {iid}（作者 TX{pos} 的位置），"
                  f"朝向 0°/0°、2.4 GHz\n")
        else:
            print(f"[skip] {args.meta} 没有 orig_instance_id 且未设 ORIG_ID："
                  "跳过作者发布包对照\n")

    rows = list(csv.DictReader(open(args.metrics, encoding="utf-8"), delimiter="\t"))
    hit = [r for r in rows if iid is not None and int(r["id"]) == int(iid)]
    if not hit:
        print(f"[skip] metrics 里没有实例 {iid}（已评测 {len(rows)} 个，"
              f"角色 {sorted({r['role'] for r in rows})}）")
        # 部分角色评测（如 ROLES=w_holdout）本来就不会包含 orig 实例，这里按角色
        # 汇总后正常退出，避免驱动把它当成崩溃去重跑整臂。
        for role in sorted({r["role"] for r in rows}):
            g = [r for r in rows if r["role"] == role]
            print(f"  {role:15s} k={len(g):3d}  "
                  f"PSNR {statistics.mean(float(x['psnr']) for x in g):7.4f}  "
                  f"SSIM {statistics.mean(float(x['ssim']) for x in g):.4f}  "
                  f"LPIPS {statistics.mean(float(x['lpips']) for x in g):.4f}")
        return
    r = hit[0]

    print(f"{'方法':26s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}  说明")
    for name, p, s, l, note in REF:
        print(f"{name:26s} {640:4d} {p:7.4f} {s:7.4f} {l:7.4f}  {note}")
    print(f"{args.tag:26s} {int(r['n']):4d} {float(r['psnr']):7.4f} "
          f"{float(r['ssim']):7.4f} {float(r['lpips']):7.4f}  角色 {r['role']}")

    d = float(r["psnr"]) - REF[0][1]
    print(f"\n相对 RF-3DGS 发布包：PSNR {d:+.4f} dB")
    if d >= 0:
        print("→ 不低于原作者发布结果")
    elif d > -1.0:
        print("→ 与发布结果同档（< 1 dB）")
    else:
        print(f"→ 落后发布结果 {-d:.2f} dB")


if __name__ == "__main__":
    main()
