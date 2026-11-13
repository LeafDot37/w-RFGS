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
    $PY code/tools/_orig_vs_baseline.py \
        --metrics results/<run>/test_inst/metrics.txt \
        --meta datasets/RF-3DGS_cbf_params/meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
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
        iid = int(m["orig_instance_id"])
        pos = next(i for i in m["instances"] if int(i["id"]) == iid)["position_id"]
        print(f"留出发射源：实例 {iid}（作者 TX{pos} 的位置），"
              f"朝向 0°/0°、2.4 GHz\n")

    rows = list(csv.DictReader(open(args.metrics, encoding="utf-8"), delimiter="\t"))
    hit = [r for r in rows if int(r["id"]) == iid]
    if not hit:
        raise SystemExit(f"metrics 里没有实例 {iid}（已评测 {len(rows)} 个）")
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
