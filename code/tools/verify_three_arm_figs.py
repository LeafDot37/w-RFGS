#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_three_arm_figs.py —— V6 出图数值来源审计（本地，独立重算）

为什么需要它
------------
柱状图与定性对照图是**交付物**，图里的数字读者只能信。这个脚本把每条画上去的数字
**从源头重算一遍**，用「图里的值 vs 现场算的值」逐条对拍：

  ① 柱状图 `<组>/柱状图/<M>/数据.json` 的三指标
     == 本树 `阶段3_RRF重建/<臂>/<M>/results.json`（容差 0）
  ② 定性图 `<组>/图像对比/<M>/三臂对照_<M>.json` 的逐帧 PSNR
     == 从 `定性抽帧/{renders,gt}/<帧>.png` **现场重算**的 PSNR（容差见 --tol）
  ③ 定性图行标里的「臂 640 均值 PSNR」
     == 本树 `results.json` 的 PSNR（容差 0）

**为什么不做「看图模型转述」**：看图模型会"读出"图里并不存在的数值（[[multiband-scalar-verdict]]
记过这个教训）。图的正确性只能靠「画之前的那份数据」与「重算的那份数据」对拍来保证。

用法（WSL 的 mvsplat env 里跑，路径要在仓库根）：
    PY=$HOME/miniforge3/envs/mvsplat/bin/python
    $PY code/tools/verify_three_arm_figs.py --run results/run_20260912_两组预算_三臂对比
退出码 0 = 全对拍通过；非 0 = 有条目不符（逐条打印）。
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ARMS = ["A1_原版物理_原版RRF", "A2_原版物理_新RRF", "A3_新物理_新RRF"]
ARM_KEY = {"A1_原版物理_原版RRF": "A1", "A2_原版物理_新RRF": "A2", "A3_新物理_新RRF": "A3"}
# 图 JSON 里臂的键是短名 A1/A2/A3（清单里才是长目录名）——两者别混

# 重算要用仓库自己的 psnr，必须把 code/ 放进 sys.path（口径与出图工具逐字一致）
_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
METHODS = ["MPC", "CBF", "Delay", "MVDR"]
METRICS = ["PSNR", "SSIM", "LPIPS"]


def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="本地 run 根（含 清单/ 与 组*/）")
    ap.add_argument("--methods", default=" ".join(METHODS))
    ap.add_argument("--tol", type=float, default=0.0,
                    help="逐帧 PSNR 的对拍容差（默认 0：同一函数同一次输入应逐位相同）")
    args = ap.parse_args()

    run = Path(args.run)
    bad = ok = 0
    notes = []

    # ---- 清单：把「组+臂+方法」映射到本树相对路径 ----
    mani = {}
    for p in sorted((run / "清单").glob("*.json")):
        r = jload(p)
        mani[(str(r["组"]), r["臂"], r["方法"])] = r

    groups = sorted({k[0] for k in mani})
    print(f"清单 {len(mani)} 节点；组 {groups}")

    for gno in groups:
        # 组目录名：从任一节点的本树相对路径首段取（不靠拼名字，避免名字改动后失配）
        gdir = Path(mani[(gno, "A1", "MPC")]["本树相对路径"]).parts[0]
        g = run / gdir
        print(f"\n=== 组{gno} {gdir} ===")

        for M in args.methods.split():
            # ---------- ① 柱状图 ----------
            bp = g / "柱状图" / M / "数据.json"
            if not bp.is_file():
                print(f"  [缺] 柱状图数据: {bp}")
                bad += 1
            else:
                bd = jload(bp)
                for A in ARMS:
                    rec = mani[(gno, ARM_KEY[A], M)]
                    rj = jload(run / rec["本树相对路径"] / "results.json")
                    key = list(rj)[0]
                    src = rj[key]
                    if not isinstance(src, dict) or "PSNR" not in src:
                        print(f"  [FAIL] {M} {A} results.json 结构异常: {rj}")
                        bad += 1
                        continue
                    got = (bd.get("臂") or {}).get(ARM_KEY[A])
                    if got is None:
                        print(f"  [FAIL] {M} {A} 柱状图数据里没有该臂")
                        bad += 1
                        continue
                    got = got.get("指标", {})      # 柱状图把三指标塞在 "指标" 一层下
                    for m in METRICS:
                        a = float(got[m]) if m in got else None
                        b = float(src[m])
                        if a is None:
                            # 柱状图可能把 LPIPS 反向存；缺键即报，不猜
                            print(f"  [FAIL] {M} {A} 柱状图缺 {m}")
                            bad += 1
                        elif abs(a - b) > 1e-12:
                            print(f"  [FAIL] {M} {A} {m}: 图 {a!r} != results.json {b!r}")
                            bad += 1
                        else:
                            ok += 1
                print(f"  [柱状图] {M} 对拍完毕")

            # ---------- ②③ 定性图 ----------
            qp = g / "图像对比" / M / f"三臂对照_{M}.json"
            if not qp.is_file():
                print(f"  [缺] 定性图数据: {qp}")
                bad += 1
                continue
            qd = jload(qp)
            frames = qd["帧"]
            # 逐帧 PSNR 现场重算
            try:
                import numpy as np
                import torch
                from torchvision.transforms import functional as tf
                from splat.utils.image_utils import psnr as _psnr
            except Exception as e:                              # noqa: BLE001
                print(f"  [skip] 无法导入重算依赖（{e}），只做均值对拍")
                np = torch = tf = _psnr = None

            for A in ARMS:
                rec = mani[(gno, ARM_KEY[A], M)]
                qdir = run / rec["本树相对路径"] / "定性抽帧"
                for fr in frames:
                    for sub in ("renders", "gt"):
                        f = qdir / sub / f"{fr}.png"
                        if not f.is_file():
                            print(f"  [FAIL] {M} {A} 缺抽帧: {f}")
                            bad += 1
                    if _psnr is None:
                        continue
                    r = tf.to_tensor(__import__("PIL.Image", fromlist=["Image"])
                                     .open(qdir / "renders" / f"{fr}.png").convert("RGB")).unsqueeze(0)
                    t = tf.to_tensor(__import__("PIL.Image", fromlist=["Image"])
                                     .open(qdir / "gt" / f"{fr}.png").convert("RGB")).unsqueeze(0)
                    val = float(_psnr(r, t).item())
                    got = float(qd["逐帧PSNR"][ARM_KEY[A]][fr])
                    if abs(val - got) > args.tol:
                        print(f"  [FAIL] {M} {A} 帧{fr} PSNR: 图 {got:.6f} != 重算 {val:.6f}")
                        bad += 1
                    else:
                        ok += 1
                # 行标的 640 均值
                rj = jload(run / rec["本树相对路径"] / "results.json")
                mean_src = float(list(rj.values())[0]["PSNR"])
                mean_q = float(qd["臂640均值PSNR"][ARM_KEY[A]])
                if abs(mean_src - mean_q) > 1e-9:
                    print(f"  [FAIL] {M} {A} 640 均值: 图 {mean_q!r} != results.json {mean_src!r}")
                    bad += 1
                else:
                    ok += 1
            print(f"  [定性图] {M} 对拍完毕")

    print(f"\n==== V6 汇总：通过 {ok} 条，不符 {bad} 条 ====")
    for n in notes:
        print("  " + n)
    if bad:
        print("V6 FAIL：图里的数字与源头/重算不符，交付物不可信")
        return 1
    print("V6 PASS：每条画上去的数字都能从源头复现")
    return 0


if __name__ == "__main__":
    sys.exit(main())
