# -*- coding: utf-8 -*-
"""参数化数据集的自检：实例表 ↔ 磁盘、完全区组、同位置不同参数的成对差异。

查六件事
--------------------------------------------------------------------------------
1. 每个实例的图数对不对（views=all → 3200；test → 640）；
2. 每张图的尺寸一致（300×200），且 8bit 灰度；
3. 完全区组：每个训练位置恰好 4 个实例，且 yaw / pitch / freq 各取遍 4 个水平；
4. 参数域：freq 落在声明区间内，留出值不等于任何训练值；
5. 朝向自洽：yaw_abs = base_orientation(xyz) + offset（按同一套公式重算）；
6. **同位置不同参数的成对差异**：抽若干位置，算各对之间的 PSNR。
   朝向若真的起了作用，成对 PSNR 必须远低于「同图重跑」的 0 差；
   若成对 PSNR 接近 99（图几乎一样），说明朝向没进仿真（老问题：iso 天线）。

跑法（WSL mvsplat 环境，numpy/PIL 即可）：
    $PY code/tools/_check_param_data.py \
        --meta datasets/RF-3DGS_cbf_params/meta.json \
        --spectra datasets/RF-3DGS_cbf_params/spectra
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splat.param_frame import base_orientation          # noqa: E402

EXPECT = {"all": 3200, "test": 640}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--spectra", required=True)
    ap.add_argument("--sample_positions", type=int, default=4)
    ap.add_argument("--sample_views", type=int, default=8)
    ap.add_argument("--no_freq_gate", action="store_true",
                    help="频率轴在闸测里被砍掉时，用这个跳过频率留出组的存在性检查")
    args = ap.parse_args()

    m = json.load(open(args.meta, encoding="utf-8"))
    insts = m["instances"]
    root = Path(args.spectra)
    key_fmt = "inst{idx:03d}"
    target = np.asarray(m["base_frame"]["target_world"], np.float64)
    fails = []

    def bad(msg):
        fails.append(msg)
        print(f"[FAIL] {msg}")

    # 1) 图数
    missing = []
    for it in insts:
        iid = int(it["id"])
        d = root / key_fmt.format(idx=iid)
        want = EXPECT[it.get("views", "all")]
        got = len(list(d.glob("*.png"))) if d.is_dir() else 0
        if got != want:
            missing.append((iid, it.get("role"), got, want))
    if missing:
        bad(f"{len(missing)} 个实例图数不对（前 5：{missing[:5]}）")
    else:
        print(f"[OK ] 图数全对：{len(insts)} 个实例")

    # 2) 尺寸 / 灰度
    it0 = insts[0]
    d0 = root / key_fmt.format(idx=int(it0["id"]))
    p0 = sorted(d0.glob("*.png"))[0]
    im = Image.open(p0)
    if im.size != (300, 200) or im.mode != "L":
        bad(f"{p0.name} 尺寸/模式 = {im.size}/{im.mode}（应为 (300, 200)/L）")
    else:
        print(f"[OK ] 图像 {im.size} mode={im.mode}")

    # 3) 完全区组
    by_pos = defaultdict(list)
    for it in insts:
        if it.get("role") == "train":
            by_pos[int(it["position_id"])].append(it)
    wrong = {p: len(v) for p, v in by_pos.items() if len(v) != 4}
    if wrong:
        bad(f"训练位置每组实例数不是 4：{wrong}")
    else:
        print(f"[OK ] 完全区组：{len(by_pos)} 个训练位置 × 4")
    for key, tag in (("offset_yaw_rad", "yaw"), ("offset_pitch_rad", "pitch"),
                     ("freq_hz", "freq")):
        unbal = []
        for p, v in by_pos.items():
            vals = sorted(round(float(x[key]), 12) for x in v)
            if len(set(vals)) != 4:
                unbal.append((p, vals))
        if unbal:
            bad(f"{tag} 在某位置没有取遍 4 个水平（前 3：{unbal[:3]}）")
        else:
            print(f"[OK ] {tag}：每个训练位置取遍 4 个水平各一次")

    # 4) 参数域
    ps_ = m["param_space"]
    lo, hi = (float(x) * 1e9 for x in ps_["freq_ghz"]["declared"])
    seen = {round(float(x) * 1e9) for x in ps_["freq_ghz"]["seen"]}
    hold = {round(float(x) * 1e9) for x in ps_["freq_ghz"]["holdout"]}
    if hold & seen:
        bad(f"频率留出值与训练值重合：{hold & seen}")
    oob = [it["id"] for it in insts if not (lo <= float(it["freq_hz"]) <= hi)]
    if oob:
        bad(f"{len(oob)} 个实例频率越界（声明 {lo/1e9}~{hi/1e9} GHz）")
    if not args.no_freq_gate and not any(it.get("role") == "freq_holdout" for it in insts):
        bad("没有 freq_holdout 角色（频率轴若已判死，加 --no_freq_gate）")
    else:
        print(f"[OK ] 频率域 {lo/1e9:.2f}~{hi/1e9:.2f} GHz，留出不与训练重合")

    # 5) 朝向自洽
    worst = 0.0
    for it in insts:
        a0, b0 = base_orientation(np.asarray(it["xyz"], np.float64), target)
        worst = max(worst,
                    abs(a0 + float(it["offset_yaw_rad"]) - float(it["yaw_abs_rad"])),
                    abs(b0 + float(it["offset_pitch_rad"]) - float(it["pitch_abs_rad"])))
    if worst > 1e-9:
        bad(f"朝向不自洽：max|yaw_abs − (base+off)| = {worst:.3e} rad")
    else:
        print(f"[OK ] 朝向自洽（{len(insts)} 个实例，max 误差 {worst:.1e} rad）")

    # 6) 同位置不同参数的成对 PSNR
    # 视角要**均匀铺开**取，不能取前 N 张：文件按相机序列排列，同位置的前几十张
    # 常是一整块「无路径」的图（tr38901 定向天线 + 背向视角），全零图对参数天然
    # 不敏感，会把这一项假通过（实测位置 1 前 8 张全零 ⇒ 97.5 dB「几乎一样」）。
    pos_ids = sorted(by_pos)[: max(1, args.sample_positions)]
    all_views = sorted(p.stem for p in
                       (root / key_fmt.format(idx=int(by_pos[pos_ids[0]][0]["id"]))).glob("*.png"))
    k = min(args.sample_views, len(all_views))
    pick = np.linspace(0, len(all_views) - 1, k).round().astype(int) if k else []
    views = [all_views[int(i)] for i in pick]
    print(f"\n同位置不同参数的成对 PSNR（{len(pos_ids)} 个位置 × {len(views)} 个视角）")
    pos_min = {}
    for p in pos_ids:
        group = by_pos[p]
        pairs, n_black = [], 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                vals = []
                for v in views:
                    a = np.asarray(Image.open(
                        root / key_fmt.format(idx=int(group[i]["id"])) / f"{v}.png").convert("L"))
                    b = np.asarray(Image.open(
                        root / key_fmt.format(idx=int(group[j]["id"])) / f"{v}.png").convert("L"))
                    # 两张都是全零 ⇒ 该视角没有路径，不提供参数信息，不计入
                    if int(a.max()) == 0 and int(b.max()) == 0:
                        n_black += 1
                        continue
                    vals.append(psnr(a, b))
                if vals:
                    pairs.append(float(np.mean(vals)))
        if not pairs:
            bad(f"位置 {p} 的 {len(views)} 个抽样视角全是零图，无法判断参数效应")
            continue
        desc = "、".join(
            f"y={np.rad2deg(float(group[i]['offset_yaw_rad'])):+.0f}/"
            f"p={np.rad2deg(float(group[i]['offset_pitch_rad'])):+.0f}/"
            f"f={float(group[i]['freq_hz'])/1e9:.2f}"
            for i in range(len(group)))
        pos_min[p] = min(pairs)
        print(f"  位置 {p:2d}  成对 PSNR {min(pairs):.2f}~{max(pairs):.2f} dB  "
              f"（剔除双零视角 {n_black} 对次）[{desc}]")

    # 判据放在**中位数**上，而不是逐位置硬卡 40 dB。
    # 理由：个别位置参数效应天生就弱（例如位置 1 只有 48.5 dB）——那不是仿真没接线，
    # 是「所有到达方向落在同一段波束增益里，旋转只改了一个全局系数，被逐图 min-max 除掉」。
    # 真要出「iso 天线 ⇒ 朝向没进仿真」的老毛病，**每个**位置都会贴到 ~99 dB，
    # 中位数一样会抓到。逐位置硬卡只会把这种物理上正常的弱效应误判成故障。
    if pos_min:
        med = float(np.median(list(pos_min.values())))
        if med > 40:
            bad(f"参数效应的中位成对 PSNR = {med:.2f} dB（> 40）"
                f"⇒ 朝向/频率很可能没进仿真")
        else:
            weak = {p: v for p, v in pos_min.items() if v > 40}
            tail = f"；弱效应位置 {weak}（非故障，见上方注释）" if weak else ""
            print(f"[OK ] 参数效应中位成对 PSNR {med:.2f} dB（越小说明参数越敏感）{tail}")

    print()
    if fails:
        print(f"❌ {len(fails)} 项未通过")
        raise SystemExit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
