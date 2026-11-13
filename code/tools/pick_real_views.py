# -*- coding: utf-8 -*-
"""
pick_real_views.py —— 从评测用的真实机位里，选出一条「全真实」的轨迹
================================================================================
回答的问题：环绕/推轨那两条轨迹的机位**数据集没拍过**，那里**没有真值**，
所以「三方法 + 原始真实结果」的四栏对比**不能**建在它们上面。

但真值是有的 —— 评测用的 640 个机位里，`gt/` 就是数据集的原图（实测：
`test/ours_<档>/gt/<i>.png` 与 `images/<test_index[i]>.png` 下采样后**逐像素差 0**，
最大差 0）。所以只要把「轨迹」换成真实机位序列，每一帧就都有真值，
而且**三臂在同一机位的渲染已经存在**（就是那 640 张）。

实测的机位结构（`cameras.json` + `test_index.txt`，640 台 test 机位）
--------------------------------------------------------------------------------
  - 640 台 = **160 个位置 × 4 个朝向**；同一位置的 4 台**位置完全相同**，
    朝向是**固定的世界方向 {270°, 0°, 90°, 180°}**（与位置无关，像一台四向机架）。
  - 位置绕场景约 356°（半径 3.29 ~ 7.89 m），天顶角恒 90°（光轴水平、零 roll）。
  - 直接按 index 顺序播放会「原地转 4 下、再跳 1~5.7 m」（相邻帧 75% 位移≈0）⇒ 观感发飘。

本脚本的选法（**每一帧都是真实相机**，只做"选"不做"造"）
--------------------------------------------------------------------------------
  1. 按渲染索引顺次扫，位置相对上一台移动 > `--pos-tol` 米就开一个新"位置"；
  2. 每个位置从它的 4 个朝向里选**最朝向场景质心**的那个
     （朝向量化到 4 个方向 ⇒ 与内向的夹角最大只到 45°，这是数据的固有分辨率，不是误差）；
  3. 选出的 160 个位置**按绕质心的方位角升序**排序 ⇒ 得到一条环绕顺序；
  4. 逐帧记录诚实性读数：与内向夹角、到最近真实机位的距离。

⚠ 排序是**我们加的**（数据本身没有这条顺序），其余（位置、朝向、图像）全部来自数据集。
   这一点写进 选择.json 的 `规则`，不靠嘴说。

用法（服务器 code/ 下；用 gs3d 的 python，需要 numpy）：
    <PY> tools/pick_real_views.py \
        --dataset /data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MPC_100 \
        --cameras-json <某个模型目录>/cameras.json \
        --out <run>/真实机位_选择.json --index-out <run>/真实机位_索引.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="从真实 test 机位里选出一条全真值轨迹")
    ap.add_argument("--dataset", required=True, help="数据集根（含 test_index.txt 与 images/）")
    ap.add_argument("--cameras-json", required=True, help="含全部相机 position/rotation 的 cameras.json")
    ap.add_argument("--out", required=True, help="输出 选择.json")
    ap.add_argument("--index-out", default="", help="另出一份给 shell 读的纯文本索引表")
    ap.add_argument("--pos-tol", type=float, default=0.05,
                    help="判定「同一位置」的位移阈值（米）；实测同位置 4 台位移恒 0")
    args = ap.parse_args()

    ds = Path(args.dataset)
    test = [l.strip() for l in open(ds / "test_index.txt", encoding="utf-8") if l.strip()]
    cams = {c["img_name"]: c for c in json.load(open(args.cameras_json, encoding="utf-8"))}
    miss = [n for n in test if n not in cams]
    if miss:
        raise SystemExit(f"cameras.json 里缺 {len(miss)} 个 test 机位，例如 {miss[:3]}")

    P = np.array([cams[n]["position"] for n in test], dtype=np.float64)
    R = np.array([cams[n]["rotation"] for n in test], dtype=np.float64)   # camera-to-world
    fwd = R[:, :, 2]                                                      # 光轴 = 第 3 列
    N = len(test)

    # ── 1) 按位置分"点"：顺次扫，位移超阈值就开新点 ──────────────────────────
    spots = []          # [(位置中心, [渲染索引...])]
    for i in range(N):
        if not spots or np.linalg.norm(P[i] - spots[-1][0]) > args.pos_tol:
            spots.append([P[i].copy(), [i]])
        else:
            spots[-1][1].append(i)
    grp_sizes = [len(s[1]) for s in spots]

    cen = P[:, :2].mean(0)
    inward = np.concatenate([cen[None, :] - P[:, :2],
                             np.zeros((N, 1))], axis=1)
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)

    # ── 2) 每个位置选最朝内的那一台 ─────────────────────────────────────────
    picked = []
    for pos, idxs in spots:
        cos = np.array([float(np.dot(fwd[i], inward[i])) for i in idxs])
        j = idxs[int(np.argmax(cos))]        # argmax 取第一个 ⇒ 平局时确定性
        picked.append(j)

    az_pos = np.degrees(np.arctan2(P[picked, 1] - cen[1], P[picked, 0] - cen[0])) % 360
    # ── 3) 按方位角升序排（kind="stable" 保平局确定性）────────────────────────
    order = np.argsort(az_pos, kind="stable")
    picked = [picked[k] for k in order]

    # ── 4) 逐帧读数 ────────────────────────────────────────────────────────
    frames = []
    for k, i in enumerate(picked):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        f = fwd[i]
        frames.append({
            "序号": k, "渲染索引": int(i), "img_name": test[i],
            "位置": [round(float(v), 6) for v in P[i]],
            "位置方位": round(float(np.degrees(np.arctan2(P[i, 1] - cen[1], P[i, 0] - cen[0])) % 360), 3),
            "半径": round(float(np.linalg.norm(P[i, :2] - cen)), 4),
            "朝向方位": round(float(np.degrees(np.arctan2(f[1], f[0])) % 360), 3),
            "天顶角": round(float(np.degrees(np.arccos(np.clip(f[2], -1, 1)))), 3),
            "与内向夹角": round(float(np.degrees(np.arccos(np.clip(np.dot(f, inward[i]), -1, 1)))), 3),
            "到最近真实机位": round(float(d.min()), 4),
        })

    pos_step = np.linalg.norm(np.diff(P[picked], axis=0), axis=1)
    ang_step = np.diff(np.array([fr["位置方位"] for fr in frames]))
    ang_step = np.where(ang_step < -180, ang_step + 360, ang_step)
    stat = {
        "位置数": len(spots), "每位置朝向数": sorted(set(grp_sizes)),
        "选取帧数": len(frames),
        "半径 min/中位/max": [round(float(np.linalg.norm(P[picked][:, :2] - cen, axis=1).min()), 3),
                              round(float(np.median(np.linalg.norm(P[picked][:, :2] - cen, axis=1))), 3),
                              round(float(np.linalg.norm(P[picked][:, :2] - cen, axis=1).max()), 3)],
        "相邻位置位移 min/中位/max": [round(float(pos_step.min()), 3),
                                      round(float(np.median(pos_step)), 3),
                                      round(float(pos_step.max()), 3)],
        "相邻方位步进 min/中位/max": [round(float(ang_step.min()), 2),
                                      round(float(np.median(ang_step)), 2),
                                      round(float(ang_step.max()), 2)],
        "与内向夹角 max": max(fr["与内向夹角"] for fr in frames),
        "天顶角去重": sorted({fr["天顶角"] for fr in frames}),
    }
    payload = {
        "规则": ("每一帧都是**真实机位**（位置、朝向、图像全部取自数据集），只做「选」不做「造」："
                 "先按渲染索引把 640 台聚成位置点（同位置 4 个朝向），每个位置取最朝向场景质心的"
                 "那一台，再把选出的位置**按绕质心的方位角升序**排序。"
                 "排序是本脚本加的（数据本身没有这条顺序），位置/朝向/图像均非合成。"
                 "故本轨迹的每一帧都有真值，可做 GT + 三臂的严格同机位四栏对比。"),
        "数据集": str(ds),
        "cameras_json": str(args.cameras_json),
        "索引映射": "渲染索引 i ↔ test_index.txt 第 i 行 ↔ images/<img_name>.png（实测下采样后逐像素差 0）",
        "统计": stat,
        "每帧": frames,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[ok] {out}  帧数 {len(frames)}")

    if args.index_out:
        p = Path(args.index_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for fr in frames:
                fh.write(f"{fr['序号']:05d} {fr['渲染索引']:05d} {fr['img_name']}\n")
        print(f"[ok] {p}")

    # 断言：选出的每一帧所在位置点必须都在原始 test 集里（本就成立，防手滑改坏）
    assert len(frames) == len(spots) == len(set(picked))
    print(f"[stat] 位置 {len(spots)} 个（每位置 {sorted(set(grp_sizes))} 个朝向）→ 选 {len(frames)} 帧")
    print(f"[stat] 半径 {stat['半径 min/中位/max']} m；相邻位移 {stat['相邻位置位移 min/中位/max']} m")
    print(f"[stat] 方位步进 {stat['相邻方位步进 min/中位/max']}°；与内向夹角 max {stat['与内向夹角 max']}°；"
          f"天顶角 {stat['天顶角去重']}")


if __name__ == "__main__":
    main()
