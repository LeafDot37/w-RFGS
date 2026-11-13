# -*- coding: utf-8 -*-
"""生成「发射源参数泛化」的正式实例表（位置 + 朝向 + 频率）。

设计：**随机化完全区组**（每个位置对每个轴的每个取值恰好见一次）
--------------------------------------------------------------------------------
每个训练位置抽 4 组参数；若三轴各自独立随机，4 个样本下各轴会互相混杂，
「留出朝向」组就归因不清。所以：

    对位置 p 的 4 次抽样 k=0..3：
        yaw   = seen_yaw[perm_y[k]]
        pitch = seen_pitch[perm_p[k]]
        freq  = seen_freq[perm_f[k]]
    perm_* 是每个位置、每个轴独立的随机置换（固定种子）

⇒ 每个位置**见到每个 yaw 取值恰好一次**（pitch/f 同理），边际完全平衡；
   跨位置看，三轴彼此不相关（参数不依赖位置）。位置留出组用同一程序、
   不同种子在 13 个留出位置上抽。

角色与视角
--------------------------------------------------------------------------------
    train          36 位置 × 4 组 = 144 实例，views=all（3200）
    pos_holdout    13 位置 × 4 组 =  52 实例，views=test（640）
    yaw_holdout    36 位置 × 1     =  36 实例，views=test
    pitch_holdout  36 位置 × 1     =  36 实例，views=test
    freq_holdout   36 位置 × 1     =  36 实例，views=test
    double_holdout 13 位置 × 1     =  13 实例，views=test
    合计 317 实例；图像 144×3200 + 173×640 = 571,520

三条留出带都取**内插值**（落在训练取值的区间内、但不是任何一个训练取值），
测的是「没见过的取值」，不是外推。

朝向语义
--------------------------------------------------------------------------------
`yaw_abs = base_alpha + offset_yaw`，`pitch_abs = base_beta + offset_pitch`；
基准帧让局部 +x（波束轴）指向 RX 质心（见 `splat/param_frame.base_orientation`）。
头吃的条件是**偏移角**，几何用绝对值。

跑法：
    python code/tools/make_tx_instances.py --out datasets/RF-3DGS_cbf_params
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splat.param_frame import base_orientation                      # noqa: E402

SRC_META = "datasets/NIST_lobby_spatial_full/meta.json"


def parse_floats(s):
    return [float(x) for x in str(s).replace("，", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/RF-3DGS_cbf_params")
    ap.add_argument("--src_meta", default=SRC_META)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw_seen", default="-50,-15,15,50")
    ap.add_argument("--yaw_hold", default="-32,0,32")
    ap.add_argument("--pitch_seen", default="-20,-7,7,20")
    ap.add_argument("--pitch_hold", default="-13.5,0,13.5")
    ap.add_argument("--freq_seen", default="2.10,2.60,3.10,3.60")
    ap.add_argument("--freq_hold", default="2.35,2.85,3.35")
    ap.add_argument("--freq_lo", type=float, default=2.0)
    ap.add_argument("--freq_hi", type=float, default=4.5)
    ap.add_argument("--tx_pattern", default="tr38901")
    ap.add_argument("--orig_yaw", type=float, default=0.0,
                    help="原 RF-3DGS 发射源（TX48）的朝向偏移，度；0 = 指向场景质心")
    ap.add_argument("--orig_pitch", type=float, default=0.0)
    ap.add_argument("--orig_freq_ghz", type=float, default=2.40)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = json.loads(Path(args.src_meta).read_text(encoding="utf-8"))
    tx = np.asarray(src["tx_world"], np.float64)
    pos_train = [int(i) for i in src["split"]["train"]]
    pos_test = [int(i) for i in src["split"]["test"]]
    orig = int(src.get("orig_tx_index", -1))
    target = np.asarray(src["rx_world"], np.float64).mean(0)

    ys = parse_floats(args.yaw_seen)
    yh = parse_floats(args.yaw_hold)
    ps = parse_floats(args.pitch_seen)
    ph = parse_floats(args.pitch_hold)
    fs = parse_floats(args.freq_seen)
    fh = parse_floats(args.freq_hold)
    for name, seen, hold in (("yaw", ys, yh), ("pitch", ps, ph), ("freq", fs, fh)):
        if len(seen) != 4:
            raise SystemExit(f"--{name}_seen 必须 4 个取值（完全区组），收到 {seen}")
        lo, hi = min(seen), max(seen)
        bad = [h for h in hold if not (lo < h < hi)]
        if bad:
            raise SystemExit(f"--{name}_hold 必须是内插值（{lo}~{hi} 之间且不等于端点）：{bad}")

    rng = np.random.default_rng(args.seed)
    insts = []

    def add(pos, yaw_off, pitch_off, freq_ghz, role, views, label):
        xyz = tx[pos]
        a0, b0 = base_orientation(xyz, target)
        insts.append({
            "id": len(insts),
            "position_id": int(pos),
            "xyz": [float(x) for x in xyz],
            "offset_yaw_rad": float(np.deg2rad(yaw_off)),
            "offset_pitch_rad": float(np.deg2rad(pitch_off)),
            "yaw_abs_rad": float(a0 + np.deg2rad(yaw_off)),
            "pitch_abs_rad": float(b0 + np.deg2rad(pitch_off)),
            "freq_hz": float(freq_ghz * 1e9),
            "role": role,
            "views": views,
            "label": label,
        })

    def block(positions, seed_offset, views_expect):
        r = np.random.default_rng(args.seed + seed_offset)
        for p in positions:
            py, pp, pf = (r.permutation(4) for _ in range(3))
            for k in range(4):
                add(p, ys[py[k]], ps[pp[k]], fs[pf[k]], "train" if views_expect == "all"
                    else "pos_holdout", views_expect, f"d{k}")

    # 训练：36 位置 × 4 组
    block(pos_train, 1, "all")
    # 位置留出：13 位置 × 4 组（同一程序，不同种子）
    block(pos_test, 2, "test")

    # 单轴留出：每个训练位置 1 个「没见过的取值」，其余两轴取训练取值
    for role, vals, idx in (("yaw_holdout", yh, 0),
                            ("pitch_holdout", ph, 1),
                            ("freq_holdout", fh, 2)):
        for i, p in enumerate(pos_train):
            v = vals[i % len(vals)]
            y = ys[int(rng.integers(len(ys)))]
            pp = ps[int(rng.integers(len(ps)))]
            f = fs[int(rng.integers(len(fs)))]
            if idx == 0:
                y = v
            elif idx == 1:
                pp = v
            else:
                f = v
            add(p, y, pp, f, role, "test", f"v{v:g}")

    # 双留出：13 个留出位置 × 三轴都取留出值
    for i, p in enumerate(pos_test):
        add(p, yh[i % len(yh)], ph[i % len(ph)], fh[i % len(fh)],
            "double_holdout", "test", "joint")

    # 原 RF-3DGS 发射源（作者 TX48）＝主读数：位置在 test 划分里，朝向取场景基准帧
    # （偏移 0°/0°，新协议下最接近原始 iso 配置），频率 2.4 GHz（原始值）。
    # 三个参数都不在训练取值里 ⇒ 同时是「未见位置 + 未见参数」。
    if orig < 0:
        raise SystemExit(f"{args.src_meta} 没有 orig_tx_index")
    if orig not in pos_test:
        raise SystemExit(f"原发射源 {orig} 不在 test 划分里")
    add(orig, args.orig_yaw, args.orig_pitch, args.orig_freq_ghz, "orig", "test", "orig")
    orig_iid = len(insts) - 1

    # 校验
    roles = {}
    for it in insts:
        roles.setdefault(it["role"], []).append(it["id"])
    ids = [it["id"] for it in insts]
    assert ids == list(range(len(ids))), "id 必须连续"

    # 参数-位置独立性自检：把每个参数对位置坐标回归，R² 应接近 0
    def r2(param, coord):
        A = np.stack([coord, np.ones_like(coord)], 1)
        coef, *_ = np.linalg.lstsq(A, param, rcond=None)
        pred = A @ coef
        ss = ((param - param.mean()) ** 2).sum()
        return 1.0 - ((param - pred) ** 2).sum() / max(ss, 1e-12)

    tr = [it for it in insts if it["role"] == "train"]
    X = np.array([it["xyz"] for it in tr])
    checks = {}
    for key in ("offset_yaw_rad", "offset_pitch_rad", "freq_hz"):
        v = np.array([it[key] for it in tr])
        checks[key] = float(max(abs(r2(v, X[:, d])) for d in range(3)))

    n_views = {"all": 3200, "test": 640}
    n_img = sum(n_views[it["views"]] for it in insts)
    meta = {
        "format": "cbf_params_v1",
        "scene": "NIST_lobby",
        "source_position_meta": args.src_meta,
        "tx_world": [[float(x) for x in p] for p in tx],
        "split": {"train": pos_train, "test": pos_test},
        "orig_tx_index": orig,
        "base_frame": {
            "type": "scene_facing",
            "target_world": [float(x) for x in target],
            "rotation": "TR38901 (7.1-4), angles in radians, roll fixed 0, local +x is boresight",
        },
        "param_space": {
            "yaw_off_deg": {"seen": ys, "holdout": yh},
            "pitch_off_deg": {"seen": ps, "holdout": ph},
            "freq_ghz": {"declared": [args.freq_lo, args.freq_hi],
                         "seen": fs, "holdout": fh},
            "power": "not modelled (per-image min-max makes it exactly unobservable)",
        },
        "design": {"type": "randomized_complete_block", "seed": args.seed,
                   "n_draws_per_position": 4},
        "protocol": {
            "tx_array": {"pattern": args.tx_pattern, "polarization": "V"},
            "rx_array": {"M": 10, "pattern": "tr38901"},
            "material_freq_hz": 6.0e10,
            "max_depth": 1, "diffuse": False,
            "width": 300, "height": 200, "time_interval_ns": 0.5,
            "tx_power_dbm": 0.0, "normalize": "per_image_minmax",
            "materials": "tutorial @60GHz, frequency_update_callback frozen",
        },
        "orig_instance_id": int(orig_iid),
        "roles": {k: {"n": len(v), "ids": v[:8]} for k, v in roles.items()},
        "n_instances": len(insts),
        "n_images": n_img,
        "independence_check_max_R2": checks,
        "instances": insts,
    }

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} 非空；加 --overwrite 才覆盖（不会删图，只重写 meta）")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    (out / "instances.json").write_text(
        json.dumps({"instances": insts}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"写出 {out}/meta.json 与 instances.json")
    print(f"  实例 {len(insts)}；预计图像 {n_img:,} 张")
    for k, v in roles.items():
        print(f"  {k:15s} {len(v):4d}")
    print(f"  参数-位置独立性 max R²: {checks}")


if __name__ == "__main__":
    main()
