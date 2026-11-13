# -*- coding: utf-8 -*-
"""给训练配置补「同配置、不同 w」的对照实例。

为什么需要
--------------------------------------------------------------------------------
现有 w 数据集里，144 个训练实例 = 144 个互不相同的场景配置
（位置, yaw, pitch, 频率），**没有一个配置在第二个 w 下出现**。而模型每个高斯带
32 维可学习嵌入，把 144 个配置逐个记住就能把训练 loss 压到最低 —— 梯度没有任何
理由去用 w 那 5 维。全数据集仅有的 36 个「同配置不同 w」对照全在 `w_holdout`
（测试集）里。

这个脚本只做一件事：为每个 `role=train` 的实例克隆出 K 个**配置完全相同、只有
w_label 不同**的实例，编号接在原有 id 之后。生成图像后，同一个配置就在训练里
出现 K+1 次、对应 K+1 个不同的 w —— 记忆无法同时拟合两个不同的 GT，模型只能
通过 w 通道区分它们。

w 的取法
--------------------------------------------------------------------------------
只用**训练码本**（theta_train × phi_train，36 个），绝不用留出码本
（theta_hold / phi_hold）—— 那是测试集。每个配置按轮转方式分配，保证 36 个 w
的出现次数尽量均匀，且不与自己的原 w 重复。

产物
--------------------------------------------------------------------------------
`<out>/meta.json`：原 354 个实例原样保留（id 0–353），追加 K×144 个新实例。
目录名仍是 `inst{id:03d}`，与 `SpectrumStore` 的 key_fmt 一致。

用法::

    $PY code/tools/make_w_pair_instances.py \
        --base datasets/RF-3DGS_mvdr_w_params/meta.json \
        --out datasets/RF-3DGS_cbf_w_pairs --k 1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="datasets/RF-3DGS_mvdr_w_params/meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="每个配置补几个额外 w")
    ap.add_argument("--roles", default="train", help="给哪些角色补对照（逗号分隔）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads(Path(args.base).read_text(encoding="utf-8"))
    insts = meta["instances"]
    by_id = {int(i["id"]): i for i in insts}
    id_max = max(by_id)
    if sorted(by_id) != list(range(len(by_id))):
        raise SystemExit(f"基础实例表 id 不连续（0..{id_max}），无法顺延编号")

    cb = meta["codebook"]
    labels = []
    for t in cb["theta_train_deg"]:
        for p in cb["phi_train_deg"]:
            lab = f"t{int(round(t)):02d}_p{int(round(p)):03d}"
            labels.append(lab)
    # 只保留基础表里真的用过、且不是 uniform 的写法
    known = {str(i.get("w_label")) for i in insts}
    weird = [l for l in labels if l not in known]
    if weird:
        print(f"[warn] 训练码本里有 {len(weird)} 个标签在基础表中没出现过："
              f"{weird[:5]} …（照常可用）")
    print(f"训练码本 {len(labels)} 个 w：{labels[:3]} … {labels[-2:]}")

    want_roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    targets = [i for i in insts if i.get("role", "train") in want_roles]
    print(f"要补对照的实例 {len(targets)} 个（角色 {sorted(want_roles)}）")

    rng = random.Random(args.seed)
    # 每个复制层各洗一次牌，轮转分配 ⇒ 各 w 出现次数差 ≤ 1
    new_insts = []
    next_id = id_max + 1
    for j in range(args.k):
        order = labels[:]
        rng.shuffle(order)
        for n, base in enumerate(targets):
            own = str(base.get("w_label", "uniform"))
            pick = order[n % len(order)]
            if pick == own:                       # 与自己的 w 撞了 → 顺延一个
                pick = order[(n + 1) % len(order)]
                if pick == own:
                    pick = order[(n + 2) % len(order)]
            if pick == own:
                raise SystemExit(f"配置 {base['id']} 找不到可用的替代 w")
            d = copy.deepcopy(base)
            d["id"] = next_id
            d["w_label"] = pick
            d["role"] = base.get("role", "train")
            d["pair_of"] = int(base["id"])
            d["pair_index"] = j + 1
            new_insts.append(d)
            next_id += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_insts = insts + new_insts
    # 自检：每个被补的配置至少有 2 个不同 w；没有任何配置被补两次同 w
    groups = {}
    for i in all_insts:
        key = (i["position_id"], round(float(i["offset_yaw_rad"]), 6),
               round(float(i["offset_pitch_rad"]), 6), round(float(i["freq_hz"]), 1))
        groups.setdefault(key, []).append(i)
    paired = [v for v in groups.values() if len(v) > 1]
    bad_same = [v for v in paired if len({str(x["w_label"]) for x in v}) < 2]
    if bad_same:
        raise SystemExit(f"[FAIL] {len(bad_same)} 个配置的多个副本 w 相同")
    print(f"配置数 {len(groups)}，其中被补成对照的 {len(paired)} 个"
          f"（每个配置的 w 数分布 {dict(Counter(len(v) for v in paired))}）")

    wc = Counter(str(i["w_label"]) for i in new_insts)
    print(f"新实例 {len(new_insts)} 个，覆盖 {len(wc)} 个 w，"
          f"每 w 出现 {min(wc.values())}–{max(wc.values())} 次")

    new_meta = {
        "format": "cbf_w_pairs_v1",
        "base_meta": args.base,
        "note": "为每个训练配置补了同配置、不同 w 的对照实例；id 0..base_max 与基础表逐字相同",
        "protocol": meta.get("protocol", {}),
        "codebook": cb,
        "pairing": {"k": args.k, "roles": sorted(want_roles),
                    "seed": args.seed, "base_n": len(insts),
                    "pair_n": len(new_insts)},
        "counts": {"instances": len(all_insts),
                   "base_instances": len(insts),
                   "pair_instances": len(new_insts),
                   "views_per_instance": 640},
        "instances": all_insts,
    }
    if "split" in meta:
        new_meta["split"] = meta["split"]
    p = out / "meta.json"
    p.write_text(json.dumps(new_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}（{len(all_insts)} 个实例）")

    ids_new = [i["id"] for i in new_insts]
    print(f"新实例 id 段 {ids_new[0]}..{ids_new[-1]}；"
          f"生成命令用 --ids，或按 --shard 切分")


if __name__ == "__main__":
    main()
