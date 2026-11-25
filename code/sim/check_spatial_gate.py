# -*- coding: utf-8 -*-
"""空间场数据集的**渲染口径闸**：在答案已知处证明这条链有分辨力。

三道检查（前两道是合成答案，第三道只是目视预览）：
    1) 常数场    逐点 u≡c 渲出来必须等于 c·α（α 由 u≡1 渲染得到）
                 —— 证明「逐点标量 → 光栅化逐通道混合 → 标量图」没有缩放/通道错位
    2) 面标签    按坐标给 6 个面各一个唯一 u，从 6 个朝向各渲一次，
                 覆盖率高的像素里主导面必须等于该朝向看向的面
                 —— 证明相机 R/T 与点云归一化系对齐、6 个朝向没接错
    3) 真实场    取 3 个 TX × 2 个 RX × 6 面，存一张拼接图供目视
                 —— 只能看形态，**不能**当数值证据

用法（WSL mvsplat 环境）：
    python code/sim/check_spatial_gate.py --root datasets/SionnaRoom \
        --out results/spatial/gate
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from corf.spatial_data import SpatialScene, CLOUD_OPACITY_LOGIT   # noqa: E402
from splat.color_scalar import apply_jet, make_jet_lut            # noqa: E402
from rfgs.coupled.train_student import _Pipe                                    # noqa: E402


def log(msg):
    print(msg, flush=True)


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = ArgumentParser()
    ap.add_argument("--root", default="datasets/SionnaRoom")
    ap.add_argument("--out", default="results/spatial/gate")
    ap.add_argument("--tx", nargs="+", type=int, default=[0, 60, 140])
    ap.add_argument("--rx", nargs="+", type=int, default=[1, 22])
    args = ap.parse_args()

    torch.cuda.set_device(0)
    os.makedirs(args.out, exist_ok=True)
    sc = SpatialScene(args.root)
    pipe = _Pipe()
    bg = torch.zeros(3, device="cuda")
    lut = make_jet_lut(256).cuda()
    P = sc.P
    log(f"[gate] P={P}  faces={sc.face_names}  H=W={sc.H}  "
        f"opacity={torch.sigmoid(torch.tensor(CLOUD_OPACITY_LOGIT)):.4f}")

    # ---------------------------------------------------------------- 1 常数场
    u_one = torch.ones(P, device="cuda")
    cam = sc.camera(0, 0)
    alpha = sc.render_scalar(cam, u_one, pipe, bg)[0]
    report = {"coverage": {}}
    with torch.no_grad():
        for c in (0.2, 0.5, 0.8):
            u = sc.render_scalar(cam, torch.full((P,), c, device="cuda"), pipe, bg)[0]
            # 有覆盖的像素上，混合值应严格等于 c·α
            m = alpha > 1e-4
            d = (u - c * alpha)[m].abs()
            report["coverage"][f"const_{c}"] = {
                "covered_frac": float(m.float().mean()),
                "max_abs": float(d.max()), "mean_abs": float(d.mean())}
            log(f"[gate] 常数场 c={c}: 覆盖 {float(m.float().mean())*100:.1f}%  "
                f"max|u−cα|={float(d.max()):.2e}  mean={float(d.mean()):.2e}")
    if not all(v["max_abs"] < 2e-3 for v in report["coverage"].values()):
        raise SystemExit("[gate] FAIL：常数场渲染与 c·α 不符")

    # ---------------------------------------------------------------- 2 面标签
    # ⚠ 点云存的是**归一化坐标**（meta.norm_unit），不是米；6 个面的位置由各轴
    # 极值给出。按「到哪个面最近」分类，避免角点被阈值重复认领（0.01 m 的
    # 内缩比 0.05 m 的格子小，纯阈值法会把地板边缘判成墙）。
    xyz = sc.xyz
    lo = xyz.min(dim=0).values
    hi = xyz.max(dim=0).values
    planes = [  # (轴, 常数) 顺序与 names 一致
        (2, lo[2]), (2, hi[2]),
        (0, hi[0]), (0, lo[0]),
        (1, hi[1]), (1, lo[1]),
    ]
    dist = torch.stack([(xyz[:, a] - c).abs() for a, c in planes], dim=1)
    labels = dist.argmin(dim=1)
    names = ["floor", "ceil", "x+", "x-", "y+", "y-"]
    if int(dist.min(dim=1).values.max() > 0.02):
        raise SystemExit("[gate] FAIL：有点离 6 个面都很远，点云与面定义不符")
    # 面序号 -> 该朝向应主要看到的面
    expect = {"px": "x+", "nx": "x-", "py": "y+", "ny": "y-", "pz": "ceil", "nz": "floor"}
    # 标签值用 k/5，相邻面差 0.2，足够区分；u 在光栅化里是线性混合，取最近标签
    u_lab = (labels.float() / 5.0)
    report["faces"] = {}
    ry = args.rx[0]
    # ⚠ 不能用「整图主导面」判朝向：95° 视场下站在墙角时，侧墙的张角
    # （~83°）比正对的墙（~43°）还大，主导面天然是侧墙。中心像素才是
    # 「相机朝哪看」的直接读数。
    cy, cx = sc.H // 2, sc.W // 2
    for f, fname in enumerate(sc.face_names):
        cam = sc.camera(ry, f)
        with torch.no_grad():
            a = sc.render_scalar(cam, u_one, pipe, bg)[0]
            ul = sc.render_scalar(cam, u_lab, pipe, bg)[0]
        patch = ul[cy - 2:cy + 3, cx - 2:cx + 3]
        apatch = a[cy - 2:cy + 3, cx - 2:cx + 3]
        k = torch.round(patch[apatch > 0.99] * 5.0).long().clamp(0, 5)
        top = int(k.float().median().round())
        ok = names[top] == expect[fname]
        # 滚转检查：侧向 4 面 down=∓z，故图上方 z 更大（u_z 更大）
        roll_ok = True
        if fname in ("px", "nx", "py", "ny"):
            uz = (xyz[:, 2] - lo[2]) / (hi[2] - lo[2])
            with torch.no_grad():
                rz = sc.render_scalar(cam, uz, pipe, bg)[0]
            col = rz[:, cx]
            roll_ok = bool(col[:8].mean() > col[-8:].mean() + 0.05)
        report["faces"][fname] = {"center": names[top], "ok": bool(ok and roll_ok),
                                  "roll_ok": roll_ok,
                                  "covered": float(apatch.mean())}
        log(f"[gate] 面 {fname:2s}: 中心 {names[top]:5s} 期望 {expect[fname]:5s}  "
            f"roll {'OK' if roll_ok else 'FAIL'}  中心覆盖 {float(apatch.mean())*100:.0f}%  "
            f"{'OK' if ok and roll_ok else 'FAIL'}")
    if not all(v["ok"] for v in report["faces"].values()):
        raise SystemExit("[gate] FAIL：朝向/滚转与点云面不对应")

    # ---------------------------------------------------------------- 3 真实场预览
    tiles = []
    for j in args.tx:
        uj = sc.u(j)
        log(f"[gate] TX{j}: u p1/p50/p99 "
            f"{float(torch.quantile(uj,0.01)):.3f}/{float(torch.quantile(uj,0.5)):.3f}/"
            f"{float(torch.quantile(uj,0.99)):.3f}  "
            f"dBm p50 {float(sc.u_to_dbm(uj).median()):.1f}")
        for r in args.rx:
            row = []
            for f in range(sc.n_face):
                cam = sc.camera(r, f)
                with torch.no_grad():
                    uu = sc.render_scalar(cam, uj, pipe, bg)[0]
                    rgb = apply_jet(uu, lut)
                row.append(to_uint8(rgb))
            tiles.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(tiles, axis=0)
    p = os.path.join(args.out, "gt_preview.png")
    Image.fromarray(sheet).save(p)
    log(f"[gate] 预览 {len(args.tx)}TX×{len(args.rx)}RX×6面 -> {p}  shape {sheet.shape}")

    report["ok"] = True
    report["note"] = ("闸通过只代表渲染链在合成答案上正确；真实场的形态只供目视，"
                      "不构成数值证据。")
    with open(os.path.join(args.out, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("[gate] PASS")


if __name__ == "__main__":
    main()
