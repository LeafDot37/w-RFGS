# -*- coding: utf-8 -*-
"""按实例目录批量算 PSNR/SSIM/LPIPS（口径与 `eval_dir_pairs.py` 逐字相同）。

目录约定（`render_rrf_angular.py --role ... --out_root <root>` 的产物）：
    <root>/inst000/renders/*.png
    <root>/inst000/gt/*.png
    ...

为什么单独写一个：`eval_dir_pairs.py` 每次调用都要重建 LPIPS 网络（数秒），
317 个实例 × 两臂会白白烧掉半小时以上。这里网络只建一次。

输出：
    <root>/metrics.txt   TSV，一行一个实例（含角色、图数、三指标）
    stdout 末尾按角色给 mean ± pstdev

跑法（WSL mvsplat 环境）：
    $PY code/tools/eval/eval_instances.py --root results/<run>/test_inst \
        --meta datasets/RF-3DGS_cbf_params/meta.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from splat.utils.image_utils import psnr            # noqa: E402
from splat.utils.loss_utils import ssim             # noqa: E402
from splat.lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402


def role_map(meta_path):
    """inst id → role；meta 没有 instances 时返回空表。"""
    if not meta_path:
        return {}
    import json
    m = json.load(open(meta_path, encoding="utf-8"))
    raw = m.get("instances")
    if raw is None:
        return {}
    return {int(i["id"]): str(i.get("role", "train")) for i in raw}


def inst_id(name):
    s = name[4:] if name.startswith("inst") else name
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="每实例最多算多少张视图")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_missing_gt", action="store_true",
                    help="gt 缺图时跳过缺的那几张而不是报错")
    ap.add_argument("--skip_flat_gt", action="store_true",
                    help="GT 整幅同色（无信号视角，源 u 全 0 ⇒ jet(0) 单色）时不计入"
                         "指标，只在 nflat 列里计数。这类视角的 PSNR 取决于模型是否"
                         "恰好也输出那一个常数色，没有度量意义（实测会跳到 99 的截断值）")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = sorted([d for d in root.iterdir()
                   if d.is_dir() and d.name.startswith("inst")], key=lambda d: inst_id(d.name))
    if not dirs:
        raise SystemExit(f"{root} 下没有 inst* 目录")

    roles = role_map(args.meta)
    criterion = LPIPS("vgg", "0.1").to(args.device).eval()

    rows = []
    for d in dirs:
        iid = inst_id(d.name)
        rd, gd = d / "renders", d / "gt"
        if not rd.is_dir() or not gd.is_dir():
            print(f"[skip] inst{iid:03d} 缺 renders/ 或 gt/", flush=True)
            continue
        rn = {p.name for p in rd.glob("*.png")}
        gn = {p.name for p in gd.glob("*.png")}
        names = sorted(rn & gn)
        missing = len(gn - rn)
        if not names:
            print(f"[skip] inst{iid:03d} 没有同名图（gt {len(gn)} / renders {len(rn)}）",
                  flush=True)
            continue
        if missing and not args.skip_missing_gt:
            print(f"[warn] inst{iid:03d} 有 {missing} 张 gt 没渲染（按同名图算）", flush=True)
        if args.limit:
            names = names[: args.limit]

        ps, ss, lps = [], [], []
        nflat = 0
        for n in names:
            r = tf.to_tensor(Image.open(rd / n)).unsqueeze(0)[:, :3].to(args.device)
            g = tf.to_tensor(Image.open(gd / n)).unsqueeze(0)[:, :3].to(args.device)
            # 退化视角判定：GT 整幅只有一个颜色（逐像素与左上角比较）。
            # 不能看像素 std，也不能看 amax-amin：jet(0)=[0,0,128] 的图 std 是
            # 0.2366、amax-amin 是 0.502，两个都判不出「整幅同色」。
            flat = bool(float((g - g[:, :, :1, :1]).abs().max()) <= 1e-6)
            if flat:
                nflat += 1
                if args.skip_flat_gt:
                    continue
            with torch.no_grad():
                # 与 GT 逐字节相同时 PSNR 是 +inf；截到 99 只为让 TSV 能写。
                # 退化 GT 上这个 99 会污染均值（实测 4/8 退化 ⇒ 单实例 55.77 dB），
                # 交付口径一律配 --skip_flat_gt 跑。
                _p = float(psnr(r, g).mean())
                ps.append(_p if math.isfinite(_p) else 99.0)
                ss.append(float(ssim(r, g).mean()))
                lps.append(float(criterion(r, g).mean()))
        if not ps:
            print(f"[skip] inst{iid:03d} 全部 {nflat} 张 GT 都是退化视角", flush=True)
            continue
        row = dict(id=iid, role=roles.get(iid, "?"), n=len(ps), nflat=nflat,
                   psnr=sum(ps) / len(ps), ssim=sum(ss) / len(ss),
                   lpips=sum(lps) / len(lps))
        rows.append(row)
        print(f"INST{iid:03d} {row['role']:15s} n={row['n']:4d} flat={nflat:3d}  "
              f"PSNR {row['psnr']:7.4f}  SSIM {row['ssim']:.4f}  "
              f"LPIPS {row['lpips']:.4f}", flush=True)

    out = Path(args.out) if args.out else root / "metrics.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("id\trole\tn\tnflat\tpsnr\tssim\tlpips\n")
        for r in rows:
            f.write(f"{r['id']}\t{r['role']}\t{r['n']}\t{r['nflat']}\t{r['psnr']:.6f}\t"
                    f"{r['ssim']:.6f}\t{r['lpips']:.6f}\n")
    print(f"\n写出 {out}（{len(rows)} 个实例）")

    print("\n按角色汇总（实例级 mean ± pstdev）")
    for role in sorted({r["role"] for r in rows}):
        g = [r for r in rows if r["role"] == role]
        if not g:
            continue
        print(f"  {role:15s} k={len(g):3d}  "
              f"PSNR {statistics.mean(r['psnr'] for r in g):7.4f} ± "
              f"{statistics.pstdev(r['psnr'] for r in g):.4f}  "
              f"SSIM {statistics.mean(r['ssim'] for r in g):.4f}  "
              f"LPIPS {statistics.mean(r['lpips'] for r in g):.4f}  "
              f"[退化视角 {sum(r['nflat'] for r in g)}]")


if __name__ == "__main__":
    main()
