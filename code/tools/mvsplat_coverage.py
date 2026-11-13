#!/usr/bin/env python3
"""mvsplat_coverage.py —— MVSplat init 点云「覆盖度」评测（CPU 秒级）。

口径与 rf_view_check.py / baseline 验证完全一致：
  - 相机内参 FX=FY=800、CX=800、CY=450（1600×900）
  - c2w 翻转 `[:, :3, 1:3] *= -1`（与 init_cloud.py 同一世界系）
  - 针孔投影，z 前向(z>0.1)；z-buffer nearest-wins 投影到黑底
  - 「画面覆盖率」= 点亮像素(去重) / 总像素 1600×900 （% 1 位小数，同 baseline 10.3~26 量级）
  - 「命中精度」= 点亮像素落在实拍合成图非黑目标区(real.sum>30)的比例（同 rf_view_check object-hit）
评测位置：cams = 环轨迹站号（transforms_train.json 帧下标），默认 0/200/400/600
（与快速查看/验证视角一致）。

输出（--out）：
  覆盖率.png                  各位置 画面覆盖率 柱状
  覆盖率视角_原图vs投影.png    4 行 2 列蒙太奇：行=位置，列=[原图(实拍合成) | 生成图(点云投影)]
  验证数据.json                每位置 覆盖率/相机前点数/点亮像素/命中精度/深度中位 + 点数 + 配置

用法（WSL mvsplat env）：
  python tools/mvsplat_coverage.py --cloud <run>/mvsplat/点云/init.ply \
      --src <blender_visual_dataset> --out <run>/mvsplat [--cams "0 200 400 600"]
配置串从 env RF_* 读（evaluate_run.sh 已 export；缺失时留空），不强行依赖 config.yaml。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

CAMS_DEFAULT = [0, 200, 400, 600]
FX = FY = 800.0
CX, CY = 800.0, 450.0
H, W = 900, 1600


def load_cloud(path):
    v = PlyData.read(path)["vertex"]
    pos = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    return pos, rgb


def load_poses(src):
    frames = json.load(open(src / "transforms_train.json"))["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float32) for f in frames])
    c2w[:, :3, 1:3] *= -1                          # same flip as init_cloud.py
    return c2w


def project(pos, rgb, c2w):
    """Pinhole project world points -> (u, v, z, rgb_subset)（y 向下，z>0 前向）。"""
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (pos - t) @ R                             # p_cam = R^T (p - t)
    z = pc[:, 2]
    m = z > 0.1
    u = FX * pc[:, 0] / z + CX
    v = FY * pc[:, 1] / z + CY
    return u[m], v[m], z[m], rgb[m], m


def zrender(u, v, z, rgb):
    """Z-buffered nearest-wins point render at 1600×900（黑底）。返回 (img, lit_mask)。"""
    img = np.zeros((H, W, 3), np.uint8)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() == 0:
        return img, np.zeros((H, W), bool)
    xs = u[inb].astype(np.int64)
    ys = v[inb].astype(np.int64)
    zs = z[inb]
    keys = xs * H + ys
    order = np.lexsort((zs, keys))
    _, first = np.unique(keys[order], return_index=True)
    near = order[first]
    img[ys[near], xs[near]] = rgb[inb][near]
    lit = np.zeros((H, W), bool)
    lit[ys[near], xs[near]] = True
    return img, lit


def composite_tif(path):
    """RGBA 合到黑底（与 rf_view_check 完全一致）。"""
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    out = im[..., :3] * im[..., 3:4]
    return (out * 255).astype(np.uint8)


def object_hit_frac(lit, real):
    obj = real.sum(2) > 30                          # 实拍非黑目标区
    sel = obj[lit]
    return float(sel.mean()) if len(sel) else float("nan")


def cfg_str():
    """从 env RF_* 读配置串（evaluate_run.sh 已 export）；缺失留空。"""
    def g(*ks):
        for k in ks:
            v = os.environ.get(k)
            if v:
                return v
        return None
    parts = []
    kvg = [g("RF_NUM_CLUSTERS"), g("RF_VIEWS_PER_CLUSTER"), g("RF_GAZES_PER_CLUSTER")]
    if all(x is not None for x in kvg):
        parts.append("K=%s/V=%s/G=%s" % tuple(kvg))
    mg = g("RF_MAX_GAUSSIANS")
    if mg:
        parts.append("RF_MAX_GAUSSIANS=%s" % mg)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def make_coverage_chart(out_png, labels, values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _cnfont
    _cnfont.configure(plt)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color="#4C72B0", width=0.55)
    for b, val in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("画面覆盖率 (%)")
    ax.set_title("MVSplat init 点云覆盖度（1600×900, z-buffer 投影）", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 + 2)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _mpl
    _mpl.close(fig)
    print(f"  写出 {out_png}")


def make_montage(out_png, rows_data):
    """rows_data: [(label, real_img(H,W,3)uint8, proj_img(H,W,3)uint8), ...]。"""
    from PIL import Image, ImageDraw
    import _cnfont
    fnt = _cnfont.pil_font(13)
    H_IMG = 150
    imgs = []
    for lab, real, proj in rows_data:
        def _th(a):
            im = Image.fromarray(a)
            w = max(1, int(im.width * H_IMG / im.height))
            return im.resize((w, H_IMG), Image.LANCZOS)
        imgs.append((lab, _th(real), _th(proj)))
    GUTTER, PAD, HEAD, GAP = 96, 10, 26, 12
    cw = max(i[1].width for i in imgs)
    W = GUTTER + cw * 2 + GAP + PAD * 2
    H = HEAD + PAD * 2 + len(imgs) * (H_IMG + 6)
    canvas = Image.new("RGB", (W, H), (246, 246, 246))
    dr = ImageDraw.Draw(canvas)
    cx = GUTTER + PAD
    dr.text((cx, 7), "原图 (实拍合成)", font=fnt, fill=(20, 20, 20))
    dr.text((cx + cw + GAP, 7), "生成图 (点云投影)", font=fnt, fill=(20, 20, 20))
    y = HEAD + PAD
    for lab, real, proj in imgs:
        dr.text((6, y + 4), lab, font=fnt, fill=(20, 20, 20))
        canvas.paste(real, (cx, y))
        canvas.paste(proj, (cx + real.width + GAP, y))
        y += H_IMG + 6
    canvas.save(out_png)
    print(f"  写出 {out_png}  ({len(imgs)} 个位置)")


def parse_cams_arg(s):
    return [int(x) for x in str(s).replace(",", " ").split()] if s is not None else CAMS_DEFAULT


def main():
    ap = argparse.ArgumentParser(description="MVSplat init 点云覆盖度评测（CPU）")
    ap.add_argument("--cloud", required=True, help="init.ply 路径")
    ap.add_argument("--src", required=True, help="blender_visual_dataset 根（含 transforms_train.json）")
    ap.add_argument("--out", required=True, help="输出目录（写 覆盖率.png / 覆盖率视角_原图vs投影.png / 验证数据.json）")
    ap.add_argument("--cams", default=None, help="评测站号（transforms_train 帧下标），默认 0 200 400 600")
    args = ap.parse_args()

    cloud = Path(args.cloud)
    src = Path(args.src)
    out = Path(args.out)
    cams = parse_cams_arg(args.cams)
    if not cloud.exists():
        raise SystemExit(f"找不到点云: {cloud}")
    if not (src / "transforms_train.json").exists():
        raise SystemExit(f"找不到 {src / 'transforms_train.json'}（--src 应为 dataset 根）")
    out.mkdir(parents=True, exist_ok=True)

    pos, rgb = load_cloud(cloud)
    c2w = load_poses(src)
    frames = json.load(open(src / "transforms_train.json"))["frames"]
    print(f"[coverage] {len(pos)} pts  bbox "
          f"{np.round(np.vstack([pos.min(0), pos.max(0)]), 1).tolist()}")

    coverage, nfront, nlit, acc, dep = {}, {}, {}, {}, {}
    rows = []
    for cam in cams:
        u, v, z, rgb_sub, front = project(pos, rgb, c2w[cam])
        img, lit = zrender(u, v, z, rgb_sub)
        npx = int(lit.sum())
        pct = round(100.0 * npx / (H * W), 1)
        real = composite_tif(src / frames[cam]["file_path"])
        acc[cam] = round(object_hit_frac(lit, real), 3)
        cov_k = f"cam {cam}"
        coverage[cov_k] = pct
        nfront[f"cam {cam}"] = int(front.sum())
        nlit[f"cam {cam}"] = npx
        dep[f"cam {cam}"] = round(float(np.median(z)), 2) if front.sum() else None
        rows.append((f"cam {cam}  {pct:.1f}%", real, img))
        print(f"[coverage] cam {cam}: 前向 {int(front.sum()):>7} pts / 点亮 {npx:>7} px / "
              f"覆盖 {pct:.1f}% / 命中精度 {acc[cam]:.2f} / 深度p50 {dep[f'cam {cam}']}")
        del u, v, z, rgb_sub, img, lit

    payload = {
        "配置": cfg_str(),
        "方法": "rf_view_check 同口径：FX=FY=800,CX=800,CY=450 @1600x900；c2w [:, :3, 1:3] *= -1；"
                "z-buffer nearest-wins 黑底投影；覆盖率=点亮像素/总像素",
        "点数": int(len(pos)),
        "画面覆盖率": coverage,
        "相机前点数": nfront,
        "点亮像素": nlit,
        "命中精度": acc,
        "深度p50": dep,
    }
    (out / "验证数据.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  写出 {out / '验证数据.json'}")

    make_coverage_chart(out / "覆盖率.png",
                        [f"cam {c}" for c in cams], [coverage[f"cam {c}"] for c in cams])
    make_montage(out / "覆盖率视角_原图vs投影.png", rows)


if __name__ == "__main__":
    main()
