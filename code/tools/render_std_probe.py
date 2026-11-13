# -*- coding: utf-8 -*-
"""
render_std_probe.py —— 渲染图统计探针（服务器侧跑，只输出 JSON，不必传图）
--------------------------------------------------------------------------------
回答形态 A× 的两个验收问题：
  1. **症状是否缓解**：`通道 std` 是否从 A' 的 37.05（CBF）/ 52.96（MVDR）靠近 GT 的
     51.08 / 54.12？（CBF 偏低 = 过平滑；MVDR 几乎持平）
     若 A× 的 std 没上去 ⇒ "瓶颈是标量头容量"这个**先验推断被证伪**，须如实写进文档。
  2. **不变量是否还在**：离板（距 jet 折线最近距离）>5 的像素占比是否仍为 0.00%？
     口径与 .diag/check_onpalette.py 一致（同一个 make_jet_lut、同一个最近距离）。

**`通道 std` 的定义已按旧文档逐位校准**（2026-09-12 反解确认）：
    逐帧 → 对每个通道单独求像素标准差 → 三通道取算术均值 → **前 N 帧取中位**（N 默认 24）
校准证据：A'/MVDR 前 24 帧 = 52.959（文档 52.96）、GT = 54.120（54.12）、
          A'/CBF = 37.048（37.05）、GT = 51.077（51.08），色数中位 664/138/798/253 亦逐位吻合。
⚠ 常见的"全帧池化"口径会给出完全不同的数（MVDR A' 池化 = 69.02）——**别混用**。

用法（服务器 code/ 下）：
    <PY> tools/render_std_probe.py --run-dir <MD> --iteration 70000 \
        --out <MD>/std_probe_70000.json [--use-gt]
--use-gt：连 GT 一起量（GT 与臂无关，每个方法跑一次即可）。
离板统计只扫前 --off-frames 帧、每 --off-stride 像素取一个（默认 64 帧 / 步长 2）：
判据是"是否恰为 0"，约 100 万像素的样本足以发现 >0.01% 的偏移，而全量全像素要 7 分钟。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
from splat.color_scalar import make_jet_lut  # noqa: E402

_LUT = make_jet_lut(256).numpy() * 255.0
_LUT2 = (_LUT ** 2).sum(1)


def off_palette(a: np.ndarray) -> np.ndarray:
    """每个像素到 jet 折线（256 锚点）的最近距离（0-255 尺度）。口径同 check_onpalette.py。"""
    a = a.reshape(-1, 3).astype(np.float64)
    d2 = (a ** 2).sum(1)[:, None] + _LUT2[None, :] - 2.0 * (a @ _LUT.T)
    return np.sqrt(np.clip(d2, 0, None).min(1))


def scan(png_dir: Path, n_frames: int = 24, off_frames: int = 64,
         off_stride: int = 2) -> dict:
    files = sorted(p for p in png_dir.glob("*.png"))
    if not files:
        return {"n_frames": 0, "dir": str(png_dir)}

    # ── 主统计：前 n_frames 帧（文档口径）──
    sel = files[:n_frames] if (n_frames and n_frames > 0) else files
    chan_std, uniq, gray_std = [], [], []
    for p in sel:
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        chan_std.append(a.reshape(-1, 3).std(0).mean())     # 逐通道 std 再对通道取均值
        gray_std.append(a.mean(-1).std())
        uniq.append(np.unique(a.reshape(-1, 3).astype(np.uint8), axis=0).shape[0])
    chan_std = np.asarray(chan_std)

    out = {
        "n_frames": len(files),
        "n_frames_used": len(sel),
        "dir": str(png_dir),
        "chan_std_median_255": float(np.median(chan_std)),      # ← 与旧文档同口径
        "chan_std_mean_255": float(chan_std.mean()),
        "gray_std_median_255": float(np.median(gray_std)),
        "n_unique_colors_median": float(np.median(uniq)),
        "def_note": "通道 std = 逐帧逐通道 std → 三通道均值 → 前 n_frames 帧中位（旧文档口径）",
    }

    # ── 离板（抽帧抽样，判据是"是否恰为 0"）──
    if off_frames != 0:
        offs = []
        sub = files[:off_frames] if off_frames > 0 else files
        for p in sub:
            a = np.asarray(Image.open(p).convert("RGB"))[::off_stride, ::off_stride]
            offs.append(off_palette(a))
        o = np.concatenate(offs)
        out.update({
            "off_note": f"前 {len(sub)} 帧，每 {off_stride} 像素取 1，共 {o.size} 像素",
            "off_n_pixels": int(o.size),
            "off_mean": float(o.mean()),
            "off_gt1_pct": float(100.0 * (o > 1).mean()),
            "off_gt5_pct": float(100.0 * (o > 5).mean()),
        })
    return out


def show(tag: str, s: dict):
    if not s.get("n_frames"):
        print(f"{tag:8s} 缺图"); return
    print(f"{tag:8s} n={s['n_frames']:4d}(用{s['n_frames_used']})  "
          f"通道std中位={s['chan_std_median_255']:7.3f}  灰度std中位={s['gray_std_median_255']:7.3f}  "
          f"色数中位={s['n_unique_colors_median']:6.0f}  "
          f"离板>5={s.get('off_gt5_pct', float('nan')):6.3f}%  "
          f"(>1 {s.get('off_gt1_pct', float('nan')):.2f}%, 均值 {s.get('off_mean', float('nan')):.4f})")


def main():
    ap = argparse.ArgumentParser(description="渲染图 通道std / 离板 探针（服务器侧）")
    ap.add_argument("--run-dir", required=True, help="模型目录（含 test/ours_<it>）")
    ap.add_argument("--iteration", type=int, default=70000)
    ap.add_argument("--out", default=None, help="输出 JSON（默认 <run-dir>/std_probe_<it>.json）")
    ap.add_argument("--n-frames", type=int, default=24, help="通道std/色数用前几帧（0=全部）")
    ap.add_argument("--off-frames", type=int, default=64, help="离板扫描前几帧（0=跳过）")
    ap.add_argument("--off-stride", type=int, default=2, help="离板扫描像素步长")
    ap.add_argument("--use-gt", action="store_true", help="连 GT 一起量")
    args = ap.parse_args()

    run = Path(args.run_dir)
    R = run / "test" / f"ours_{args.iteration}"
    if not (R / "renders").is_dir():
        raise SystemExit(f"缺渲染目录: {R / 'renders'}")

    payload = {"run_dir": str(run), "iteration": args.iteration,
               "renders": scan(R / "renders", args.n_frames, args.off_frames, args.off_stride)}
    if args.use_gt and (R / "gt").is_dir():
        payload["gt"] = scan(R / "gt", args.n_frames, args.off_frames, args.off_stride)
    payload["def_note"] = payload["renders"].get("def_note")

    print(f"[{run.name}] ours_{args.iteration}")
    show("渲染", payload["renders"])
    if "gt" in payload:
        show("GT", payload["gt"])

    out = Path(args.out) if args.out else run / f"std_probe_{args.iteration}.json"
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
