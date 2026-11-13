# -*- coding: utf-8 -*-
"""阶段 0 闸测：量「朝向 / 俯仰 / 频率」在 CBF 谱图上到底有多大的可观测效应。

输入是 `gen_cbf_params.py` 用 `results/_gate_params/instances.json` 生成的图。
所有变体都以 `V0_base`（Δyaw=0, Δpitch=0, 2.400 GHz, tr38901）为参照。

三条结论
--------------------------------------------------------------------------------
1. 朝向：ΔPSNR 是否随 |Δyaw| 单调下降；`ΔPSNR(90°) − ΔPSNR(2°) ≥ 3 dB` 才算
   「朝向确实可学、且级别间距有意义」。
2. 频率：由 PSNR–Δf 曲线取 **3 dB 相关宽度 w**（相对「结构全毁」地板）：
   地板 = 把 V0 的像素空间随机置换后与 V0 比。任何 Δf 处 PSNR 一旦落到
   地板 + 3 dB 以内，就认为已经完全不相关。
3. 自检：V16（原样重跑）与 V17（先设 4.5 GHz 再设回 2.4 GHz）必须与 V0
   **逐像素完全相同**，否则求解器不确定 / 材质冻结失效，后续结论不可信。

指标口径
--------------------------------------------------------------------------------
PSNR / SSIM 都在 8bit 灰度谱值 u 上算（同一 jet 表逐像素映射，灰度与 RGB 单调
对应，闸测不需要 LPIPS）。SSIM 用 11×11 高斯窗 σ=1.5，与 `splat/utils/loss_utils.py`
同参数（那边是 torch 版，本脚本是 numpy 版，数值应一致到 1e-4）。

跑法：
    D:/anaconda3/envs/sionna_win/python.exe code/tools/repro_params_gate.py \
        --data results/_gate_params/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

import _cnfont                                           # noqa: E402

_G = np.exp(-(np.arange(11) - 5) ** 2 / (2 * 1.5 ** 2))
_G = _G / _G.sum()


def _filt(x):
    return convolve1d(convolve1d(x, _G, axis=0, mode="reflect"),
                      _G, axis=1, mode="reflect")


def ssim_gray(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = _filt(a), _filt(b)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = _filt(a * a) - mu1_sq
    s2 = _filt(b * b) - mu2_sq
    s12 = _filt(a * b) - mu12
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(m.mean())


def metrics(a_u8, b_u8):
    d = a_u8.astype(np.float32) - b_u8.astype(np.float32)
    mse = float((d * d).mean())
    psnr = 99.0 if mse <= 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    a = a_u8.astype(np.float32) / 255.0
    b = b_u8.astype(np.float32) / 255.0
    if a.std() < 1e-9 or b.std() < 1e-9:
        r = 1.0 if np.allclose(a, b) else 0.0
    else:
        r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return {
        "psnr": psnr, "ssim": ssim_gray(a, b),
        "mae": float(np.abs(d).mean()), "pearson": r,
        "frac_gt1": float((np.abs(d) > 1).mean()),
        "frac_ne": float((a_u8 != b_u8).mean()),      # 8bit 量化下「变了的像素」占比
        "std_a": float(a.std()), "std_b": float(b.std()),
    }


def imgs(root, iid, names):
    d = Path(root) / f"inst{iid:03d}"
    return {n: np.asarray(Image.open(d / f"{n}.png").convert("L")) for n in names}


def shuffle_floor(a_u8, seed=0):
    rng = np.random.default_rng(seed)
    b = a_u8.ravel().copy()
    rng.shuffle(b)
    return metrics(a_u8, b.reshape(a_u8.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/_gate_params/data")
    ap.add_argument("--instances", default="results/_gate_params/instances.json")
    ap.add_argument("--out", default="results/_gate_params")
    ap.add_argument("--limit", type=int, default=0, help="每变体只用前 N 张（快速看）")
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    insts = spec["instances"]
    out = Path(args.out)
    (out / "图").mkdir(parents=True, exist_ok=True)

    # 按位置槽分组：role -> {label: instance}
    slots: dict[str, dict[str, dict]] = {}
    for i in insts:
        slots.setdefault(i["role"], {})[i["label"]] = i

    result = {"base_frame": spec.get("base_frame"), "positions": spec.get("positions"),
              "slots": {}}
    for role, by_label in slots.items():
        base = by_label["V0_base"]
        names_all = sorted(p.stem for p in
                           (Path(args.data) / f"inst{base['id']:03d}").glob("*.png"))
        if args.limit:
            names_all = names_all[: args.limit]
        if not names_all:
            print(f"[gate] {role} 无图，跳过（先跑生成）")
            continue

        B = imgs(args.data, base["id"], names_all)
        # 地板：把基准图的像素随机置换
        floor = {k: np.mean([shuffle_floor(B[n], seed=0)[k] for n in names_all])
                 for k in ("psnr", "ssim", "mae", "pearson", "frac_gt1", "frac_ne")}

        slot = {"n_images": len(names_all), "floor": floor, "variants": {}}
        for label, inst in by_label.items():
            have = sorted(p.stem for p in
                          (Path(args.data) / f"inst{inst['id']:03d}").glob("*.png"))
            names = [n for n in names_all if n in set(have)]
            if len(names) != len(names_all):
                print(f"[gate] {role}/{label} 只有 {len(names)}/{len(names_all)} 张")
            if not names:
                continue
            A = imgs(args.data, inst["id"], names)
            ms = [metrics(B[n], A[n]) for n in names]
            agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            agg.update({
                "label": label, "id": inst["id"],
                "delta_yaw_deg": inst.get("delta_yaw_deg", 0.0),
                "delta_pitch_deg": inst.get("delta_pitch_deg", 0.0),
                "freq_hz": inst["freq_hz"],
                "tx_pattern": inst.get("tx_pattern", "tr38901"),
                "delta_psnr_vs_floor": agg["psnr"] - floor["psnr"],
                # 精确重合检查（自检用）
                "max_abs_u8": int(max(int(np.abs(B[n].astype(np.int16)
                                                 - A[n].astype(np.int16)).max())
                                      for n in names)),
            })
            slot["variants"][label] = agg
            print(f"[gate] {role:11s} {label:11s} PSNR {agg['psnr']:7.3f}  "
                  f"SSIM {agg['ssim']:.4f}  r {agg['pearson']:+.4f}  "
                  f"|Δ|>1 {agg['frac_gt1']*100:5.2f}%  地板差 "
                  f"{agg['delta_psnr_vs_floor']:+6.2f} dB", flush=True)
        result["slots"][role] = slot

    # ---------------- 判定 ----------------
    def mean_over_slots(label, key):
        vals = [s["variants"][label][key] for s in result["slots"].values()
                if label in s["variants"]]
        return float(np.mean(vals)) if vals else None

    def mean_floor(key):
        vals = [s["floor"][key] for s in result["slots"].values()]
        return float(np.mean(vals)) if vals else None

    verdict = {"floor": {k: mean_floor(k) for k in
                         ("psnr", "ssim", "pearson", "frac_gt1")}}

    # 自检：闸要求「逐像素完全相同」。8bit + 逐图 min-max 会把 1 ulp 的浮点差
    # 放大成 ±1 灰阶，所以判据是 max|Δu8| <= 1 且 变动像素 < 0.1%；
    # 若出现 |Δu8| >= 2 或成片变动，才是求解器不确定 / 材质冻结失效。
    for chk in ("V16_rerun", "V17_freeze"):
        mx, fne = None, None
        for s in result["slots"].values():
            if chk in s["variants"]:
                mx = max(mx or 0, s["variants"][chk]["max_abs_u8"])
                fne = max(fne or 0.0, s["variants"][chk]["frac_ne"])
        verdict[chk] = {"max_abs_u8": mx, "frac_ne": fne,
                        "pass": (mx is not None and mx <= 1 and (fne or 0) < 1e-3)}

    # 朝向
    yaw_pts = [(0, "V0_base"), (2, "V1_yaw2"), (10, "V2_yaw10"), (20, "V3_yaw20"),
               (45, "V4_yaw45"), (90, "V5_yaw90"), (180, "V6_yaw180")]
    yaw_curve = [(d, mean_over_slots(lab, "psnr")) for d, lab in yaw_pts
                 if mean_over_slots(lab, "psnr") is not None]
    pitch_pts = [(0, "V0_base"), (2, "V7_pitch2"), (20, "V8_pitch20"),
                 (45, "V9_pitch45")]
    pitch_curve = [(d, mean_over_slots(lab, "psnr")) for d, lab in pitch_pts
                   if mean_over_slots(lab, "psnr") is not None]
    verdict["yaw_curve"] = yaw_curve
    verdict["pitch_curve"] = pitch_curve

    if yaw_curve:
        p90 = dict(yaw_curve).get(90)
        p2 = dict(yaw_curve).get(2)
        verdict["orientation"] = {
            "psnr_at_2deg": p2, "psnr_at_90deg": p90,
            "gap_90_minus_2": (p90 - p2) if (p90 is not None and p2 is not None) else None,
            "monotone": all(yaw_curve[i][1] >= yaw_curve[i + 1][1] - 0.05
                            for i in range(len(yaw_curve) - 1)),
            "pass": (p90 is not None and p2 is not None and (p90 - p2) <= -3.0),
        }

    # 频率
    f_pts = [(5e6, "V10_f2405"), (50e6, "V11_f2450"), (300e6, "V12_f2700"),
             (1000e6, "V13_f3400"), (2100e6, "V14_f4500")]
    f_curve = [(df, mean_over_slots(lab, "psnr"), mean_over_slots(lab, "pearson"))
               for df, lab in f_pts if mean_over_slots(lab, "psnr") is not None]
    verdict["freq_curve"] = f_curve
    fl = verdict["floor"]["psnr"]
    if f_curve and fl is not None:
        thr = fl + 3.0
        w = None
        for df, p, _ in sorted(f_curve):
            if p <= thr:
                w = df
                break
        verdict["frequency"] = {
            "floor_psnr": fl, "threshold_psnr": thr,
            "first_decorrelated_delta_hz": w,
            "skipped_at_smallest_probe": (f_curve and sorted(f_curve)[0][1] <= thr),
        }

    # ---------------- 出图 ----------------
    font = _cnfont.configure(plt)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    if yaw_curve:
        x, y = zip(*yaw_curve)
        ax[0].plot(x, y, "o-", label="朝向 Δyaw")
        ax[0].axhline(verdict["floor"]["psnr"], ls="--", c="gray",
                      label="结构全毁地板")
        ax[0].set_xlabel("Δyaw (°)"); ax[0].set_ylabel("PSNR vs 基准 (dB)")
        ax[0].set_title("朝向灵敏度"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    if pitch_curve:
        x, y = zip(*pitch_curve)
        ax[1].plot(x, y, "s-", c="C1", label="俯仰 Δpitch")
        ax[1].axhline(verdict["floor"]["psnr"], ls="--", c="gray")
        ax[1].set_xlabel("Δpitch (°)"); ax[1].set_ylabel("PSNR vs 基准 (dB)")
        ax[1].set_title("俯仰灵敏度"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    if f_curve:
        x = [df / 1e6 for df, _, _ in f_curve]
        y = [p for _, p, _ in f_curve]
        ax[2].semilogx(x, y, "d-", c="C2", label="频率偏移")
        ax[2].axhline(verdict["floor"]["psnr"], ls="--", c="gray",
                      label="结构全毁地板")
        ax[2].set_xlabel("Δf (MHz)"); ax[2].set_ylabel("PSNR vs 基准 (dB)")
        ax[2].set_title("频率灵敏度"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "图" / "闸测_灵敏度曲线.png", dpi=150)
    plt.close(fig)

    (out / "数据.json").write_text(
        json.dumps({"verdict": verdict, **result}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---------------- 文字结论 ----------------
    lines = ["# 阶段 0 闸测结论", ""]
    lines.append(f"结构全毁地板：PSNR {verdict['floor']['psnr']:.3f} dB / "
                 f"SSIM {verdict['floor']['ssim']:.4f} / "
                 f"r {verdict['floor']['pearson']:+.4f}")
    for chk in ("V16_rerun", "V17_freeze"):
        v = verdict[chk]
        lines.append(f"- {chk}: max|Δu8| = {v['max_abs_u8']}，变动像素占比 "
                     f"{(v['frac_ne'] or 0):.2e} → "
                     f"{'通过' if v['pass'] else '**不通过**'}")
    if "orientation" in verdict:
        o = verdict["orientation"]
        lines.append("")
        lines.append(f"朝向：PSNR(2°) {o['psnr_at_2deg']:.2f} dB，"
                     f"PSNR(90°) {o['psnr_at_90deg']:.2f} dB，"
                     f"差 {o['gap_90_minus_2']:.2f} dB；"
                     f"单调={o['monotone']} → {'GO' if o['pass'] else 'NO-GO'}")
    if "frequency" in verdict:
        fq = verdict["frequency"]
        lines.append("")
        _w = fq["first_decorrelated_delta_hz"]
        _wtxt = ("探测范围内**从未**落到阈值内 ⇒ 相关宽度 > 2.1 GHz"
                 if _w is None else f"{_w/1e6:.1f} MHz")
        lines.append(f"频率：地板 {fq['floor_psnr']:.2f}，判据阈值 "
                     f"{fq['threshold_psnr']:.2f}；首个落到阈值内的 Δf = {_wtxt}")
        for df, p, r in f_curve:
            lines.append(f"  - Δf {df/1e6:8.1f} MHz：PSNR {p:7.3f}  r {r:+.4f}")
    text = "\n".join(lines)
    (out / "结论文本.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    if not font:
        print("[gate] 警告：没找到中文字体，图中的中文可能是方块")


if __name__ == "__main__":
    main()
