#!/usr/bin/env bash
# run_conf_probe.sh —— 「置信度—误差相关性诊断」服务器端驱动（零 GPU，纯 CPU）
#
# 目的
# ----
# 在花任何 GPU 之前，先回答一个问题：MVSplat 的匹配置信度（= init_view.ply 的
# opacity 字段，理由见 tools/conf_error_probe.py 的模块头）与 RRF 各臂的逐像素
# 重建误差，到底有没有关系、往哪个方向？
#
# 预注册口径（见 docs/MVSplat置信度引导分配_方案.md §5 测试一）
# ------------------------------------------------------------
# 主判据 = ρ(置信度, 逐像素误差)，取【MVSplat 物理那条臂】的读数。
#   |ρ| < 0.1，或按置信度分档后误差不单调  ⇒  判死，这条线到此为止，不花任何 GPU。
# 否则看差值曲线的斜率，判定「往高置信投还是往低置信投」。
#
# 三根柱子（都是 ours_60000、640 张全量渲染，**只读复用，不重训不重渲**）
# ---------------------------------------------------------------------
#   原版物理 + 原版 SH   ← 最原版对照（用户指定加入）
#   原版物理 + 标量头
#   MVSplat 物理 + 标量头
#
# 已实测的前提（本次侦察，见 results/run_20260914_置信度引导分配/诊断/说明.md）
# ---------------------------------------------------------------------------
#   * 三臂的 cameras.json **逐位相同**（position/rotation 最大差 0.000e+00）
#     ⇒ 三臂共用一份 cameras.json 是合法的，投影不会错位。
#   * 渲染文件是 `{下标:05d}.png`（00000..00639），且 `cameras.json[i].img_name`
#     与原图 MAE = 0.000000 / corr = 1.000000（抽查 i = 0/1/100/319/639）。
#   * 服务器 init_view.ply 有 opacity 与 f_dc_0..2 两类字段（N = 1276957）。
#
# 用法（服务器 code/ 下）
# ----------------------
#     bash tools/run_conf_probe.sh                 # 两个方法都跑，stride 8
#     STRIDE=4 bash tools/run_conf_probe.sh        # 更密
#     METHODS="MVDR" bash tools/run_conf_probe.sh  # 只跑一个方法
set -uo pipefail

PY=${PY:-/data/yuexf/miniconda3/envs/gs3d/bin/python}
RB=${RB:-/data/yuexf/3dgs/results}
B="$RB/run_20260912_两组预算_三臂对比/组1_物理30000_RRF30000/阶段3_RRF重建"
TOP=${TOP:-$RB/run_20260914_置信度引导分配/诊断}
INIT=${INIT:-$RB/run_20260907_185429_parity档_视觉30k基座/mvsplat/点云/init_view.ply}
STRIDE=${STRIDE:-8}
METHODS=${METHODS:-"MVDR CBF"}

CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE" || exit 1

echo "=============================================================="
echo " 置信度—误差相关性诊断（零 GPU）"
echo "   python    $PY"
echo "   点云      $INIT"
echo "   输出      $TOP"
echo "   相机步长  $STRIDE（⇒ 每方法 $((640 / STRIDE)) 台）"
echo "=============================================================="

[ -f "$INIT" ] || { echo "❌ 点云不存在：$INIT"; exit 1; }

# ---------------------------------------------------------------------------
# 前置闸（决定性）：把 `project()` 与仓库渲染器真正用的矩阵 `P @ Rt` 对拍。
# 为什么口径不能靠图像来验：本数据集的 GT 是 RF 频谱伪彩图，「非黑区」是整片 RF 场的
# 投影、覆盖大半画面 ⇒ 正确口径的 IoU 0.879，而转置 / 上下翻 / 左右翻**也都是 0.879**，
# 拿 IoU 当闸完全不判别（实测）。所以口径必须在图像之外验，且这一验是可判定的：
# 正确口径偏差 float32 舍入级，错法 120~180 px。
# 闸不过就**两个方法都不跑** —— 在一个没验过的投影上做统计，出来的数一个都不能用。
# ---------------------------------------------------------------------------
echo ""
echo "===== 前置闸：投影口径 vs 仓库渲染器 ====="
mkdir -p "$TOP"
# 位姿已实测三臂/两方法逐位相同（position/rotation 最大差 0.000e+00），故取一份即可。
PCAMS="$B/A3_新物理_新RRF/MVDR/cameras.json"
[ -f "$PCAMS" ] || PCAMS="$(ls "$B"/A3_*/[A-Z]*/cameras.json 2>/dev/null | head -1)"
"$PY" tools/verify_cam_projection.py --cameras "$PCAMS" --cloud "$INIT" \
    > "$TOP/投影口径对拍.log" 2>&1
PG=$?
cat "$TOP/投影口径对拍.log"
if [ "$PG" != 0 ]; then
  echo "❌ 投影口径对拍 FAIL ⇒ 按预注册，诊断到此为止，不跑任何统计。"
  exit 1
fi
echo "✅ 前置闸通过（日志：$TOP/投影口径对拍.log）"

FAIL=0
for M in $METHODS; do
  A1="$RB/run_rrf40k_sh/$M/test/ours_60000"                 # 原版物理 + 原版 SH
  A2="$B/A2_原版物理_新RRF/$M/test/ours_60000"               # 原版物理 + 标量头
  A3="$B/A3_新物理_新RRF/$M/test/ours_60000"                 # MVSplat 物理 + 标量头
  CAMS="$B/A3_新物理_新RRF/$M/cameras.json"

  MISS=0
  for d in "$A1/renders" "$A1/gt" "$A2/renders" "$A2/gt" "$A3/renders" "$A3/gt"; do
    [ -d "$d" ] || { echo "❌ $M 缺目录：$d"; MISS=1; }
  done
  [ -f "$CAMS" ] || { echo "❌ $M 缺 cameras.json：$CAMS"; MISS=1; }
  [ "$MISS" = 1 ] && { FAIL=1; continue; }

  OUT="$TOP/$M"
  mkdir -p "$OUT"
  echo ""
  echo "----- $M -----"
  "$PY" tools/conf_error_probe.py \
      --cloud "$INIT" \
      --cameras "$CAMS" \
      --arms "mvsplat=MVSplat 物理 + 标量头:$A3" \
             "orig=原版物理 + 标量头:$A2" \
             "origsh=原版物理 + 原版 SH:$A1" \
      --pairs "mvsplat-orig" "mvsplat-origsh" \
      --verdict-arm mvsplat \
      --method "$M" --out "$OUT" --stride "$STRIDE" \
    || { echo "❌ $M 诊断脚本失败"; FAIL=1; continue; }

  # 收尾哨兵：三张图 + json 必须都在，且非空（否则「跑过了」是假的）
  for f in "诊断.json" "诊断_置信度分层误差_$M.png" \
           "诊断_误差差值分层_$M.png" "诊断_覆盖率与逐相机_$M.png"; do
    if [ ! -s "$OUT/$f" ]; then echo "❌ $M 缺产物或为空：$f"; FAIL=1; fi
  done

  # **「文件非空」拦不住空图**：`诊断_覆盖率与逐相机_*.png` 曾经是 135 KB 的
  # 纯坐标轴 + 图例（一个数据点都没有，图例写着 `ρ = +nan（0 台）`），`-s`
  # 判它完全合格，于是这张空图一路交付到总结文档里。字节数不是「有没有画东西」的代理。
  # ⇒ 改判 `诊断.json` 里出图函数自己回报的 `fig_stats`（真正画上去的数据点数）。
  FIGCHK=$("$PY" - "$OUT/诊断.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
fs = d.get("fig_stats") or {}
if not fs:
    print("BAD 诊断.json 里没有 fig_stats（图是空的、或脚本被改回了旧版）")
    raise SystemExit(0)
bad = [k for k, v in fs.items() if (not v) or any(x <= 0 for x in v.values())]
if bad:
    print("BAD 空图/无数据：" + "; ".join(f"{k}={fs[k]}" for k in bad))
else:
    for k, v in fs.items():
        print(f"    {k}: {v}")
    print("OK")
PYEOF
)
  echo "$FIGCHK" | grep -q '^OK$' || { echo "❌ $M 收尾哨兵不过："; echo "$FIGCHK"; FAIL=1; }
  echo "✔ $M 完成 ⇒ $OUT"
done

echo ""
if [ "$FAIL" = 0 ]; then echo "✅ 全部完成：$TOP"; else echo "❌ 有失败项，见上"; fi
exit $FAIL
