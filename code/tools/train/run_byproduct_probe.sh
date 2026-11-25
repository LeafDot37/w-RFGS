#!/usr/bin/env bash
# run_byproduct_probe.sh —— 「物理阶段次生产物 → RRF 误差」相关性诊断（零 GPU，纯 CPU）
#
# 命题
# ----
# 用户提出：既然物理阶段的**终态**对 RRF 几乎无影响（A2 原版物理 vs A3 MVSplat 物理：
# MVDR +0.0216 dB / CBF +0.1614 dB），能否把物理阶段的**次生产物**带进 RRF 提高效果？
#
# 本诊断回答它的第一步（与「置信度引导」那条线同一套方法论：先诊断，再决定花不花 GPU）：
#     denom / max_radii2D 与 RRF 的逐像素误差有没有相关性、往哪个方向？
#
# 预注册判据（**沿用 run_conf_probe.sh 已写死的那条，不改**）
# ----------------------------------------------------------
#   主判据 = ρ(次生产物, 逐像素误差)，取 A3（MVSplat 物理 + 标量头）那条臂的读数。
#     |ρ| < 0.1，或按该量分档后误差不单调  ⇒  判死，这条线到此为止，不花任何 GPU。
#   否则看差值曲线斜率，判定「该往高投还是往低投」。
#
# ⚠ 几何自洽性（本诊断与前一条线唯一的实质差别，必须照做）
# ------------------------------------------------------------
#   点云必须是**该臂渲染时真正用的那套几何**。A3 的物理 = parity 档 30k ckpt，
#   而 RRF 阶段 `--densify_until_iter 0` ⇒ 几何冻死 ⇒ A3 的几何 = 那个 ckpt 逐位相同。
#   ⇒ 点云只能取 `chkpnt30000.pth` 导出的 byproduct.ply。
#
#   反例（**不要照 run_conf_probe.sh 抄**）：那条线拿 init_view.ply（MVSplat 前馈点云、
#   未经过任何 3DGS 训练）去投 A1/A2 的渲染，而 A1/A2 的物理是**原版 COLMAP 种子** ——
#   点云与渲染的几何不同源，投影本身就没有一一对应。本驱动因此**不接 A1/A2**：
#   它们没有与 byproduct.ply 同源的几何，接上去只会得到一个不可解释的数。
#
# 为什么只做 30k 档
# ------------------
#   `densify_and_prune` 每次调用清零这三项，而短训轴的存档点（500/1000/2000/4000/8000）
#   **全是 100 的倍数** ⇒ 短档这三项 100% 是零（实测 1430603/1430603）。
#   只有 30000 档（> densify_until_iter=15000）留着真值。详见 make_byproduct_ply.py 文件头。
#
# 用法（服务器 code/ 下）
# ----------------------
#     bash tools/run_byproduct_probe.sh                       # 两个方法 × 两个字段
#     METHODS="MVDR" bash tools/run_byproduct_probe.sh
#     FIELDS="denom" bash tools/run_byproduct_probe.sh
set -uo pipefail

PY=${PY:-/data/yuexf/miniconda3/envs/gs3d/bin/python}
RB=${RB:-/data/yuexf/3dgs/results}
CK=${CK:-$RB/run_20260907_185429_parity档_视觉30k基座/rf-3dgs_1/chkpnt30000.pth}
TOP=${TOP:-$RB/run_20260915_次生产物诊断}
B="$RB/run_20260912_两组预算_三臂对比/组1_物理30000_RRF30000/阶段3_RRF重建"
CLOUD="$TOP/点云/byproduct.ply"
STRIDE=${STRIDE:-8}
METHODS=${METHODS:-"MVDR CBF"}
FIELDS=${FIELDS:-"denom max_radii2D"}

CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE" || exit 1

echo "=============================================================="
echo " 物理阶段次生产物 — RRF 误差相关性诊断（零 GPU）"
echo "   ckpt      $CK"
echo "   点云      $CLOUD   ← 与 A3 几何同源（冻结）"
echo "   输出      $TOP"
echo "   字段      $FIELDS"
echo "   相机步长  $STRIDE（⇒ 每方法 $((640 / STRIDE)) 台）"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 一、导出次生产物点云（幂等：已存在就复用，但**必须重跑非退化断言**）
# ---------------------------------------------------------------------------
mkdir -p "$TOP/点云"
if [ ! -s "$CLOUD" ]; then
  echo ""
  echo "===== 导出次生产物 ply ====="
  "$PY" tools/make_byproduct_ply.py --ckpt "$CK" --out "$CLOUD" || {
    echo "❌ 导出失败"; exit 1; }
else
  echo ""
  echo "===== 复用已有 $CLOUD（重跑非退化断言）====="
  "$PY" tools/make_byproduct_ply.py --ckpt "$CK" --out "$CLOUD" > /dev/null || {
    echo "❌ 复用校验失败"; exit 1; }
  echo "✅ 非退化断言通过"
fi

# ---------------------------------------------------------------------------
# 二、前置闸（决定性）：投影口径 vs 仓库渲染器。
# 用一个**新点云**跑诊断就必须重新过闸 —— 闸验的是 project() 与 P@Rt 的一致性。
# ---------------------------------------------------------------------------
echo ""
echo "===== 前置闸：投影口径 vs 仓库渲染器 ====="
PCAMS="$B/A3_新物理_新RRF/MVDR/cameras.json"
[ -f "$PCAMS" ] || PCAMS="$(ls "$B"/A3_*/[A-Z]*/cameras.json 2>/dev/null | head -1)"
"$PY" tools/verify_cam_projection.py --cameras "$PCAMS" --cloud "$CLOUD" \
    > "$TOP/投影口径对拍.log" 2>&1
PG=$?
cat "$TOP/投影口径对拍.log"
if [ "$PG" != 0 ]; then
  echo "❌ 投影口径对拍 FAIL ⇒ 按预注册，诊断到此为止，不跑任何统计。"
  exit 1
fi
echo "✅ 前置闸通过（日志：$TOP/投影口径对拍.log）"

# ---------------------------------------------------------------------------
# 三、诊断
# ---------------------------------------------------------------------------
FAIL=0
for M in $METHODS; do
  A3="$B/A3_新物理_新RRF/$M/test/ours_60000"
  CAMS="$B/A3_新物理_新RRF/$M/cameras.json"
  for d in "$A3/renders" "$A3/gt"; do
    [ -d "$d" ] || { echo "❌ $M 缺目录：$d"; FAIL=1; }
  done
  [ -f "$CAMS" ] || { echo "❌ $M 缺 cameras.json：$CAMS"; FAIL=1; }
  [ "$FAIL" = 1 ] && continue

  for F in $FIELDS; do
    OUT="$TOP/诊断/$M/$F"
    mkdir -p "$OUT"
    echo ""
    echo "----- $M / 字段 $F -----"
    "$PY" tools/conf_error_probe.py \
        --cloud "$CLOUD" --conf-field "$F" \
        --cameras "$CAMS" \
        --arms "a3=MVSplat 物理 + 标量头（几何同源）:$A3" \
        --verdict-arm a3 \
        --method "$M" --out "$OUT" --stride "$STRIDE" \
      || { echo "❌ $M/$F 诊断失败"; FAIL=1; continue; }
    # 哨兵：json 与三张图都得在且非空
    for f in "诊断.json" "诊断_置信度分层误差_$M.png" \
             "诊断_误差差值分层_$M.png" "诊断_覆盖率与逐相机_$M.png"; do
      [ -s "$OUT/$f" ] || { echo "❌ $M/$F 缺产物或为空：$f"; FAIL=1; }
    done
    echo "✔ $M/$F 完成 ⇒ $OUT"
  done
done

echo ""
if [ "$FAIL" = 0 ]; then echo "✅ 全部完成：$TOP"; else echo "❌ 有失败项，见上"; fi
exit "$FAIL"
