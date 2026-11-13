#!/bin/bash
# =============================================================================
# evaluate_run.sh —— 对已有端到端 run 补 baseline 等价全量评测
# -----------------------------------------------------------------------------
# 用法（WSL bash，仓库根）：
#     bash evaluate_run.sh run_<时间戳> [profile]
#     bash evaluate_run.sh results/run_<时间戳> [profile]     # 也可给完整/相对路径
# 参数 2 profile 缺省时读取该 run 的 profile.txt（run_end_to_end.sh 写入）。
#
# 评测套件（内容 = baseline run_20260905_190255_baseline 的评测，全自动/幂等）：
#   1. mvsplat 覆盖度：对 run/mvsplat/点云/init.ply 投影到 4 个位置(0/200/400/600)
#      → mvsplat/覆盖率.png + 验证数据.json + 覆盖率视角_原图vs投影.png（CPU 秒级）
#   2. 阶段一 多档迭代扫描：自动发现 rf-3dgs_1/point_cloud/iteration_*，逐档
#      render.py --iteration --eval（已渲染跳过）→ metrics.py 一次 → 评测/ 汇总
#      + 每档「4 视角原图|渲染」快速查看图
#   3. 阶段二 双档（10000RRF 对齐档 + 40000总 档）：逐方法逐档 render（幂等）→ metrics
#   4. 汇总（make_run_eval.py，读 config.yaml eval.reference 嵌原版参考）：阶段一扫描图、
#      阶段二评测.json、三方对比/三方对比.png+对比数据.json
#   5. 参数量(高斯球数)-质量曲线：三方对比/参数量曲线_我的方法vs原版.png/.json
#      （我方多档 vs 原版 blender_visual_trained，评测法同 baseline）
# 幂等：test/ours_<iter>/renders 已存在则跳过该档 render（省 GPU）。
# =============================================================================
set -euo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_DIR/load_config.sh"

# 先用默认 profile 取 P_OUT 等路径（路径与 profile 无关）
load_config full
RUNREF="${1:?用法: evaluate_run.sh run_<时间戳> [profile]}"
PROFILE_ARG="${2:-}"

if [[ "$RUNREF" = /* ]]; then
  RUN="$RUNREF"
else
  RUN="$P_OUT/$RUNREF"
fi
[ -d "$RUN/rf-3dgs_1" ] || { echo "[eval] 找不到 run：$RUN"; exit 1; }

if [ -n "$PROFILE_ARG" ]; then
  load_config "$PROFILE_ARG"
else
  P2="$(cat "$RUN/profile.txt" 2>/dev/null || echo full)"
  load_config "$P2"
fi
echo "[eval] RUN=$RUN  profile=$PROFILE  ITR1=$ITR1  METHODS='$METHODS'"
echo "        阶段B 存档档: ${STAGE1_SAVE:-$ITR1}"
if [ -n "${TOTAL2:-}" ]; then
  echo "        阶段C 双档: 10000RRF 对齐绝对 $ITR2 / 40000总 绝对 $TOTAL2"
else
  echo "        阶段C 单档: 对齐绝对 $ITR2"
fi
mkdir -p "$RUN/logs"

# 自动发现模型里已存的所有迭代档（point_cloud/iteration_<i>），数值升序；无则回退单档
discover_iters() {   # $1 = 模型目录
  if [ -d "$1/point_cloud" ]; then
    ls -d "$1"/point_cloud/iteration_* 2>/dev/null | sed 's#.*/iteration_##' | sort -n | tr '\n' ' '
  fi
}

# ---- 1) mvsplat 覆盖度（CPU，秒级）-----------------------------------------
if [ -f "$RUN/mvsplat/点云/init.ply" ]; then
  echo; echo "[eval] mvsplat 覆盖度（4 位置投影）"
  if ! "$PY" "$P_TOOLS/mvsplat_coverage.py" --cloud "$RUN/mvsplat/点云/init.ply" \
        --src "$P_VISUAL" --out "$RUN/mvsplat" > "$RUN/logs/stageA_coverage.log" 2>&1; then
    echo "  [覆盖度失败]"; tail -30 "$RUN/logs/stageA_coverage.log"; exit 1
  fi
else
  echo; echo "[eval] 跳过覆盖度：无 $RUN/mvsplat/点云/init.ply"
fi

# ---- 2) 阶段一（阶段B 模型）多档评测：161 held-out 测试视角 --------------------
echo; echo "[eval] 阶段一  rf-3dgs_1 多档迭代扫描（resolution $RES）"
B_MODEL="$RUN/rf-3dgs_1"
B_ITERS="$(discover_iters "$B_MODEL")"
[ -n "$B_ITERS" ] || B_ITERS="$ITR1"
echo "  发现档位: $B_ITERS"
cd "$P_CODE"
for i in $B_ITERS; do
  if [ ! -d "$B_MODEL/test/ours_$i/renders" ]; then
    echo "  render ours_$i ..."
    if ! "$PY" render.py -s "$P_VISUAL" -m "$B_MODEL" --iteration "$i" \
            --eval --skip_train --resolution "$RES" > "$RUN/logs/stageB_render_$i.log" 2>&1; then
      echo "  [阶段一 ours_$i render 失败]"; tail -30 "$RUN/logs/stageB_render_$i.log"; exit 1
    fi
  else
    echo "  renders ours_$i 已存在，跳过 render"
  fi
done
echo "  metrics（一次聚合所有档）..."
"$PY" metrics.py -m "$B_MODEL" > "$RUN/logs/stageB_metrics.log" 2>&1 \
  || { echo "  [阶段一 metrics 失败]"; tail -30 "$RUN/logs/stageB_metrics.log"; exit 1; }

# ---- 3) 阶段二 各方法双档 RRF 评测（640 频谱，原生分辨率）----------------------
for M in $METHODS; do
  MD="$RUN/rf-3dgs_2/$M"
  [ -d "$MD/point_cloud" ] || { echo "[eval] 跳过缺失模型 $M: $MD"; continue; }
  C_ITERS="$(discover_iters "$MD")"
  [ -n "$C_ITERS" ] || C_ITERS="$ITR2"
  echo; echo "[eval] 阶段二  $M 档位: $C_ITERS"
  for i in $C_ITERS; do
    if [ ! -d "$MD/test/ours_$i/renders" ]; then
      echo "  render ours_$i ..."
      if ! "$PY" render.py -s "$P_SPECTRUM/3dgs_${M}_100" -m "$MD" --iteration "$i" \
              --eval --skip_train > "$RUN/logs/stageC_render_${M}_$i.log" 2>&1; then
        echo "  [$M ours_$i render 失败]"; tail -30 "$RUN/logs/stageC_render_${M}_$i.log"; exit 1
      fi
    else
      echo "  renders ours_$i 已存在，跳过 render"
    fi
  done
  echo "  metrics ..."
  "$PY" metrics.py -m "$MD" > "$RUN/logs/stageC_metrics_${M}.log" 2>&1 \
    || { echo "  [$M metrics 失败]"; tail -30 "$RUN/logs/stageC_metrics_${M}.log"; exit 1; }
done

# ---- 4) 汇总：阶段一扫描图 / 阶段二评测 / 三方对比（含原版参考）-----------------
echo; echo "[eval] 聚合汇总（make_run_eval.py）"
if ! "$PY" "$P_TOOLS/make_run_eval.py" --run "$RUN" > "$RUN/logs/eval_aggregate.log" 2>&1; then
  echo "  [聚合失败]"; tail -40 "$RUN/logs/eval_aggregate.log"; exit 1
fi

# ---- 5) 参数量(高斯球数)-质量曲线（我的方法 vs 原版）进 三方对比 -----------------
NKEY="$("$PY" - "$B_MODEL/results.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    d = {}
print(sum(1 for k in d if str(k).startswith("ours_")) if isinstance(d, dict) else 0)
PYEOF
)"
echo; echo "[eval] 参数量曲线"
if [ "$NKEY" -ge 2 ] && [ -f "$P_ORIG_VIS/results.json" ]; then
  if ! "$PY" "$P_TOOLS/make_params_curve.py" --run "$RUN" \
        --orig "$P_ORIG_VIS" > "$RUN/logs/params_curve.log" 2>&1; then
    echo "  [曲线失败]"; tail -40 "$RUN/logs/params_curve.log"; exit 1
  fi
elif [ "$NKEY" -lt 2 ]; then
  echo "  我方仅 $NKEY 个有指标的档（需 ≥2 才能画曲线）；先完成多档渲染/评测"
else
  echo "  缺原版模型 results.json（$P_ORIG_VIS），跳过曲线"
fi

echo "[eval] 完成"
echo "       覆盖度：$RUN/mvsplat/{覆盖率.png,覆盖率视角_原图vs投影.png,验证数据.json}"
echo "       阶段一：$RUN/rf-3dgs_1/评测/（阶段一评测.json + 阶段一_指标图.png + 快速查看/）"
echo "       阶段二：$RUN/rf-3dgs_2/评测/ + $RUN/rf-3dgs_2/三方对比/"
