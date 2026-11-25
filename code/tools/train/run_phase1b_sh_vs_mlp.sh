#!/bin/bash
# =============================================================================
# run_phase1b_sh_vs_mlp.sh —— Phase1b：同 LR 协议下 SH 原版 vs MLP 颜色头（4 方法，梯度+曲线）
# =============================================================================
# 目的：判定 Phase1 A2E0(+3.96dB) 的收益来自"换颜色表达(MLP)"还是"纯换 LR 协议(exp)"。
#   方法 = 双引擎同跑 A* 协议（exp 8e-3->8e-5, max_steps=40000, RRF 本地到 1 万步=abs40000）：
#     ENG=sh  原版逐点 SH 颜色头（train_rrf_sh.py 复刻 train.py RF reset；render_sh_rrf.py）
#     ENG=mlp embedding+共享 MLP 颜色头（train_rrf_mlp.py + render_rrf_mlp.py，embed 2.5e-3 固定）
#   其余全一致：同 chkpnt30000、冻结几何、opacity 可训、densify=0、4 方法、同 LR 曲线。
#   产物 RUNBASE/<M>/：train 梯度档 point_cloud/iteration_{32000 35000 38000 40000}、
#       test/ours_<i>（中间档 48 帧抽帧 + 末档全量 640）、curve.json(48 子集)、results.json(640)。
#   收敛曲线由 tools/make_curve_phase1b.py 叠加：SH@A*(本次) vs MLP@A*(本次) +
#       固定-LR 参考线（run_rrf40k_sh/mlp curve.json 同档位）→ 2x2 判据。
# 恢复：各段产物存在即跳过（可中断续跑）。判据 = 每档 test/ours_<i>/renders 目录 + curve.json。
# 用法（服务器 ~/3dgs，nohup，与另一引擎各占一卡并行）：
#   CUDA_VISIBLE_DEVICES=0 ENG=sh  nohup bash run_phase1b_sh_vs_mlp.sh \
#       > results/run_mlp_v2_阶段1b阶段2与长训/阶段1b长训_SH头/phase1b_sh.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 ENG=mlp nohup bash run_phase1b_sh_vs_mlp.sh \
#       > results/run_mlp_v2_阶段1b阶段2与长训/阶段1b_MLP头/phase1b_mlp.log 2>&1 &
#
# 目录归属（2026-09-11 整理后）：SH@A* 臂的正式记录在 `阶段1b长训_SH头`
#   （短跑到 abs40000 的档位与长跑同档位数值重合，ΔPSNR ≤ 0.002 dB，短跑产物已删并入）。
#   本脚本 ENG=sh 直接续写该目录：已有 4 档会自动跳过，只补缺。
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
ENG="${ENG:?需指定 ENG=sh|mlp}"
[ "$ENG" = sh ] || [ "$ENG" = mlp ] || { echo "ENG 需为 sh 或 mlp"; exit 2; }

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DATA=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
TOP=/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训
case "$ENG" in
  sh)  RUNBASE="$TOP/阶段1b长训_SH头" ;;   # SH@A* 臂正式归属（短跑产物已并入）
  mlp) RUNBASE="$TOP/阶段1b_MLP头" ;;
esac
METHODS="MPC CBF Delay MVDR"
LADDER="32000 35000 38000 40000"
FINAL=40000
SUBSET_CURVE=48
# A* 协议（A2E0）：exp 8e-3 -> 8e-5, max_steps=40000（embed/opacity 默认固定）
A_INIT=8e-3
A_FINAL=8e-5

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$RUNBASE"
echo "=========== [$(date +%F_%T)] PHASE1b ENG=$ENG  卡=$CUDA_VISIBLE_DEVICES  A*=$(echo $LADDER | wc -w)档到 $FINAL ==========="

for M in $METHODS; do
  P_SPC="$DATA/3dgs_${M}_100"
  [ -d "$P_SPC" ] || { echo "[$M] 数据缺失: $P_SPC"; exit 1; }
  MD="$RUNBASE/$M"; mkdir -p "$MD"
  echo "===================================================================="
  echo "[$(date +%F_%T)] ==== [$ENG] $M: RRF -> $FINAL @A* ===="
  echo "===================================================================="

  # ---------- 训练（final 云已存在则跳过整段）----------
  if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
    if [ "$ENG" = sh ]; then
      echo "[T] $M train_rrf_sh.py ..."
      "$PY" train_rrf_sh.py -s "$P_SPC" -m "$MD" \
          --start_checkpoint "$SEED" --iterations "$FINAL" \
          --save_iterations $LADDER \
          --densify_until_iter 0 --data_device cpu --quiet \
          --lr_schedule exp --color_lr_init "$A_INIT" --color_lr_final "$A_FINAL" \
          --lr_max_steps 40000 \
          > "$RUNBASE/log_train_${M}.txt" 2>&1 \
        || { echo "  [$M] TRAIN_FAIL"; tail -25 "$RUNBASE/log_train_${M}.txt"; exit 1; }
    else
      echo "[T] $M train_rrf_mlp.py ..."
      "$PY" train_rrf_mlp.py -s "$P_SPC" -m "$MD" \
          --start_checkpoint "$SEED" --iterations "$FINAL" \
          --save_iterations $LADDER \
          --densify_until_iter 0 --data_device cpu --quiet \
          --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
          --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
          --lr_max_steps 40000 \
          > "$RUNBASE/log_train_${M}.txt" 2>&1 \
        || { echo "  [$M] TRAIN_FAIL"; tail -25 "$RUNBASE/log_train_${M}.txt"; exit 1; }
    fi
    echo "[T] $M 完成（$FINAL 云已写）"
  else
    echo "[T] $M final 云已存在，跳过训练"
  fi

  # ---------- 逐档渲染：中间档抽帧 48（曲线），末档全量 640（results.json）----------
  for i in $LADDER; do
    sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
    if [ ! -d "$MD/test/ours_$i/renders" ]; then
      if [ "$ENG" = sh ]; then
        "$PY" render_sh_rrf.py -s "$P_SPC" -m "$MD" --iteration "$i" \
            --eval --skip_train --subset "$sub" --quiet \
            > "$RUNBASE/log_render_${M}_${i}.txt" 2>&1 \
          || { echo "  [$M] RENDER_FAIL_$i"; tail -12 "$RUNBASE/log_render_${M}_${i}.txt"; exit 1; }
      else
        "$PY" render_rrf_mlp.py -s "$P_SPC" -m "$MD" --iteration "$i" \
            --eval --skip_train --subset "$sub" --quiet \
            > "$RUNBASE/log_render_${M}_${i}.txt" 2>&1 \
          || { echo "  [$M] RENDER_FAIL_$i"; tail -12 "$RUNBASE/log_render_${M}_${i}.txt"; exit 1; }
      fi
      echo "  [R] ours_$i 已渲染（subset=$sub）"
    else
      echo "  [R] ours_$i 已存在，跳过"
    fi
  done

  # ---------- 梯度档抽帧评测 -> curve.json（缺才跑）----------
  if [ ! -f "$MD/curve.json" ]; then
    echo "[C] $M 抽帧($SUBSET_CURVE)评测 $LADDER -> curve.json"
    "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
        --iterations $LADDER --subset "$SUBSET_CURVE" \
        --out "$MD/curve.json" > "$RUNBASE/log_curve_${M}.txt" 2>&1 \
      || { echo "  [$M] CURVE_FAIL"; tail -15 "$RUNBASE/log_curve_${M}.txt"; exit 1; }
  else
    echo "[C] $M curve.json 已存在，跳过"
  fi

  # ---------- 末档全量评测 -> results.json ----------
  if [ ! -f "$MD/results.json" ]; then
    echo "[F] $M 末档 $FINAL 全量(640)评测 -> results.json"
    "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
        --iterations "$FINAL" --subset 0 \
        --out "$MD/results.json" > "$RUNBASE/log_full_${M}.txt" 2>&1 \
      || { echo "  [$M] FULL_FAIL"; tail -15 "$RUNBASE/log_full_${M}.txt"; exit 1; }
  else
    echo "[F] $M results.json 已存在，跳过"
  fi

  echo "[$M] curve.json (48子集):"
  "$PY" - "$MD/curve.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    print("  %s  PSNR %.3f  SSIM %.4f  LPIPS %.4f" % (k, v.get("PSNR",-1), v.get("SSIM",-1), v.get("LPIPS",-1)))
PYEOF
  echo "[$M] results.json (640):"
  "$PY" - "$MD/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    print("  %s  PSNR %.3f  SSIM %.4f  LPIPS %.4f" % (k, v.get("PSNR",-1), v.get("SSIM",-1), v.get("LPIPS",-1)))
PYEOF
done

echo "[$(date +%F_%T)] ALL_PHASE1B_${ENG}_DONE"
