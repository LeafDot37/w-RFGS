#!/bin/bash
# =============================================================================
# run_rrf40k_driver.sh —— 双引擎 RRF 长跑驱动（4 方法 30k→abs70000 = RRF 4万）
# =============================================================================
# 目标：把阶段二 RRF 颜色重建成像从 1 万步拉长到 4 万步（绝对 30000→70000），
#       双引擎平行对比，每方法沿途存 10 档 + 渲染 + 评测，供收敛曲线/末档柱状图：
#   引擎 SH  = train.py（原版逐点 SH 色头，RF reset 语义）        → render.py
#   引擎 MLP = train_rrf_mlp.py（embedding+共享 MLP 色头，单变量）→ render_rrf_mlp.py
#   起点     = orig_seed_chkpnt30000.pth（原版视觉，与既有 run_mlp_rrf_full / conv_MPC 一致）
#   存档档位 = 32000 35000 38000 40000 45000 50000 55000 60000 65000 70000
#              （RRF 2k/5k/8k/10k/15k/20k/25k/30k/35k/40k；两引擎同档 → 同图可比）
#   评测     = tools/metrics_fast.py：每档抽帧48 → curve.json；末档全量640 → results.json
# 说明：
#   * train.py 每次 --start_checkpoint 都执行 RF reset，故不跨 chkpnt 续跑，从 seed 整段重训。
#   * 已存在产物自动跳过（可中断续跑）。产物判据 = point_cloud/iteration_70000/point_cloud.ply。
#   * 不写 --checkpoint_iterations（省盘/IO；收敛图只需点云档）。
# 用法（在 ~/3dgs 下）：
#   CUDA_VISIBLE_DEVICES=2 ENG=mlp bash run_rrf40k_driver.sh   # MLP 引擎
#   CUDA_VISIBLE_DEVICES=0 ENG=sh  bash run_rrf40k_driver.sh   # 原版 SH 引擎
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
ENG="${ENG:?需指定 ENG=sh|mlp}"
[ "$ENG" = sh ] || [ "$ENG" = mlp ] || { echo "ENG 需为 sh 或 mlp"; exit 2; }

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DATA=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
RUNBASE=/data/yuexf/3dgs/results/run_rrf40k_${ENG}
METHODS="MPC CBF Delay MVDR"
LADDER="32000 35000 38000 40000 45000 50000 55000 60000 65000 70000"
FINAL=70000
EMBED_DIM=32
MLP_HIDDEN=128
MLP_DEPTH=2
SUBSET_CURVE=48

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$RUNBASE"
echo "=========== [$(date +%F_%T)] ENG=$ENG  卡=$CUDA_VISIBLE_DEVICES  RRF 30000->$FINAL ($(echo $LADDER | wc -w) 档) ==========="

for M in $METHODS; do
  P_SPC="$DATA/3dgs_${M}_100"
  [ -d "$P_SPC" ] || { echo "[$M] 数据缺失: $P_SPC"; exit 1; }
  MD="$RUNBASE/$M"; mkdir -p "$MD"
  echo "===================================================================="
  echo "[$(date +%F_%T)] ==== [$ENG] $M: RRF -> $FINAL ===="
  echo "===================================================================="

  # ---------- 训练（final 云已存在则跳过整段）----------
  if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
    if [ "$ENG" = sh ]; then
      echo "[T] $M train.py ..."
      "$PY" train.py -s "$P_SPC" -m "$MD" \
          --start_checkpoint "$SEED" --iterations "$FINAL" \
          --save_iterations $LADDER \
          --densify_until_iter 0 --data_device cpu \
          > "$RUNBASE/log_train_${M}.txt" 2>&1 \
        || { echo "  [$M] TRAIN_FAIL"; tail -25 "$RUNBASE/log_train_${M}.txt"; exit 1; }
    else
      echo "[T] $M train_rrf_mlp.py ..."
      "$PY" train_rrf_mlp.py -s "$P_SPC" -m "$MD" \
          --start_checkpoint "$SEED" --iterations "$FINAL" \
          --save_iterations $LADDER \
          --densify_until_iter 0 --data_device cpu --quiet \
          --embed_dim "$EMBED_DIM" --mlp_hidden "$MLP_HIDDEN" --mlp_depth "$MLP_DEPTH" \
          > "$RUNBASE/log_train_${M}.txt" 2>&1 \
        || { echo "  [$M] TRAIN_FAIL"; tail -25 "$RUNBASE/log_train_${M}.txt"; exit 1; }
    fi
    echo "[T] $M 完成（$FINAL 云已写）"
  else
    echo "[T] $M final 云已存在，跳过训练"
  fi

  # ---------- 逐档渲染（缺哪档补哪档）----------
  for i in $LADDER; do
    if [ ! -d "$MD/test/ours_$i/renders" ]; then
      if [ "$ENG" = sh ]; then
        "$PY" render.py -s "$P_SPC" -m "$MD" --iteration "$i" --eval --skip_train \
            > "$RUNBASE/log_render_${M}_${i}.txt" 2>&1 \
          || { echo "  [$M] RENDER_FAIL_$i"; tail -12 "$RUNBASE/log_render_${M}_${i}.txt"; exit 1; }
      else
        "$PY" render_rrf_mlp.py -s "$P_SPC" -m "$MD" --iteration "$i" --eval --skip_train --quiet \
            > "$RUNBASE/log_render_${M}_${i}.txt" 2>&1 \
          || { echo "  [$M] RENDER_FAIL_$i"; tail -12 "$RUNBASE/log_render_${M}_${i}.txt"; exit 1; }
      fi
      echo "  [R] ours_$i 已渲染"
    else
      echo "  [R] ours_$i 已存在，跳过"
    fi
  done

  # ---------- 抽帧评测：整条档位 -> curve.json（缺才跑，LPIPS 慢）----------
  if [ ! -f "$MD/curve.json" ]; then
    echo "[C] $M 抽帧($SUBSET_CURVE)评测 10 档 -> curve.json"
    "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
        --iterations $LADDER --subset "$SUBSET_CURVE" \
        --out "$MD/curve.json" > "$RUNBASE/log_curve_${M}.txt" 2>&1 \
      || { echo "  [$M] CURVE_FAIL"; tail -15 "$RUNBASE/log_curve_${M}.txt"; exit 1; }
  else
    echo "[C] $M curve.json 已存在，跳过"
  fi

  # ---------- 末档全量评测 -> results.json（柱状图/文档锚点）----------
  if [ ! -f "$MD/results.json" ]; then
    echo "[F] $M 末档 $FINAL 全量(640)评测 -> results.json"
    "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
        --iterations "$FINAL" --subset 0 \
        --out "$MD/results.json" > "$RUNBASE/log_full_${M}.txt" 2>&1 \
      || { echo "  [$M] FULL_FAIL"; tail -15 "$RUNBASE/log_full_${M}.txt"; exit 1; }
  else
    echo "[F] $M results.json 已存在，跳过"
  fi

  echo "[$M] results.json:"
  "$PY" - "$MD/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    if isinstance(v, dict):
        print("  %s  PSNR %.3f  SSIM %.4f  LPIPS %.4f" % (k, v.get("PSNR",-1), v.get("SSIM",-1), v.get("LPIPS",-1)))
PYEOF
done

echo "[$(date +%F_%T)] ALL_RRF40K_${ENG}_DONE"
