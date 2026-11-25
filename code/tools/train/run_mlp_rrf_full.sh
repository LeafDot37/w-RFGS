#!/bin/bash
# =============================================================================
# run_mlp_rrf_full.sh —— MLP-RRF 本地全量（4 方法，从原版视觉 chkpnt30000 起）
# =============================================================================
# 单变量对比"SH 颜色头 → embedding+共享 MLP 颜色头"：
#   起点     = 原版 RF-3DGS 视觉重建结果（datasets/.../blender_visual_trained/chkpnt30000.pth）
#   训练     = code/rfgs/rrf/train_rrf_mlp.py（冻结几何 + opacity/embedding/MLP 可训，densify=0）
#   绝对末档 = 40000（= 视觉 30000 + RRF 10000，对齐 config.yaml eval.reference.stage2_orig）
#   评测     = render_rrf_mlp.py --iteration 40000 + metrics.py
# 沿途额外存 36000/38000 档作为断点保护（不评测，可选画收敛曲线）。
#
# 说明：本脚本面向"本地 WSL /mnt/d"跑（不经服务器），路径全为 WSL 侧绝对路径。
# 已存在产物自动跳过（可中断续跑）：判据 = point_cloud/iteration_40000/point_cloud.ply。
# 用法（在 Windows Git Bash 里）：
#   wsl.exe -d Ubuntu-22.04 bash /mnt/d/GS3D/mvsplat/code/tools/train/run_mlp_rrf_full.sh
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true

PY=/home/leafd_dot/miniforge3/envs/mvsplat/bin/python
CODE=/mnt/d/GS3D/mvsplat/code
DATA=/mnt/d/GS3D/mvsplat/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/mnt/d/GS3D/mvsplat/datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
RUN=/mnt/d/GS3D/mvsplat/results/run_mlp_rrf_full_MLP首轮全量
METHODS="MPC CBF Delay MVDR"
ITER_FINAL=40000
ITER_EXTRA="36000 38000"          # 断点保护档（不评测）
EMBED_DIM=32
MLP_HIDDEN=128
MLP_DEPTH=2

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$RUN"

for M in $METHODS; do
  P_SPC="$DATA/3dgs_${M}_100"
  [ -d "$P_SPC" ] || { echo "[$M] 数据缺失: $P_SPC"; exit 1; }
  MD="$RUN/$M"
  mkdir -p "$MD"
  echo "===================================================================="
  echo "[$(date +%F_%T)] ==== $M: RRF 30000 -> ${ITER_FINAL} (embedding+MLP) ===="
  echo "===================================================================="

  # ---- 训练（已出 final 档则跳过）----
  if [ ! -f "$MD/point_cloud/iteration_${ITER_FINAL}/point_cloud.ply" ]; then
    echo "[T] $M train_rrf_mlp.py ..."
    "$PY" train_rrf_mlp.py -s "$P_SPC" -m "$MD" \
        --start_checkpoint "$SEED" \
        --iterations "$ITER_FINAL" \
        --save_iterations $ITER_EXTRA "$ITER_FINAL" \
        --densify_until_iter 0 --data_device cpu --quiet \
        --embed_dim "$EMBED_DIM" --mlp_hidden "$MLP_HIDDEN" --mlp_depth "$MLP_DEPTH" \
        > "$RUN/log_train_${M}.txt" 2>&1 \
      || { echo "  [$M] TRAIN_FAIL"; tail -25 "$RUN/log_train_${M}.txt"; exit 1; }
    echo "[T] $M 完成（final 档已写）"
  else
    echo "[T] $M final 档已存在，跳过训练"
  fi

  # ---- 渲染 final 档 ----
  if [ ! -d "$MD/test/ours_${ITER_FINAL}/renders" ]; then
    echo "[R] $M render ours_${ITER_FINAL}"
    "$PY" render_rrf_mlp.py -s "$P_SPC" -m "$MD" \
        --iteration "$ITER_FINAL" --eval --skip_train --quiet \
        > "$RUN/log_render_${M}.txt" 2>&1 \
      || { echo "  [$M] RENDER_FAIL"; tail -15 "$RUN/log_render_${M}.txt"; exit 1; }
  else
    echo "[R] $M renders 已存在，跳过"
  fi

  # ---- metrics（缺 final 键才跑）----
  need=1
  if [ -f "$MD/results.json" ]; then
    grep -q "\"ours_${ITER_FINAL}\"" "$MD/results.json" && need=0
  fi
  if [ "$need" = 1 ]; then
    echo "[M] $M metrics"
    "$PY" "$CODE/metrics.py" -m "$MD" > "$RUN/log_metrics_${M}.txt" 2>&1 \
      || { echo "  [$M] METRICS_FAIL"; tail -15 "$RUN/log_metrics_${M}.txt"; exit 1; }
  else
    echo "[M] $M results.json 已含 ours_${ITER_FINAL}，跳过"
  fi

  echo "[$M] results:"
  "$PY" - "$MD/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0):
    if isinstance(v, dict):
        print("  %s  PSNR %.3f  SSIM %.4f  LPIPS %.4f" % (k, v.get("PSNR", -1), v.get("SSIM", -1), v.get("LPIPS", -1)))
PYEOF
done

echo "ALL_MLP_RRF_FULL_DONE"
