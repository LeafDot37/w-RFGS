#!/bin/bash
# =============================================================================
# run_mlp_v2_probe_a4.sh —— Phase1 追加探档 A4E0：exp 1.6e-2 -> 1.6e-4
# =============================================================================
# 目的：A2E0(8e-3) 位处扫档上沿且呈"悬崖"式跳升，补探再翻倍初值是否还能涨。
#   若 A4E0 > A2E0   => A* 上移；
#   若打平/回落      => 8e-3 确认甜点。
# 方法 MVDR、RRF->abs40000、X0D0R0、embed 默认 2.5e-3（E0），口径同 Phase1。
# 恢复：results.json 齐即跳过；train/render/eval 各段断点续跑。
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
SPC=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
BASE=/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训/阶段1_LR协议
FIN=40000
X="--lr_schedule exp --mlp_lr_init 1.6e-2 --mlp_lr_final 1.6e-4 --lr_max_steps 40000"

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
[ -d "$SPC" ] || { echo "数据缺失: $SPC"; exit 1; }
mkdir -p "$BASE"
echo "=========== [$(date +%F_%T)] PHASE1-PROBE A4E0 卡=$CUDA_VISIBLE_DEVICES ==========="

c=A4E0
MD="$BASE/$c/MVDR"
mkdir -p "$MD"
echo "===================================================================="
echo "[$(date +%F_%T)] ==== $c  MVDR  RRF->$FIN ==== extra: $X"
echo "===================================================================="

if [ -f "$MD/results.json" ]; then
  echo "[$c] results.json 已存在，整档跳过"
else
  if [ ! -f "$MD/point_cloud/iteration_${FIN}/point_cloud.ply" ]; then
    echo "[T] $c train_rrf_mlp.py ..."
    # shellcheck disable=SC2086
    "$PY" "$CODE/train_rrf_mlp.py" -s "$SPC" -m "$MD" \
        --start_checkpoint "$SEED" --iterations "$FIN" \
        --save_iterations "$FIN" \
        --densify_until_iter 0 --data_device cpu --quiet \
        --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
        $X > "$BASE/log_train_${c}.txt" 2>&1 \
      || { echo "[$c] TRAIN_FAIL"; tail -25 "$BASE/log_train_${c}.txt"; exit 1; }
    echo "[T] $c 完成（$FIN 云已写）"
  else
    echo "[T] $c 云已存在，跳过训练"
  fi
  if [ ! -d "$MD/test/ours_${FIN}/renders" ]; then
    echo "[R] $c render ..."
    "$PY" "$CODE/render_rrf_mlp.py" -s "$SPC" -m "$MD" \
        --iteration "$FIN" --eval --skip_train --quiet \
        > "$BASE/log_render_${c}.txt" 2>&1 \
      || { echo "[$c] RENDER_FAIL"; tail -15 "$BASE/log_render_${c}.txt"; exit 1; }
    echo "[R] $c ours_${FIN} 已渲染"
  else
    echo "[R] $c 渲染已存在，跳过"
  fi
  echo "[E] $c metrics_fast 全量640 -> results.json"
  "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
      --iterations "$FIN" --subset 0 \
      --out "$MD/results.json" > "$BASE/log_eval_${c}.txt" 2>&1 \
    || { echo "[$c] EVAL_FAIL"; tail -15 "$BASE/log_eval_${c}.txt"; exit 1; }
  echo "[E] $c results.json 已写"
fi

echo "[$c] 结果:"
"$PY" - "$MD/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    if isinstance(v, dict):
        print("  %s  PSNR %.4f  SSIM %.4f  LPIPS %.4f" % (k, v.get("PSNR", -1), v.get("SSIM", -1), v.get("LPIPS", -1)))
PYEOF
echo "[$(date +%F_%T)] PHASE1_PROBE_A4_DONE"
