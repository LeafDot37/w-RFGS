#!/bin/bash
# =============================================================================
# run_mlp_v2_phase1.sh —— v2 实验阶段1：LR 主筛（只动 MLP lr 协议 / embed lr）
# =============================================================================
# 目标（docs/RRF-MLP改进方案_对照Scaffold-GS.md §6.3 阶段1）：
#   方法 = MVDR（MLP 差距最大、最敏感）；预算 abs40000（RRF 1 万本地步）
#   V00 A0E0  MLP lr 固定 1.25e-4（现状，同批复跑 → run 噪声对照）
#   V01 A1E0  exp 4e-3 -> 4e-5，max_steps=40000
#   V02 A2E0  exp 8e-3 -> 8e-5，max_steps=40000
#   V03 A3E0  exp 2e-3 -> 2e-5，max_steps=40000
#   V04 A1E5  V01 + embed_lr 5e-3（验证 embed lr 是否值得）
#   X0/D0/R0 全关（无位置编码，LR 单变量）。末档 abs40000 全量 640 -> results.json。
# 注意：lr_max_steps=40000 = RRF 本地总长（设计全程固定），扫档在调度第 1 万步(t=0.25)取值。
# 恢复：每档判据 = results.json（整档完成）。train/render/eval 各段均可断点续跑。
# 用法（服务器 ~/3dgs，nohup）：
#   CUDA_VISIBLE_DEVICES=0 CFGS='A0E0 A1E0 A2E0' nohup bash run_mlp_v2_phase1.sh \
#       > results/run_mlp_v2_阶段1b阶段2与长训/阶段1_LR协议/phase1_g0.log 2>&1 &
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
SPC=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
BASE=/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训/阶段1_LR协议
FIN=40000
CFGS="${CFGS:-A0E0 A1E0 A2E0 A3E0 A1E5}"

# 每档的 MLP/embed lr 相关 CLI（A0 为空串 = 走 fixed 默认 1.25e-4；A 系均 X0D0R0）
declare -A EXTRA=(
  [A0E0]=""
  [A1E0]="--lr_schedule exp --mlp_lr_init 4e-3 --mlp_lr_final 4e-5 --lr_max_steps 40000"
  [A2E0]="--lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 40000"
  [A3E0]="--lr_schedule exp --mlp_lr_init 2e-3 --mlp_lr_final 2e-5 --lr_max_steps 40000"
  [A1E5]="--lr_schedule exp --mlp_lr_init 4e-3 --mlp_lr_final 4e-5 --lr_max_steps 40000 --embed_lr 5e-3"
)

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
[ -d "$SPC" ] || { echo "数据缺失: $SPC"; exit 1; }
mkdir -p "$BASE"
echo "=========== [$(date +%F_%T)] PHASE1 LR 主筛 卡=$CUDA_VISIBLE_DEVICES CFGS='$CFGS' ==========="

for c in $CFGS; do
  X="${EXTRA[$c]:-}"
  [ -n "$X" ] || [ "$c" = A0E0 ] || { echo "[$c] 未定义 flags"; exit 2; }
  MD="$BASE/$c/MVDR"
  mkdir -p "$MD"
  echo "===================================================================="
  echo "[$(date +%F_%T)] ==== $c  MVDR  RRF->$FIN ==== extra: ${X:-（fixed 默认）}"
  echo "===================================================================="

  if [ -f "$MD/results.json" ]; then
    echo "[$c] results.json 已存在，整档跳过"
  else
    # ---------- 训练 ----------
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
    # ---------- 渲染 ----------
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
    # ---------- 末档全量 640 评测 ----------
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
done
echo "[$(date +%F_%T)] PHASE1_DONE"
