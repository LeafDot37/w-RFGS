#!/bin/bash
# =============================================================================
# run_scalar_rrf.sh —— 形态 A'：标量头 + 固定 jet 查表（先混合再查表）
# =============================================================================
# 假设：GT 像素 100% 在 jet 调色板上，而原版/我们的 MLP 臂都在 **RGB 域**混合：
#     Σ wᵢ·LUT(sᵢ)  ≠  LUT(Σ wᵢ·sᵢ)          （jet 非线性，与线性混合不可交换）
#   ⇒ 只要 Σ wᵢ < 1，原版必然渲染出调色板外的颜色，而 GT 从不偏离。
#
# 诊断依据（.diag/palette_offline_probe.py + .diag/snap_test.py，作者自己的渲染集）：
#     MVDR —— 93% 像素离板（>5 占 53.7%）；**逐像素吸附回调色板后 LPIPS 变好 65.5%**
#             ⇒ 调色板约束是更优的假设类，A' 有靶子
#     CBF  —— 65% 离板但幅度极小（中位 1.41）；吸附后 LPIPS 反而略差（+0.0013）
#             ⇒ 预期净零，作**回归检查**（A' 不应把它搞坏）
#     MPC  —— 阴性对照：调色板 43811 色、GT 自身 4.2% 离板，吸附后 LPIPS 明显变差
#             ⇒ 本组不跑 MPC（非 rank-1，A' 前提不成立）
#
# 单变量设计：PE 三开关、LR 调度、起点、冻结协议与 V14 臂**完全一致**，
#   唯一差异 = 颜色头输出标量 + 查表（vs 输出 RGB）。
#   对照线（同 TOP 之外，已存在）：阶段2长训_位置编码_X5D3R1/<M>（V14）与
#   阶段1b长训_SH头/<M>（SH@A*）。
#
# 产物：$TOP/<M>/ 下 test/ours_<i>(48 抽帧) + 末档全量 640、curve.json、results.json。
#   各段存在即跳过（可续跑）。
#
# 用法（服务器 code/ 下，占一卡）：
#   CUDA_VISIBLE_DEVICES=2 METHODS="MVDR CBF" nohup bash tools/run_scalar_rrf.sh \
#       > /data/yuexf/3dgs/results/run_20260911_形态A标量头查表/日志/驱动.log 2>&1 &
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

METHODS="${METHODS:-MVDR CBF}"

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
TOP=/data/yuexf/3dgs/results/run_20260911_形态A标量头查表
LOG="$TOP/日志"
LADDER="32000 35000 38000 40000 45000 50000 55000 60000 65000"
FINAL=70000
SUBSET_CURVE=48
A_INIT=8e-3
A_FINAL=8e-5

[ -d "$DS" ] || { echo "数据缺失: $DS"; exit 1; }
[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$LOG"
echo "==== [$(date +%F_%T)] 形态A' 标量头+jet查表  卡=$CUDA_VISIBLE_DEVICES  methods='$METHODS' ===="

FAIL=0
for M in $METHODS; do
  DATA="$DS/3dgs_${M}_100"
  MD="$TOP/$M"
  mkdir -p "$MD"
  [ -d "$DATA" ] || { echo "  [$M] 数据缺失，跳过: $DATA"; FAIL=1; continue; }
  echo "[$(date +%F_%T)] ===== $M =====  -> $MD"

  if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
    echo "[$(date +%F_%T)]  训练 -> $FINAL（RRF 本地 $((FINAL-30000)) 步，A* 调度）"
    "$PY" train_rrf_scalar.py -s "$DATA" -m "$MD" \
        --start_checkpoint "$SEED" --iterations "$FINAL" --save_iterations $LADDER \
        --densify_until_iter 0 --data_device cpu --quiet \
        --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
        --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
        --lr_max_steps 40000 --pe_xyz_L 5 --pe_dir_L 3 --use_dist \
        > "$LOG/train_${M}.txt" 2>&1 \
      || { echo "  TRAIN_FAIL_$M"; tail -25 "$LOG/train_${M}.txt"; FAIL=1; continue; }
  else
    echo "  final 档已存在，跳过训练"
  fi

  for i in $LADDER $FINAL; do
    sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
    [ -d "$MD/test/ours_$i/renders" ] || "$PY" render_rrf_scalar.py -s "$DATA" -m "$MD" \
        --iteration "$i" --eval --skip_train --subset "$sub" --quiet \
        > "$LOG/render_${M}_$i.txt" 2>&1 \
      || { echo "  RENDER_FAIL_${M}_$i"; tail -12 "$LOG/render_${M}_$i.txt"; FAIL=1; break; }
    echo "  [R] $M ours_$i 已渲染 (subset=$sub)"
  done

  [ -f "$MD/curve.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
      --iterations $LADDER "$FINAL" --subset "$SUBSET_CURVE" --out "$MD/curve.json" \
      > "$LOG/curve_${M}.txt" 2>&1
  [ -f "$MD/results.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
      --iterations "$FINAL" --subset 0 --out "$MD/results.json" \
      > "$LOG/full_${M}.txt" 2>&1
  echo "[$(date +%F_%T)]  $M 完成：$MD"
done

echo "==== [$(date +%F_%T)] ALL_SCALAR_DONE (FAIL=$FAIL) ===="
