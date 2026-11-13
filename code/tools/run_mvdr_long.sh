#!/bin/bash
# =============================================================================
# run_mvdr_long.sh —— 仅 MVDR：V14(MLP+PE 全开) 与 SH@A* 各自长训到 RRF 40000
# =============================================================================
# 目的：MVDR 的 PE 臂趋势与其余方法不一致，怀疑是"未收敛"（现只跑到 RRF 1 万步 /
#   调度 t=0.25）。本驱动把两条线延长到 RRF 40000 = abs70000（调度终点 t=1.0）：
#     A) V14 = MLP 头 + PE(xyzL5 + dirL3) + dist  (阶段2_位置编码_X5D3R1 的同配置长训版)
#     B) SH@A* = SH 原版 + exp LR  (阶段1b长训_SH头 的同配置长训版)
#   对照的第 3 条线 = SH 原版固定(rrf40k_sh) 已有到 abs70000 的曲线，直接复用。
#   三线同轴（abs30000→70000，RRF 本地 0→40000），看 MVDR 趋势是否改变。
# 训练：与 phase2 的 A*(exp 8e-3->8e-5, lr_max_steps=40000) 一致；从同一视觉
#   chkpnt30000 起、一次跑到 70000（lr_max_steps=40000 ⇒ 终点正好 t=1.0），
#   densify=0、冻结几何。梯度档 32k..65k 每 3k 一档 + 70k。
# 产物：阶段2长训_位置编码_X5D3R1/MVDR 与 阶段1b长训_SH头/MVDR 下 test/ours_<i>(48 抽帧,
#   末档全量 640) + curve.json(48) + results.json(640)。各段存在即跳过（可续跑）。
# 用法（服务器 code/ 下，占一卡）：
#   CUDA_VISIBLE_DEVICES=2 nohup bash tools/run_mvdr_long.sh \
#       > results/run_mlp_v2_阶段1b阶段2与长训/长训MVDR_g2.log 2>&1 &
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DATA=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
TOP=/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训
LADDER="32000 35000 38000 40000 45000 50000 55000 60000 65000"
FINAL=70000
SUBSET_CURVE=48
A_INIT=8e-3
A_FINAL=8e-5

[ -d "$DATA" ] || { echo "数据缺失: $DATA"; exit 1; }
[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$TOP/阶段2长训MVDR_日志"
echo "==== [$(date +%F_%T)] MVDR 长训 卡=$CUDA_VISIBLE_DEVICES ladder->$FINAL RRF=$((FINAL-30000)) ===="

# ------------------------- A) V14：MLP + PE(xyzL5+dirL3)+dist -------------------------
MD="$TOP/阶段2长训_位置编码_X5D3R1/MVDR"; mkdir -p "$MD"
echo "[$(date +%F_%T)] A) V14 长训 -> $FINAL"
if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
  "$PY" train_rrf_mlp.py -s "$DATA" -m "$MD" \
      --start_checkpoint "$SEED" --iterations "$FINAL" --save_iterations $LADDER \
      --densify_until_iter 0 --data_device cpu --quiet \
      --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
      --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
      --lr_max_steps 40000 --pe_xyz_L 5 --pe_dir_L 3 --use_dist \
      > "$TOP/阶段2长训MVDR_日志/train_mlp_v14.txt" 2>&1 \
    || { echo "TRAIN_MLP_FAIL"; tail -25 "$TOP/阶段2长训MVDR_日志/train_mlp_v14.txt"; exit 1; }
else
  echo "  final 云已存在，跳过训练"
fi
for i in $LADDER $FINAL; do
  sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
  [ -d "$MD/test/ours_$i/renders" ] || "$PY" render_rrf_mlp.py -s "$DATA" -m "$MD" \
      --iteration "$i" --eval --skip_train --subset "$sub" --quiet \
      > "$TOP/阶段2长训MVDR_日志/render_mlp_$i.txt" 2>&1 \
    || { echo "RENDER_MLP_FAIL_$i"; tail -12 "$TOP/阶段2长训MVDR_日志/render_mlp_$i.txt"; exit 1; }
  echo "  [RA] ours_$i 已渲染(subset=$sub)"
done
[ -f "$MD/curve.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
    --iterations $LADDER "$FINAL" --subset "$SUBSET_CURVE" --out "$MD/curve.json" \
    > "$TOP/阶段2长训MVDR_日志/curve_mlp.txt" 2>&1
[ -f "$MD/results.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
    --iterations "$FINAL" --subset 0 --out "$MD/results.json" \
    > "$TOP/阶段2长训MVDR_日志/full_mlp.txt" 2>&1
echo "[$(date +%F_%T)] A) 完成：$MD"

# ------------------------- B) SH@A*：SH 原版 + exp LR -------------------------
MD="$TOP/阶段1b长训_SH头/MVDR"; mkdir -p "$MD"
echo "[$(date +%F_%T)] B) SH@A* 长训 -> $FINAL"
if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
  "$PY" train_rrf_sh.py -s "$DATA" -m "$MD" \
      --start_checkpoint "$SEED" --iterations "$FINAL" --save_iterations $LADDER \
      --densify_until_iter 0 --data_device cpu --quiet \
      --lr_schedule exp --color_lr_init "$A_INIT" --color_lr_final "$A_FINAL" \
      --lr_max_steps 40000 \
      > "$TOP/阶段2长训MVDR_日志/train_sh.txt" 2>&1 \
    || { echo "TRAIN_SH_FAIL"; tail -25 "$TOP/阶段2长训MVDR_日志/train_sh.txt"; exit 1; }
else
  echo "  final 云已存在，跳过训练"
fi
for i in $LADDER $FINAL; do
  sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
  [ -d "$MD/test/ours_$i/renders" ] || "$PY" render_sh_rrf.py -s "$DATA" -m "$MD" \
      --iteration "$i" --eval --skip_train --subset "$sub" --quiet \
      > "$TOP/阶段2长训MVDR_日志/render_sh_$i.txt" 2>&1 \
    || { echo "RENDER_SH_FAIL_$i"; tail -12 "$TOP/阶段2长训MVDR_日志/render_sh_$i.txt"; exit 1; }
  echo "  [RB] ours_$i 已渲染(subset=$sub)"
done
[ -f "$MD/curve.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
    --iterations $LADDER "$FINAL" --subset "$SUBSET_CURVE" --out "$MD/curve.json" \
    > "$TOP/阶段2长训MVDR_日志/curve_sh.txt" 2>&1
[ -f "$MD/results.json" ] || "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
    --iterations "$FINAL" --subset 0 --out "$MD/results.json" \
    > "$TOP/阶段2长训MVDR_日志/full_sh.txt" 2>&1
echo "[$(date +%F_%T)] B) 完成：$MD"

echo "[$(date +%F_%T)] ALL_MVDR_LONG_DONE"
