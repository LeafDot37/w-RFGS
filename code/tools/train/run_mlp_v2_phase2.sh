#!/bin/bash
# =============================================================================
# run_mlp_v2_phase2.sh —— Phase2：位置编码主筛（LR 锁 A*，逐档加 PE）5 臂 × 多方法
# =============================================================================
# 目的：Phase1b 判出 A* LR 是主收益源、MLP 形式仅在 Delay/MVDR 略优；本阶段给
#   MLP 颜色头逐档加位置编码(PE)，看 PE + A* LR 能否让 MLP 在落后方法(MPC/CBF)
#   翻超 SH@A*、并压过 SH 原版。5 臂（编码=产物目录后缀，见 §6.2）：
#     V10 X5D0R0  PE(xyz)L5                    V11 X3D0R0  PE(xyz)L3(带宽敏感)
#     V12 X0D3R0  PE(dir)L3(分离 dir-PE)       V13 X5D3R0  xyz-PE+dir-PE 叠加
#     V14 X5D3R1  再+距离条件(use_dist) → A2 全部
#   所有臂其余全一致 = Phase1b MLP@A* 配置（A2E0: exp 8e-3->8e-5 lr_max 40000、
#   embed 2.5e-3 固定、embed32/hidden128/depth2、同 chkpnt30000、冻结几何、
#   densify=0）⇒ 与 阶段1b_MLP头(<M>/curve+results.json) 单变量差 = PE。
# 基准（复用已有，不重跑）：阶段1b长训_SH头 (SH@A*)、阶段1b_MLP头 (MLP@A* 无PE)、
#   run_rrf40k_sh/mlp (固定 LR)、config stage2_orig (SH 原版 640)。
# 产物 阶段2_位置编码_<ENC>/<M>/：train 梯度档 point_cloud/iteration_{32000..40000}、
#   test/ours_<i>（中间 48 抽帧 + 末档全量 640）、curve.json(48)、results.json(640)。
# 恢复：各段产物存在即跳过（可中断续跑）。
# 用法（服务器 ~/3dgs，每实例占一卡；ENC×METHOD 由调用方分配以并行）：
#   CUDA_VISIBLE_DEVICES=0 ENCS="X5D0R0 X3D0R0 X0D3R0 X5D3R0 X5D3R1" \
#     METHODS="MPC CBF"   nohup bash run_mlp_v2_phase2.sh > results/run_mlp_v2_阶段1b阶段2与长训/阶段2_g0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 ENCS="X5D0R0 X3D0R0 X0D3R0 X5D3R0 X5D3R1" \
#     METHODS="Delay MVDR" nohup bash run_mlp_v2_phase2.sh > results/run_mlp_v2_阶段1b阶段2与长训/阶段2_g2.log 2>&1 &
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
ENCS="${ENCS:?需指定 ENCS 编码列表，如 X5D0R0 X3D0R0}"
METHODS="${METHODS:?需指定 METHODS 列表，如 MPC CBF}"
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DATA=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
SEED=/data/yuexf/3dgs/results/orig_seed_chkpnt30000.pth
TOP=/data/yuexf/3dgs/results/run_mlp_v2_阶段1b阶段2与长训
LADDER="32000 35000 38000 40000"
FINAL=40000
SUBSET_CURVE=48
# A*（A2E0 锁定，Phase1b 同款）：MLP 组 exp；embed 2.5e-3 固定（E0 不传 --embed_lr）
A_INIT=8e-3
A_FINAL=8e-5

[ -f "$SEED" ] || { echo "seed 缺失: $SEED"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$TOP/阶段2_日志"
echo "======== [$(date +%F_%T)] PHASE2 卡=$CUDA_VISIBLE_DEVICES  ENCS=[$ENCS] METHODS=[$METHODS] A*=$A_INIT->$A_FINAL ========"

enc_pe_args() {  # 编码 -> PE 开关（编码=目录/run 后缀，见 docs §6.2）
  case "$1" in
    X5D0R0) echo "--pe_xyz_L 5";;
    X3D0R0) echo "--pe_xyz_L 3";;
    X0D3R0) echo "--pe_dir_L 3";;
    X5D3R0) echo "--pe_xyz_L 5 --pe_dir_L 3";;
    X5D3R1) echo "--pe_xyz_L 5 --pe_dir_L 3 --use_dist";;
    *) echo "__BAD_ENC__";;
  esac
}

for E in $ENCS; do
  PE_ARGS=$(enc_pe_args "$E")
  [ "$PE_ARGS" = "__BAD_ENC__" ] && { echo "未知编码 $E"; exit 2; }
  for M in $METHODS; do
    P_SPC="$DATA/3dgs_${M}_100"
    [ -d "$P_SPC" ] || { echo "[$E/$M] 数据缺失"; exit 1; }
    MD="$TOP/阶段2_位置编码_$E/$M"; mkdir -p "$MD"
    echo "===================================================================="
    echo "[$(date +%F_%T)] ==== [$E] $M: RRF -> $FINAL @A*+PE ===="
    echo "===================================================================="

    # ---------- 训练（final 云已存在则跳过整段）----------
    if [ ! -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
      echo "[T] $E/$M train_rrf_mlp.py $PE_ARGS ..."
      "$PY" train_rrf_mlp.py -s "$P_SPC" -m "$MD" \
          --start_checkpoint "$SEED" --iterations "$FINAL" \
          --save_iterations $LADDER \
          --densify_until_iter 0 --data_device cpu --quiet \
          --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
          --lr_schedule exp --mlp_lr_init "$A_INIT" --mlp_lr_final "$A_FINAL" \
          --lr_max_steps 40000 $PE_ARGS \
          > "$TOP/阶段2_日志/log_train_${E}_${M}.txt" 2>&1 \
        || { echo "  [$E/$M] TRAIN_FAIL"; tail -25 "$TOP/阶段2_日志/log_train_${E}_${M}.txt"; exit 1; }
      echo "[T] $E/$M 完成（$FINAL 云已写）"
    else
      echo "[T] $E/$M final 云已存在，跳过训练"
    fi

    # ---------- 逐档渲染：中间 48（曲线），末档全量 0（results.json）----------
    for i in $LADDER; do
      sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
      if [ ! -d "$MD/test/ours_$i/renders" ]; then
        "$PY" render_rrf_mlp.py -s "$P_SPC" -m "$MD" --iteration "$i" \
            --eval --skip_train --subset "$sub" --quiet \
            > "$TOP/阶段2_日志/log_render_${E}_${M}_${i}.txt" 2>&1 \
          || { echo "  [$E/$M] RENDER_FAIL_$i"; tail -12 "$TOP/阶段2_日志/log_render_${E}_${M}_${i}.txt"; exit 1; }
        echo "  [R] ours_$i 已渲染（subset=$sub）"
      else
        echo "  [R] ours_$i 已存在，跳过"
      fi
    done

    # ---------- 48 子集曲线 -> curve.json ----------
    if [ ! -f "$MD/curve.json" ]; then
      echo "[C] $E/$M 抽帧($SUBSET_CURVE)评测 $LADDER -> curve.json"
      "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
          --iterations $LADDER --subset "$SUBSET_CURVE" \
          --out "$MD/curve.json" > "$TOP/阶段2_日志/log_curve_${E}_${M}.txt" 2>&1 \
        || { echo "  [$E/$M] CURVE_FAIL"; tail -15 "$TOP/阶段2_日志/log_curve_${E}_${M}.txt"; exit 1; }
    else
      echo "[C] $E/$M curve.json 已存在，跳过"
    fi

    # ---------- 末档全量 640 -> results.json ----------
    if [ ! -f "$MD/results.json" ]; then
      echo "[F] $E/$M 末档 $FINAL 全量(640)评测 -> results.json"
      "$PY" "$CODE/tools/metrics_fast.py" -m "$MD" \
          --iterations "$FINAL" --subset 0 \
          --out "$MD/results.json" > "$TOP/阶段2_日志/log_full_${E}_${M}.txt" 2>&1 \
        || { echo "  [$E/$M] FULL_FAIL"; tail -15 "$TOP/阶段2_日志/log_full_${E}_${M}.txt"; exit 1; }
    else
      echo "[F] $E/$M results.json 已存在，跳过"
    fi

    echo "[$E/$M] curve.json (48):"
    "$PY" - "$MD/curve.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    print("  %s  PSNR %.3f" % (k, v.get("PSNR", -1)))
PYEOF
    echo "[$E/$M] results.json (640):"
    "$PY" - "$MD/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[1])):
    print("  %s  PSNR %.3f" % (k, v.get("PSNR", -1)))
PYEOF
  done
done

echo "[$(date +%F_%T)] ALL_PHASE2_${CUDA_VISIBLE_DEVICES}_DONE"
