#!/bin/bash
# 波束权 w 轴的小样本适应（本机 4060 侧驱动）。
#
# 口径：留出 w 实例 318（w_label=t45_p270，base_instance=0，role=w_holdout，
# 640 个 test 视角）。适应集 = test 视角里**均匀随机** 320 个（train_view_random
# 320 / seed 0，位置与朝向同时无偏），评估 = 另外 320 个。评测端 `.tmp/_tta_eval.py`
# 传同一对数 (--random 320 --seed 0) 才能对上同一条划分。
#
# 三条臂（同一实例、同一 320 留出集、同一套指标，可同表排名）：
#   zs  零样本：180k 波束条件臂原样渲染
#   ad  +320 视角 20k 步适应
#   or  全 640 视角 20k 步（样本内上限；它在留出 320 上是见过答案的）
#
# ⚠ 长跑期间不得编辑本文件（bash 按字节偏移顺序读脚本）。
set -uo pipefail
# 谱图 LRU 上限必须 ≥ 适应集视角数，否则每步都从 9p 盘重读 PNG（实测 3 倍变慢）。
export RRF_SPECTRA_CACHE=700
PY=/home/leafd_dot/miniforge3/envs/mvsplat/bin/python3.10
R=/mnt/d/GS3D/mvsplat
RUN=$R/results/run_20261114_本机4060_波束权小样本适应
CK=$RUN/ckpt/w180k/angular_180000.pth
DATA=$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100
SPC=$R/datasets/RF-3DGS_mvdr_w_params/spectra
META=$R/datasets/RF-3DGS_mvdr_w_params/meta.json
SEED=$R/datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
I=${I:-318}
STEPS=${STEPS:-20000}
ELR=${ELR:-3e-4}
MLP_LR=${MLP_LR:-2e-3}
mkdir -p "$RUN"/{gt,renders,train,eval,logs}
cd "$R/code" || exit 9

echo "=== [$(date '+%F %T')] w-TTA inst=$I steps=$STEPS ==="
[ -f "$CK" ] || { echo "[FAIL] 缺检查点 $CK"; exit 9; }

# ---- 1) GT 导出（640 test 视角，jet 域 RGB）----
if [ ! -f "$RUN/.gt.done" ]; then
  "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
      -m "$RUN/train/_gt" --checkpoint "$CK" --instance "$I" --split test \
      --limit 640 --export_gt --start_checkpoint "$SEED" \
      --out_root "$RUN/gt/inst$I" > "$RUN/logs/gt.log" 2>&1 \
    && touch "$RUN/.gt.done" || { echo "[FAIL] gt"; tail -5 "$RUN/logs/gt.log"; exit 8; }
fi

# ---- 2) 零样本渲染 ----
if [ ! -f "$RUN/.rd_zs.done" ]; then
  "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
      -m "$RUN/train/_zs" --checkpoint "$CK" --instance "$I" --split test \
      --limit 640 --start_checkpoint "$SEED" --out_root "$RUN/renders/zs$I" \
      > "$RUN/logs/rd_zs.log" 2>&1 \
    && touch "$RUN/.rd_zs.done" || { echo "[FAIL] render zs"; tail -5 "$RUN/logs/rd_zs.log"; exit 7; }
fi

# ---- 3) 两条适应臂 ----
for ARM in ad or; do
  if [ "$ARM" = "ad" ]; then NV=320; else NV=640; fi
  MD=$RUN/train/${ARM}$I; RD=$RUN/renders/${ARM}$I
  if [ ! -f "$RUN/.tr_$ARM.done" ]; then
    echo "--- [$(date '+%F %T')] 训练 $ARM（N=$NV, $STEPS 步）"
    "$PY" train_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$MD" --start_checkpoint "$SEED" --resume_angular "$CK" \
        --train_roles w_holdout --train_instance_ids "$I" --allow_single_w 1 \
        --train_views test --train_view_random "$NV" --train_view_seed 0 \
        --iterations "$STEPS" --views_per_step 2 \
        --use_tx_orient 1 --use_tx_local_dir 1 --use_tx_freq 1 --use_tx_w 1 \
        --init_u 0.05 --lr_schedule cosine --lr_max_steps "$STEPS" \
        --lr_delay_steps 0 --mlp_lr_init "$MLP_LR" --mlp_lr_final 1e-5 \
        --opacity_lr 0 --embed_lr "$ELR" --log_every 1000 \
        > "$RUN/logs/${ARM}$I.train.log" 2>&1 \
      && touch "$RUN/.tr_$ARM.done" || { echo "[FAIL] train $ARM"; tail -8 "$RUN/logs/${ARM}$I.train.log"; exit 8; }
  fi
  if [ ! -f "$RUN/.rd_$ARM.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$MD" --checkpoint "$MD/angular_${STEPS}.pth" --instance "$I" --split test \
        --limit 640 --start_checkpoint "$SEED" --out_root "$RD" \
        > "$RUN/logs/rd_$ARM.log" 2>&1 \
      && touch "$RUN/.rd_$ARM.done" || { echo "[FAIL] render $ARM"; tail -5 "$RUN/logs/rd_$ARM.log"; exit 7; }
  fi
done

# ---- 4) 三条臂在同一 320 留出集上评测 ----
for ARM in zs ad or; do
  case $ARM in zs) RD=$RUN/renders/zs$I ;; ad) RD=$RUN/renders/ad$I ;; or) RD=$RUN/renders/or$I ;; esac
  "$PY" "$R/.tmp/_tta_eval.py" --renders "$RD" --gt "$RUN/gt/inst$I" \
      --poses_root "$DATA" --random 320 --seed 0 --skip_flat_gt \
      --out "$RUN/eval/${ARM}$I.tsv" > "$RUN/logs/eval_$ARM.log" 2>&1
  echo -n "  $ARM: "; tail -1 "$RUN/logs/eval_$ARM.log"
done
echo "=== [$(date '+%F %T')] 完成 ==="
