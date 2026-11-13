#!/bin/bash
# resume_stageC.sh —— 在已跑完 阶段A+B 的 run 目录上只重跑 阶段C(RRF 频谱重建)，不重算 A+B。
# 用法(服务器 ~/3dgs/ 下):
#     CUDA_VISIBLE_DEVICES=<卡> RF_VRAM_GB=<GiB> bash resume_stageC.sh results/run_<ts>
# 背景: 阶段C 读 3200 张图/方法, dataset_readers 用 PIL 惰性打开不及时关 → 默认 fd 软限 1024 不够
#       (OSError Errno 24 Too many open files)。本脚本先抬 fd 上限再逐方法跑(需 hard≥65535)。
#       显存包裹同 run_end_to_end: RF_VRAM_GB>0 → 经 code/tools/_prealloc.py 同进程预占。
# 产出: 写回同一 run 目录 rf-3dgs_2/<METHOD> + logs/stageC_rrf_<M>.log；阶段B 目录不动。
set -euo pipefail
ulimit -n 65535 2>/dev/null || true
RUN="${1:?用法: resume_stageC.sh <run目录, 如 results/run_20260907_005205_full档_端到端>}"
[ -d "$RUN" ] || RUN="results/$RUN"
RUN="$(cd "$RUN" && pwd)"   # 绝对化: 后续 cd "$P_CODE" 后重定向/检查仍有效
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_DIR/load_config.sh"; load_config full

[ -f "$RUN/rf-3dgs_1/chkpnt${ITR1}.pth" ] || { echo "[resume] 缺 $RUN/rf-3dgs_1/chkpnt${ITR1}.pth (阶段B 未完成?)"; exit 1; }

if [ -n "${RF_VRAM_GB:-}" ]; then TRAIN_PY=( "$PY" "$P_CODE/tools/_prealloc.py" train.py ); else TRAIN_PY=( "$PY" train.py ); fi

rm -rf "$RUN/rf-3dgs_2"; mkdir -p "$RUN/rf-3dgs_2" "$RUN/logs"
rm -f "$RUN"/logs/stageC_rrf_*.log   # 清掉上次失败残留, 避免 tail 误导
FINAL2="${TOTAL2:-$ITR2}"; SAVE2="$ITR2"; [ "$FINAL2" != "$ITR2" ] && SAVE2="$ITR2 $FINAL2"
echo "[resume stageC] RUN=$RUN  DENSIFY=$DENSIFY  末迭代=$FINAL2  存档档=[$SAVE2]  start=chkpnt${ITR1}"
cd "$P_CODE"
for M in $METHODS; do
  echo "  - $M  ($P_SPECTRUM/3dgs_${M}_100  → $RUN/rf-3dgs_2/$M)"
  if ! "${TRAIN_PY[@]}" -s "$P_SPECTRUM/3dgs_${M}_100" -m "$RUN/rf-3dgs_2/$M" \
        --iterations "$FINAL2" --start_checkpoint "$RUN/rf-3dgs_1/chkpnt${ITR1}.pth" \
        --densify_until_iter "$DENSIFY" --save_iterations $SAVE2 \
        --data_device cpu \
        > "$RUN/logs/stageC_rrf_${M}.log" 2>&1; then
    echo "[resume stageC] $M 失败, 末尾日志:"; tail -25 "$RUN/logs/stageC_rrf_${M}.log"; exit 1
  fi
done
echo "full" > "$RUN/profile.txt"; echo "replace" > "$RUN/cloud_mode.txt"
echo "[resume stageC] 完成: $RUN (rf-3dgs_2 下 4 方法双档 22000/40000)"
