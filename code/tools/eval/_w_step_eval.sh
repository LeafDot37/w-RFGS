#!/bin/bash
# 单步评测（只做原样条件，**不做 w 置换**）：给定 ROOT 与 STEP，导出 GT → 渲染 → 三指标。
#
# 用途：比较同一臂在不同步数上的读数（120k vs 180k），判断长训尾巴值不值得跑。
# 与 `_eval_w_arms.sh` 的区别只有一个：不做 wswap 对照（那是判「头用不用 w」，
# 这里判「多跑 60k 步有没有用」，两件事不能混）。
#
# 产物：$OUT/{inst*/{gt,renders}, metrics.txt}；同一步重跑靠 .gt_*.done 跳过。
#
# 用法（服务器）：
#   ROOT=/root/autodl-tmp/rf/results/<run>/训练180k/w STEP=120000 bash 本脚本
set -uo pipefail
PY=${PY:-/root/miniconda3/bin/python}
R=${R:-/root/autodl-tmp/rf}

ROOT=${ROOT:?}
STEP=${STEP:?}
ROLES=${ROLES:-w_holdout}
DATA=${DATA:-$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100}
SPC=${SPC:-$R/datasets/RF-3DGS_mvdr_w_params/spectra}
META=${META:-$R/datasets/RF-3DGS_mvdr_w_params/meta.json}
CK=${CK:-$ROOT/angular_$STEP.pth}
OUT=${OUT:-$ROOT/test_step$STEP}

[ -f "$CK" ] || { echo "[FAIL] 缺检查点 $CK"; exit 9; }
mkdir -p "$OUT"
cd "$R/code" || exit 9
echo "ROOT=$ROOT STEP=$STEP OUT=$OUT ROLES=$ROLES"

for role in $ROLES; do
  if [ ! -f "$OUT/.gt_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" --export_gt \
        --out_root "$OUT" > "$OUT/gt_$role.log" 2>&1 \
      && touch "$OUT/.gt_$role.done" \
      || { echo "[FAIL] gt $role"; tail -8 "$OUT/gt_$role.log"; exit 9; }
  fi
  if [ ! -f "$OUT/.render_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" \
        --out_root "$OUT" > "$OUT/render_$role.log" 2>&1 \
      && touch "$OUT/.render_$role.done" \
      || { echo "[FAIL] render $role"; tail -8 "$OUT/render_$role.log"; exit 9; }
  fi
done

"$PY" tools/eval_instances.py --root "$OUT" --meta "$META" --skip_missing_gt
