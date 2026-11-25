#!/bin/bash
# 训练集内探针：同一位置、**都见过**的 4 组 w，模型输出差多少 vs GT 差多少。
#
# 为什么要这一步
# ---------------------------------------------------------------------------
# `w_holdout` 上的置换对照已经给出阴性（头对 w 几乎不响应）。但阴性有两个来源：
#   (a) 表示能力不够：逐高斯标量场根本表达不了「换波束方向 ⇒ 整幅图变」；
#   (b) 泛化不够：见过的 (位置,w) 上其实学得动，只是没推到没见过的 w。
# 判据用 `w_holdout` 分不开这两者。办法是回到**训练集**：36 个位置每个都有 4 个
# 实例（4 组不同的 w，位置和 w 都在训练分布内）。若连这 4 个实例之间模型都不
# 拉开差距，就是 (a) 表示问题，与泛化无关。
#
# 用法（服务器）：
#   ROOT=<arm 目录> bash code/tools/eval/_w_train_probe.sh
# 产物：$ROOT/test_trainprobe/{inst*/renders,inst*/gt}、gt_train.log、render_train.log
set -uo pipefail
PY=${PY:-/root/miniconda3/bin/python}
R=${R:-/root/autodl-tmp/rf}
cd "$R/code" || exit 9

ROOT=${ROOT:?}
N=${N:-4}
DATA=${DATA:-$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100}
SPC=${SPC:-$R/datasets/RF-3DGS_mvdr_w_params/spectra}
META=${META:-$R/datasets/RF-3DGS_mvdr_w_params/meta.json}
CK=${CK:-$ROOT/angular_60000.pth}
OUT=$ROOT/test_trainprobe
mkdir -p "$OUT"

[ -f "$CK" ] || { echo "[FAIL] 缺检查点 $CK"; exit 9; }

# 只取前 N 个训练实例，它们属于同一个位置（实例表按位置连续编号，每位置 4 个）
if [ ! -f "$OUT/.gt.done" ]; then
  "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
      -m "$ROOT" --checkpoint "$CK" --role train --export_gt \
      --max_instances "$N" --out_root "$OUT" > "$OUT/gt_train.log" 2>&1 \
    && touch "$OUT/.gt.done" || { echo "[FAIL] GT 导出（见 $OUT/gt_train.log）"; exit 9; }
fi
if [ ! -f "$OUT/.render.done" ]; then
  "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
      -m "$ROOT" --checkpoint "$CK" --role train \
      --max_instances "$N" --out_root "$OUT" > "$OUT/render_train.log" 2>&1 \
    && touch "$OUT/.render.done" || { echo "[FAIL] 渲染（见 $OUT/render_train.log）"; exit 9; }
fi

ls -d "$OUT"/inst* 2>/dev/null | sed 's|^|  |'
