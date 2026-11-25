#!/bin/bash
# MVDR 参数臂 180k 的全量评测：主读数（原发射源）+ 位置留出 52 实例，各 640 test 视角。
# 与 CBF 臂同一套脚本与指标；GT 与渲染都换成 MVDR 谱图。
#
#   bash code/tools/eval/_eval_mvdr_180k.sh
set -uo pipefail
R=/mnt/d/GS3D/mvsplat
ROOT="$R/results/run_20261025_本机4060_MVDR参数泛化/长训180k/full"
STEP=180000 ROLES="orig pos_holdout" MAXINST=0 ROOT="$ROOT" \
  SPC="$R/datasets/RF-3DGS_mvdr_params/spectra" \
  META="$R/datasets/RF-3DGS_mvdr_params/meta.json" \
  bash "$R/code/tools/eval/_eval_all_instances.sh"
echo "[eval exit=$?]"
