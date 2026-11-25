#!/bin/bash
# 参数化数据集的按角色全量评测：每个实例 640 个 test 视角，三指标一起报。
#
# 用法（WSL）：
#   ROOT=/mnt/d/.../results/run_xxx_full STEP=180000 bash 本脚本
#   ROLES="yaw_holdout" MAXINST=3 bash 本脚本      # 冒烟
#
# 产物：$ROOT/test_inst/inst<ID>/{renders,gt}/、$ROOT/test_inst/metrics.txt
#
# 注意：几何只加载一次（--role 批量口径），比逐实例起进程快一个量级。
set -uo pipefail
# PY/R 可从环境覆盖（服务器侧见 _srv_eval_mvdr_180k.sh）
PY=${PY:-/home/leafd_dot/miniforge3/envs/mvsplat/bin/python3.10}
R=${R:-/mnt/d/GS3D/mvsplat}
cd "$R/code" || exit 9

ROOT=${ROOT:?}
STEP=${STEP:-60000}
CK=${CK:-$ROOT/angular_$STEP.pth}
DATA=${DATA:-$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100}
SPC=${SPC:-$R/datasets/RF-3DGS_cbf_params/spectra}
META=${META:-$R/datasets/RF-3DGS_cbf_params/meta.json}
# 默认只评「主读数（原 TX48）+ 位置留出」两组：参数留出组不再是闸，
# 需要时用 ROLES="... yaw_holdout ..." 显式加。
ROLES=${ROLES:-orig pos_holdout}
MAXINST=${MAXINST:-0}
# 退化 GT（整幅 jet(0)，源 u 全 0）会让 PSNR 跳到 99 dB 截断值。交付口径一律
# 跳过它们（见 eval_instances.py 的说明）；默认保持历史行为，需要时 SKIPFLAT=1。
SKIPFLAT=${SKIPFLAT:-0}
FLATARG=""
[ "$SKIPFLAT" = "1" ] && FLATARG="--skip_flat_gt"
OUT=$ROOT/test_inst
mkdir -p "$OUT"

[ -f "$CK" ] || { echo "[FAIL] 缺检查点 $CK"; exit 9; }
[ -d "$SPC" ] || { echo "[FAIL] 缺谱图 $SPC"; exit 9; }
[ -f "$META" ] || { echo "[FAIL] 缺实例表 $META"; exit 9; }

echo "ROOT=$ROOT STEP=$STEP"
echo "ROLES=$ROLES MAXINST=$MAXINST"

for role in $ROLES; do
  # 1) GT 导出（查 jet 表）；一个角色做完再下一个，避免重复扫盘
  if [ ! -f "$OUT/.gt_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" --export_gt \
        --max_instances "$MAXINST" --out_root "$OUT" \
        > "$OUT/gt_$role.log" 2>&1 \
      && touch "$OUT/.gt_$role.done" \
      || { echo "[FAIL] gt $role（见 $OUT/gt_$role.log）"; continue; }
  fi
  # 2) 渲染
  if [ ! -f "$OUT/.render_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" \
        --max_instances "$MAXINST" --out_root "$OUT" \
        > "$OUT/render_$role.log" 2>&1 \
      && touch "$OUT/.render_$role.done" \
      || { echo "[FAIL] render $role（见 $OUT/render_$role.log）"; continue; }
  fi
  echo "[OK] $role"
done

echo "=== 逐实例指标 ==="
"$PY" tools/eval_instances.py --root "$OUT" --meta "$META" --skip_missing_gt $FLATARG

echo
echo "=== 主读数：留出发射源 vs 原始 RF-3DGS ==="
"$PY" tools/_orig_vs_baseline.py --metrics "$OUT/metrics.txt" --meta "$META"
