#!/bin/bash
# 波束权 w 轴的评测：GT 导出 → 渲染 → 三指标，外加一条**w 置换对照**。
#
# 为什么必须有 w 置换对照
# ---------------------------------------------------------------------------
# 只看 `w_holdout`（没见过的 w）的绝对指标，无法区分两件事：
#   (a) 头真的在用 w 条件，只是没学到没见过的那几个方向；
#   (b) 头根本没在用 w，输出的图与 w 无关 —— 这时指标可能还不算太差，
#       因为它可以靠「位置 + 其它参数」猜个大概。
# 对照做法：把 `w_holdout` 实例的 w 换成它 `base_instance`（同位置、同视角、
# 训练时见过的 w），**别的条件一律不动**，再渲染一遍。
#   - 若头在用 w：换 w 后图会明显变、且对 GT 变得更差（因为换成的是**错的** w）；
#   - 若头没用 w：两张图逐位相同，指标一字不差。
# 这条对照不能反过来说「用了 w 就一定对」，它只负责否掉 (b)。
#
# 用法（服务器）：
#   ROOT=/root/autodl-tmp/rf/results/<run>/训练/w STEP=60000 bash 本脚本
#
# 产物：$ROOT/test_inst/{inst*/{renders,gt}}、metrics.txt、
#       $ROOT/test_wswap/{inst*/renders}、metrics.txt
# 注意：长跑中途不要编辑本文件（bash 按字节偏移读脚本）。
set -uo pipefail
PY=${PY:-/home/leafd_dot/miniforge3/envs/mvsplat/bin/python3.10}
R=${R:-/mnt/d/GS3D/mvsplat}
cd "$R/code" || exit 9

ROOT=${ROOT:?}
STEP=${STEP:-60000}
CK=${CK:-$ROOT/angular_$STEP.pth}
DATA=${DATA:-$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100}
SPC=${SPC:-$R/datasets/RF-3DGS_mvdr_w_params/spectra}
META=${META:-$R/datasets/RF-3DGS_mvdr_w_params/meta.json}
# 主读数：orig（作者 TX48 位置 + 一个留出 w）+ w_holdout（36 个训练位置 × 留出 w）
ROLES=${ROLES:-orig w_holdout}
MAXINST=${MAXINST:-0}
# 退化 GT（整幅 jet(0)，源 u 全 0）会把 PSNR 顶到 99 的截断值，口径见
# `eval_instances.py`；交付口径一律跳过，历史读数需要复现时 SKIPFLAT=0。
SKIPFLAT=${SKIPFLAT:-1}
FLATARG=""
[ "$SKIPFLAT" = "1" ] && FLATARG="--skip_flat_gt"
OUT=$ROOT/test_inst
WSWAP=$ROOT/test_wswap
mkdir -p "$OUT" "$WSWAP"

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
  # 2) 渲染（条件按实例原样）
  if [ ! -f "$OUT/.render_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" \
        --max_instances "$MAXINST" --out_root "$OUT" \
        > "$OUT/render_$role.log" 2>&1 \
      && touch "$OUT/.render_$role.done" \
      || { echo "[FAIL] render $role（见 $OUT/render_$role.log）"; continue; }
  fi
  # 3) w 置换对照：只把 tx_w 换成 base_instance 的（训练域 w），其余不动。
  #    只对 w_holdout 有意义（orig 也有 base，但没有可换的训练域同源实例时会被跳过）。
  if [ "$role" = "w_holdout" ] && [ ! -f "$WSWAP/.render_$role.done" ]; then
    "$PY" render_rrf_angular.py -s "$DATA" --spectra_root "$SPC" --meta "$META" \
        -m "$ROOT" --checkpoint "$CK" --role "$role" --param_mode sibling_w \
        --max_instances "$MAXINST" --out_root "$WSWAP" \
        > "$WSWAP/render_$role.log" 2>&1 \
      && touch "$WSWAP/.render_$role.done" \
      || echo "[WARN] wswap $role 失败（见 $WSWAP/render_$role.log）"
  fi
  echo "[OK] $role"
done

echo "=== 逐实例指标（原样条件）==="
"$PY" tools/eval_instances.py --root "$OUT" --meta "$META" --skip_missing_gt $FLATARG

# 守卫必须用 `find -print -quit`：写成 `ls "$WSWAP"/inst*/renders/*.png` 会在
# 23,040 个文件时超 ARG_MAX，`ls` 返回 126 而不是「没有文件」，整段被静默跳过，
# 日志里看不到任何报错。
if [ -n "$(find "$WSWAP" -path '*/renders/*.png' -print -quit 2>/dev/null)" ]; then
  # wswap 只重渲了 renders，没有 gt ⇒ 把 GT 软链/复制过来一次，让评测脚本能配对
  for d in "$WSWAP"/inst*/; do
    iid=$(basename "$d")
    if [ -d "$OUT/$iid/gt" ] && [ ! -e "$d/gt" ]; then ln -s "$OUT/$iid/gt" "$d/gt"; fi
  done
  echo
  echo "=== 逐实例指标（w 置换对照）==="
  "$PY" tools/eval_instances.py --root "$WSWAP" --meta "$META" --skip_missing_gt $FLATARG
  echo
  echo "=== w 置换前后的配对差 ==="
  "$PY" tools/_w_swap_report.py --as_is "$OUT/metrics.txt" \
      --wswap "$WSWAP/metrics.txt" --meta "$META"
fi
