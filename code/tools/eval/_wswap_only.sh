#!/bin/bash
# 只跑「w 置换对照」那一段：wswap 渲染已完成，缺的是 metrics + 配对报告。
#
# 为什么单独一个脚本
# ---------------------------------------------------------------------------
# `_eval_w_arms.sh` 里那半段的守卫写的是
#     ls "$WSWAP"/inst*/renders/*.png >/dev/null 2>&1
# 一次展开 23,040 个路径参数会超 ARG_MAX，`ls` 返回 126（不是「没有文件」），
# 守卫判假，整段被静默跳过 —— 日志里看不出任何报错。这里改用 `find -print -quit`，
# 只看「有没有」，不把文件名全铺开。
#
# 用法（服务器）：
#   ROOT=<arm 目录> bash code/tools/eval/_wswap_only.sh
#
# 前提：`$ROOT/test_inst/{inst*/renders,inst*/gt}` 与 `$ROOT/test_wswap/inst*/renders` 都在。
set -uo pipefail
PY=${PY:-/root/miniconda3/bin/python}
R=${R:-/root/autodl-tmp/rf}
cd "$R/code" || exit 9

ROOT=${ROOT:?}
META=${META:-$R/datasets/RF-3DGS_mvdr_w_params/meta.json}
OUT=$ROOT/test_inst
WSWAP=$ROOT/test_wswap

[ -f "$OUT/metrics.txt" ] || { echo "[FAIL] 缺 $OUT/metrics.txt（原样条件还没评）"; exit 9; }
first=$(find "$WSWAP" -path '*/renders/*.png' -print -quit 2>/dev/null)
[ -n "$first" ] || { echo "[FAIL] $WSWAP 下没有渲染图"; exit 9; }
n=$(find "$WSWAP" -path '*/renders/*.png' | wc -l)
echo "wswap 渲染图 $n 张"

# GT 软链（wswap 只重渲了 renders，GT 与 as_is 共用同一份）
for d in "$WSWAP"/inst*/; do
  iid=$(basename "$d")
  if [ -d "$OUT/$iid/gt" ] && [ ! -e "$d/gt" ]; then ln -s "$OUT/$iid/gt" "$d/gt"; fi
done

echo "=== 逐实例指标（w 置换对照）==="
"$PY" tools/eval_instances.py --root "$WSWAP" --meta "$META" --skip_missing_gt || exit 9

echo
echo "=== w 置换前后的配对差 ==="
"$PY" tools/_w_swap_report.py --as_is "$OUT/metrics.txt" \
    --wswap "$WSWAP/metrics.txt" --meta "$META"
