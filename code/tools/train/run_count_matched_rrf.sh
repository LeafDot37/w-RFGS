#!/bin/bash
# =============================================================================
# run_count_matched_rrf.sh —— 计数配平：把 8000 档剪到 1000 档的点数，同配置重跑
# =============================================================================
# 为什么必须做这一步（**斜率轴上唯一还没排除的致命混杂**）
# -----------------------------------------------------------------------------
# 斜率轴（run_20260915_视觉质量因果斜率）跑完后，MVDR 的 PSNR 排序与点数排序
# **完全一致**（秩相关 = 1.0）：
#
#     档位     1000    2000    4000    30000   8000
#     点数P   1191k < 1241k < 1249k < 1262k < 1431k
#     PSNR    19.57 < 20.41 < 20.59 < 20.60 < 20.68
#
# 而 A× 标量头有一条**逐点自由 embedding** `embed_p` = P×32
# ⇒ **点数差就是容量差**。光看那张表，「视觉质量在起作用」与「点更多所以容量更大」
# 分不开，指标变好也读不出是哪一半。
#
# 本驱动做**计数配平**：把 8000 档（指标最好的那档）均匀抽样到 1000 档的点数，
# 得到一根「同点数、不同视觉步数」的臂。
#
# 预登记的判读（写在看到数据之前）
# -----------------------------------------------------------------------------
#   剪到 1191636 后 MVDR PSNR 掉回 ≈19.57（= 1000 档水平）
#     ⇒ 之前的"显著斜率"**全部是点数效应**，视觉质量对 RF 无因果贡献。
#   不掉（仍在 ≈20.6 ~ 20.7 区间）
#     ⇒ 在**同点数**下视觉步数仍在起作用，这才是"引导成立"的证据。
#
# 已知的残余混杂（**必须如实写进说明.md，不许省**）
# -----------------------------------------------------------------------------
#   均匀抽样在改点数的同时也**改了密度** ⇒ 本臂不是 1000 档的完美对照，
#   只是唯一便宜可行的拆法。这与 run_20260914_协方差初始化 的计数配平臂
#   是同一套口径，那里的结论也是这么读的（该线的结论：伤害来自形状不是容量）。
#
#   追加对照（可选，见环境变量 TARGETS）：把同一份 8000 档剪到**几乎不动**的点数
#   （如 1400000，只抽掉 2%）。若它指标不动 ⇒ 抽样本身无害 ⇒ 主臂的落差可归因到点数。
#
# 与斜率轴的口径关系（**必须逐条相同，否则又多一个变量**）
# -----------------------------------------------------------------------------
#   RRF 超参、LR 协议、本地步数、绝对终点（first_iter 8000 + 本地 10000 = 18000）、
#   频谱集、`--densify_until_iter 0` —— 全部从 run_vis_slope_rrf.sh 逐字照抄。
#   唯一变量 = 起点 ckpt 的点数与内容。
#
# 用法（服务器 code/ 下）
# -----------------------------------------------------------------------------
#   全量：CUDA_VISIBLE_DEVICES=3 nohup bash tools/run_count_matched_rrf.sh \
#           > /tmp/run_count_matched.log 2>&1 &
#   只建 ckpt（零 GPU）：SEGMENTS="PREP" bash tools/run_count_matched_rrf.sh
#   追加"几乎不动"对照：TARGETS="1191636 1400000" ...
# =============================================================================
set -u

# 频谱集有近千张图，readColmapCameras 逐张 open 读尺寸 ⇒ fd 上限不够就
# OSError: [Errno 24] Too many open files（斜率轴第一版就栽在这，14 个节点全灭）。
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null)" 2>/dev/null || true

SEGMENTS="${SEGMENTS:-PREP RRF RRFEVAL}"
SRC_STEP="${SRC_STEP:-8000}"          # 被剪的那一档（视觉短训步数）
REF_STEP="${REF_STEP:-1000}"          # 目标点数取自哪一档
RRF_ITERS="${RRF_ITERS:-10000}"
RRF_METHODS="${RRF_METHODS:-MVDR CBF}"
SEED="${SEED:-0}"

ROOT=/data/yuexf/3dgs
cd "$ROOT" || exit 9
source ./load_config.sh
load_config full

SLOPE="${SLOPE:-$P_OUT/run_20260915_视觉质量因果斜率}"
TOP="${TOP:-$SLOPE/计数配平}"
LOG="$TOP/日志"
SRC_CKPT="$SLOPE/视觉短训/rf-3dgs_vis/chkpnt${SRC_STEP}.pth"
SRC_SRC="$SLOPE/来源.txt"
DS="${DS:-${P_SPECTRUM:-}}"
PY="$PY"
ABS=$((SRC_STEP + RRF_ITERS))         # 绝对终点，与斜率轴同口径
mkdir -p "$LOG" "$TOP/点云"

echo "==== [$(date +%F_%T)] 计数配平（斜率轴 档 $SRC_STEP 剪到 档 $REF_STEP 的点数）===="
echo "     SEGMENTS='$SEGMENTS'  CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-（未设）}'"
echo "     源 ckpt  = $SRC_CKPT"
echo "     目标点数 = 取自 $SRC_SRC 的档 $REF_STEP"
echo "     RRF      = 本地 $RRF_ITERS 步，绝对终点 $ABS，方法 [$RRF_METHODS]"
echo "     输出     = $TOP"

FAIL=0
want() { case " $SEGMENTS " in *" $1 "*) return 0;; esac; return 1; }

# ---------------------------------------------------------------------------
# 一、目标点数：**从斜率轴的 来源.txt 里读**，不在这里再硬编码一个数
#     来源.txt 的表头是「档位 first_iter 点数P s_lr ckpt_MB md5」，$1/$2/$3。
# ---------------------------------------------------------------------------
[ -f "$SRC_SRC" ] || { echo "❌ 斜率轴来源.txt 不存在：$SRC_SRC"; exit 1; }
TARGETS="${TARGETS:-$(awk -v s="$REF_STEP" '$1==s {print $3}' "$SRC_SRC")}"
[ -n "$TARGETS" ] || { echo "❌ 在 $SRC_SRC 里找不到档 $REF_STEP 的点数"; exit 1; }
REF_P=$(echo "$TARGETS" | awk '{print $1}')
echo "     档 $REF_STEP 的点数 = $REF_P（读自来源.txt）  本次目标 [$TARGETS]"

# ---------------------------------------------------------------------------
# 二、段 PREP：剪 ckpt（CPU，零 GPU）
#     产物名**刻意非标准**（不带 chkpnt<6位数字>.pth 的模式），
#     免得被任何"按文件名找末档"的脚本误认成训练产物。
# ---------------------------------------------------------------------------
declare -A CKPT
for T in $TARGETS; do
  CKPT[$T]="$TOP/点云/chkpnt${SRC_STEP}_剪到${T}.pth"
done

if want PREP; then
  [ -f "$SRC_CKPT" ] || { echo "❌ 源 ckpt 缺失：$SRC_CKPT"; exit 1; }
  echo; echo "[$(date +%F%T)] ==== 段 PREP：剪 ckpt（零 GPU）===="
  cd "$P_CODE" || exit 9
  for T in $TARGETS; do
    OUT_C="${CKPT[$T]}"
    echo "----- $SRC_STEP 档 → $T 点 -----"
    "$PY" "$P_TOOLS/make_count_matched_ckpt.py" \
        --src "$SRC_CKPT" --target "$T" --seed "$SEED" --out "$OUT_C" \
        > "$LOG/prep_剪到${T}.txt" 2>&1
    rc=$?
    tail -25 "$LOG/prep_剪到${T}.txt"
    [ $rc -eq 0 ] || { echo "  PREP_FAIL 剪到 $T"; FAIL=1; }
    [ -s "$OUT_C" ] || { echo "  【哨兵】剪到 $T 的 ckpt 不存在或为空"; FAIL=1; }
  done
  [ "$FAIL" -eq 0 ] || { echo "❌ PREP 有失败项 ⇒ 不进 GPU 段"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 三、段 RRF：同配置重训（GPU）
# ---------------------------------------------------------------------------
if want RRF; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "RRF 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  echo; echo "[$(date +%F%T)] ==== 段 RRF：[$(echo $TARGETS)] × [$RRF_METHODS]（卡 $CUDA_VISIBLE_DEVICES）===="
  cd "$P_CODE" || exit 9
  for M in $RRF_METHODS; do
    case "$M" in
      MVDR|CBF) ;;
      *) echo "  [拒绝] RRF_METHODS 里有 '$M'：本脚本只跑标量头（MVDR / CBF）。"; exit 3;;
    esac
  done
  for T in $TARGETS; do
    BASE="${CKPT[$T]}"
    [ -f "$BASE" ] || { echo "  [剪到 $T] 起点 ckpt 缺失: $BASE"; FAIL=1; continue; }
    for M in $RRF_METHODS; do
      DATA="$DS/3dgs_${M}_100"; MD="$TOP/剪到${T}/$M"
      # 同配置重跑是逐位确定的；已有末档 ply 就跳过（幂等续跑）
      if [ -f "$MD/point_cloud/iteration_${ABS}/point_cloud.ply" ]; then
        echo "  [剪到 $T/$M] 末档已在，跳过训练"
      else
        echo "[$(date +%F%T)] RRF 剪到 $T / $M（起点 $(basename "$BASE")，绝对 $ABS = 本地 $RRF_ITERS）"
        unset RF_POINTCLOUD
        "$PY" train_rrf_scalar.py -s "$DATA" -m "$MD" \
            --start_checkpoint "$BASE" --iterations "$ABS" \
            --save_iterations "$ABS" --densify_until_iter 0 --data_device cpu --quiet \
            --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 \
            --lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 40000 \
            --pe_xyz_L 5 --pe_dir_L 3 --use_dist --n_bands 2 --band_mix mult \
            > "$LOG/rrf_剪到${T}_${M}.txt" 2>&1 \
          || { echo "  RRF_FAIL 剪到 $T/$M"; tail -30 "$LOG/rrf_剪到${T}_${M}.txt"; FAIL=1; continue; }
        grep -aE 'Number of points|头参数量|Training complete' "$LOG/rrf_剪到${T}_${M}.txt" \
          | sed 's/^/    /' | head -4
      fi
      [ -f "$MD/point_cloud/iteration_${ABS}/point_cloud.ply" ] \
        || { echo "  [剪到 $T/$M] 缺末档 ply"; FAIL=1; }
    done
  done
fi

# ---------------------------------------------------------------------------
# 四、段 RRFEVAL：640 全量渲染 + 三指标（LPIPS 为主判据）
# ---------------------------------------------------------------------------
if want RRFEVAL; then
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "RRFEVAL 段缺 CUDA_VISIBLE_DEVICES"; exit 2; }
  echo; echo "[$(date +%F%T)] ==== 段 RRFEVAL ===="
  cd "$P_CODE" || exit 9
  pngn() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
  for T in $TARGETS; do
    for M in $RRF_METHODS; do
      DATA="$DS/3dgs_${M}_100"; MD="$TOP/剪到${T}/$M"
      [ -d "$MD/point_cloud/iteration_${ABS}" ] || { echo "  [skip] 剪到 $T/$M 无 abs$ABS"; continue; }
      unset RF_POINTCLOUD
      if [ "$(pngn "$MD/test/ours_${ABS}/renders")" -lt 640 ]; then
        echo "[$(date +%F%T)] 渲染 剪到 $T/$M 640 张"
        "$PY" render_rrf_scalar.py -s "$DATA" -m "$MD" --iteration "$ABS" \
              --eval --skip_train --subset 0 --quiet \
            > "$LOG/rrfrender_剪到${T}_${M}.txt" 2>&1 \
          || { echo "  RENDER_FAIL 剪到 $T/$M"; tail -20 "$LOG/rrfrender_剪到${T}_${M}.txt"; FAIL=1; continue; }
      fi
      [ "$(pngn "$MD/test/ours_${ABS}/renders")" -ge 640 ] \
        || { echo "  [剪到 $T/$M] 渲染不足 640 张（$(pngn "$MD/test/ours_${ABS}/renders")）"; FAIL=1; }
      "$PY" "$P_TOOLS/metrics_fast.py" -m "$MD" --iterations "$ABS" --subset 0 \
            --out "$MD/results.json" > "$LOG/rrfmetrics_剪到${T}_${M}.txt" 2>&1 \
        || { echo "  METRICS_FAIL 剪到 $T/$M"; tail -20 "$LOG/rrfmetrics_剪到${T}_${M}.txt"; FAIL=1; continue; }
      "$PY" - "$MD" "$ABS" "$T" "$M" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1] + "/results.json", encoding="utf-8"))
for k, v in d.items():
    if k.startswith("ours_" + sys.argv[2]) and "PSNR" in v:
        print("    剪到 %-9s %-6s PSNR %.4f  SSIM %.5f  LPIPS %.5f"
              % (sys.argv[3], sys.argv[4], v["PSNR"], v["SSIM"], v["LPIPS"]))
PYEOF
    done
  done
fi

# ---------------------------------------------------------------------------
# 五、来源.txt（含**对照读数**：把斜率轴上被比的两档一并写进去，免得事后另找）
# ---------------------------------------------------------------------------
if [ "$FAIL" -eq 0 ]; then
  {
    echo "计数配平（斜率轴 档 $SRC_STEP → 档 $REF_STEP 的点数）"
    echo "生成 $(date +%F_%T)"
    echo
    echo "源 ckpt   = $SRC_CKPT"
    echo "目标点数  = $TARGETS（档 $REF_STEP 的点数 = $REF_P，读自 $SRC_SRC）"
    echo "抽样      = make_count_matched_ckpt.py --seed $SEED（均匀随机，取了再排序）"
    echo "RRF       = 本地 $RRF_ITERS 步，绝对终点 $ABS"
    echo "超参      = --embed_dim 32 --mlp_hidden 128 --mlp_depth 2 --pe_xyz_L 5"
    echo "            --pe_dir_L 3 --use_dist --n_bands 2 --band_mix mult"
    echo "            --lr_schedule exp --mlp_lr_init 8e-3 --mlp_lr_final 8e-5 --lr_max_steps 40000"
    echo "            --densify_until_iter 0（几何冻死）"
    echo "            ⇒ 与 run_vis_slope_rrf.sh 的 RRF 段**逐条相同**，唯一变量 = 起点 ckpt"
    echo
    echo "残余混杂（必须如实写进说明.md）：均匀抽样同时改了密度 ⇒ 本臂不是档 $REF_STEP"
    echo "的完美对照，只是唯一便宜可行的拆法（与 run_20260914_协方差初始化 同口径）。"
    echo
    echo "判读（预登记）：剪到 $REF_P 后 MVDR PSNR 掉回档 $REF_STEP 的水平 ⇒ 之前的斜率"
    echo "全部是点数效应；不掉 ⇒ 同点数下视觉步数仍在起作用。"
    echo
    echo "对照读数（来自斜率轴，本次未重跑）见 $SLOPE/RRF/vis*/<方法>/results.json"
  } > "$TOP/来源.txt"
  echo
  echo "  来源.txt 已写：$TOP/来源.txt"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "==== [$(date +%F_%T)] 全部段通过 ✓  产物在 $TOP"
else
  echo "==== [$(date +%F_%T)] 有失败项（FAIL=1），看上面逐行 [xxx_FAIL] 与 $LOG"
fi
exit "$FAIL"
