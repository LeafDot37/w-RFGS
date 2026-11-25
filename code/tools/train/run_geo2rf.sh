#!/bin/bash
# =============================================================================
# 前馈「光学点云 → 射频点云」驱动（本机 WSL，RTX 4060 8 GB，单卡）
#
# 输入 = 一份**已优化**的光学 3DGS 点云（MVSplat 初始化 + 光学短训，1 435 598 点）
# 输出 = 同一份点上的射频不透明度 + 射频球谐（逐点，一次前向，零逐场景优化）
#
# 段（各段幂等，可重入）：
#   TRAIN   逐臂训头
#   RENDER  渲射频测试集（640 视角 / 160 个留出位置）
#   METRIC  metrics_fast 出 PSNR+SSIM+LPIPS
#   POST    写说明.md
#
# 臂（ARMS 环境变量，空格分隔）：
#   geo_app   位置 + 56 维光学属性（尺度/旋转/不透明度/48 个 SH 系数）
#   geo_xyz   只用位置（阴性对照：光学外观到底贡献多少）
#   geo_xyz_dist  仅位置 + 老师逐点属性蒸馏
#   geo_app_dist  位置 + 光学属性 + 老师逐点属性蒸馏
#   geo_xyz_h128  128 宽 / 4 层 / PE10（容量臂，三变量同时变）
#   geo_xyz_w128  只加宽到 128（3 层 / PE6）—— 容量臂单变量拆解
#   geo_xyz_pe10  只升到 PE10（64 宽 / 3 层）—— 容量臂单变量拆解
#   geo_xyz_pe4   只降到 PE4（64 宽 / 3 层）—— 反方向验证频率是杠杆
#   geo_xyz_jet   标量头 A'（单带）：光栅化在标量域混合，混合后查一次 jet 表
#   geo_xyz_jet2  标量头 A×（双带乘性调制）
#   geo_xyz_jet_i3    标量头 A'，初值 0.3（GT 标量均值约 0.31）
#   geo_xyz_jet_pe16  标量头 A' + PE16（检验平滑映射的空间分辨率是否不够）
#   geo_xyz_jet_dir   标量头 A' + 视线条件（dir + γ₃(dir) + 距离）——诊断直指 64% 方向效应
#   geo_xyz_jet_dir_nd 同上但**不带距离**（拆解距离那一项是否必要）
#
# 实测（640 测试视角）：geo_xyz 13.797/0.6777/0.4169 反而**好于**
# geo_app 13.3590/0.6614/0.4406 ⇒ 光学外观输入在本规模下是噪声，蒸馏臂以 xyz 为主。
#
# ⚠ 启动后**绝不能再编辑本文件**（bash 按字节偏移读脚本）。
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null)" 2>/dev/null || true

ROOT="${ROOT:-/mnt/d/GS3D/mvsplat}"
cd "$ROOT" || exit 9
PY=/home/leafd_dot/miniforge3/envs/mvsplat/bin/python3.10

SEGMENTS="${SEGMENTS:-TRAIN RENDER METRIC POST}"
ARMS="${ARMS:-geo_xyz_dist geo_app_dist}"
# 老师逐点属性蒸馏：make_teacher_match.py 的产物，把阶段 1 老师的外观按最近邻搬到本点云
TMATCH="${TMATCH:-$ROOT/results/student_feats/teacher_attr_geo.pt}"
LAMBDA_ATTR="${LAMBDA_ATTR:-0.1}"
FINAL="${FINAL:-20000}"
LADDER="${LADDER:-10000 15000}"
LR="${LR:-5e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SH_BOUND="${SH_BOUND:-2.0}"
HIDDEN="${HIDDEN:-64}"
LAYERS="${LAYERS:-3}"
PE_BANDS="${PE_BANDS:-6}"
LOG_EVERY="${LOG_EVERY:-200}"
SUBSET="${SUBSET:-0}"
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0.2}"

PLY="${PLY:-$ROOT/results/run_20260907_005205_full档_端到端/rf-3dgs_1/point_cloud/iteration_12000/point_cloud.ply}"
DATA="${DATA:-$ROOT/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100}"
TOP="${TOP:-$ROOT/results/run_20261017_本机4060_点云到射频点云前馈_有界输出}"
LOG="$TOP/日志"
mkdir -p "$LOG" "$TOP"

echo "=============================================================="
echo "点云 -> 射频点云 前馈模型   $(date '+%F %T')"
echo "  臂 = $ARMS   最终迭代 $FINAL   段 = $SEGMENTS"
echo "  头 hidden=$HIDDEN layers=$LAYERS pe=$PE_BANDS  lr=$LR  clip=$GRAD_CLIP  sh_bound=$SH_BOUND  dssim=$LAMBDA_DSSIM"
echo "  点云 = $PLY"
echo "  数据 = $DATA"
echo "  蒸馏 = ${TMATCH}  λ=$LAMBDA_ATTR"
echo "=============================================================="

[ -f "$PLY" ] || { echo "[abort] 点云不存在"; exit 3; }
[ -d "$DATA" ] || { echo "[abort] 频谱数据不存在"; exit 3; }
case " $ARMS " in
  *_dist*) [ -f "$TMATCH" ] || { echo "[abort] 蒸馏臂缺老师匹配文件：$TMATCH"; exit 3; } ;;
esac

arm_args() {
  case "$1" in
    geo_app)      echo "--use_app 1" ;;
    geo_xyz)      echo "--use_app 0" ;;
    geo_app_dist) echo "--use_app 1 --teacher_attr $TMATCH --lambda_attr $LAMBDA_ATTR" ;;
    geo_xyz_dist) echo "--use_app 0 --teacher_attr $TMATCH --lambda_attr $LAMBDA_ATTR" ;;
    geo_xyz_h128) echo "--use_app 0 --hidden 128 --n_layers 4 --pe_bands 10" ;;
    geo_xyz_h256) echo "--use_app 0 --hidden 256 --n_layers 4 --pe_bands 10" ;;
    # 容量臂的**单变量拆解**（相对 geo_xyz 的 64×3/PE6 每次只动一项）
    geo_xyz_w128) echo "--use_app 0 --hidden 128 --n_layers 3 --pe_bands 6" ;;
    geo_xyz_pe10) echo "--use_app 0 --hidden 64 --n_layers 3 --pe_bands 10" ;;
    # 反向：降频（若 PE10 的伤害来自频率过高，这一臂应持平或小涨）
    geo_xyz_pe4)  echo "--use_app 0 --hidden 64 --n_layers 3 --pe_bands 4" ;;
    # 标量头：逐点标量 → 标量域混合 → 查 jet 表（形态 A' / A×）
    geo_xyz_jet)  echo "--use_app 0 --out_kind scalar --n_bands 1" ;;
    geo_xyz_jet2) echo "--use_app 0 --out_kind scalar --n_bands 2" ;;
    # 标量头的两个失败假设拆开测
    geo_xyz_jet_i3)   echo "--use_app 0 --out_kind scalar --n_bands 1 --init_u 0.3" ;;
    geo_xyz_jet_pe16) echo "--use_app 0 --out_kind scalar --n_bands 1 --pe_bands 16" ;;
    # 视线条件：逐场景标量头的输入是 cat[γ(xyz), dir, γ₃(dir), r]，这里对齐它
    geo_xyz_jet_dir)   echo "--use_app 0 --out_kind scalar --n_bands 1 --use_dir 1 --pe_dir_bands 3 --use_dist 1" ;;
    geo_xyz_jet_dir_nd) echo "--use_app 0 --out_kind scalar --n_bands 1 --use_dir 1 --pe_dir_bands 3" ;;
    *)            echo "" ; return 1 ;;
  esac
}

arm_label() {
  case "$1" in
    geo_app)      echo "点云+光学外观" ;;
    geo_xyz)      echo "仅位置" ;;
    geo_app_dist) echo "点云+光学外观+老师蒸馏" ;;
    geo_xyz_dist) echo "仅位置+老师蒸馏" ;;
    geo_xyz_h128) echo "仅位置+宽头128×4+PE10" ;;
    geo_xyz_h256) echo "仅位置+宽头256×4+PE10" ;;
    geo_xyz_w128) echo "仅位置+只加宽128(3层/PE6)" ;;
    geo_xyz_pe10) echo "仅位置+只升PE10(64×3)" ;;
    geo_xyz_pe4)  echo "仅位置+只降PE4(64×3)" ;;
    geo_xyz_jet)  echo "仅位置+标量头A'(单带)" ;;
    geo_xyz_jet2) echo "仅位置+标量头A×(双带乘性)" ;;
    geo_xyz_jet_i3)   echo "标量头A'+初值0.3" ;;
    geo_xyz_jet_pe16) echo "标量头A'+PE16(空间分辨率)" ;;
    geo_xyz_jet_dir)   echo "标量头A'+视线条件(方向+距离)" ;;
    geo_xyz_jet_dir_nd) echo "标量头A'+视线条件(仅方向)" ;;
    *)            echo "$1" ;;
  esac
}

FAIL=0

# ------------------------------------------------------------------ TRAIN
if [[ " $SEGMENTS " == *" TRAIN "* ]]; then
  for arm in $ARMS; do
    aargs="$(arm_args "$arm")" || { echo "[skip] 未知臂 $arm"; FAIL=$((FAIL+1)); continue; }
    MD="$TOP/$arm"; mkdir -p "$MD"
    if [ -f "$MD/head_${FINAL}.pth" ]; then
      echo "[$(date '+%F_%T')] TRAIN 跳过（$arm 已有 head_${FINAL}.pth）"
      continue
    fi
    echo "[$(date '+%F_%T')] TRAIN $arm  $aargs"
    "$PY" code/rfgs/geo2rf/train_geo2rf.py -m "$MD" \
        --cloud_ply "$PLY" \
        --rf_source_path "$DATA" --rf_data_device cpu \
        --iterations "$FINAL" --save_iterations $LADDER "$FINAL" \
        --hidden "$HIDDEN" --n_layers "$LAYERS" --pe_bands "$PE_BANDS" \
        --lr "$LR" --lambda_dssim "$LAMBDA_DSSIM" \
        --grad_clip "$GRAD_CLIP" --sh_bound "$SH_BOUND" \
        --log_every "$LOG_EVERY" \
        $aargs \
        > "$LOG/train_${arm}.txt" 2>&1
    rc=$?
    tr '\r' '\n' < "$LOG/train_${arm}.txt" | grep -a "\[g2r\]" | grep -a "s/it" | tail -2
    if [ $rc -ne 0 ]; then
      echo "[fail] TRAIN $arm rc=$rc"
      tail -30 "$LOG/train_${arm}.txt"
      FAIL=$((FAIL+1)); continue
    fi
  done
fi

# ------------------------------------------------------------------ RENDER
if [[ " $SEGMENTS " == *" RENDER "* ]]; then
  for arm in $ARMS; do
    MD="$TOP/$arm"
    [ -f "$MD/head_${FINAL}.pth" ] || { echo "[render] 跳过 $arm（无 head）"; continue; }
    OR="$MD/eval"
    if [ -f "$OR/rf/test/ours_${FINAL}/renders/00000.png" ]; then
      echo "[$(date '+%F_%T')] RENDER 跳过（$arm 已在）"
      continue
    fi
    echo "[$(date '+%F_%T')] RENDER $arm"
    "$PY" code/rfgs/geo2rf/render_geo2rf.py -m "$MD" --head "head_${FINAL}.pth" \
        --cloud_ply "$PLY" --rf_source_path "$DATA" --rf_data_device cpu \
        --out_root "$OR" --iteration "$FINAL" --subset "$SUBSET" \
        > "$LOG/render_${arm}.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[fail] RENDER $arm rc=$rc"; tail -20 "$LOG/render_${arm}.txt"
      FAIL=$((FAIL+1))
    fi
  done
fi

# ------------------------------------------------------------------ METRIC
if [[ " $SEGMENTS " == *" METRIC "* ]]; then
  for arm in $ARMS; do
    OR="$TOP/$arm/eval"
    [ -d "$OR/rf/test/ours_${FINAL}/renders" ] || continue
    OUTJ="$OR/rf_results.json"
    [ -f "$OUTJ" ] && { echo "[metric] 跳过 $arm（已有）"; continue; }
    echo "[$(date '+%F_%T')] METRIC $arm"
    "$PY" code/tools/diag/metrics_fast.py -m "$OR/rf" \
        --iterations "$FINAL" --subset "$SUBSET" --out "$OUTJ" \
        > "$LOG/metric_${arm}.txt" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "[fail] METRIC $arm rc=$rc"; FAIL=$((FAIL+1)); }
    tr '\r' '\n' < "$LOG/metric_${arm}.txt" | grep -a "ours_" | tail -2
  done
fi

# ------------------------------------------------------------------ POST
if [[ " $SEGMENTS " == *" POST "* ]]; then
  "$PY" - "$TOP" "$FINAL" "$ARMS" <<'PY'
import json, os, sys
top, final, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
LABEL = {"geo_app": "点云+光学外观", "geo_xyz": "仅位置",
         "geo_app_dist": "点云+光学外观+老师蒸馏", "geo_xyz_dist": "仅位置+老师蒸馏",
         "geo_xyz_h128": "仅位置+宽头128×4+PE10",
         "geo_xyz_h256": "仅位置+宽头256×4+PE10",
         "geo_xyz_w128": "仅位置+只加宽128(3层/PE6)",
         "geo_xyz_pe10": "仅位置+只升PE10(64×3)",
         "geo_xyz_pe4": "仅位置+只降PE4(64×3)",
         "geo_xyz_jet": "仅位置+标量头A'(单带)",
         "geo_xyz_jet2": "仅位置+标量头A×(双带乘性)",
         "geo_xyz_jet_i3": "标量头A'+初值0.3",
         "geo_xyz_jet_pe16": "标量头A'+PE16(空间分辨率)",
         "geo_xyz_jet_dir": "标量头A'+视线条件(方向+距离)",
         "geo_xyz_jet_dir_nd": "标量头A'+视线条件(仅方向)"}
def rd(p):
    try:
        d = json.load(open(p)); k = sorted(d.keys())[-1]; return d[k]
    except Exception:
        return None
lines = ["# 光学点云 -> 射频点云 前馈模型 · 读数汇总", "",
         "口径：PSNR / SSIM / LPIPS 同时报，**LPIPS 为主判据**。",
         "射频 = 640 个测试视角（160 个**留出位置** × 4 朝向）。",
         "输入只有光学点云（逐点尺度/旋转/不透明度/48 个 SH 系数），"
         "**不含任何射频侧信息**；几何与光学外观全部冻结，不优化。", "",
         "| 方法 | 输入 | PSNR | SSIM | LPIPS | n |", "|---|---|---|---|---|---|"]
for arm in arms:
    r = rd(os.path.join(top, arm, "eval", "rf_results.json"))
    if r is None:
        continue
    lines.append(f"| {LABEL.get(arm, arm)} | 光学点云 | {r['PSNR']:.4f} | "
                 f"{r['SSIM']:.4f} | {r['LPIPS']:.4f} | {r['n_views']} |")
lines += ["", "参考（同一批 640 个测试视角）：",
          "- 阶段 1「冻结几何 / MVDR」逐场景优化老师：PSNR 16.1964 / SSIM 0.7334 / LPIPS 0.3581",
          "- RF-3DGS 原始管线（逐场景）：约 13.3–13.8 dB"]
open(os.path.join(top, "说明.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("写出", os.path.join(top, "说明.md"))
PY
fi

echo "=============================================================="
if [ $FAIL -eq 0 ]; then
  echo "ALL_GEO2RF_DONE (FAIL=0)  $(date '+%F %T')"
else
  echo "GEO2RF_DONE_WITH_FAILURES FAIL=$FAIL  $(date '+%F %T')"
fi
echo "=============================================================="
