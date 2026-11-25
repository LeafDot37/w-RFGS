#!/bin/bash
# =============================================================================
# run_corf.sh —— 阶段 1：CORF 耦合优化（**老师**），三臂 × 两方法
# =============================================================================
# 目标：在本数据集（RF-3DGS_dataset）上复现 CORF-GS 的耦合优化，作为阶段 2
#       前馈学生的「标准答案」。对应 CORF-GS 论文 Table IV 的消融阶梯。
#
# 三条臂（**同一个起点**，唯一差异 = 哪些参数可训 / 哪些损失开着）：
#   冻结几何    几何冻结 + 光学外观冻结，只训射频外观，L = L_rf      （≈ 本仓既有管线）
#   射频调几何  几何可训   + 光学外观冻结，只训射频外观，L = L_rf
#   完整耦合    几何可训   + 两面外观都可训，L = L_opt + 0.1·L_rf    （论文 Ours）
#
# 起点为什么用 seed 而不是 MVSplat 点云
# -----------------------------------------------------------------------------
# 三条臂必须**同起点**才可归因。MVSplat 点云（未收敛几何）起头时，「冻结几何」臂
# 训的是射频外观而「完整耦合」臂同时在动几何，两臂连「起点」都不同，比出来的是
# 「起点差 + 协议差」的混合。
# 用已收敛的光学模型（orig_seed_chkpnt30000.pth，本仓基线历史产物，**只读**）起头，
# 三条臂共享同一套几何与同一套光学外观 ⇒ 干净单变量。
# 代价：完整耦合臂的几何已趋饱和、耦合收益会被压缩。这一条是**已知的口径差异**，
#      写进 docs/CORF复现与前馈蒸馏_实施记录.md，不当成「论文没复现出来」。
#
# 段：TRAIN RENDER METRIC SRC POST（各段幂等，可重入）
#   TRAIN   训练到 $FINAL（绝对迭代），存档 $LADDER $FINAL
#   RENDER  射频：$LADDER 抽 48 帧、$FINAL 全量 640；光学：$FINAL 全量
#   METRIC  metrics_fast.py 出 rf_results.json / rf_curve.json / opt_results.json
#   SRC     源码快照 + md5
#   POST    收尾哨兵 + 说明.md
#
# 用法（服务器，占一卡）：
#   CUDA_VISIBLE_DEVICES=1 nohup bash tools/run_corf.sh \
#       > /data/yuexf/3dgs/results/run_20261008_CORF老师三臂/日志/驱动.log 2>&1 &
#   冒烟（只跑冻结几何 + MVDR 的 40000 档）：
#     MODES="frozen" CMETHODS="MVDR" FINAL=40000 LADDER="34000" \
#       CUDA_VISIBLE_DEVICES=1 bash tools/run_corf.sh
#
# ⚠ 启动后**绝不能再编辑本文件**（bash 按字节偏移读脚本）。
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn 2>/dev/null)" 2>/dev/null || true
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES（本驱动占一卡）"; exit 2; }
case "$CUDA_VISIBLE_DEVICES" in
  *,*) echo "CUDA_VISIBLE_DEVICES 必须**恰好一张卡**，收到 '$CUDA_VISIBLE_DEVICES'"; exit 2 ;;
esac

# 仓库根：服务器是 /data/yuexf/3dgs，本机 WSL 是 /mnt/d/GS3D/mvsplat。
# 两边的 config.yaml 各自写着自己那套路径，load_config.sh 会读对应的那份 ⇒
# **同一个驱动、同一套口径**，两端只差 CORF_ROOT 一个变量。
ROOT="${CORF_ROOT:-/data/yuexf/3dgs}"
cd "$ROOT" || exit 9
source ./load_config.sh
load_config full

SEGMENTS="${SEGMENTS:-TRAIN RENDER METRIC SRC POST}"
MODES="${MODES:-frozen rf_tuned joint}"
CMETHODS="${CMETHODS:-MVDR CBF}"
FINAL="${FINAL:-70000}"                 # 绝对迭代（seed 在 30000 ⇒ 本地 40000 步）
LADDER="${LADDER:-40000 55000}"         # 中间档（画收敛曲线用，抽帧评）
LAMBDA_RF="${LAMBDA_RF:-0.1}"           # CORF 论文取 0.1
SUBSET_CURVE="${SUBSET_CURVE:-48}"
SEED="${SEED:-$P_OUT/orig_seed_chkpnt30000.pth}"
# 服务器上 seed 是 results/orig_seed_chkpnt30000.pth；本机没有它，退回作者发布包
# blender_visual_trained/chkpnt30000.pth。**两者是否逐位相同未验证** —— SRC 段会把
# 实际用到的那个 seed 的 md5 写进 源码/MD5.md，读数要连同 md5 一起引用。
if [ ! -f "$SEED" ] && [ -f "${P_VISUAL%/blender_visual_dataset}/blender_visual_trained/chkpnt30000.pth" ]; then
  SEED="${P_VISUAL%/blender_visual_dataset}/blender_visual_trained/chkpnt30000.pth"
  echo "[warn] 未找到 results/orig_seed_chkpnt30000.pth，退回作者发布包: $SEED"
fi
INIT_PLY="${INIT_PLY:-$P_CLOUDS/init.ply}"
TOP="${TOP:-$P_OUT/run_20261008_CORF老师三臂}"
LOG="$TOP/日志"
SRC="$TOP/源码"

[ -d "$P_SPECTRUM" ] || { echo "频谱集缺失: $P_SPECTRUM"; exit 1; }
[ -f "$SEED" ]       || { echo "seed 缺失: $SEED"; exit 1; }
[ -f "$INIT_PLY" ]   || { echo "init 点云缺失: $INIT_PLY"; exit 1; }
cd "$P_CODE" || exit 9
mkdir -p "$LOG"

case " $SEGMENTS " in *" TRAIN "*)   need_train=1;; *) need_train=0;; esac
case " $SEGMENTS " in *" RENDER "*)  need_render=1;; *) need_render=0;; esac
case " $SEGMENTS " in *" METRIC "*)  need_metric=1;; *) need_metric=0;; esac
case " $SEGMENTS " in *" SRC "*)     need_src=1;; *) need_src=0;; esac
case " $SEGMENTS " in *" POST "*)    need_post=1;; *) need_post=0;; esac

echo "==== [$(date +%F_%T)] CORF 老师（耦合优化）  TOP=$TOP"
echo "     SEGMENTS='$SEGMENTS'  卡=$CUDA_VISIBLE_DEVICES"
echo "     MODES='$MODES'  CMETHODS='$CMETHODS'"
echo "     迭代 $FINAL（绝对）  中间档 '$LADDER'  lambda_rf=$LAMBDA_RF"
echo "     seed   = $SEED"
echo "     init云 = $INIT_PLY"
echo "     视觉集 = $P_VISUAL   resolution=$RES"
echo "     频谱集 = $P_SPECTRUM"

FAIL=0

# ------------------------------------------------------------------ TRAIN
if [ "$need_train" = 1 ]; then
  for MODE in $MODES; do
    case "$MODE" in
      frozen)   LABEL="冻结几何" ;;
      rf_tuned) LABEL="射频调几何" ;;
      joint)    LABEL="完整耦合" ;;
      *) echo "未知 MODE='$MODE'（可选 frozen / rf_tuned / joint）"; FAIL=1; continue ;;
    esac
    for M in $CMETHODS; do
      DATA="$P_SPECTRUM/3dgs_${M}_100"
      MD="$TOP/$LABEL/$M"
      mkdir -p "$MD"
      [ -d "$DATA" ] || { echo "  [$LABEL/$M] 数据缺失，跳过: $DATA"; FAIL=1; continue; }
      echo "[$(date +%F_%T)] ===== $LABEL / $M =====  -> $MD"
      if [ -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ]; then
        echo "  末档已存在，跳过训练"
        continue
      fi
      # RF_POINTCLOUD 必须指向已存在的文件：否则引擎会退回数据集内 points3d.ply，
      # 找不到就**往数据集里写一份随机点云**（改了数据集）。带 --start_checkpoint 时
      # 这份点云读完即被 seed 覆盖，但这一步不能省。
      export RF_POINTCLOUD="$INIT_PLY"
      "$PY" train_corf.py -s "$P_VISUAL" -m "$MD" \
          --start_checkpoint "$SEED" \
          --rf_source_path "$DATA" --rf_data_device cpu --rf_resolution -1 \
          --resolution "$RES" --data_device cpu \
          --mode "$MODE" --warmup_iters 0 \
          --iterations "$FINAL" --save_iterations $LADDER "$FINAL" \
          --lambda_rf "$LAMBDA_RF" --log_every 500 --quiet \
          > "$LOG/train_${LABEL}_${M}.txt" 2>&1 \
        || { echo "  TRAIN_FAIL $LABEL/$M"; tail -30 "$LOG/train_${LABEL}_${M}.txt"; FAIL=1; continue; }
      echo "  训练完成（末行）：$(tail -1 "$LOG/train_${LABEL}_${M}.txt")"
    done
  done
else
  echo "[SKIP] TRAIN"
fi

# ------------------------------------------------------------------ RENDER
if [ "$need_render" = 1 ]; then
  for MODE in $MODES; do
    case "$MODE" in
      frozen)   LABEL="冻结几何" ;;
      rf_tuned) LABEL="射频调几何" ;;
      joint)    LABEL="完整耦合" ;;
      *) continue ;;
    esac
    for M in $CMETHODS; do
      DATA="$P_SPECTRUM/3dgs_${M}_100"
      MD="$TOP/$LABEL/$M"
      [ -f "$MD/point_cloud/iteration_${FINAL}/point_cloud.ply" ] \
        || { echo "  [$LABEL/$M] 无末档，跳过渲染"; FAIL=1; continue; }

      # —— 射频：中间档抽帧 + 末档全量（落在 $MD/rf_eval，避免覆盖光学那 640 张）——
      for i in $LADDER "$FINAL"; do
        sub=$SUBSET_CURVE; [ "$i" = "$FINAL" ] && sub=0
        [ -d "$MD/rf_eval/test/ours_$i/renders" ] && continue
        "$PY" render_corf.py -s "$P_VISUAL" -m "$MD" --rf_source_path "$DATA" \
            --modality rf --out_root "$MD/rf_eval" --iteration "$i" \
            --eval --skip_train --subset "$sub" --quiet \
            > "$LOG/rfrender_${LABEL}_${M}_$i.txt" 2>&1 \
          || { echo "  RFRENDER_FAIL $LABEL/$M ours_$i"; tail -15 "$LOG/rfrender_${LABEL}_${M}_$i.txt"; FAIL=1; break; }
        echo "  [R-rf] $LABEL/$M ours_$i (subset=$sub)"
      done

      # —— 光学：只评末档全量（引擎口径，落 $MD/test）——
      if [ ! -d "$MD/test/ours_${FINAL}/renders" ]; then
        "$PY" render_corf.py -s "$P_VISUAL" -m "$MD" --rf_source_path "$DATA" \
            --modality opt --iteration "$FINAL" --eval --skip_train --quiet \
            > "$LOG/optrender_${LABEL}_${M}.txt" 2>&1 \
          || { echo "  OPTRENDER_FAIL $LABEL/$M"; tail -15 "$LOG/optrender_${LABEL}_${M}.txt"; FAIL=1; }
        echo "  [R-opt] $LABEL/$M ours_$FINAL"
      fi
    done
  done
else
  echo "[SKIP] RENDER"
fi

# ------------------------------------------------------------------ METRIC
if [ "$need_metric" = 1 ]; then
  for MODE in $MODES; do
    case "$MODE" in
      frozen)   LABEL="冻结几何" ;;
      rf_tuned) LABEL="射频调几何" ;;
      joint)    LABEL="完整耦合" ;;
      *) continue ;;
    esac
    for M in $CMETHODS; do
      MD="$TOP/$LABEL/$M"
      [ -d "$MD/rf_eval/test/ours_${FINAL}" ] || { echo "  [$LABEL/$M] 无射频渲染档，跳过评测"; continue; }
      [ -f "$MD/rf_results.json" ] || "$PY" "$P_TOOLS/metrics_fast.py" -m "$MD/rf_eval" \
          --iterations "$FINAL" --subset 0 --out "$MD/rf_results.json" \
          > "$LOG/rfmetric_${LABEL}_${M}.txt" 2>&1 \
        || { echo "  RFMETRIC_FAIL $LABEL/$M"; tail -10 "$LOG/rfmetric_${LABEL}_${M}.txt"; FAIL=1; }
      [ -f "$MD/rf_curve.json" ] || "$PY" "$P_TOOLS/metrics_fast.py" -m "$MD/rf_eval" \
          --iterations $LADDER "$FINAL" --subset "$SUBSET_CURVE" --out "$MD/rf_curve.json" \
          > "$LOG/rfcurve_${LABEL}_${M}.txt" 2>&1 \
        || { echo "  RFCURVE_FAIL $LABEL/$M"; FAIL=1; }
      [ -f "$MD/opt_results.json" ] || "$PY" "$P_TOOLS/metrics_fast.py" -m "$MD" \
          --iterations "$FINAL" --subset 0 --out "$MD/opt_results.json" \
          > "$LOG/optmetric_${LABEL}_${M}.txt" 2>&1 \
        || { echo "  OPTMETRIC_FAIL $LABEL/$M"; FAIL=1; }
      echo "  [M] $LABEL/$M 三指标已出"
      [ -f "$MD/rf_results.json" ] && grep -o '"PSNR": [0-9.]*' "$MD/rf_results.json" | head -2
      [ -f "$MD/opt_results.json" ] && grep -o '"PSNR": [0-9.]*' "$MD/opt_results.json" | head -1
    done
  done
else
  echo "[SKIP] METRIC"
fi

# ------------------------------------------------------------------ SRC
if [ "$need_src" = 1 ]; then
  mkdir -p "$SRC"
  for f in train_corf.py render_corf.py corf/__init__.py corf/coupled_model.py \
           corf/dual_scene.py tools/run_corf.sh; do
    if [ -f "$P_CODE/$f" ]; then
      d="$SRC/$(dirname "$f")"; mkdir -p "$d"
      cp "$P_CODE/$f" "$d/$(basename "$f")"
    fi
  done
  { echo "# 源码快照 $(date +%F_%T)"; echo
    echo "| 文件 | md5 |"; echo "|---|---|"
    ( cd "$SRC" && find . -type f -name '*.py' -o -type f -name '*.sh' | sort | while read -r p; do
        echo "| ${p#./} | $(md5sum "$p" | cut -d' ' -f1) |"; done )
    echo; echo "seed  $SEED  md5=$(md5sum "$SEED" | cut -d' ' -f1)"
    echo "init  $INIT_PLY  md5=$(md5sum "$INIT_PLY" | cut -d' ' -f1)"
  } > "$SRC/MD5.md"
  echo "[SRC] 快照 -> $SRC"
else
  echo "[SKIP] SRC"
fi

# ------------------------------------------------------------------ POST
if [ "$need_post" = 1 ]; then
  # 说明.md **只在本组目录放一份**（本仓约定），各臂目录不另放。
  {
    echo "# CORF 老师（耦合优化）三臂 —— 说明"
    echo
    echo "时间：$(date +%F_%T)    卡：$CUDA_VISIBLE_DEVICES"
    echo "驱动：\`$P_CODE/tools/run_corf.sh\`（源码快照见 \`源码/\`）"
    echo
    echo "## 目录"
    echo
    echo '```'
    echo "$TOP/"
    echo "  冻结几何/<方法>/     几何冻结 + 光学外观冻结，只训射频外观，L = L_rf"
    echo "  射频调几何/<方法>/   几何可训 + 光学外观冻结，只训射频外观，L = L_rf"
    echo "  完整耦合/<方法>/     几何可训 + 两面外观可训，L = L_opt + lambda_rf·L_rf"
    echo "  日志/  源码/"
    echo '```'
    echo
    echo "每个 \`<方法>/\` 下："
    echo
    echo "- \`point_cloud/iteration_<档>/point_cloud.ply\` 光学半边（标准点云格式）"
    echo "- \`point_cloud/iteration_<档>/coupled_rf_<档>.pth\` 射频外观 sidecar"
    echo "- \`corf_experiment.json\` 该臂的完整超参"
    echo "- \`rf_eval/test/ours_<档>/{renders,gt}\` 射频渲染（评测口径输入）"
    echo "- \`test/ours_<档>/{renders,gt}\` 光学渲染"
    echo "- \`rf_results.json\` / \`rf_curve.json\` / \`opt_results.json\` 三指标"
    echo
    echo "## 本次配置"
    echo
    echo "| 项 | 值 |"
    echo "|---|---|"
    echo "| 起点 seed | \`$SEED\` |"
    echo "| 初始化点云（读完即被 seed 覆盖） | \`$INIT_PLY\` |"
    echo "| 视觉集 | \`$P_VISUAL\`  resolution=$RES |"
    echo "| 频谱集 | \`$P_SPECTRUM\` |"
    echo "| 方法 | $CMETHODS |"
    echo "| 绝对迭代 | $FINAL（seed 在 30000） |"
    echo "| 中间档 | $LADDER |"
    echo "| lambda_rf | $LAMBDA_RF |"
    echo "| 深度正则 eta·L_d | **未实现**（无 Depth-Anything-2 先验） |"
    echo
    echo "## 读数"
    echo
    echo "三指标（峰值信噪比 / 结构相似度 / 感知相似度）一律并报，**感知相似度是主判据**。"
    echo "射频数字取自 \`rf_results.json\`（640 测试视角全量），光学取自 \`opt_results.json\`。"
    echo
    echo "| 臂 | 方法 | 射频 PSNR | 射频 SSIM | 射频 LPIPS | 光学 PSNR | 光学 SSIM | 光学 LPIPS |"
    echo "|---|---|---|---|---|---|---|---|"
    for MODE in $MODES; do
      case "$MODE" in
        frozen) LABEL="冻结几何";; rf_tuned) LABEL="射频调几何";; joint) LABEL="完整耦合";; *) continue;;
      esac
      for M in $CMETHODS; do
        R="$TOP/$LABEL/$M/rf_results.json"; O="$TOP/$LABEL/$M/opt_results.json"
        rv="$(grep -o '"PSNR": [0-9.]*\|"SSIM": [0-9.]*\|"LPIPS": [0-9.]*' "$R" 2>/dev/null | head -3 | grep -o '[0-9.]*$' | paste -sd/ -)"
        ov="$(grep -o '"PSNR": [0-9.]*\|"SSIM": [0-9.]*\|"LPIPS": [0-9.]*' "$O" 2>/dev/null | head -3 | grep -o '[0-9.]*$' | paste -sd/ -)"
        echo "| $LABEL | $M | ${rv:-未出数} | | | ${ov:-未出数} | | |"
      done
    done
    echo
    echo "（上表射频/光学各填一格 \`PSNR/SSIM/LPIPS\`；未出数的写「未出数」，**不用别的数字顶替**。）"
  } > "$TOP/说明.md"
  echo "[POST] 说明.md -> $TOP/说明.md"
  echo "==== [$(date +%F_%T)] ALL_CORF_DONE (FAIL=$FAIL) ===="
else
  echo "==== [$(date +%F_%T)] 段清单未含 POST，未写收尾哨兵 (FAIL=$FAIL) ===="
fi
exit "$FAIL"
