#!/bin/bash
# run_student.sh —— 阶段 2 · 前馈学生的驱动（服务器与本机共用同一个脚本）
# -----------------------------------------------------------------------------
# 设计见 docs/CORF复现与前馈蒸馏_实施记录.md §2。
#
# 学生只训一个权重共享的射频头；几何与光学外观**钉死**取自种子（从不接触射频梯度）。
# 本驱动铺的是**乙实验（样本效率）**：唯一变量是「有没有视觉支路」与「监督用几个频谱视角」。
#
# 段：FEAT TRAIN RENDER METRIC SRC POST（段幂等，中断直接重跑）
#
# 臂名（文字名，见全仓纪律「方案/说明文档中测试与臂一律文字名」）：
#   有视觉     --use_visual
#   无视觉     （什么都不加）
#   打乱        --use_visual --shuffle_vis     ← 阳性对照：特征在、只是与高斯错配
#
# 默认只铺 有视觉/无视觉 × N=全部/100 四条（先看两条线分不分得开，分不开不必铺满）。
# 要跑阴性对照：ARMS="有视觉 无视觉 打乱"
#
# ⚠ 驱动一旦启动就**绝不能再编辑**（bash 按字节偏移读脚本）。
set -uo pipefail

ROOT="${STUDENT_ROOT:-/data/yuexf/3dgs}"
cd "$ROOT" || exit 9
source ./load_config.sh
load_config full

PY="${PY:-python}"
cd "$P_CODE" || exit 9
TOOLS="$P_CODE/tools"

SEGMENTS="${SEGMENTS:-FEAT TRAIN RENDER METRIC SRC POST}"
FINAL="${FINAL:-30000}"
LADDER="${LADDER:-10000 20000}"
ARMS="${ARMS:-有视觉 无视觉}"
NS="${NS:-0 100}"
LR="${LR:-1e-3}"
SUBSET="${SUBSET:-0}"
SEED_RF="${SEED_RF:-$P_OUT/orig_seed_chkpnt30000.pth}"

# 射频数据集（MVDR）。跨方法绝对数值不可比，本阶段只跑 MVDR。
# `$P_SPECTRUM` 由 load_config.sh 导出（与 run_corf.sh 同源），不自己拼路径。
DATA="${DATA:-$P_SPECTRUM/3dgs_MVDR_100}"
[ -d "$DATA" ] || { echo "[abort] 频谱数据集不存在：$DATA"; exit 3; }

# ⚠ 用**去重后**的缓存（809 个唯一视点）。先前那份 vis_opt_all_r2.pt 是 970 个「相机」——
# 其中 161 个是 transforms_test.json 指回 train/ 的重复视点，会把 mean/max/std 聚偏。
FEAT_CACHE="${FEAT_CACHE:-$P_OUT/student_feats/vis_opt_uniq_r2.pt}"
STAGE1_TOP="${STAGE1_TOP:-$P_OUT/run_20261008_本机4060_CORF老师三臂}"

# 老师靶子（只给「可抄性」臂用）。**运行时解析，不写死档位** ——
# 阶段 1 的 FINAL 与本阶段未必同源，写死 iteration_70000 会在阶段 1 以别的档位落盘时
# 变成一条永远 FileNotFoundError 的死路。优先 70000，否则退回最新的一档。
TEACHER="${TEACHER:-}"
if [ -z "$TEACHER" ]; then
    for c in "$STAGE1_TOP/冻结几何/MVDR/point_cloud/iteration_70000/coupled_rf_70000.pth" \
             $(ls -t "$STAGE1_TOP"/冻结几何/MVDR/point_cloud/iteration_*/coupled_rf_*.pth 2>/dev/null); do
        if [ -f "$c" ]; then TEACHER="$c"; break; fi
    done
fi
[ -n "$TEACHER" ] || echo "[warn] 找不到老师靶子：$STAGE1_TOP/…（只有「可抄性」臂需要它）"

TOP="${TOP:-$P_OUT/run_20261008_前馈学生N扫描}"
mkdir -p "$TOP/日志" "$TOP/源码"

# ⚠ 用 ${VAR:-} —— 直接写 "$CUDA_VISIBLE_DEVICES" 在 `set -u` 下「未设置」会先被 bash
# 判成 unbound variable 而中止（那句友好的 abort 根本来不及打印），只剩「设了但为空」能走到。
case "${CUDA_VISIBLE_DEVICES:-}" in
    *,*) echo "[abort] 本仓约定：一次只占一张卡（CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES）"; exit 2 ;;
    "")  echo "[abort] 未设 CUDA_VISIBLE_DEVICES（或设成了空）"; exit 2 ;;
esac

# 种子若不在受保护位置上，退回作者发布包（本机的情形）
if [ ! -f "$SEED_RF" ]; then
    ALT="$(dirname "$P_VISUAL")/blender_visual_trained/chkpnt30000.pth"
    if [ -f "$ALT" ]; then
        echo "[warn] 未找到 $SEED_RF，退回作者发布包: $ALT"
        SEED_RF="$ALT"
    else
        echo "[abort] 找不到种子：$SEED_RF"; exit 3
    fi
fi

echo "==== [$(date +%F_%T)] 阶段 2 · 前馈学生  TOP=$TOP"
echo "     SEGMENTS='$SEGMENTS'  卡=$CUDA_VISIBLE_DEVICES"
echo "     臂='$ARMS'   N='$NS'   迭代 $FINAL   LR=$LR"
echo "     特征缓存 = $FEAT_CACHE"
echo "     频谱集   = $DATA"
echo "     种子     = $SEED_RF"
echo "     老师靶子 = $TEACHER"

FAIL=0
RUNS=0

# --------------------------------------------------------------------------
if [[ " $SEGMENTS " == *" FEAT "* ]]; then
    if [ -f "$FEAT_CACHE" ]; then
        echo "[$(date +%F_%T)] FEAT 跳过（缓存已在：$(du -h "$FEAT_CACHE" | cut -f1)）"
        # ⚠ 跳过不等于对得上：下游 train/render 用的是 cache["meta"]["ckpt"] 记的那个种子，
        # 不是 $SEED_RF。两者分叉 ⇒ 源码/MD5.md 里登记的哈希指向的**不是被训的模型**。
        # 所以这里比一次 md5，不一致就中止（宁可中止也不要一条错误的溯源链）。
        CACHE_CKPT="$("$PY" -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu')['meta']['ckpt'])" "$FEAT_CACHE" 2>/dev/null)"
        if [ -z "$CACHE_CKPT" ] || [ ! -f "$CACHE_CKPT" ]; then
            echo "[abort] 缓存里记的种子读不到：'$CACHE_CKPT'（缓存与数据集可能被挪过）"; exit 3
        fi
        if [ "$(md5sum < "$CACHE_CKPT" | cut -d' ' -f1)" != "$(md5sum < "$SEED_RF" | cut -d' ' -f1)" ]; then
            echo "[abort] 种子分叉：缓存由 $CACHE_CKPT 算出，但本次指定的是 $SEED_RF"
            echo "        要么删掉缓存重算，要么把 SEED_RF 指回缓存里那个。"
            exit 3
        fi
        echo "          种子对拍通过（缓存 ↔ \$SEED_RF 同 md5）"
    else
        # ⚠ 必须显式传 --ckpt：下游 train/render 都是从 cache["meta"]["ckpt"] 取种子
        # （`train_student.py:load_seed(cache["meta"]["ckpt"], …)`），所以**这一处**决定了
        # 全阶段真正被训的是哪个种子。不给就会回落到脚本自己的默认路径，
        # 于是 $SEED_RF 的 fallback 与 源码/MD5.md 里那个哈希指向的都不是被训的模型。
        echo "[$(date +%F_%T)] FEAT 预计算视觉支路特征（几分钟；装配阶段会短暂碰一下 CUDA）"
        mkdir -p "$(dirname "$FEAT_CACHE")"
        "$PY" "$TOOLS/make_student_features.py" --ckpt "$SEED_RF" \
            --n_cams 0 --resolution 2 --device cpu --out "$FEAT_CACHE" \
            > "$TOP/日志/feat.txt" 2>&1 || { echo "[fail] FEAT"; FAIL=$((FAIL + 1)); }
        tr '\r' '\n' < "$TOP/日志/feat.txt" | grep -av "Reading camera" | tail -4
    fi
fi

run_one () {
    local arm="$1" nkey="$2" extra="$3"; shift 3
    local ntag
    if [ "$nkey" = "0" ]; then ntag="N全部"; else ntag="N${nkey}"; fi
    local MD="$TOP/$arm/$ntag"
    RUNS=$((RUNS + 1))
    echo ""
    echo "[$(date +%F_%T)] ===== $arm / $ntag =====  -> $MD"
    mkdir -p "$MD"

    if [[ " $SEGMENTS " == *" TRAIN "* ]]; then
        # shellcheck disable=SC2086
        "$PY" "$P_CODE/train_student.py" -m "$MD" \
            --feature_cache "$FEAT_CACHE" --rf_source_path "$DATA" \
            --rf_data_device cpu --n_keyframes "$nkey" \
            --iterations "$FINAL" --save_iterations $LADDER "$FINAL" \
            --lr "$LR" --log_every 500 $extra \
            > "$TOP/日志/train_${arm}_${ntag}.txt" 2>&1
        rc=$?
        tr '\r' '\n' < "$TOP/日志/train_${arm}_${ntag}.txt" | grep -a "\[student\]" | tail -3
        if [ $rc -ne 0 ]; then echo "[fail] TRAIN $arm/$ntag rc=$rc"; FAIL=$((FAIL + 1)); return; fi
    fi

    if [[ " $SEGMENTS " == *" RENDER "* ]]; then
        "$PY" "$P_CODE/render_student.py" -m "$MD" --head "head_${FINAL}.pth" \
            --feature_cache "$FEAT_CACHE" --rf_source_path "$DATA" \
            --rf_data_device cpu --out_root "$MD/rf_eval" --iteration "$FINAL" \
            --subset "$SUBSET" --skip_train --quiet \
            > "$TOP/日志/render_${arm}_${ntag}.txt" 2>&1 \
            || { echo "[fail] RENDER $arm/$ntag"; FAIL=$((FAIL + 1)); return; }
        echo "     渲染 $(find "$MD/rf_eval" -name '*.png' | wc -l) 张"
    fi

    if [[ " $SEGMENTS " == *" METRIC "* ]]; then
        "$PY" "$TOOLS/metrics_fast.py" -m "$MD/rf_eval" --iterations "$FINAL" \
            --subset "$SUBSET" --out "$MD/rf_results.json" \
            > "$TOP/日志/metric_${arm}_${ntag}.txt" 2>&1 \
            || { echo "[fail] METRIC $arm/$ntag"; FAIL=$((FAIL + 1)); return; }
        cat "$MD/rf_results.json"
    fi
}

if [[ " $SEGMENTS " == *" TRAIN "* || " $SEGMENTS " == *" RENDER "* || " $SEGMENTS " == *" METRIC "* ]]; then
    for N in $NS; do
        for A in $ARMS; do
            case "$A" in
                有视觉)  run_one "有视觉" "$N" "--use_visual" ;;
                无视觉)  run_one "无视觉" "$N" "" ;;
                打乱)    run_one "阴性对照_特征打乱" "$N" "--use_visual --shuffle_vis" ;;
                可抄性)  run_one "可抄性_属性蒸馏" "$N" "--use_visual --w_attr ${W_ATTR:-0.1} --teacher_ckpt $TEACHER" ;;
                *)       echo "[warn] 未知臂：$A（可选 有视觉/无视觉/打乱/可抄性）" ;;
            esac
        done
    done
fi

# --------------------------------------------------------------------------
if [[ " $SEGMENTS " == *" SRC "* ]]; then
    {
        echo "# 源码与种子快照（阶段 2）"
        echo
        echo "生成：$(date +%F_%T)"
        echo
        echo '```'
        echo "seed   $SEED_RF"
        md5sum "$SEED_RF" 2>/dev/null || echo "（种子读不到）"
        [ -f "$FEAT_CACHE" ] && { echo "feat   $FEAT_CACHE"; md5sum "$FEAT_CACHE"; }
        [ -f "$TEACHER" ]    && { echo "teacher $TEACHER"; md5sum "$TEACHER"; }
        echo '```'
        echo
        echo "## 本阶段新增/改动的源码"
        echo
        echo '```'
        for f in train_student.py render_student.py corf/rf_head.py corf/dual_scene.py \
                 tools/make_student_features.py tools/probe_projection.py tools/run_student.sh; do
            [ -f "$P_CODE/$f" ] && md5sum "$P_CODE/$f"
        done
        echo '```'
    } > "$TOP/源码/MD5.md"
    echo "[$(date +%F_%T)] SRC -> $TOP/源码/MD5.md"
fi

# --------------------------------------------------------------------------
if [[ " $SEGMENTS " == *" POST "* ]]; then
    cat > "$TOP/说明.md" <<MDEOF
# 阶段 2 · 前馈学生（N 扫描）

**这是一次实验的目录，不是交付物。** 设计见
\`docs/CORF复现与前馈蒸馏_实施记录.md\` §2。

## 这次跑的是什么

学生只训一个**权重共享的射频头**；几何与光学外观全程钉死，取自 30 000 步纯光学种子。
本驱动铺的是**乙实验（样本效率）**：唯一变量是「有没有视觉支路」与「监督用几个频谱视角」。

| 臂 | 头的输入 | 说明 |
|---|---|---|
| 有视觉 | 位置 + 视觉支路（12 维） | |
| 无视觉 | 位置 | 两条线唯一的差别就是那 12 维 |
| 阴性对照_特征打乱 | 位置 + **打乱后**的视觉支路 | 特征在、只是与高斯错配。用来区分「视觉有用」与「头在按位置查表」 |

## 口径

- 频谱方法：**MVDR**（跨方法绝对数值不可比，本阶段只跑 MVDR）
- 监督视角 N：\`$NS\`（0 = 全部 2560）
- 迭代：$FINAL，学习率 $LR
- 评测：频谱**测试集**全部视角（\`--subset $SUBSET\`），三指标并报，**LPIPS 为主判据**
- 种子与特征缓存的 md5 见 \`源码/MD5.md\`

## 产物

\`\`\`
<臂>/<N标签>/head_${FINAL}.pth          学生唯一可训的部分
<臂>/<N标签>/student_experiment.json    全部超参（含实际用到的关键帧索引）
<臂>/<N标签>/rf_eval/test/ours_${FINAL}/{renders,gt}
<臂>/<N标签>/rf_results.json            三指标
日志/                                    每臂的训练与渲染日志
\`\`\`

## 怎么读

**先看两条线分不分得开，再看阴性对照。**
若「有视觉」与「无视觉」在 N 小时分开、在 N 大时重合 ⇒ 视觉提供的是先验（补数据）。
若两条线全程重合，**不要直接写「视觉无用」**——先看阴性对照臂：
若打乱臂也追平，说明头在按位置查表，这个实验没测到视觉的价值。
MDEOF
    echo "[$(date +%F_%T)] POST -> $TOP/说明.md"
fi

echo ""
echo "==== [$(date +%F_%T)] 阶段 2 收尾：跑 $RUNS 个配置，失败 $FAIL ===="
echo "ALL_STUDENT_DONE (FAIL=$FAIL)"
