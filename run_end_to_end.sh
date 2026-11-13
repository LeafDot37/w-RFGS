#!/bin/bash
# =============================================================================
# run_end_to_end.sh —— 端到端驱动（MVSplat init 点云 → RF-3DGS 短训 → RRF 频谱重建）
# -----------------------------------------------------------------------------
# 用法（WSL bash，仓库根）：
#     bash run_end_to_end.sh [profile] [mode]
#       profile: full(默认) | smoke
#       mode:    replace（默认）＝新云覆盖 latest、原 latest 压入 archive
#              | no-replace ＝ latest 不动，新云直接压入 archive（主云不受污染）
# 产出：results/run_<时间戳>/{mvsplat, rf-3dgs_1, rf-3dgs_2, logs}
#       + 更新 results/mvsplat_clouds/（replace 刷新 latest；no-replace 仅入 archive）
# 说明：本脚本只训练，不做评测（评测见 evaluate_run.sh，对 run_<时间戳> 单独跑）。
# 所有超参数/路径来自仓库根 config.yaml（load_config.sh 装载），唯一事实来源。
# =============================================================================
set -euo pipefail
# 阶段C 频谱数据集每方法 3200 张图, dataset_readers 用 PIL 惰性打开不及时关 → 默认 fd 软限 1024
# 不够(OSError Errno 24, ~第975张挂)。把进程 fd 上限抬到高限(服务器 hard=1048576)。
ulimit -n 65535 2>/dev/null || true

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_DIR/load_config.sh"

PROFILE="${1:-full}"
MODE="${2:-replace}"                       # replace | no-replace
case "$MODE" in
  replace|no-replace) : ;;
  *) echo "[e2e] 参数 2 须为 replace|no-replace（默认 replace）" >&2; exit 2 ;;
esac
load_config "$PROFILE"

# ---- 本次 run 目录 ----------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
RUN="$P_OUT/run_$TS"
mkdir -p "$RUN/mvsplat/点云" "$RUN/rf-3dgs_1" "$RUN/rf-3dgs_2" "$RUN/logs"
echo "[e2e] profile=$PROFILE  mode=$MODE  RUN=$RUN"
echo "[e2e] K=${RF_NUM_CLUSTERS} V=${RF_VIEWS_PER_CLUSTER} G=${RF_GAZES_PER_CLUSTER} "
echo "      MAXG=${RF_MAX_GAUSSIANS}  ITR1=$ITR1  METHODS='$METHODS'"
echo "      阶段B 存档档: $STAGE1_SAVE"
if [ -n "${TOTAL2:-}" ]; then
  echo "      阶段C: 10000RRF 对齐档绝对 $ITR2 / 40000总 档绝对 $TOTAL2（单次 RRF 跑到 $TOTAL2，两档快照）"
else
  echo "      阶段C: 仅对齐档（绝对末迭代 $ITR2）"
fi

# 保险：STAGE1_SAVE 末尾必须含 ITR1（保证末档 point_cloud + chkpnt 语义）
if [ "${STAGE1_SAVE##* }" != "$ITR1" ]; then
  STAGE1_SAVE="$STAGE1_SAVE $ITR1"
  export STAGE1_SAVE
fi

# ---- 阶段 A：前馈 init 点云（cwd=code/ → 产物落 code/outputs/）----
echo; echo "[stage A] 前馈 init 点云 (init_cloud.py)"
cd "$P_CODE"
rm -rf outputs/rf_init
if ! "$PY" init_cloud.py mode=test +experiment=re10k \
        checkpointing.load="$P_CKPT" > "$RUN/logs/stageA_mvsplat.log" 2>&1; then
  echo "[stage A] 失败，末尾日志："; tail -40 "$RUN/logs/stageA_mvsplat.log"; exit 1
fi
A_PLY="$P_CODE/outputs/rf_init/init.ply"
[ -f "$A_PLY" ] || { echo "[stage A] 未产出 init.ply"; tail -40 "$RUN/logs/stageA_mvsplat.log"; exit 1; }

# 高斯计数（ply 头 element vertex）
N_GAUSS="$("$PY" -c 'import re,sys;d=open(sys.argv[1],"rb").read(4096);
m=re.search(rb"element vertex (\d+)",d);print(m.group(1).decode() if m else "0")' "$A_PLY")"
echo "[stage A] init 云 $N_GAUSS 高斯"
cp "$A_PLY" "$RUN/mvsplat/点云/init.ply"
cp "$P_CODE/outputs/rf_init/init_view.ply" "$RUN/mvsplat/点云/init_view.ply"
cp "$P_CODE/outputs/rf_init/clusters.json" "$RUN/mvsplat/点云/clusters.json"

# ---- 点云仓库刷新（results/mvsplat_clouds latest + archive + provenance）----
# replace：新云覆盖 latest，原 latest 先被压入 archive（archive 里有旧版）
# no-replace：latest 不动，新云直接压入 archive（不污染主云，供并行/对比 run）
mkdir -p "$P_CLOUDS/archive"
TAG="${TS}_K${RF_NUM_CLUSTERS}V${RF_VIEWS_PER_CLUSTER}G${RF_GAZES_PER_CLUSTER}"
ARCH_DIR="$P_CLOUDS/archive/$TAG"
mkdir -p "$ARCH_DIR"
cp "$A_PLY" "$ARCH_DIR/init.ply"                                   # 新云必然入 archive
cp "$P_CODE/outputs/rf_init/clusters.json" "$ARCH_DIR/clusters.json"
if [ "$MODE" = replace ]; then
  cp "$A_PLY" "$P_CLOUDS/init.ply"                                 # 仅 replace 才覆盖 latest
  cp "$P_CODE/outputs/rf_init/init_view.ply" "$P_CLOUDS/init_view.ply"
  cp "$P_CODE/outputs/rf_init/clusters.json" "$P_CLOUDS/clusters.json"
fi

# provenance.json 更新（latest 仅在 replace 时改；history 两种模式都记一条）
CFG_K="$RF_NUM_CLUSTERS" CFG_V="$RF_VIEWS_PER_CLUSTER" CFG_G="$RF_GAZES_PER_CLUSTER" \
CFG_MAXG="$RF_MAX_GAUSSIANS" CFG_TAG="$TAG" CFG_N="$N_GAUSS" CFG_MODE="$MODE" \
CFG_ORIGIN="$RUN/mvsplat/点云/" \
"$PY" - "$P_CLOUDS/provenance.json" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
mode = os.environ["CFG_MODE"]
entry = {
    "tag": os.environ["CFG_TAG"],
    "vertex_count": int(os.environ["CFG_N"]),
    "config": "K=%s V=%s G=%s RF_MAX_GAUSSIANS=%s" % (
        os.environ["CFG_K"], os.environ["CFG_V"], os.environ["CFG_G"], os.environ["CFG_MAXG"]),
    "files": ["init.ply", "clusters.json"],
    "mode": mode,
    "note": "新云覆盖 latest（原 latest 已入 archive）" if mode == "replace"
            else "仅归档（no-replace），latest 未变",
}
try:
    prov = json.load(open(path, encoding="utf-8"))
except Exception:
    prov = {"latest": {}, "history": []}
if mode == "replace":
    prov["latest"] = {
        "cloud": "init.ply", "vertex_count": entry["vertex_count"],
        "source_archive": entry["tag"], "origin_run": os.environ["CFG_ORIGIN"],
        "note": "本次端到端（profile 见对应 run README）刷新",
    }
prov.setdefault("history", []).insert(0, entry)
json.dump(prov, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("provenance updated:", entry["tag"], entry["vertex_count"], "gaussians", f"(mode={mode})")
PYEOF

# RF-3DGS 阶段一读“本次 run 的新云”（run 自带副本，dataset_readers RF_POINTCLOUD 优先；
# 与上方 replace/no-replace 无关——mode 只决定共享仓库 latest 是否被刷新，本 run 总用自己的新云）
export RF_POINTCLOUD="$RUN/mvsplat/点云/init.ply"

# ---- 显存包裹：RF_VRAM_GB>0 → 经 code/tools/_prealloc.py 同进程预占 ~GB 后跑 train.py ----
#   （训练在预占的 ~12GB 包裹内复用、峰值不超，且该内存第 0 秒归本进程，后到租户抢不走；
#    未设 RF_VRAM_GB 时 TRAIN_PY 就是 `$PY train.py`，与原先完全一致）
if [ -n "${RF_VRAM_GB:-}" ]; then
  TRAIN_PY=( "$PY" "$P_CODE/tools/_prealloc.py" train.py )
else
  TRAIN_PY=( "$PY" train.py )
fi

# ---- 阶段 B：RF-3DGS 视觉短训 ---------------------------------------------
echo; echo "[stage B] RF-3DGS 视觉短训 ${ITR1} 步 (data=$P_VISUAL, res=$RES)"
cd "$P_CODE"
if ! "${TRAIN_PY[@]}" -s "$P_VISUAL" -m "$RUN/rf-3dgs_1" --resolution "$RES" \
        --iterations "$ITR1" --save_iterations $STAGE1_SAVE \
        --checkpoint_iterations "$ITR1" > "$RUN/logs/stageB_shorttrain.log" 2>&1; then
  echo "[stage B] 失败，末尾日志："; tail -40 "$RUN/logs/stageB_shorttrain.log"; exit 1
fi
[ -f "$RUN/rf-3dgs_1/chkpnt${ITR1}.pth" ] || { echo "[stage B] 未产出 chkpnt${ITR1}.pth"; exit 1; }

# ---- 阶段 C：RF-3DGS RRF 频谱重建（每方法独立模型）----------------------------
unset RF_POINTCLOUD      # 频谱数据用自身 colmap 点云，不走外部 init
# 双档语义：冻结几何(0)确定性续训 → 单次 RRF 跑到 FINAL2，save 对齐档 ITR2 与 40000总 档 FINAL2 两处快照。
#   TOTAL2 有定义（full）→ FINAL2=TOTAL2=40000，两档快照 = ours_ITR2(10000RRF对齐) + ours_40000(40000总)
#   TOTAL2 缺省（smoke）→ FINAL2=ITR2，仅一档（行为同既往）
FINAL2="${TOTAL2:-$ITR2}"
SAVE2="$ITR2"
if [ "$FINAL2" != "$ITR2" ]; then SAVE2="$ITR2 $FINAL2"; fi
echo; echo "[stage C] RRF 频谱重建（共 $(echo "$METHODS" | wc -w) 方法，DENSIFY=$DENSIFY，"
echo "           最终绝对迭代 $FINAL2，存档档 [$SAVE2]，start chkpnt${ITR1}）"
for M in $METHODS; do
  echo "  - $M  ($P_SPECTRUM/3dgs_${M}_100  → $SAVE2, start chkpnt${ITR1})"
  # --data_device cpu: 2560 张频谱图留内存(服务器 RAM 富余), 加载期不上 GPU。
  #   默认 cuda 会把图全搬显存, 叠加释放的巨段预占块会触发分配器 raw cudaMalloc OOM(实测);
  #   cpu 仅改图驻留位置, 数值完全不变(评测等价), 且显著降显存。
  if ! "${TRAIN_PY[@]}" -s "$P_SPECTRUM/3dgs_${M}_100" -m "$RUN/rf-3dgs_2/$M" \
          --iterations "$FINAL2" --start_checkpoint "$RUN/rf-3dgs_1/chkpnt${ITR1}.pth" \
          --densify_until_iter "$DENSIFY" --save_iterations $SAVE2 \
          --data_device cpu \
          > "$RUN/logs/stageC_rrf_${M}.log" 2>&1; then
    echo "[stage C] $M 失败，末尾日志："; tail -40 "$RUN/logs/stageC_rrf_${M}.log"; exit 1
  fi
done

# ---- 收尾：run README / profile 记录 ----------------------------------------
echo "$PROFILE" > "$RUN/profile.txt"
echo "$MODE" > "$RUN/cloud_mode.txt"
cat > "$RUN/README.md" <<EOF
# run_$TS — 端到端（$PROFILE profile, 点云 $MODE）

> 驱动：\`bash run_end_to_end.sh $PROFILE $MODE\`（只训练，评测用 \`bash evaluate_run.sh $RUN\`）。
> 配置：仓库根 \`config.yaml\`（profile=$PROFILE，唯一事实来源）。
> 点云模式：$MODE（replace=新云已覆盖 latest；no-replace=新云仅入 archive，latest 未变）

## 关键参数

| 项 | 值 |
|---|---|
| K / V / G | $RF_NUM_CLUSTERS / $RF_VIEWS_PER_CLUSTER / $RF_GAZES_PER_CLUSTER |
| max_gaussians | $RF_MAX_GAUSSIANS |
| init 高斯数 | $N_GAUSS |
| 阶段一迭代 | $ITR1（末迭代；resolution $RES） |
| 阶段一存档档 | $STAGE1_SAVE（迭代扫描用） |
| 阶段二方法 | $METHODS |
| 阶段二迭代 | 10000RRF 对齐档绝对 $ITR2 / 40000总 档绝对 ${TOTAL2:-无}（densify_until_iter=$DENSIFY） |

## 内部布局

- mvsplat/点云/   阶段 A init 点云（init.ply + init_view.ply + clusters.json）
- rf-3dgs_1/      阶段 B 短训模型（chkpnt$ITR1.pth / point_cloud/iteration_<档位>/）
- rf-3dgs_2/      阶段 C 各方法 RRF 模型（\`<METHOD>/\` 下 point_cloud/iteration_{$ITR2${TOTAL2:+,$TOTAL2}}/）
- logs/           三阶段日志

评测：先 \`bash run_end_to_end.sh\`，再 \`bash evaluate_run.sh run_$TS $PROFILE\`。
评测套件（=baseline 等价）：mvsplat 覆盖度(覆盖率.png/验证数据.json/4视角对比) + 阶段一
多档迭代扫描(评测/阶段一评测.json+指标图+快速查看4视角对比) + 阶段二双档三方对比
(评测/阶段二评测.json + 三方对比/三方对比.png+对比数据.json) + 参数量-质量曲线
(三方对比/参数量曲线_我的方法vs原版.png/.json)。
EOF
echo
echo "[e2e] 完成：$RUN"
if [ "$MODE" = replace ]; then
  echo "      点云仓库已刷新 results/mvsplat_clouds（latest=$TAG, $N_GAUSS 高斯）"
else
  echo "      no-replace：results/mvsplat_clouds/latest 未变，新云入 archive/$TAG（$N_GAUSS 高斯）"
fi
echo "      评测：bash $P_ROOT/evaluate_run.sh run_$TS $PROFILE"
