#!/bin/bash
# 波束权 w 轴的两臂训练：**波束条件臂** vs **波束盲臂**。
# 两臂逐字相同，只差一个开关 `--use_tx_w`（5 维：sin/cos θ_w, sin/cos φ_w, uniform 标志）。
#
# 为什么两臂逐字相同很重要
# ---------------------------------------------------------------------------
# `synthetic_array=True` 下路径集合与 w 无关，w→图 对 CBF 是**闭式重加权**。所以
# 「波束条件臂」能不能赢不是判据 —— 判据是它离闭式解有多远（评测侧另行算）。
# 盲臂的作用只有一个：给出「同一个头、同一份数据、完全不给 w」的下界，
# 用来确认条件臂的提升**确实来自 w 输入**，而不是多出来的 5 维容量。
#
# 数据口径
# ---------------------------------------------------------------------------
# w 数据集每个实例只有 640 个 test 视角（4×4 阵让多径涨 4 倍，3200 视角要 3–4 天），
# 所以必须 `--train_views test`，否则完整性闸按 2560 张判会直接报「谱图不全」。
# 泛化轴是 w 不是视角：留出组 `w_holdout` 是训练位置 + 见过的视角 + **没见过的 w**。
#
# 用法（WSL 本机）：
#   ITERS=60000 SAVES="30000 60000" ARMS="w blind" bash 本脚本
#   ITERS=180000 SAVES="60000 180000" ARMS="w" bash 本脚本        # 胜者长训
#
# 服务器：见 `run_w_arms.sh` with `PY`/`R` overridden（只换 PY/R/BASE，协议逐字相同）。
# 产物：$BASE/<arm>/{angular_<step>.pth, angular_experiment.json, train.log}
# 注意：长跑中途不要编辑本文件（bash 按字节偏移读脚本）。
set -uo pipefail
PY=${PY:-python3}
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$R/code" || exit 9

BASE=${BASE:-$R/results/_w_arms}
ITERS=${ITERS:-60000}
SAVES=${SAVES:-$ITERS}
ARMS=${ARMS:-w}
VIEWS=${VIEWS:-2}
LR0=${LR0:-8e-3}
LR1=${LR1:-8e-5}
INITU=${INITU:-0.05}
# 感知项：默认 0 = 历史行为。sign-Adam 下 λ 是阈值不是比例（实测等量点 ≈3），
# 见 results/_w_lpips_probe/说明.md。
LPW=${LPW:-0}
LPN=${LPN:-vgg}

DATA=${DATA:-$R/datasets/RF-3DGS_dataset/training-rf-spectrum/3dgs_CBF_100}
SPC=${SPC:-$R/datasets/RF-3DGS_mvdr_w_params/spectra}
META=${META:-$R/datasets/RF-3DGS_mvdr_w_params/meta.json}
SEED=${SEED:-$R/datasets/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth}

[ -f "$META" ] || { echo "[FAIL] 缺实例表 $META"; exit 9; }
[ -d "$SPC" ]  || { echo "[FAIL] 缺谱图 $SPC"; exit 9; }
[ -f "$SEED" ] || { echo "[FAIL] 缺种子检查点 $SEED"; exit 9; }
# 实例表必须带 instances 且带 w_label，否则 load_instances 会退回旧口径
"$PY" - "$META" <<'PYEOF' || exit 9
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
raw = m.get("instances")
if raw is None:
    raise SystemExit("[FAIL] meta 没有 instances ⇒ 会走旧「发射源」口径，训练端会报错")
labs = {str(i.get("w_label", "uniform")) for i in raw}
if len(labs) < 2:
    raise SystemExit(f"[FAIL] 实例表里 w_label 只有 {len(labs)} 种 ⇒ 波束轴是常量")
roles = {}
for i in raw:
    roles[i.get("role", "train")] = roles.get(i.get("role", "train"), 0) + 1
print(f"[data] 实例 {len(raw)}，w 标签 {len(labs)} 种，角色 {roles}")
PYEOF

mkdir -p "$BASE"
for arm in $ARMS; do
  case "$arm" in
    w)     W=1 ;;
    blind) W=0 ;;
    *) echo "[FAIL] 未知臂 $arm（只认 w / blind）"; exit 9 ;;
  esac
  M=$BASE/$arm
  if [ -f "$M/.done" ]; then echo "[skip] $arm 已完成"; continue; fi
  mkdir -p "$M"
  echo "=== 臂 $arm  use_tx_w=$W  iters=$ITERS ==="
  "$PY" train_rrf_angular.py -s "$DATA" -m "$M" \
      --spectra_root "$SPC" --meta "$META" --start_checkpoint "$SEED" \
      --iterations "$ITERS" --views_per_step "$VIEWS" --train_views test \
      --lr_schedule cosine --mlp_lr_init "$LR0" --mlp_lr_final "$LR1" \
      --lr_max_steps "$ITERS" --init_u "$INITU" \
      --use_tx_orient 1 --use_tx_local_dir 1 --use_tx_freq 1 --use_tx_w "$W" \
      --train_roles train --lpips_w "$LPW" --lpips_net "$LPN" \
      --save_iterations $SAVES --log_every 200 \
      > "$M/train.log" 2>&1
  rc=$?
  echo "train rc=$rc"
  grep -E "^\[ang\]" "$M/train.log" | tail -5
  if [ $rc -eq 0 ]; then
    touch "$M/.done"
  else
    echo "[FAIL] 臂 $arm 训练失败（见 $M/train.log）"
    tail -20 "$M/train.log"
  fi
done
echo "=== 各臂检查点 ==="
for arm in $ARMS; do
  ls -1 "$BASE/$arm"/angular_*.pth 2>/dev/null | sed "s|^|  |"
done
