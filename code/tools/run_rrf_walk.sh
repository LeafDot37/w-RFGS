#!/bin/bash
# =============================================================================
# run_rrf_walk.sh —— 形态 A× 的点云涂色之外的**第二件交付物**：新视角漫游
# =============================================================================
# 为什么需要它：A× 的评测只覆盖数据集自带的 640 张 test 图（train 3200 / test 640，
# 机位是 800 位置 × 4 个相隔 90° 的方位角）。用户要看的是**连续性**：绕场景一圈、
# 或从一端推到另一端，标量场是否稳定、伪彩图是否连贯。评测指标回答不了这个。
#
# 关键口径（我此前一度读错、已推翻，见 render_rrf_walk.py 模块头）：
#   数据集 3200 台相机的**天顶角恒为 90.0000°、up 恒 (0,0,1)、roll 为零**，
#   变化的轴是**方位角**，每位置 4 个采样（间隔 90°）。
#   ⇒ 水平环绕属**沿方位轴内插**（最坏离实测方位角 45°）；**只有光轴俯仰与 roll 是真外推**。
#   故 `orbit`（水平）与 `dolly`（实测机位间直线）默认跑；`orbit_pitch` 单独作为
#   标注清楚的外推对照臂，**默认不跑**。诚实性由 轨迹.json 的逐帧
#   `dist_nearest_real_cam` 与 `azimuth_gap_deg_max` 给证据，不靠嘴说。
#
# 渲染链与评测路径**共用同一个函数**（render_rrf_scalar.synthesize）⇒ 漫游视频展示的
# 一定就是被评测的那个模型，结构上杜绝两条路径漂移。
#
# 两道闸（花钱的渲染之前先过）：
#   V2 相机等价：--selfcheck 用真实 test 相机位姿重建合成 Camera，两次渲染逐元素比
#                （阈值 1e-6）。**不与已存 PNG 比 —— PNG 是 8 bit 量化的弱判据。**
#   V3 抽取回归：把 render_rrf_scalar.py 抽 synthesize() 的行为保持性验一遍 ——
#                临时 model 目录（point_cloud 软链、cfg_args 拷贝）重渲 4 帧，
#                与 $MD/test/ours_70000/renders/ 里**重构之前**存的 PNG 对拍。
#
# 产物：$VIS/<M>/漫游/<traj>/{00000.png..., 轨迹.json}（视频由本地 make_walk_video.py 合成）
#       各 (方法, 轨迹) 的 轨迹.json 存在即跳过（可续跑）。
#
# 用法（服务器 code/ 下，占一卡）：
#   CUDA_VISIBLE_DEVICES=2 METHODS="MVDR CBF" nohup bash tools/run_rrf_walk.sh \
#       > /data/yuexf/3dgs/results/run_20260912_多频段标量分支/日志/walk_驱动.log 2>&1 &
# 冒烟（单方法、8 帧）：
#   CUDA_VISIBLE_DEVICES=2 METHODS=MVDR TRAJ=orbit FRAMES=8 bash tools/run_rrf_walk.sh
# 只过闸不渲染：GATES_ONLY=1 bash tools/run_rrf_walk.sh
# =============================================================================
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { echo "缺 CUDA_VISIBLE_DEVICES"; exit 2; }

METHODS="${METHODS:-MVDR CBF}"
TRAJ="${TRAJ:-orbit dolly}"
FRAMES="${FRAMES:-120}"
GATES_ONLY="${GATES_ONLY:-0}"

PY=/data/yuexf/miniconda3/envs/gs3d/bin/python
CODE=/data/yuexf/3dgs/code
DS=/data/yuexf/3dgs/datasets/RF-3DGS_dataset/training-rf-spectrum
TOP="${TOP:-/data/yuexf/3dgs/results/run_20260912_多频段标量分支}"
VIS="$TOP/阶段2_多频段_三维可视化"
LOG="$TOP/日志"
ITER="${ITER:-70000}"

[ -d "$DS" ] || { echo "数据缺失: $DS"; exit 1; }
[ -f "$CODE/tools/render_rrf_walk.py" ] || { echo "缺 tools/render_rrf_walk.py"; exit 1; }
cd "$CODE" || exit 9
mkdir -p "$LOG" "$VIS"
echo "==== [$(date +%F_%T)] 形态A× 新视角漫游  卡=$CUDA_VISIBLE_DEVICES  methods='$METHODS'  traj='$TRAJ'  frames=$FRAMES ===="

FAIL=0

# ── 闸 V2：合成相机构造等价性（真实 test 相机位姿 → 合成 Camera → 两次渲染必须一致）
for M in $METHODS; do
  DATA="$DS/3dgs_${M}_100"
  MD="$TOP/$M"
  [ -d "$DATA" ] || { echo "  [$M] 数据缺失，跳过: $DATA"; FAIL=1; continue; }
  echo "[$(date +%F_%T)] ----- V2 相机等价自检 $M -----"
  # --eval 必须给：cfg_args 里存的是 eval=False（训练时没开），而 Scene 在 eval=False 时
  # 把 test 集置空 ⇒ selfcheck 取 views[0] 会 IndexError。子集划分由 --eval 决定，不是
  # 由已存在的 test/ 目录决定。
  "$PY" tools/render_rrf_walk.py -s "$DATA" -m "$MD" --iteration "$ITER" --eval --selfcheck \
      > "$LOG/walk_selfcheck_${M}.txt" 2>&1
  rc=$?
  grep -a '\[walk\]' "$LOG/walk_selfcheck_${M}.txt" | sed 's/^/    /'
  if [ $rc -ne 0 ]; then
    echo "  SELFCHECK_FAIL_$M (rc=$rc) —— 渲染中止，先查相机约定"
    tail -20 "$LOG/walk_selfcheck_${M}.txt"; FAIL=1
  fi
done

# ── 闸 V3：抽取 synthesize() 的回归（与**重构之前**存的 PNG 对拍）
for M in $METHODS; do
  DATA="$DS/3dgs_${M}_100"
  MD="$TOP/$M"
  [ -d "$DATA" ] || continue
  STORED="$MD/test/ours_$ITER/renders"
  [ -d "$STORED" ] || { echo "  [V3 $M] 无已存 renders，跳过回归"; continue; }
  TMP="/tmp/walk_regress_$M"
  rm -rf "$TMP"; mkdir -p "$TMP"
  ln -s "$MD/point_cloud" "$TMP/point_cloud"          # 软链：不复制 240 MB
  cp -f "$MD/cfg_args" "$TMP/cfg_args" 2>/dev/null    # 保住渲染相关配置（white_background 等）
  cp -f "$MD/cameras.json" "$TMP/cameras.json" 2>/dev/null
  echo "[$(date +%F_%T)] ----- V3 抽取回归 $M（临时目录 $TMP）-----"
  "$PY" render_rrf_scalar.py -s "$DATA" -m "$TMP" --iteration "$ITER" --eval --skip_train \
      --subset 4 --quiet > "$LOG/walk_regress_${M}.txt" 2>&1 \
    || { echo "  REGRESS_RENDER_FAIL_$M"; tail -20 "$LOG/walk_regress_${M}.txt"; FAIL=1; rm -rf "$TMP"; continue; }
  # 先比**字节**：同一条 torchvision.save_image 路径 + 确定性光栅化 ⇒ PNG 应逐字节相同。
  # 这是最强判据，且不引入 PIL 依赖。只有字节不同才退回数值比对去量化差异。
  "$PY" - "$TMP/test/ours_$ITER/renders" "$STORED" <<'PYEOF'
import sys
from pathlib import Path

new, old = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(new.glob("*.png"))
if not files:
    print("  [V3] 新渲帧为空"); sys.exit(1)
bad = []
for f in files:
    g = old / f.name
    if not g.is_file():
        print(f"  [V3] 旧档缺同索引帧 {f.name}"); sys.exit(1)
    if f.read_bytes() != g.read_bytes():
        bad.append(f.name)
print(f"  [V3] {len(files)} 帧逐字节比对：{'全部相同' if not bad else '不同 ' + str(bad)}")
if not bad:
    print("  [V3] PASS（抽取 synthesize() 未改任何数值）")
    sys.exit(0)
nexact = len(bad)
try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"  [V3] FAIL：字节不同，且无法量化（缺 {e.name}）"); sys.exit(1)
worst = 0
for nm in bad:
    a = np.asarray(Image.open(new / nm).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(old / nm).convert("RGB"), dtype=np.int16)
    d = int(np.abs(a - b).max())
    worst = max(worst, d)
    print(f"  [V3] {nm}  逐字节不同，最大通道差 {d}/255")
print(f"  [V3] FAIL：{nexact} 帧字节不同，最大通道差 {worst}/255（重构改变了数值）")
sys.exit(1)
PYEOF
  [ $? -ne 0 ] && FAIL=1
  rm -rf "$TMP"
done

if [ "$GATES_ONLY" = "1" ]; then
  echo "==== [$(date +%F_%T)] GATES_ONLY —— 不渲染漫游 (FAIL=$FAIL) ===="
  exit 0
fi

# ── 漫游渲染
for M in $METHODS; do
  DATA="$DS/3dgs_${M}_100"
  MD="$TOP/$M"
  [ -d "$DATA" ] || { FAIL=1; continue; }
  for T in $TRAJ; do
    OUT="$VIS/$M/漫游"
    if [ -f "$OUT/$T/轨迹.json" ]; then
      echo "  [$M/$T] 已完成（轨迹.json 在），跳过"
      continue
    fi
    echo "[$(date +%F_%T)] ===== $M / $T（$FRAMES 帧）====="
    "$PY" tools/render_rrf_walk.py -s "$DATA" -m "$MD" --iteration "$ITER" \
        --traj "$T" --frames "$FRAMES" --out "$OUT" \
        > "$LOG/walk_${M}_${T}.txt" 2>&1
    rc=$?
    grep -a '\[walk\]' "$LOG/walk_${M}_${T}.txt" | tail -8 | sed 's/^/    /'
    if [ $rc -ne 0 ] || [ ! -f "$OUT/$T/轨迹.json" ]; then
      echo "  WALK_FAIL_${M}_${T} (rc=$rc)"; tail -20 "$LOG/walk_${M}_${T}.txt"; FAIL=1
    fi
  done
done

echo "==== [$(date +%F_%T)] ALL_WALK_DONE (FAIL=$FAIL) ===="
