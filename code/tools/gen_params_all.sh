#!/bin/bash
# 全量生成参数化谱图（571,520 张）。在 **Windows 的 sionna_win 环境**跑，
# 从 Git Bash 里启动；不要进 WSL（Sionna 装在 Windows 侧）。
#
# 用法（Git Bash）：
#   bash code/tools/_gen_params_all.sh                 # 2 个分片并行
#   SHARDS=4 bash code/tools/_gen_params_all.sh
#   ROLES="train" bash code/tools/_gen_params_all.sh    # 只补训练角色
#
# 特性：
#   - 按实例 id 分片（--num_shards/--shard），每片单进程；
#   - --skip_existing 续跑（单实例 3200 张是一个原子单元）；
#   - 每个分片单独日志；结束后逐分片报 rc。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 9

PY=${PY:-D:/anaconda3/envs/sionna_win/python.exe}
INST=${INST:-datasets/RF-3DGS_cbf_params/instances.json}
OUT=${OUT:-datasets/RF-3DGS_cbf_params/spectra}
LOGDIR=${LOGDIR:-results/_gen_params}
SHARDS=${SHARDS:-2}
ROLES=${ROLES:-}
EXTRA=${EXTRA:-}

[ -f "$INST" ] || { echo "[FAIL] 缺实例表 $INST（先跑 make_tx_instances.py）"; exit 9; }
mkdir -p "$LOGDIR" "$OUT"

ARGS=(code/sim/gen_cbf_params.py --instances "$INST" --out "$OUT" --skip_existing)
if [ -n "$ROLES" ]; then ARGS+=(--roles $ROLES); fi
if [ -n "$EXTRA" ]; then ARGS+=($EXTRA); fi

echo "实例表 $INST"
echo "输出   $OUT"
echo "分片   $SHARDS"
echo "角色   ${ROLES:-全部}"

pids=()
for k in $(seq 0 $((SHARDS - 1))); do
  PYTHONIOENCODING=utf-8 "$PY" "${ARGS[@]}" --num_shards "$SHARDS" --shard "$k" \
      > "$LOGDIR/shard${k}.log" 2>&1 &
  pids+=($!)
  echo "  分片 $k → pid ${pids[$k]}  日志 $LOGDIR/shard${k}.log"
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[OK] 分片 $i"
  else
    echo "[FAIL] 分片 $i（见 $LOGDIR/shard${i}.log 末尾）"
    tail -5 "$LOGDIR/shard${i}.log"
    fail=1
  fi
done

echo "=== 磁盘图数 ==="
"$PY" - "$OUT" <<'PYEOF'
import sys
from pathlib import Path
root = Path(sys.argv[1])
ds = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("inst"))
tot = sum(len(list(d.glob("*.png"))) for d in ds)
print(f"  {len(ds)} 个实例目录，共 {tot:,} 张")
PYEOF
exit $fail
