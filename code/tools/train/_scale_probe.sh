#!/bin/bash
# G1 探针：只为 4 个实例各生成 64 个 test 视角的**尺度元数据**
# （--save_scale 会把归一化前的 min/max/mean 写进 scales_instXXX.jsonl）。
# 目的不是造数据，而是判断 log 尺度能否由几何闭式解释（残差 < 1 dB）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 9

PY=${PY:-D:/anaconda3/envs/sionna_win/python.exe}
OUT=${OUT:-results/_scale_probe}
OUT="$OUT" "$PY" - <<'PYEOF'
import os
from pathlib import Path
Path(os.environ["OUT"]).mkdir(parents=True, exist_ok=True)
PYEOF

echo "[probe] 实例 0,1(train) 144(pos_holdout) 317(orig)  每实例 64 视角"
PYTHONIOENCODING=utf-8 "$PY" code/sim/gen_cbf_params.py \
    --instances datasets/RF-3DGS_cbf_params/instances.json \
    --out "$OUT" \
    --ids 0 1 144 317 \
    --views test \
    --limit 64 \
    --save_scale \
    --skip_existing 2>&1 | tee "$OUT/probe.log"

echo "=== 产物 ==="
ls -1 "$OUT"/scales_inst*.jsonl 2>/dev/null
