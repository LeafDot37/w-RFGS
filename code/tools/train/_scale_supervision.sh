#!/bin/bash
# 尺度监督集：32 个实例 × 前 64 个 test 视角（= 16 位置 × 4 朝向），
# 只导出归一化前的量级（--save_scale），用于训练/评估尺度头。
# 选例：12 个 train + 各角色各若干（含 orig 317 留出不参与训练）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 9

PY=${PY:-D:/anaconda3/envs/sionna_win/python.exe}
OUT=${OUT:-results/_scale_super}
mkdir -p "$OUT"

# 32 个 id：train 前 12 个 + 各留出角色各 4 个 + orig
IDS=$(PYTHONIOENCODING=utf-8 python - <<'PY'
import json
d = json.load(open("datasets/RF-3DGS_cbf_params/instances.json", encoding="utf-8"))
inst = d["instances"] if isinstance(d, dict) else d
by = {}
for i in inst:
    by.setdefault(i["role"], []).append(i["id"])
sel = by.get("train", [])[:12]
for r in ("pos_holdout", "pitch_holdout", "freq_holdout", "double_holdout"):
    sel += by.get(r, [])[:4]
sel += by.get("orig", [])[:1]
print(" ".join(map(str, sel)))
PY
)
echo "[scale] ids = $IDS"
for id in $IDS; do
  PYTHONIOENCODING=utf-8 "$PY" code/sim/gen_cbf_params.py \
      --instances datasets/RF-3DGS_cbf_params/instances.json \
      --out "$OUT" --ids "$id" --views test --limit 64 \
      --save_scale --skip_existing >> "$OUT/gen.log" 2>&1
  echo "  id $id rc=$?"
done
echo "=== 产物 ==="; ls -1 "$OUT"/scales_inst*.jsonl | wc -l
