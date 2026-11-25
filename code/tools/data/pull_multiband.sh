#!/bin/bash
# =============================================================================
# pull_multiband.sh —— 拉取形态 A× 出图所需的最小数据集到本地暂存树
# =============================================================================
# 只拉"画图真正要用的东西"，**不整目录同步**（渲染图 640 张/臂，本地只需要 4 帧）：
#   1. 服务器侧先跑 render_std_probe.py（通道 std + 离板占比）—— 只回传 JSON，不传图。
#      这是"过平滑是否缓解 / 输出是否仍在 jet 上"两条验收的唯一数据源。
#   2. curve.json / results.json（4 臂 × 各方法）
#   3. 4 帧 renders(+gt)（A× / A' / V14 / SH@A*）
#   4. A× 的训练日志里抽出的 λ 轨迹行
#
# 暂存树保持服务器相对结构，故 make_multiband_figs.py 在本地跑时路径语义与服务器一致：
#   <STAGE>/<run_dir>/<METHOD>/{curve.json,results.json,std_probe_70000.json,test/...}
#
# 用法（本地仓库根）：
#   bash code/tools/data/pull_multiband.sh
#   METHODS="MVDR" FRAMES="00000 00130" bash code/tools/data/pull_multiband.sh
# =============================================================================
set -uo pipefail

HOST="${HOST:-yuexf@10.250.183.254}"
RR="${RR:-/data/yuexf/3dgs/results}"
RCODE="${RCODE:-/data/yuexf/3dgs/code}"
PY="${PY:-/data/yuexf/miniconda3/envs/gs3d/bin/python}"
STAGE="${STAGE:-results/_stage_multiband}"
METHODS="${METHODS:-MVDR CBF}"
FRAMES="${FRAMES:-00000 00130 00260 00390}"
IT="${IT:-70000}"

RELAX="run_20260912_多频段标量分支"          # 形态 A×（本步新跑）
RELAP="run_20260911_形态A标量头查表"          # 形态 A'
REL14="run_mlp_v2_阶段1b阶段2与长训/阶段2长训_位置编码_X5D3R1"   # V14 长训
RELSH="run_mlp_v2_阶段1b阶段2与长训/阶段1b长训_SH头"            # SH@A* 长训

TMPTGZ="${TMPTGZ:-/tmp/mb_stage.tgz}"
FLOCAL="${FLOCAL:-/tmp/mb_stage_local.tgz}"

echo "==== [$(date +%F_%T)] 拉取形态 A× 数据  host=$HOST  methods='$METHODS'  帧='$FRAMES' ===="
echo "     暂存根: $STAGE"

# ── 服务器侧：探针 + 打包（诊断一律走 stderr，stdout 只留给 tar 流）──────────────
# 单次 ssh：heredoc 喂远程脚本的 stdin，stdout 重定向到本地 tgz，stderr 直达终端。
ssh "$HOST" "RR='$RR' RCODE='$RCODE' PY='$PY' METHODS='$METHODS' FRAMES='$FRAMES' \
IT='$IT' RELAX='$RELAX' RELAP='$RELAP' REL14='$REL14' RELSH='$RELSH' \
OUTTGZ='$TMPTGZ' bash -s" > "$FLOCAL" <<'REMOTE'
set -u
cd "$RCODE" || { echo "远程 code 目录缺失: $RCODE" >&2; exit 1; }

# 1) 探针：只回传 JSON
for M in $METHODS; do
  for A in "$RELAX" "$RELAP" "$REL14" "$RELSH"; do
    D="$RR/$A/$M"
    [ -d "$D/test/ours_$IT/renders" ] || { echo "  [probe] 跳过（无渲染）: $A/$M" >&2; continue; }
    GT=""; [ "$A" = "$RELAX" ] && GT="--use-gt"      # GT 与臂无关，量一次即可
    "$PY" tools/render_std_probe.py --run-dir "$D" --iteration "$IT" \
        --out "$D/std_probe_$IT.json" $GT >&2 \
      || echo "  [probe] 失败: $A/$M" >&2
  done
done

# 2) 组装文件清单（存在才入列）
LIST=$(mktemp)
for M in $METHODS; do
  for A in "$RELAX" "$RELAP" "$REL14" "$RELSH"; do
    for f in curve.json results.json "std_probe_$IT.json"; do
      [ -f "$RR/$A/$M/$f" ] && echo "$A/$M/$f" >> "$LIST"
    done
  done
  for F in $FRAMES; do
    echo "$RELAX/$M/test/ours_$IT/renders/$F.png" >> "$LIST"
    echo "$RELAX/$M/test/ours_$IT/gt/$F.png"      >> "$LIST"
    echo "$RELAP/$M/test/ours_$IT/renders/$F.png" >> "$LIST"
    echo "$REL14/$M/test/ours_$IT/renders/$F.png" >> "$LIST"
    echo "$RELSH/$M/test/ours_$IT/renders/$F.png" >> "$LIST"
  done
done
# 清单里剔除不存在的（--ignore-failed-read 不是所有 tar 都有）
LIST2=$(mktemp)
while read -r f; do [ -f "$RR/$f" ] && echo "$f" >> "$LIST2"; done < "$LIST"
echo "  [tar] $(wc -l < "$LIST2") 个文件" >&2

tar -czf "$OUTTGZ" -C "$RR" -T "$LIST2" && echo "  [tar] $OUTTGZ ($(du -h "$OUTTGZ" | cut -f1))" >&2
cat "$OUTTGZ"
REMOTE
RC=$?
[ $RC -eq 0 ] || { echo "远程打包失败（rc=$RC）"; exit $RC; }

# ── 本地：解包到暂存树 ────────────────────────────────────────────────────────
mkdir -p "$STAGE"
tar -xzf "$FLOCAL" -C "$STAGE" || { echo "解包失败: $FLOCAL"; exit 1; }
echo "[$(date +%F_%T)] 解包完成 -> $STAGE"

# ── λ 轨迹（A× 训练日志里的存档点读数）───────────────────────────────────────
for M in $METHODS; do
  echo "--- $M λ 轨迹:"
  ssh "$HOST" "grep -o 'λ=[0-9.]*' '$RR/$RELAX/日志/train_${M}.txt' 2>/dev/null | tr '\n' ' '" || true
  echo
done

echo "--- 暂存树:"
find "$STAGE" -maxdepth 4 -type d | sort
echo "==== [$(date +%F_%T)] PULL_DONE ===="
echo "下一步：python code/tools/figures/make_multiband_figs.py --stage $STAGE"
