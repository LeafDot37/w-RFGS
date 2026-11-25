#!/bin/bash
# =============================================================================
# dl_graf_sim.sh —— 从 hf-mirror 拉 GRaF-sim（SOTA 对标用的数据集）
# -----------------------------------------------------------------------------
# 为什么不写 manifest 再照单下
# -----------------------------------------------------------------------------
# 试过：tree API 分页。实测两个坑 ——
#   ① 镜像对 urllib 默认 UA 返回 403；
#   ② 分页游标在 **HTTP `Link` 响应头**里，不在响应体里。只读 body 会**静默**停在
#      第一页（拿到 964 个，且顶层按字典序排 `.gitattributes, D10, …` ⇒ 停在 D10
#      完全看不出少了 D2~D9、D11~D19）。改成跟 Link 之后，97 页里途中有一次
#      `urlopen error timed out`，重试 5 次仍失败 —— 枚举整棵树这条路不通。
#
# 改成**按结构直取**，并且每一步都自证：
#   每场景的文件数**不是常数**（D2 有 5340 个 Tx，D10 只有约 961 个），
#   所以不能硬编码。谱图数与 tx_pos.csv 的行数一一对应 ⇒ 先下 tx_pos.csv、
#   数行、再按下标构造谱图 URL。下完**逐场景核对张数**，不等即报错。
#
# 为什么走 hf-mirror：服务器到 huggingface.co **不通**（curl 直接超时），
# 到 hf-mirror.com 通（HTTP 200 / 0.88 s）。镜像不支持 Xet，所以要关掉 Xet。
#
# 用法：nohup bash tools/dl_graf_sim.sh > /tmp/dl_graf.log 2>&1 &
# =============================================================================
set -u

DEST="${DEST:-/data/yuexf/3dgs/datasets/GRaF/GRaF-sim}"
CKPT="${CKPT:-/data/yuexf/3dgs/datasets/GRaF/GRaF-pretrained}"
MIR="https://hf-mirror.com/datasets/kyang73/GRaF-sim/resolve/main"
MIRM="https://hf-mirror.com/kyang73/GRaF-pretrained/resolve/main"
PAR="${PAR:-64}"                      # xargs 并发；实测是**延迟受限**不是带宽受限
UA="Mozilla/5.0 (compatible; graf-fetch/1.0)"

mkdir -p "$DEST" "$CKPT" /tmp/grafurls

fetch() {   # $1=url $2=out —— 已存在且非空就跳过（可中断续跑）
  [ -s "$2" ] && return 0
  curl -sL --retry 3 --retry-delay 2 -A "$UA" -o "$2" "$1"
}
export -f fetch 2>/dev/null || true

echo "==== [$(date +%F_%T)] GRaF-sim -> $DEST  并发 $PAR ===="

TOT=0
for i in $(seq 2 19); do
  S="D$i"
  mkdir -p "$DEST/$S/spectrum"
  fetch "$MIR/$S/gateway_info.yml" "$DEST/$S/gateway_info.yml"
  fetch "$MIR/$S/tx_pos.csv"       "$DEST/$S/tx_pos.csv"
  if [ ! -s "$DEST/$S/tx_pos.csv" ]; then
    echo "  [$S] tx_pos.csv 没下来 —— 跳过本场景"; continue
  fi
  N=$(( $(wc -l < "$DEST/$S/tx_pos.csv") - 1 ))
  # 谱图 URL 表：文件名是 5 位零填充
  U="/tmp/grafurls/$S.txt"
  : > "$U"
  for k in $(seq 1 "$N"); do
    printf "%s/%s/spectrum/%05d.png\n" "$MIR" "$S" "$k" >> "$U"
  done
  # 已下好的从表里剔掉（续跑时省掉绝大部分请求）
  : > "$U.need"
  while read -r u; do
    b="$(basename "$u")"
    [ -s "$DEST/$S/spectrum/$b" ] || echo "$u" >> "$U.need"
  done < "$U"
  NEED=$(wc -l < "$U.need")
  echo "[$(date +%F%T)] [$S] Tx=$N  待下谱图 $NEED"
  if [ "$NEED" -gt 0 ]; then
    xargs -P "$PAR" -n 1 -a "$U.need" sh -c \
      "curl -sL --retry 3 --retry-delay 2 -A '$UA' -o '$DEST/$S/spectrum/'\$(basename \$0) \$0"
  fi
  GOT=$(find "$DEST/$S/spectrum" -name '*.png' | wc -l)
  if [ "$GOT" -ne "$N" ]; then
    echo "  ✗ [$S] 张数不符：下到 $GOT，应为 $N"
  else
    echo "  ✓ [$S] $GOT/$N"
  fi
  TOT=$((TOT + GOT))
done
echo "[$(date +%F%T)] 谱图共 $TOT 张"

echo "==== [$(date +%F%T)] GRaF 预训练权重 ===="
for w in type2_catA.pth type3_AB_testC.pth type4_sim2real.pth; do
  echo "  拉 $w"
  fetch "$MIRM/$w" "$CKPT/$w"
  ls -la "$CKPT/$w" 2>/dev/null | awk '{printf "    %s  %.1f MB\n", $9, $5/1e6}'
done
fetch "$MIRM/README.md" "$CKPT/README.md"

echo "==== [$(date +%F_%T)] DONE ===="
du -sh "$DEST" "$CKPT" 2>/dev/null
