#!/bin/bash
# Package the generated spectra datasets into release-sized archives + checksums.
#
#   bash scripts/package_datasets.sh --root datasets --out /data/release --split-mb 1900
#
# Output (in --out):
#   wrfgs_<dataset>.tar.zst.part-00 … part-NN   plain byte splits, `cat part-*` restores the stream
#   SHA256SUMS.txt                              one line per part: <sha256>  <filename>
#   MANIFEST.txt                                per-dataset file/image counts and total bytes
#
# Datasets are only packed if they exist under --root, so the same command works on a machine
# that holds a subset. `--codec gzip` swaps zstd for gzip (slower, universally available).
#
# Note on --split-mb: GitHub release assets are capped at 2 GiB, hence the 1900 default. For a
# personal cloud drive use --split-mb 0 (single archive per dataset).
set -uo pipefail

ROOT="datasets"
OUT="dist"
SPLIT_MB=1900
CODEC="zstd"
DATASETS="RF-3DGS_cbf_params RF-3DGS_mvdr_w_params RF-3DGS_mvdr_params RF-3DGS_cbf_w_pairs"

while [ $# -gt 0 ]; do
  case "$1" in
    --root)     ROOT="$2"; shift 2;;
    --out)      OUT="$2"; shift 2;;
    --split-mb) SPLIT_MB="$2"; shift 2;;
    --codec)    CODEC="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    -h|--help)  sed -n '2,18p' "$0"; exit 0;;
    *) echo "未知参数：$1"; exit 2;;
  esac
done

mkdir -p "$OUT"
MAN="$OUT/MANIFEST.txt"
SUM="$OUT/SHA256SUMS.txt"
: > "$MAN"
: > "$SUM"
printf "# generated %s\n" "$(date '+%F %T')" >> "$MAN"

for D in $DATASETS; do
  SRC="$ROOT/$D"
  if [ ! -d "$SRC" ]; then
    echo "[skip] $D 不存在（$SRC）"
    continue
  fi
  NAME="wrfgs_${D#*-3DGS_}"
  case "$CODEC" in zstd) EXT=zst;; gzip) EXT=gz;; *) EXT=$CODEC;; esac
  echo "=== $D  →  $NAME.tar.$EXT"

  N_FILES=$(find "$SRC" -type f | wc -l)
  N_PNG=$(find "$SRC" -name '*.png' | wc -l)
  N_INST=$(find "$SRC" -mindepth 1 -maxdepth 2 -type d -name 'tx*' -o -mindepth 1 -maxdepth 2 -type d -name 'inst*' 2>/dev/null | wc -l)
  printf "%s\tfiles=%s\timages=%s\tinstance_dirs=%s\n" "$D" "$N_FILES" "$N_PNG" "$N_INST" | tee -a "$MAN"

  # tar → compressor → optional split
  if [ "$CODEC" = "gzip" ]; then
    COMP=(gzip -1 -c)
  else
    if command -v zstd > /dev/null 2>&1; then COMP=(zstd -3 -T0 -c); else
      echo "[warn] 没有 zstd，回退 gzip"; CODEC=gzip; EXT=gz; COMP=(gzip -1 -c)
    fi
  fi

  if [ "$SPLIT_MB" -gt 0 ]; then
    BYTES=$(( SPLIT_MB * 1024 * 1024 ))
    tar -C "$ROOT" -cf - "$D" | "${COMP[@]}" | split -b "$BYTES" -d -a 2 - "$OUT/$NAME.tar.$EXT.part-"
    PARTS=$(ls "$OUT/$NAME.tar.$EXT.part-"* 2>/dev/null | wc -l)
    echo "[ok] $NAME.tar.$EXT → $PARTS 段"
  else
    tar -C "$ROOT" -cf - "$D" | "${COMP[@]}" > "$OUT/$NAME.tar.$EXT"
    echo "[ok] $NAME.tar.$EXT → $(du -h "$OUT/$NAME.tar.$EXT" | cut -f1)"
  fi
done

echo "=== 校验和"
if command -v sha256sum > /dev/null 2>&1; then
  ( cd "$OUT" && sha256sum wrfgs_*.tar.* > SHA256SUMS.txt )
  SUMS="SHA256SUMS.txt"
else
  ( cd "$OUT" && shasum -a 256 wrfgs_*.tar.* > SHA256SUMS.txt )
  SUMS="SHA256SUMS.txt"
fi
echo "[ok] $OUT/$SUMS（$(wc -l < "$OUT/SHA256SUMS.txt") 行）"
echo
echo "上传：把 $OUT/ 下的 wrfgs_* 与 SHA256SUMS.txt 一起作为 release 资产（tag: datasets-v1），"
echo "然后按下式还原："
echo "    cat wrfgs_xxx.tar.zst.part-* | zstd -d | tar -xf - -C <repo>/datasets"
echo "    （gzip 编解码时把 zstd 换成 gzip）"
