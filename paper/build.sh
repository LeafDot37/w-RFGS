#!/bin/bash
# 一键构建：出图 → pdflatex/bibtex 三遍。TeX Live 2025 在 D 盘的默认安装路径。
#
#   bash latex/build.sh          # 只编译
#   bash latex/build.sh --figs   # 先重跑 make_figs.py 再编译
set -euo pipefail
export PATH="/d/LaTeX/texlive/texlive/2025/bin/windows:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ "${1:-}" = "--figs" ]; then
  "${PYTHON:-/c/Python314/python}" make_figs.py
fi

pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > _last.log
grep -E "Output written|^!|Overfull" _last.log || true
