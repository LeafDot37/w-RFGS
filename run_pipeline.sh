#!/bin/bash
# One-command pipeline: train -> evaluate -> figures.
#   bash run_pipeline.sh [extra pipeline.py flags]
set -euo pipefail
exec python pipeline.py --stage all "$@"
