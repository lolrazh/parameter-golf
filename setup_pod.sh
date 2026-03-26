#!/bin/bash
# One-shot pod setup for Parameter Golf
# Usage: bash setup_pod.sh [--proxy|--prod]
#   --proxy: 10 shards (fast, for iteration)
#   --prod:  80 shards (full data, for submission)
set -euo pipefail

MODE="${1:---prod}"
SHARDS=80
[[ "$MODE" == "--proxy" ]] && SHARDS=10

# Detect home dir (RunPod uses /workspace, others use $HOME)
WORKDIR="/home/ubuntu"
[[ -d "/workspace" ]] && WORKDIR="/workspace"
cd "$WORKDIR"

echo "=== 1/4 Clone repo ==="
if [ -d "parameter-golf" ]; then
    cd parameter-golf && git pull
else
    git clone https://github.com/lolrazh/parameter-golf.git && cd parameter-golf
fi

echo "=== 2/4 Install deps ==="
pip install -r requirements.txt

echo "=== 3/4 Install FA3 (pre-built wheel) ==="
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100

echo "=== 4/4 Download data ($SHARDS shards) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$SHARDS"

echo ""
echo "=== SETUP COMPLETE ($SHARDS shards) ==="
if [[ "$MODE" == "--prod" ]]; then
    echo "Run 3-seed submission:"
    echo "  ./run.sh --prod --seed 1337"
    echo "  ./run.sh --prod --seed 42"
    echo "  ./run.sh --prod --seed 2025"
else
    echo "Run smoke test:"
    echo "  MAX_WALLCLOCK_SECONDS=120 WARMDOWN_ITERS=500 ./run.sh --proxy"
fi
