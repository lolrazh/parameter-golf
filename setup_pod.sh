#!/bin/bash
# One-shot pod setup for Parameter Golf smoke test
# Usage: paste this into a RunPod H100 terminal
set -euo pipefail

echo "=== 1/4 Clone repo ==="
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/lolrazh/parameter-golf.git
fi
cd parameter-golf

echo "=== 2/4 Install deps ==="
pip install -r requirements.txt

echo "=== 3/4 Install FA3 (pre-built wheel) ==="
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100

echo "=== 4/4 Download data (10 shards for proxy) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

echo ""
echo "=== READY. Run smoke test with: ==="
echo "MAX_WALLCLOCK_SECONDS=120 WARMDOWN_ITERS=500 ./run.sh --proxy"
