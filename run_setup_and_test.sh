#!/bin/bash
set -e
cd /workspace/parameter-golf

# Kill any stale training processes
pkill -9 -f sota_train_gpt 2>/dev/null || true
pkill -9 -f train_gpt 2>/dev/null || true
sleep 2

# Pull the CUDA graph fix
git pull origin main

# Build FA3 (if not already built)
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "=== Building FA3 from source (this takes 15-30 min) ==="
    cd /workspace/flash-attention/hopper
    pip install --break-system-packages --no-build-isolation -e . 2>&1 | tail -5
    cd /workspace/parameter-golf
    echo "=== FA3 build done ==="
fi

# Sanity test: 6L ralph_030 recipe, 15 steps
echo "=== Starting sanity test ==="
export PYTHONUNBUFFERED=1
export RUN_ID=sanity_6L_v3
export NUM_LAYERS=6
export TRAIN_BATCH_TOKENS=131072
export WARMDOWN_ITERS=500
export MUON_WD=0.04
export ADAM_WD=0.04
export XSA_LAST_N=2
export ROPE_BASE=50000
export ROPE_DIMS=0
export LN_SCALE_ENABLED=0
export EMA_ENABLED=0
export MUON_USE_CUDA_GRAPH=0
export ITERATIONS=15
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export SKIP_SERIALIZATION=1

python3 sota_train_gpt.py 2>&1 | tee /workspace/sanity_v3.log
echo "=== Sanity test done (exit: $?) ==="
