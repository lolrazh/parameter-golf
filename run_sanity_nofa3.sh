#!/bin/bash
# Sanity test WITHOUT FA3 (uses SDPA fallback)
cd /workspace/parameter-golf

pkill -9 -f 'RUN_ID=sanity' 2>/dev/null || true
sleep 1

export PYTHONUNBUFFERED=1
export RUN_ID=sanity_6L_nofa3
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

python3 sota_train_gpt.py
echo "EXIT: $?"
