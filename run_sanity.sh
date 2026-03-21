#!/bin/bash
# Sanity test: 6L ralph_030 recipe on H100 SXM
cd /workspace/parameter-golf

pkill -9 -f sota_train_gpt 2>/dev/null
sleep 2

export PYTHONUNBUFFERED=1
export RUN_ID=sanity_6L_v2
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
export ITERATIONS=15
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export SKIP_SERIALIZATION=1

python3 sota_train_gpt.py > /workspace/sanity_v2.log 2>&1
echo "DONE: exit code $?"
