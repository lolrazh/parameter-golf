#!/bin/bash
# Test upstream train_gpt.py to isolate if NaN is our code or the environment
cd /workspace/parameter-golf

pkill -9 -f python3 2>/dev/null
sleep 2

# Clone upstream if not already there
if [ ! -d /workspace/upstream-pgolf ]; then
    git clone https://github.com/openai/parameter-golf.git /workspace/upstream-pgolf
fi

# Download data for upstream if needed
cd /workspace/upstream-pgolf
if [ ! -d data/datasets/fineweb10B_sp1024 ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

# Test upstream
export PYTHONUNBUFFERED=1
export RUN_ID=upstream_test
export ITERATIONS=15
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0

python3 train_gpt.py > /workspace/upstream_test.log 2>&1
echo "UPSTREAM EXIT: $?"

# Also test our fork with minimal changes
cd /workspace/parameter-golf
export RUN_ID=fork_debug
export NUM_LAYERS=6
export TRAIN_BATCH_TOKENS=131072
export WARMDOWN_ITERS=500
export ROPE_DIMS=0
export LN_SCALE_ENABLED=0
export EMA_ENABLED=0

python3 sota_train_gpt.py > /workspace/fork_debug.log 2>&1
echo "FORK EXIT: $?"
