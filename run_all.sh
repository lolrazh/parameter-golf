#!/bin/bash
cd /workspace/parameter-golf

# Kill stale processes
pkill -9 -f python3 2>/dev/null || true
pkill -9 -f nvcc 2>/dev/null || true
sleep 2

# Pull latest (CUDA graph fix)
git pull origin main 2>&1

# === FA3 BUILD IN BACKGROUND (parallel compilation) ===
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "=== Starting FA3 build with MAX_JOBS=32 ==="
    (cd /workspace/flash-attention/hopper && \
     MAX_JOBS=32 pip install --break-system-packages --no-build-isolation -e . \
     > /workspace/fa3_build.log 2>&1 && \
     echo "FA3 BUILD DONE" >> /workspace/fa3_build.log || \
     echo "FA3 BUILD FAILED" >> /workspace/fa3_build.log) &
    FA3_PID=$!
    echo "FA3 building in background (PID: $FA3_PID), MAX_JOBS=32"
else
    echo "FA3 already installed!"
fi

# === 10-MIN BASELINE (runs immediately, no FA3 needed) ===
echo "=== Starting 10-min baseline ==="
export PYTHONUNBUFFERED=1
export RUN_ID=baseline_sxm_6L_10m
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
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=200

python3 sota_train_gpt.py 2>&1 | tee /workspace/baseline_10m.log
echo "=== Baseline done (exit: $?) ==="

# Check if FA3 finished
if [ -n "$FA3_PID" ]; then
    echo "Checking FA3 build status..."
    if kill -0 $FA3_PID 2>/dev/null; then
        echo "FA3 still building. Check /workspace/fa3_build.log"
    else
        tail -5 /workspace/fa3_build.log
    fi
fi
