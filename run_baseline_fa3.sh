#!/bin/bash
# 10-min baseline: 6L sp1024 with FA3 on H100 SXM
cd /workspace/parameter-golf

pkill -9 -f python3 2>/dev/null || true
sleep 1

export PYTHONUNBUFFERED=1
export RUN_ID=baseline_sxm_6L_fa3_10m
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

python3 sota_train_gpt.py
echo "EXIT: $?"
