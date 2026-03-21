#!/bin/bash
# 10L sp4096 + Int5 MLP + SWA + warmdown=3000 + zstd
# Matches SOTA recipe techniques on sp4096 tokenizer
# Estimated: ~7000-8000 steps at ~80ms/step
cd /workspace/parameter-golf
git pull origin main 2>/dev/null

export PYTHONUNBUFFERED=1
export RUN_ID=sub_10L_sp4096_int5_swa
export NUM_LAYERS=10
export QUANT_PRESET=front3_back1_8_middle6
export INT5_MLP_ENABLED=1
export SWA_ENABLED=1
export SWA_START_FRAC=0.4
export SWA_EVERY=50
export WARMDOWN_ITERS=3000

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
echo "EXIT: $?"
