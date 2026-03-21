#!/bin/bash
# 11L sp1024 FULL STACK: Partial RoPE + LN Scale + EMA + XSA4 + front3_back1 quant
# Matches PR #315 recipe (1.1250 BPB pending SOTA)
# Target: Thunder Compute 1xH100 PCIe, 10 min
cd ~/parameter-golf

# Install FA3 if not present
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || \
    pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

export PYTHONUNBUFFERED=1
export RUN_ID=fullstack_11L_sp1024_10m
export QUANT_PRESET=front3_back1_8_middle6
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=200

# Use the pre-cleanup version that has Partial RoPE + LN Scale + EMA
# Defaults: NUM_LAYERS=11, ROPE_DIMS=16, LN_SCALE_ENABLED=1, EMA_ENABLED=1, XSA_LAST_N=4
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
    python3 sota_train_gpt_11L.py
echo "EXIT: $?"
