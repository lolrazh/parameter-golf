#!/bin/bash
# 9L sp4096 + Int5 MLP + zstd, 8xH100 SXM — Result: PENDING
cd /workspace/parameter-golf

export PYTHONUNBUFFERED=1
export RUN_ID=sub_9L_sp4096_int5
export NUM_LAYERS=9
export QUANT_PRESET=front3_back1_8_middle6
export INT5_MLP_ENABLED=1

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
