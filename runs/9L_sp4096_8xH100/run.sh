#!/bin/bash
# 9L sp4096, 8xH100 SXM, 10 min — Result: PENDING (running now)
cd /workspace/parameter-golf

export PYTHONUNBUFFERED=1
export RUN_ID=submission_9L_sp4096
export NUM_LAYERS=9
export QUANT_PRESET=front3_back1_8_middle6

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
