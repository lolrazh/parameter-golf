#!/bin/bash
# 6L sp4096, 8xH100 SXM, 10 min — Result: 1.1818 sliding BPB
# Artifact: 13.6 MB | Steps: ~10,200 | Step time: 59ms
cd /workspace/parameter-golf

export PYTHONUNBUFFERED=1
export RUN_ID=submission_6L_sp4096
export QUANT_PRESET=front3_back1_8_middle6

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
