#!/bin/bash
# TTT hyperparameter sweep on existing checkpoint
# Usage: bash ttt_sweep.sh
set -e

COMMON="PYTHONUNBUFFERED=1 NUM_LAYERS=11 VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
XSA_LAST_N=4 ROPE_DIMS=16 SKIP_POSTQUANT_EVAL=1 \
TTT_EPOCHS=1 TTT_LACT=0 \
VAL_TOKENS_LIMIT=1048576 \
CHECKPOINT_PATH=final_model.int8.ptz"

echo "=== TTT Hyperparameter Sweep ==="
echo "Checkpoint: final_model.int8.ptz (Run 10)"
echo "Val tokens: 1M (proxy)"
echo ""

# Baseline
echo "--- [1/10] Baseline: lr=0.01 chunk=256 rank=8 min_doc=256 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|ttt_config|time:'

# LR sweep
echo "--- [2/10] LR=0.005 ---"
eval $COMMON TTT_LORA_LR=0.005 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo "--- [3/10] LR=0.02 ---"
eval $COMMON TTT_LORA_LR=0.02 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo "--- [4/10] LR=0.05 ---"
eval $COMMON TTT_LORA_LR=0.05 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

# Chunk size sweep (at best LR from above, default 0.01 for now)
echo "--- [5/10] chunk=128 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=128 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo "--- [6/10] chunk=512 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=512 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

# Rank sweep
echo "--- [7/10] rank=4 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=4 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo "--- [8/10] rank=16 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=16 TTT_MIN_DOC_LEN=256 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

# Min doc len sweep
echo "--- [9/10] min_doc=64 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=64 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo "--- [10/10] min_doc=128 ---"
eval $COMMON TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256 TTT_LORA_RANK=8 TTT_MIN_DOC_LEN=128 python3 run_ttt.py 2>&1 | grep -E 'ttt_bpb|time:'

echo ""
echo "=== Sweep complete ==="
