#!/bin/bash
# Usage: ./run.sh --proxy    (1xH100, fast iteration)
#        ./run.sh --prod     (8xH100, submission run)
#        ./run.sh --prod --seed 42
set -euo pipefail

MODE="${1:?Usage: ./run.sh --proxy|--prod [--seed N]}"
SEED=1337

# Parse args
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Shared config (matches SOTA PR #549 + our additions)
export PYTHONUNBUFFERED=1
export NUM_LAYERS=11 VOCAB_SIZE=1024 MODEL_DIM=512
export NUM_HEADS=8 NUM_KV_HEADS=4
export XSA_LAST_N=4 ROPE_DIMS=16 ROPE_BASE=10000
export LATE_QAT_THRESHOLD=0.15 ENTROPY_REG=0.01
export QUANT_PRESET=front3_back1_6_middle5
export EVAL_STRIDE=64
export TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0
export SEED="$SEED"

if [[ "$MODE" == "--proxy" ]]; then
    echo "=== PROXY MODE (1xH100, fast iteration) ==="
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
    export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1500}"
    export VAL_TOKENS_LIMIT="${VAL_TOKENS_LIMIT:-1048576}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
    python3 train_gpt.py 2>&1 | tee "proxy_seed${SEED}.log"

elif [[ "$MODE" == "--prod" ]]; then
    echo "=== PROD MODE (8xH100, submission run) ==="
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
    export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "seed${SEED}.log"

else
    echo "Usage: ./run.sh --proxy|--prod [--seed N]"
    exit 1
fi
