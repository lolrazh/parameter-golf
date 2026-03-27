#!/bin/bash
# Usage: ./run_diffusion.sh --proxy    (1xH100, fast iteration)
#        ./run_diffusion.sh --prod     (8xH100, submission run)
#        ./run_diffusion.sh --prod --seed 42
set -euo pipefail

MODE="${1:?Usage: ./run_diffusion.sh --proxy|--prod [--seed N]}"
SEED=1337

# Parse args
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Shared config
export PYTHONUNBUFFERED=1
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
export MLP_MULT=3 ROPE_DIMS=16 ROPE_BASE=10000
export LOGIT_SOFTCAP=30.0 QK_GAIN_INIT=1.5
export EMBED_DIM=64 T_MIN=1.0 T_MAX=300.0
export SELF_COND_PROB=0 SCORE_TEMP=0.5
export SAMPLE_STEPS=200 SAMPLE_LEN=256
export TRAIN_BLOCK_SIZE=4
export EVAL_BLOCK_SIZE=4 EVAL_T_SAMPLES=8 EVAL_CONTEXT_LEN=2048
export MATRIX_LR=0.025 SCALAR_LR=0.025 EMBED_LR=0.05
export MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5
export MUON_WD=0.04 ADAM_WD=0.04
export BETA1=0.9 BETA2=0.95 GRAD_CLIP_NORM=0.3
export EMA_DECAY=0.997
export SWA_ENABLED=1 SWA_EVERY=50
export WARMUP_STEPS=20
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="$SEED"

if [[ "$MODE" == "--proxy" ]]; then
    echo "=== PROXY MODE (1xH100, fast iteration) ==="
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1500}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
    export VAL_TOKENS_LIMIT="${VAL_TOKENS_LIMIT:-1048576}"
    export RUN_ID="${RUN_ID:-diffusion_proxy_${SEED}}"
    python3 train_diffusion.py 2>&1 | tee "diffusion_proxy_seed${SEED}.log"

elif [[ "$MODE" == "--prod" ]]; then
    echo "=== PROD MODE (8xH100, submission run) ==="
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    export RUN_ID="${RUN_ID:-diffusion_prod_${SEED}}"
    torchrun --standalone --nproc_per_node=8 train_diffusion.py 2>&1 | tee "diffusion_prod_seed${SEED}.log"

else
    echo "Usage: ./run_diffusion.sh --proxy|--prod [--seed N]"
    exit 1
fi
