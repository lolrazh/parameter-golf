#!/bin/bash
# Single-experiment runner for Ralph Loop / autonomous research.
# Usage: ./run_experiment.sh <run_id> [ENV_VAR=value ...]
# Outputs one-line result to stdout and appends to results.tsv.

set -euo pipefail

RUN_ID="${1:?Usage: ./run_experiment.sh <run_id> [ENV_VAR=value ...]}"
shift

# Fixed env for single-GPU, no torchrun
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501

# Pass through all remaining args as env vars
for arg in "$@"; do
    export "$arg"
done

export RUN_ID

cd ~/parameter-golf

# Run training (all output to log file)
python3 sota_train_gpt.py > "logs/${RUN_ID}_stderr.log" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED run_id=${RUN_ID} exit_code=${EXIT_CODE}"
    echo -e "${RUN_ID}\tFAILED\t${EXIT_CODE}\t$(date -Iseconds)" >> results.tsv
    exit 1
fi

# Parse results from the script's own log
LOG="logs/${RUN_ID}.txt"
if [ ! -f "$LOG" ]; then
    echo "FAILED run_id=${RUN_ID} no_log_file"
    exit 1
fi

# Extract key metrics
PREQUANT=$(grep "^step:.*val_bpb:" "$LOG" | tail -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/')
STEPS=$(grep "^step:.*val_bpb:" "$LOG" | tail -1 | sed 's/.*step:\([0-9]*\).*/\1/')
POSTQUANT=$(grep "^final_int6_roundtrip_exact" "$LOG" | sed 's/.*val_bpb:\([0-9.]*\).*/\1/')
ARTIFACT=$(grep "^Total submission size int6" "$LOG" | sed 's/.*: \([0-9]*\) bytes.*/\1/')
STEP_AVG=$(grep "^step:.*step_avg:" "$LOG" | tail -1 | sed 's/.*step_avg:\([0-9.]*\).*/\1/')

# Single-line result
echo "OK run_id=${RUN_ID} steps=${STEPS} prequant_bpb=${PREQUANT} postquant_bpb=${POSTQUANT} artifact=${ARTIFACT} step_avg=${STEP_AVG}ms"

# Append to TSV
echo -e "${RUN_ID}\t${STEPS}\t${PREQUANT}\t${POSTQUANT}\t${ARTIFACT}\t${STEP_AVG}\t$(date -Iseconds)" >> results.tsv
