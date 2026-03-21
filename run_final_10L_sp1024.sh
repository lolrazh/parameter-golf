#!/bin/bash
# FINAL: 10L sp1024 int5+zstd, our tuning, 8xH100 SXM
set -e
cd /workspace

# Clone + setup
rm -rf parameter-golf 2>/dev/null
git clone https://github.com/lolrazh/parameter-golf.git
cd parameter-golf

# FA3 + zstd (parallel installs)
pip install --break-system-packages flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 \
    zstandard 2>&1 | tail -3

# sp1024 data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 2>&1 | tail -3

# Verify
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
python3 -c "import zstandard; print('zstd OK')"
ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin

echo "=== LAUNCHING 10L sp1024 ==="
export PYTHONUNBUFFERED=1
export RUN_ID=final_10L_sp1024_int5
export NUM_LAYERS=10
export VOCAB_SIZE=1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export QUANT_PRESET=front3_back1_8_middle6
export INT5_MLP_ENABLED=1

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
echo "EXIT: $?"
