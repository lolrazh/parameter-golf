#!/bin/bash
# === SUBMISSION RUN: 6L sp4096, 8xH100 SXM, 10 min ===
# Estimated cost: ~$7 (8 GPUs × $1.75/hr × 0.5hr)
set -e

echo "=== Setup ==="
cd /workspace

# Clone if needed
if [ ! -d parameter-golf ]; then
    git clone https://github.com/lolrazh/parameter-golf.git
fi
cd parameter-golf
git pull origin main

# Install FA3 (6 seconds, pre-built wheel)
pip install --break-system-packages flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 | tail -3
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"

# Download sp4096 data if not present
if [ ! -f data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin ]; then
    echo "Downloading sp4096 data..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    local_dir='./data',
    allow_patterns=['datasets/fineweb10B_sp4096/*', 'tokenizers/fineweb_4096_bpe.*'],
    local_dir_use_symlinks=False)
print('sp4096 data downloaded')
"
fi

# Also need sp1024 val shard for BPB calculation if not present
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Downloading sp1024 data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

# Verify data
echo "=== Verifying data ==="
ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin | wc -l
ls data/datasets/fineweb10B_sp4096/fineweb_val_*.bin | wc -l
ls data/tokenizers/fineweb_4096_bpe.model

echo "=== Starting 8xH100 submission run ==="
export PYTHONUNBUFFERED=1
export RUN_ID=submission_6L_sp4096
export QUANT_PRESET=front3_back1_8_middle6

torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
echo "=== Done (exit: $?) ==="
