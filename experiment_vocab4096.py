"""
experiment_vocab4096.py - Research and implementation notes for larger vocabulary (4096+)

=========================================================================================
RESEARCH SUMMARY: Vocabulary Size vs BPB in Parameter Golf
=========================================================================================

## The Core Tradeoff

BPB = val_loss_nats / (bytes_per_token * ln(2))

Larger vocabulary:
  + More bytes per token (better text compression) -> directly lowers the denominator
  + Each token step covers more text -> more text "seen" per training step
  - More classes to predict -> val_loss (nats) goes up
  - Larger embedding table -> more artifact bytes, fewer layers possible
  - Sparser token distributions -> harder to learn, need more training steps

The bet: BPB improves if the bytes-per-token gain outweighs the val_loss increase.

## Empirical Data

Bytes per token (from val set, same 50K FineWeb docs):
  sp1024:  2.47 bytes/token  (official baseline)
  sp2048: ~2.90 bytes/token  (estimated)
  sp4096: ~3.34 bytes/token  (estimated; PR #293 reported 2.75 on test sentence)
  sp8192:  3.77 bytes/token  (actual from val data)

Break-even analysis - what val_loss is needed to beat 1.2244 BPB (current baseline)?
  sp1024: val_loss <= 1.697 nats (this IS the current baseline)
  sp2048: val_loss <= 2.461 nats (45% higher allowed!)
  sp4096: val_loss <= 2.834 nats (67% higher allowed!)
  sp8192: val_loss <= 3.203 nats (89% higher allowed!)

The larger the vocab, the more "slack" you get on val_loss. But the question is
whether the model can actually stay within that slack.

## PR #293 Reference (sp4096, 1xH100, 8 layers)
  - val_loss = 2.943 nats, val_bpb = 1.283 (did NOT beat 1.224 baseline)
  - Only 1,927 steps on 1xH100 vs 13,780 steps on 8xH100
  - Used older codebase without SmearGate, EMA, QAT, etc.
  - Author estimates 1.18-1.20 BPB on equivalent 8xH100 hardware
  - Artifact: 14.78 MB (fits 16 MB budget)

## Scaling Laws Research (arXiv:2407.13623 - NeurIPS 2024)
  - Optimal vocab params scale as Nv^opt ~ Nnv^0.83
  - For ~25M non-vocab params: optimal vocab ~39K tokens (way more than 1024!)
  - Current 1024 vocab is severely under-allocated by scaling law predictions
  - BUT: scaling laws assume unbounded artifact size. Our 16MB constraint changes the math.

## BPB Scenarios for sp4096

If we achieve these val_loss values with sp4096 (bpt ~3.34), the BPB would be:
  val_loss=2.0 -> BPB ~0.864  (huge win, probably unrealistic)
  val_loss=2.2 -> BPB ~0.950  (very ambitious)
  val_loss=2.4 -> BPB ~1.036
  val_loss=2.6 -> BPB ~1.123
  val_loss=2.8 -> BPB ~1.209  (competitive with baseline!)
  val_loss=3.0 -> BPB ~1.296  (PR #293 territory)
  val_loss=3.2 -> BPB ~1.382

To beat the pending SOTA of 1.127 BPB, need val_loss <= ~2.61 with sp4096.

=========================================================================================
ARTIFACT SIZE ANALYSIS
=========================================================================================

## Current Setup (11L x 512d, sp1024, FP16 embed)
  Artifact: ~14 MB with front3_back1_8_middle6 quant preset
  Headroom: ~2 MB

## Per-component costs
  Per 1024 extra vocab tokens (FP16 passthrough + zstd): ~0.88 MB
  Per 1024 extra vocab tokens (int8 quantized + zstd):   ~0.16 MB
  Per transformer layer removed (int6/int8 mix + zstd):  ~1.1 MB

  (Estimates calibrated from real artifact data, front3_back1_8_middle6 preset)

## Vocab 4096 configurations (starting from ~13.5 MB baseline)

  With FP16 embedding (FP16_EMBED=1):
    4096v + 11L: ~16.2 MB  OVER BUDGET
    4096v + 10L: ~15.4 MB  FITS (tight)
    4096v +  9L: ~14.6 MB  FITS

  With int8 embedding (FP16_EMBED=0):
    4096v + 11L: ~14.0 MB  FITS (2 MB headroom!)
    4096v + 10L: ~13.3 MB  FITS (comfortable)
    4096v +  9L: ~12.5 MB  FITS (lots of headroom)

  RECOMMENDATION: Use FP16_EMBED=0 (int8 embedding) with 4096 vocab.
  Int8 has 127 quantization levels which is plenty for embedding rows.
  The current FP16 passthrough was added when vocab was only 1024 and
  embedding was a tiny fraction of the artifact. With 4x more vocab,
  int8 saves ~2.2 MB vs fp16 on the embedding alone.

## Vocab 2048 (conservative option)
  With FP16 embed: 2048v + 11L: ~14.4 MB  FITS
  With int8 embed: 2048v + 11L: ~13.3 MB  FITS

## Vocab 8192 (aggressive option)
  With int8 embed: 8192v + 11L: ~15.4 MB  FITS (tight)
  With int8 embed: 8192v + 10L: ~14.6 MB  FITS
  With int8 embed: 8192v +  9L: ~13.9 MB  FITS

=========================================================================================
DATA PREPARATION
=========================================================================================

Pre-built tokenizers AND pre-tokenized data exist at:
  HuggingFace: sproos/parameter-golf-tokenizers (model repo, NOT dataset repo)

Available variants: sp2048, sp4096, sp8192
Each has: 80 train shards + 1 val shard + .model + .vocab files

## Download Method 1: Direct HuggingFace Hub (recommended)

The official cached_challenge_fineweb.py only knows about the official willdepueoai repo
which only has sp1024. For sp4096, download from sproos repo directly:

```python
from huggingface_hub import hf_hub_download
import os

REPO = "sproos/parameter-golf-tokenizers"
DEST = "data"

# Download tokenizer
for fname in ["tokenizers/fineweb_4096_bpe.model", "tokenizers/fineweb_4096_bpe.vocab"]:
    hf_hub_download(repo_id=REPO, filename=fname, repo_type="model",
                    local_dir=DEST, local_dir_use_symlinks=False)

# Download val shard
hf_hub_download(repo_id=REPO, filename="datasets/fineweb10B_sp4096/fineweb_val_000000.bin",
                repo_type="model", local_dir=DEST, local_dir_use_symlinks=False)

# Download train shards (10 is enough for experiments, 80 for full runs)
for i in range(10):
    hf_hub_download(repo_id=REPO,
                    filename=f"datasets/fineweb10B_sp4096/fineweb_train_{i:06d}.bin",
                    repo_type="model", local_dir=DEST, local_dir_use_symlinks=False)
```

## Download Method 2: Train your own tokenizer (from scratch)

Requires docs_selected.jsonl (~48 GB). Steps:
  1. python3 data/cached_challenge_fineweb.py --variant sp1024 --with-docs
  2. Train SentencePiece BPE tokenizer (see train_tokenizer() below)
  3. Re-tokenize all shards (see preprocess_fineweb_4096.py in PR #293)

=========================================================================================
IMPLEMENTATION PLAN
=========================================================================================

## Changes needed in sota_train_gpt.py

1. Make vocab_size configurable (it's currently hardcoded to 1024):
   - Change: vocab_size = 1024
   - To:     vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))

2. Set FP16_EMBED=0 for larger vocab (int8 is fine, saves ~2 MB)

3. Adjust bigram_hash_buckets for larger vocab:
   - Current: 10240 (5x of 2048... wait, 10x of 1024)
   - For 4096 vocab: could use 20480 or keep at 10240
   - The manifest recommends: 5120 for sp1024, 40960 for sp8192
   - For sp4096: ~20480 seems reasonable

4. Adjust tied_embed_init_std:
   - Current: 0.005 (for 1024 rows)
   - For 4096: might want smaller init, like 0.003-0.004
   - Rule of thumb: 1/sqrt(vocab_size) ~ 0.0156 for 4096, but tied embeddings
     need much smaller init since they're shared with the output projection

5. Adjust tied_embed_lr:
   - Current: 0.05
   - For 4096: the larger embedding learns slower (more rows to move)
   - Try: 0.03-0.04 initially

6. Data/tokenizer paths:
   - DATA_PATH=./data/datasets/fineweb10B_sp4096
   - TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model

=========================================================================================
EXPERIMENT COMMANDS
=========================================================================================

## Step 0: Download sp4096 data
```bash
python3 experiment_vocab4096.py download --vocab-size 4096 --train-shards 10
```

## Step 1: Quick local test (MLX, sanity check only - NOT for hyperparameter tuning)
```bash
RUN_ID=vocab4096_local \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
FP16_EMBED=0 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=4096 \
GRAD_ACCUM_STEPS=1 \
TRAIN_SEQ_LEN=512 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
VAL_TOKENS_LIMIT=131072 \
WARMUP_STEPS=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
SKIP_SERIALIZATION=1 \
python3 train_gpt_mlx.py
```

## Step 2: Modal experiment (1xH100, 2 min, algorithmic comparison)
NOTE: This requires a Modal image that downloads sp4096 data. See below.
```bash
modal run train_modal.py --run-id vocab4096_2m --max-wallclock 120 \
    --overrides 'VOCAB_SIZE=4096,DATA_PATH=./data/datasets/fineweb10B_sp4096,TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model,FP16_EMBED=0'
```

## Step 3: Full 10-min run on 8xH100 (submission candidate)
```bash
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
FP16_EMBED=0 \
NUM_LAYERS=11 \
MUON_WD=0.04 ADAM_WD=0.04 \
XSA_LAST_N=4 ROPE_BASE=50000 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
```

=========================================================================================
RISK ASSESSMENT
=========================================================================================

## Why this might work
1. BPB formula gives 38-67% slack on val_loss for sp4096
2. Current 1024 vocab is severely under-allocated per scaling laws
3. Each training step covers 35% more text bytes (2.47 -> ~3.34 bpt)
4. With int8 embedding, 4096 vocab + 11L fits in 16 MB
5. Orthogonal to all existing optimizations (can stack with XSA, EMA, etc.)
6. Pre-tokenized data available - no custom preprocessing needed

## Why this might NOT work
1. 4x more output classes -> harder prediction task
2. Token frequency distribution gets sparser (long tail problem)
3. Embedding quality at int8 quantization with 4x more rows
4. Less data available per token (each token type appears ~4x less often)
5. Bigram hash interaction: current 10240 buckets tuned for 1024 vocab
6. QAT behavior with larger vocab (embedding excluded from QAT currently)
7. Training dynamics: convergence might be slower with more classes

## Mitigation strategies
- Start with sp2048 (conservative, guaranteed to fit, smaller quality risk)
- Use int8 embedding to maximize layer count
- Adjust bigram hash buckets proportionally
- May need to reduce tied_embed_lr (larger table, slower convergence)
- Run quick sanity check before committing to expensive GPU time

=========================================================================================
OPTIMAL STRATEGY RECOMMENDATION
=========================================================================================

Ranked by expected value (probability * impact):

1. **sp4096 + 11L + int8 embed** (~14.0 MB artifact, ~2 MB headroom)
   - Highest expected BPB improvement
   - Keeps all 11 layers (proven optimal architecture)
   - int8 embedding (127 levels) is sufficient quality
   - Comfortable margin under 16 MB budget

2. **sp2048 + 11L + int8 embed** (~13.3 MB artifact, ~2.7 MB headroom)
   - Conservative, safe bet
   - Smaller BPB improvement (~18% more bytes/token vs 35%)
   - Most headroom for other artifact additions
   - Lowest implementation risk

3. **sp8192 + 11L + int8 embed** (~15.4 MB artifact, ~0.6 MB headroom)
   - Aggressive: 53% more bytes/token but much harder prediction
   - 8192 token types with 11 layers
   - Highest reward ceiling but tight on artifact space
   - Risk: both quality and budget concerns

4. **sp4096 + 10L + fp16 embed** (~15.4 MB artifact, ~0.6 MB headroom)
   - Only consider if int8 embedding proves too lossy on 4096 vocab
   - Losing a layer is worse than int8 embed quality loss (probably)
"""

# =========================================================================================
# EXECUTABLE CODE: Download sp4096 data from sproos/parameter-golf-tokenizers
# =========================================================================================

import argparse
import os
import sys
from pathlib import Path


def download_variant(vocab_size: int, train_shards: int = 10):
    """Download tokenizer and data for a given vocab size from sproos HF repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface-hub")
        sys.exit(1)

    REPO = "sproos/parameter-golf-tokenizers"
    variant = f"sp{vocab_size}"
    dest = Path("data")

    print(f"Downloading {variant} tokenizer and data from {REPO}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train shards: {train_shards}")
    print()

    # Download tokenizer files
    for suffix in [".model", ".vocab"]:
        fname = f"tokenizers/fineweb_{vocab_size}_bpe{suffix}"
        print(f"  Downloading {fname}...")
        hf_hub_download(
            repo_id=REPO,
            filename=fname,
            repo_type="model",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

    # Download val shard
    val_fname = f"datasets/fineweb10B_{variant}/fineweb_val_000000.bin"
    print(f"  Downloading {val_fname}...")
    hf_hub_download(
        repo_id=REPO,
        filename=val_fname,
        repo_type="model",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    # Download train shards
    for i in range(train_shards):
        train_fname = f"datasets/fineweb10B_{variant}/fineweb_train_{i:06d}.bin"
        print(f"  Downloading {train_fname}...")
        hf_hub_download(
            repo_id=REPO,
            filename=train_fname,
            repo_type="model",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

    print()
    print(f"Download complete!")
    print(f"  Tokenizer: data/tokenizers/fineweb_{vocab_size}_bpe.model")
    print(f"  Data: data/datasets/fineweb10B_{variant}/")
    print()
    print("To train with this vocab, set these env vars:")
    print(f"  VOCAB_SIZE={vocab_size}")
    print(f"  DATA_PATH=./data/datasets/fineweb10B_{variant}")
    print(f"  TOKENIZER_PATH=./data/tokenizers/fineweb_{vocab_size}_bpe.model")
    print(f"  FP16_EMBED=0  # use int8 embedding to save artifact space")


def verify_tokenizer(vocab_size: int):
    """Verify a downloaded tokenizer works correctly."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: pip install sentencepiece")
        sys.exit(1)

    model_path = f"data/tokenizers/fineweb_{vocab_size}_bpe.model"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run download first.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=model_path)
    actual_vocab = int(sp.vocab_size())
    print(f"Tokenizer: {model_path}")
    print(f"  Vocab size: {actual_vocab}")
    assert actual_vocab == vocab_size, f"Expected {vocab_size}, got {actual_vocab}"

    # Test encoding
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models compress knowledge into parameters.",
        "In 2024, large language models became household tools.",
    ]

    # Compare with baseline if available
    baseline_path = "data/tokenizers/fineweb_1024_bpe.model"
    sp_baseline = None
    if os.path.exists(baseline_path):
        sp_baseline = spm.SentencePieceProcessor(model_file=baseline_path)

    print()
    total_bytes = 0
    total_tokens_new = 0
    total_tokens_base = 0

    for text in test_texts:
        tokens = sp.encode(text)
        text_bytes = len(text.encode("utf-8"))
        bpt = text_bytes / len(tokens)
        total_bytes += text_bytes
        total_tokens_new += len(tokens)

        line = f"  '{text[:50]}...' -> {len(tokens)} tokens, {bpt:.2f} bytes/token"
        if sp_baseline:
            tokens_base = sp_baseline.encode(text)
            bpt_base = text_bytes / len(tokens_base)
            total_tokens_base += len(tokens_base)
            line += f" (baseline: {len(tokens_base)} tokens, {bpt_base:.2f} bpt)"
        print(line)

    if sp_baseline and total_tokens_base > 0:
        avg_bpt_new = total_bytes / total_tokens_new
        avg_bpt_base = total_bytes / total_tokens_base
        improvement = (avg_bpt_new - avg_bpt_base) / avg_bpt_base * 100
        print(f"\n  Average: {avg_bpt_new:.2f} vs baseline {avg_bpt_base:.2f} "
              f"bytes/token ({improvement:+.1f}%)")

    print("\nTokenizer verified OK.")


def estimate_artifact(vocab_size: int, num_layers: int = 11, model_dim: int = 512,
                      fp16_embed: bool = False):
    """Estimate compressed artifact size for a given configuration.

    Uses differentiated compression ratios calibrated from real artifact data:
    - int6 block weights ([-31,31] in int8): ~0.32x with zstd (top bits always 0)
    - int8 block weights ([-127,127] in int8): ~0.65x with zstd (full range)
    - fp16/fp32 passthrough: ~0.84x with zstd
    - Assumes front3_back1_8_middle6 quant preset (blocks 0,1,2,last get int8)

    Calibrated against:
    - Exp #58: 11L sp1024, no preset -> 9.88 MB (predicted: 10.36 MB, +5%)
    - Current SOTA: 11L sp1024, front3_back1 -> ~14 MB (predicted: 13.53 MB, -3%)
    """
    d = model_dim
    n_heads = 8
    n_kv_heads = 4
    head_dim = d // n_heads
    kv_dim = n_kv_heads * head_dim
    mlp_mult = 3
    hidden = mlp_mult * d

    # Compression ratios (calibrated from experiment #58)
    R_INT6 = 0.32    # int6 in int8 with zstd: top 2 bits always 0
    R_INT8 = 0.65    # int8 full range with zstd
    R_FP = 0.84      # fp16/fp32 with zstd

    # Per block raw sizes
    block_2d = d * (d + 2 * kv_dim) + d * d + d * hidden + hidden * d
    block_scales = (d + d + d + hidden) * 2  # fp16 per-row scales
    block_control = (n_heads + d + d + 2 * d) * 4  # fp32 (control tensor patterns)

    # Quant preset: front3_back1_8_middle6 -> blocks 0,1,2,last get int8 range
    last = num_layers - 1
    int8_blocks = {0, 1, 2, last}
    n_int8_blocks = len(int8_blocks)
    n_int6_blocks = num_layers - n_int8_blocks

    # Block 2D data compressed differently based on quant range
    blocks_2d_compressed = (n_int6_blocks * block_2d * R_INT6 +
                            n_int8_blocks * block_2d * R_INT8)
    blocks_scales_compressed = num_layers * block_scales * R_FP
    blocks_control_compressed = num_layers * block_control * R_FP

    # Embedding: fp16 passthrough or int8 quantized
    if fp16_embed:
        embed_compressed = vocab_size * d * 2 * R_FP
    else:
        # int8 quantized: data at int8 range, per-row fp16 scales
        embed_compressed = (vocab_size * d) * R_INT8 + (vocab_size * 2) * R_FP

    # Bigram hash embedding: int8 quantized (full range, >65536 params)
    bigram_compressed = (10240 * 128) * R_INT8 + (10240 * 2) * R_FP
    # Bigram proj: fp16 passthrough (numel=65536, at threshold)
    bigram_proj_compressed = 128 * d * 2 * R_FP

    # Skip weights: fp32, control tensor pattern
    skip_compressed = (num_layers // 2) * d * 4 * R_FP
    # Misc: bigram_scale (fp32) + smear_gate (fp32)
    misc_compressed = (4 + d * 4) * R_FP

    # Total
    torch_overhead = 50_000
    code_bytes = 60_000
    estimated_model = (blocks_2d_compressed + blocks_scales_compressed +
                       blocks_control_compressed + embed_compressed +
                       bigram_compressed + bigram_proj_compressed +
                       skip_compressed + misc_compressed + torch_overhead)
    total_artifact = estimated_model + code_bytes

    # Param count
    embed_params = vocab_size * d  # tied, count once
    block_params = (d * (d + 2 * kv_dim) + d * d + d * hidden + hidden * d +
                    n_heads + d + d + 2 * d) * num_layers
    skip_params = (num_layers // 2) * d
    bigram_params = 10240 * 128 + 128 * d + 1 + d
    total_params = embed_params + block_params + skip_params + bigram_params

    print(f"\nConfiguration: {num_layers}L x {d}d, vocab {vocab_size}, "
          f"{'fp16' if fp16_embed else 'int8'} embed")
    print(f"  Total params: {total_params:,}")
    print(f"  Embedding: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
    print(f"  Estimated model bytes: {estimated_model/1e6:.2f} MB")
    print(f"  Estimated artifact: {total_artifact/1e6:.2f} MB")
    print(f"  Budget remaining: {(16_000_000 - total_artifact)/1e6:.2f} MB")
    print(f"  Status: {'FITS' if total_artifact < 16_000_000 else 'OVER BUDGET'}")

    return total_artifact


def main():
    parser = argparse.ArgumentParser(
        description="Vocabulary size experiment tools for parameter-golf"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Download command
    dl = subparsers.add_parser("download", help="Download tokenizer and data")
    dl.add_argument("--vocab-size", type=int, default=4096,
                    choices=[2048, 4096, 8192])
    dl.add_argument("--train-shards", type=int, default=10,
                    help="Number of train shards to download (max 80)")

    # Verify command
    vf = subparsers.add_parser("verify", help="Verify tokenizer")
    vf.add_argument("--vocab-size", type=int, default=4096,
                    choices=[2048, 4096, 8192])

    # Estimate command
    est = subparsers.add_parser("estimate", help="Estimate artifact sizes")
    est.add_argument("--vocab-size", type=int, default=4096)
    est.add_argument("--num-layers", type=int, default=11)
    est.add_argument("--fp16-embed", action="store_true")

    args = parser.parse_args()

    if args.command == "download":
        download_variant(args.vocab_size, args.train_shards)
    elif args.command == "verify":
        verify_tokenizer(args.vocab_size)
    elif args.command == "estimate":
        estimate_artifact(args.vocab_size, args.num_layers, fp16_embed=args.fp16_embed)
        # Also show comparison configs
        print("\n--- Comparison ---")
        for v, nl, fp16 in [
            (1024, 11, True),   # current baseline
            (2048, 11, True),   # conservative
            (2048, 11, False),  # conservative + int8
            (4096, 11, False),  # recommended
            (4096, 10, True),   # alternative
            (8192, 10, False),  # aggressive
        ]:
            estimate_artifact(v, nl, fp16_embed=fp16)
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("QUICK START")
        print("=" * 70)
        print()
        print("1. Download sp4096 data:")
        print("   python3 experiment_vocab4096.py download --vocab-size 4096")
        print()
        print("2. Verify tokenizer:")
        print("   python3 experiment_vocab4096.py verify --vocab-size 4096")
        print()
        print("3. Estimate artifact sizes:")
        print("   python3 experiment_vocab4096.py estimate --vocab-size 4096")
        print()
        print("4. Make vocab_size configurable in sota_train_gpt.py:")
        print("   Change line 89: vocab_size = 1024")
        print("   To:     vocab_size = int(os.environ.get('VOCAB_SIZE', 1024))")
        print()
        print("5. Run experiment:")
        print("   VOCAB_SIZE=4096 \\")
        print("   DATA_PATH=./data/datasets/fineweb10B_sp4096 \\")
        print("   TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \\")
        print("   FP16_EMBED=0 \\")
        print("   <your usual training command>")


if __name__ == "__main__":
    main()
