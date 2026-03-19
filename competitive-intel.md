# Competitive Intelligence (from issue #83 + top PRs)

Current SOTA (standard training): **1.1630 BPB** (PR #65)
Baseline: **1.2244 BPB**

## The Winning Stack (what top competitors all use)

### 1. Sliding Window Eval (~-0.034 BPB) — FREE, ZERO ARTIFACT COST
- Overlapping eval windows: instead of evaluating each 1024-token chunk independently, slide a window with stride=64
- Each scored token gets ~960 tokens of context instead of 0-1023 average
- PR #50 first demonstrated this. PR #65 uses stride=64 (best). PR #70 uses stride=256.
- **This alone beats the baseline** (1.2244 → ~1.192)
- Takes ~70s on 8xH100 (within 10-min eval budget)

### 2. MLP 3x Expansion (~-0.019 BPB)
- Hidden dim from 1024 (2x) to 1536 (3x)
- Adds capacity but makes model bigger — enabled by int6 quantization freeing ~4MB
- PR #65, #66, #70 all use this

### 3. int6 Per-Row Quantization (saves ~4MB, only +0.001-0.010 BPB)
- 31-level per-row quantization on block weights (MLP + attention)
- Much more aggressive than baseline int8 (127 levels)
- Only +0.001 BPB penalty with STE fake-quant (QAT)!
- Embeddings kept at int8 or fp16 (most quant-sensitive)
- zstd-22 compression instead of zlib (better ratio on int6)
- The 4MB savings is what ENABLES MLP 3x — without it, bigger model doesn't fit in 16MB

### 4. fp16 Tied Embedding (~-0.007 BPB)
- Keep embedding/output head at fp16 instead of quantizing
- Most quant-sensitive tensor — worth the ~523KB cost
- PR #42 first showed this

### 5. Longer Context Training (seq4096, ~-0.01 BPB)
- Train at 4096 instead of 1024 — matches sliding window eval distribution
- Costs ~64ms/step vs ~48ms (fewer steps in 10 min), but quality compensates
- PR #61, #66, #75 use this

### 6. Muon Momentum=0.99 + Lower LR (~-0.005 BPB)
- MATRIX_LR=0.020, SCALAR_LR=0.020, TIED_EMBED_LR=0.030
- MUON_MOMENTUM=0.99 (vs 0.95 baseline)
- MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500
- WARMDOWN_ITERS=3000
- Smoother optimization reduces quantization gap

### 7. Vocab 8192 + Drop 1 Layer (PR #78, BPB=1.186)
- Custom 8192-token SentencePiece tokenizer
- Sacrifice 1 layer (9→8) to fit larger embedding table
- seq_len=4096
- NorMuon optimizer
- int6 weights, int8 embeddings
- Tokenizers publicly available: huggingface.co/sproos/parameter-golf-tokenizers

### 8. LoRA Test-Time Training (PR #77, ~-0.004 BPB on top)
- Rank-8 LoRA adapters trained per-document at eval time
- Score chunk, then train LoRA on that chunk (no leakage)
- Reset between documents
- Uses ~1/10 eval budget
- Small but real improvement on top of sliding window

## The Recipe (from PR #65, current #1 standard)
```
Training:
  MLP_MULT=3, seq_len=1024, batch=524K, 12395 steps in 10 min
  + STE fake-quant during training (QAT for int6)

Quantization:
  int6 per-row on blocks (31 levels)
  int8 per-row on embeddings
  zstd-22 compression

Evaluation:
  Sliding window stride=64
  Each token gets ~960 context tokens
```

## What We Should Implement (priority order)
1. **Sliding window eval** — instant -0.034 BPB, zero training change
2. **int6 quantization + zstd** — unlocks MLP 3x by freeing 4MB
3. **MLP 3x** — once int6 frees space
4. **QAT (STE fake-quant)** — makes int6 penalty near-zero
5. **Longer training context (seq4096)** — matches eval distribution
6. **Muon tuning** — momentum=0.99, lower LR, longer warmdown
7. **Bigger vocab** — 8192 tokens (if tokenizer available)
