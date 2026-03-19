# Research Ideas

## Phase 1: Architecture (DONE — local M4)
- [x] LR sweep → 3x best locally
- [x] Width vs depth → 6L×640d best (confirmed on H100)
- [x] SwiGLU → worse than relu² at this scale
- [x] MLP 3x (without int6) → no improvement (but WORKS with int6 per competitive intel)
- [x] Value embeddings → noise locally (needs bigger batches)
- [x] Depth recurrence → worse (2x compute cost)
- [x] LAWA → noise at 200 steps
- [x] Smear module → noise
- [x] Softcap 15 → noise
- [x] Drop first MLP → noise
- [x] Grad clip, Muon momentum, head count, LR retune → all noise locally

## Phase 2: Proven Competition Techniques (implement in train_gpt.py, test on Modal)

### Tier 1: Free / near-free wins
- [ ] **Sliding window eval (stride=64)** — -0.034 BPB, ZERO artifact cost, ZERO training change
  - Overlap eval windows so each scored token gets ~960 context
  - PR #50 proved it, PR #65 optimized stride=64
  - Implement in train_gpt.py eval loop
- [ ] **Muon tuning** — -0.005 BPB, env vars only
  - MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
  - MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500
  - WARMDOWN_ITERS=3000, GRAD_CLIP_NORM=0.3
  - All from PR #70 and #61
- [ ] **Train at seq2048** — PR #61 proved 2048 > 4096 (more steps, same BPB with sliding window)
  - TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=786432

### Tier 2: Quantization (unlocks bigger models)
- [ ] **int6 per-row quantization** — saves ~4MB artifact space, only +0.01 BPB without QAT
  - 31-level per-row quantization on block weights
  - Keep embeddings at int8 or fp16
  - zstd-22 compression instead of zlib
- [ ] **QAT / STE fake-quant** — reduces int6 penalty from +0.01 to +0.001 BPB
  - Add fake quantization noise in forward pass during training
  - Straight-through estimator for gradients
  - PR #65 demonstrated this
- [ ] **fp16 tied embedding passthrough** — -0.007 BPB
  - Don't quantize the embedding — keep at fp16
  - Costs ~523KB but embedding is most quant-sensitive

### Tier 3: Architecture (enabled by int6 space savings)
- [ ] **MLP 3x** — -0.019 BPB (now fits in 16MB with int6)
- [ ] **10 layers** — extra layer for capacity (PR #39, #64)
  - Middle layers at int6, outer at int8
- [ ] **Vocab 8192** — PR #78, custom tokenizer available at huggingface.com/sproos/parameter-golf-tokenizers
  - Sacrifice 1 layer (9→8) to fit embedding
  - NorMuon optimizer
  - Tokenized data needs re-download

### Tier 4: Eval tricks (stack on top)
- [ ] **LoRA test-time training** — -0.004 BPB on top of sliding window
  - Rank-8 LoRA on lm_head, c_q, c_v
  - Train per-document at eval, reset between docs
  - Uses ~1/10 eval budget
- [ ] **Document-isolated eval** — -0.011 BPB
  - Don't let attention leak across document boundaries
  - PR #77 showed this

## Winning PR Configs (for reference)

### PR #65 (SOTA, 1.1630 BPB)
```
MLP_MULT=3, QAT int6, sliding window stride=64
seq_len=1024, batch=524K, 12395 steps, ~48ms/step on 8xH100
```

### PR #70 (1.1659 BPB)
```
MATRIX_LR=0.020 SCALAR_LR=0.020 TIED_EMBED_LR=0.030
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000 MLP_MULT=3 int6+zstd-22, sliding window stride=256
```

### PR #61 (1.1793 BPB — key insight: train 2048 not 4096)
```
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
sliding window stride=512
```

## Cloud Experiment Plan

**Platform**: Modal (L4 ~$0.02/run, H100 ~$0.15/run)

**Test durations on 1xH100**:
- Full 10 min: ~1700 steps, ~$0.55 (final validation only)
- 5 min: ~850 steps, ~$0.27 (good signal)
- 2 min: ~340 steps, ~$0.11 (quick comparison)

**Implementation order**:
1. Sliding window eval → test on H100 2-min
2. Muon tuning (env vars) → test on H100 2-min
3. Stack sliding window + Muon → test on H100 5-min
4. int6 quant + zstd → test artifact size
5. QAT → test quant penalty
6. MLP 3x (enabled by int6) → test on H100 5-min
7. Full stack → H100 10-min submission run
