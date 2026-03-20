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

## Phase 2: Proven Competition Techniques (Modal / SOTA fork status)

### Tier 1: Free / near-free wins
- [x] **Sliding window eval (stride=64)** — implemented in `sota_train_gpt.py`
  - Overlap eval windows so each scored token gets ~960 context
  - PR #50 proved it, PR #65 optimized stride=64
- [~] **Muon tuning** — mostly aligned, still missing some frontier details
  - MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
  - MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500
  - WARMDOWN_ITERS=3000 is in
  - `GRAD_CLIP_NORM=0.3` not yet adopted in the SOTA fork
  - All from PR #70 and #61
- [ ] **Train at seq2048** — PR #61 proved 2048 > 4096 (more steps, same BPB with sliding window)
  - TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=786432

### Tier 2: Quantization (unlocks bigger models)
- [x] **int6 per-row quantization + zstd** — implemented
  - 31-level per-row quantization on block weights
  - Keep embeddings at int8 or fp16
  - zstd-22 compression instead of zlib
- [x] **Selective precision / front-heavy mixed quant** — implemented and validated
  - `QUANT_PRESET=front3_back1_8_middle6` is the current best integrated export path
  - Real 5-min `9x512` result: pre `1.4444`, post `1.7629`, gap `+0.3185`
- [ ] **Late QAT / STE fake-quant schedule** — frontier version still missing
  - Add fake quantization noise in forward pass during training
  - Straight-through estimator for gradients
  - New frontier consensus is late activation (~70-85% of wallclock), not full-run QAT
- [ ] **fp16 tied embedding passthrough (retest on current stack)** — still worth one revisit
  - Don't quantize the embedding — keep at fp16
  - Costs ~523KB but embedding is most quant-sensitive

### Tier 3: Architecture (enabled by int6 space savings)
- [x] **MLP 3x** — already the default in the SOTA fork
- [x] **SmearGate + BigramHash** — implemented
  - Current cheap baseline signal: pre slightly worse, post-quant better (`1.7629 → 1.7074`)
- [ ] **10L / 11L on the new stack** — retry only after stronger training recipe
  - Naive `11x640` at 5 min / 1xH100 undertrained badly despite fitting the artifact budget
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

## Phase 3: Missed Frontier Ideas (from live issue #140)

- [ ] **Muon weight decay**
  - Repeatedly cited as part of the official merged leader lineage
  - Distinct from our current Muon settings; still not in the fork
- [ ] **SWA / checkpoint averaging**
  - Top SmearGate submissions use SWA-like averaging late in training
  - Cheap enough to matter, not yet tried here
- [ ] **Seq2048 on the current stack**
  - Real frontier consensus now leans `2048` as the honest next long-context baseline
  - Should be tested with: current quant preset + SmearGate + sliding eval
- [ ] **Overtone init + residual-mix init**
  - Early official leader lineage used this as a real foundation, not a random trick
  - We have not implemented it in the SOTA fork
- [ ] **NorMuon**
  - Appears in several strong PRs
  - Especially relevant if we revisit larger vocabs or longer context
- [ ] **Vocab 4096 / 8192 + SmearGate**
  - Larger vocab submissions have not been combined with our current stack here
- [ ] **Late-K selective precision**
  - Keep `c_k` in the last layers higher precision
  - Related to our front-heavy preset, but not the same idea
- [ ] **MTP (multi-token prediction)**
  - Auxiliary heads only during training
  - Potential sample-efficiency win without inference artifact cost
- [ ] **TTT on top of the real stack**
  - Expensive and not first priority, but still one of the few ideas with large upside left

## Current Fast-Honest Stack

- [x] `9x512`
- [x] `MLP_MULT=3`
- [x] int6 per-row quant + zstd-22
- [x] front-heavy selective precision (`front3_back1_8_middle6`)
- [x] SmearGate + BigramHash (improved: zero-init, learnable scale, better hash)
- [x] sliding-window eval implementation available
- [x] seq2048 (default in sota_train_gpt.py)
- [x] Muon weight decay (muon_wd=0.02, adam_wd=0.01)
- [x] SWA (checkpoint averaging during warmdown)
- [x] Orthogonal init + projection scaling
- [x] Grad clipping 0.3
- [x] Higher LRs (matrix=0.04, scalar=0.04, embed=0.05)
- [x] Aggressive Muon momentum (0.95, warmup from 0.85 over 500 steps)
- [ ] late QAT (always-on STE is current approach — late activation not yet tried)
- [ ] Flash Attention 3 (H100-only, ~10-20% throughput gain)
- [ ] FP16 tied embedding passthrough (not yet integrated)
- [ ] real frontier-sized 8xH100 run

## Winning PR Configs (for reference)

### Best validated pending (1.1326 BPB, 3-seed mean 1.1326)
```
NUM_LAYERS=11, 512d, MLP_MULT=3, seq2048, batch=786K
SmearGate + BigramHash(2048x128), OrthoInit
MUON_WD=0.04, ADAM_WD=0.04, GRAD_CLIP=0.3
MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 (warmup 0.92→0.99 over 1500), WARMDOWN=3000
SWA every 200 steps during warmdown, int6+zstd-22, FA3
sliding window stride=64, 7412 steps @ 81ms/step on 8xH100
```

### PR #65 (former SOTA, 1.1630 BPB)
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
