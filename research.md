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

## Phase 3: Frontier Ideas — Status & Remaining

### Confirmed / Implemented
- [x] **Muon weight decay 0.04** — confirmed improvement, in our stack
- [x] **SWA / checkpoint averaging** — implemented, but only helps with enough warmdown steps
- [x] **Seq2048** — default, confirmed
- [x] **XSA (last 3-4 layers)** — confirmed ~0.002 BPB gain
- [x] **RoPE base 50K** — confirmed improvement from PR #290

### Tested Negative / Neutral
- [x] **Temperature scaling** — no effect (softcap handles calibration). CONFIRMED.
- [x] **Low-rank Q (512→192→512)** — neutral: QAT overhead cancels matmul savings. CONFIRMED.
- [x] **Seq curriculum (512→2048)** — broken by torch.compile shape recompilation. BLOCKED.
- [x] **EMA (any decay)** — hurts at <1000 steps (CPU overhead + dampens learning). CONFIRMED for short runs.
- [x] **Late QAT (START_FRAC>0)** — worse than always-on. CONFIRMED.
- [x] **NorMuon** — 110ms/step overhead not justified under 600s (from leaderboard intel). SKIP.
- [x] **MTP (multi-token prediction)** — no BPB improvement (PR #212 ablation). SKIP.
- [x] **SwiGLU** — worse than relu² at this scale. CONFIRMED locally.
- [x] **Depth recurrence** — 2x compute, same params, not competitive (PR #103). CONFIRMED locally.

### Untried — High Value (in the bag)
- [ ] **PPM-C context mixing at eval** — blend classical byte-level model with neural probs
  - PR #283 showed ~0.015 BPB on baseline
  - Complex to implement but eval-time only, 100% transferable
  - Neural + classical make different errors → ensemble helps
- [ ] **OptRot pre-quant rotation** (arXiv:2512.24124) — rotate weights before quantizing
  - Redistributes outliers across dims, 30-50% less quant gap
  - Fuses into adjacent layers at eval → zero artifact cost
  - Our quant gap is already tiny (0.003), so expected gain is small
- [ ] **Differential Attention** (arXiv:2410.05258) — attention as difference of two softmaxes
  - 65% fewer params for same quality, reduces activation outliers
  - Requires significant architecture change
  - Estimated -0.005 to -0.015 BPB but high implementation effort
- [ ] **Entropy-regularized QAT** — compression penalty in loss
  - Cluster weights around fewer distinct values for better zstd
  - Simple loss term addition, 100% transferable
  - Estimated savings: 0.5-1.5 MB artifact
- [ ] **Int5 MLP + extra layer** — use int5 savings (~1.86MB) to fund 12th layer
  - Proven trade: int5 costs +0.008 BPB per layer, extra layer gains more
  - Merged SOTA (#180) does exactly this
  - Ready to test: INT5_MLP_ENABLED=1 + NUM_LAYERS=12
- [ ] **Mousse optimizer** (arXiv:2603.09697) — curvature-aware Muon
  - 12% more effective training at 3% overhead
  - Estimated -0.003 to -0.008 BPB
  - Significant code change (optimizer replacement)
- [ ] **Vocab 4096/8192 + SmearGate** — larger vocab for better compression
  - Sacrifice 1 layer to fit embedding
  - PR #78 has custom tokenizer at huggingface
  - Needs re-downloading tokenized data
- [ ] **TTT with optimized hyperparams** — cosine LR, per-layer LRs, freeze first 2 blocks
  - PR #290 config: 3-epoch full-model SGD, lr=0.002, momentum=0.9
  - Our prior TTT test was on undertrained model, retry on properly trained one
- [ ] **Seq curriculum via torch.compile(dynamic=True)** — retry with dynamic shapes
  - The idea is sound, implementation was blocked by compile
  - Need to test if dynamic=True performance is acceptable
- [ ] **Lattice vector quantization** (arXiv:2603.11021) — Leech lattice for groups of 24 weights
  - 15-32% less bitstring waste, saves 2-4 MB
  - Requires custom dequant kernels — high implementation effort

## Current Implemented Stack

- [x] 11L×512d, MLP 3x, GQA (8h/4kv), RoPE, relu², U-Net skips
- [x] SmearGate + BigramHash(10240×128) — zero-init, learnable scale, improved hash
- [x] XSA (Exclusive Self Attention) on last 3-4 layers — zero-param eval gain
- [x] int6 per-row quant + zstd-22 + front-heavy preset (`front2_back1_8_middle6`)
- [x] FP16 tied embedding passthrough (FP16_EMBED=1)
- [x] Sliding window eval (stride=64, compiled forward_logits, batch=64)
- [x] seq2048 training, batch 786K tokens/step (or 131K for proxy)
- [x] Muon WD=0.04, AdamW WD=0.04, grad clip 0.3
- [x] Orthogonal init + 1/sqrt(2L) projection scaling
- [x] SWA (checkpoint averaging during warmdown, SWA_EVERY=200)
- [x] Higher LRs (matrix=0.04, scalar=0.04, embed=0.05)
- [x] QAT always-on (QAT_START_FRAC=0.0 — late QAT hurts, confirmed)
- [x] Flash Attention 3 (conditional import, SDPA fallback)
- [x] TTT (test-time training, full-model SGD on val data)
- [x] Depth recurrence (NUM_RECURRENCE × blocks + optional LoRA adapters)
- [x] EMA weight averaging (implemented, but only helps at 7000+ steps)
- [x] Temperature scaling search (implemented, but T=1.0 optimal due to softcap)
- [x] Mixed int5/int6 quant (INT5_MLP_ENABLED — trade-off: -550KB, +0.008 BPB)
- [x] Low-rank Q factorization (Q_LOW_RANK — neutral on proxy, QAT overhead issue)
- [x] Seq length curriculum (SEQ_CURRICULUM_ENABLED — broken by torch.compile shape change)
- [x] ROPE_BASE=50000 (from PR #290, confirmed improvement)
- [ ] FA3 package installed on cloud
- [ ] Larger vocab (4096/8192)
- [ ] Real 8xH100 SXM submission run (RunPod)

## Competition State (as of 2026-03-21, updated with research agent intel)

**Merged SOTA**: 1.1428 BPB (thwu1 — 10L, mixed int5/int6, BigramHash(10240), SWA, Muon WD)
**Best pending**: 1.1271 BPB (PR #287 — jfprincz, 11L, XSA(4), EMA(0.997), WD=0.04, FA3, 3-seed)
**Best TTT pending**: 1.1354 BPB (PR #290 — XSA + TTT + BatchOpt + RoPE 50K)
**Paid prefix banned**: Organizers ruled it out-of-scope

### The Meta Stack (every top submission uses all of these)
- 11L×512d, MLP 3x, GQA, SmearGate + BigramHash, OrthoInit
- Int6 (or mixed int5/int6) + zstd-22 + FP16 embedding
- Muon WD 0.04, SWA or EMA, sliding window stride=64, FA3
- QAT always-on (late QAT confirmed worse by PR #76 and our testing)

### Frontier Differentiators (what separates top from pack)
- **XSA (last 4 layers)** — ~0.002-0.005 BPB, zero params (PR #265, #287)
- **EMA (decay=0.997)** — replaces SWA, smoother averaging (PR #287, needs 7000+ steps)
- **TTT (full-model SGD, freeze first 2 blocks)** — ~0.01 BPB eval gain (PR #254, #290)
- **Mixed int5/int6** — int5 MLP + int6 attn, saves 1.86MB for more capacity (merged SOTA)
- **RoPE base 50K** — extended positional range (PR #290, confirmed by our testing)
- **Batch 524K** — 22% more gradient updates vs 786K (PR #236)
- **BigramHash(10240)** — larger hash (merged SOTA, confirmed better than 2048 by our testing)

## Cloud Experiment Plan

**Current GPU**: Thunder Compute H100 PCIe at 69.19.136.6:32581
**Launch prefix**: `RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501`

### Next algorithmic experiments (transferable, on proxy)

#### 1. Int5 MLP + extra layer (12L instead of 11L)
```bash
RUN_ID=algo_int5_12L NUM_LAYERS=12 INT5_MLP_ENABLED=1 \
EVAL_STRIDE=64 MUON_WD=0.04 ADAM_WD=0.04 XSA_LAST_N=4 ROPE_BASE=50000 \
MAX_WALLCLOCK_SECONDS=180 python3 sota_train_gpt.py
```

#### 2. PPM-C context mixing (requires implementation)
- Implement classical PPM-C model alongside neural model
- Blend probabilities at eval time

#### 3. Entropy-regularized QAT
```bash
# Add L_entropy = -lambda * sum(softmax(w/T) * log(softmax(w/T))) to loss
# Clusters weights for better zstd compression
```

#### 4. Seq curriculum with dynamic compile
```bash
RUN_ID=algo_seqcurr_dynamic COMPILE_MODE=reduce-overhead \
SEQ_CURRICULUM_ENABLED=1 SEQ_CURRICULUM_START=512 \
# Needs code change: torch.compile(dynamic=True)
```

### Final submission pipeline (RunPod 8xH100 SXM)
```bash
# Best SOTA config with all confirmed improvements:
NUM_LAYERS=11 MUON_WD=0.04 ADAM_WD=0.04 XSA_LAST_N=4 \
ROPE_BASE=50000 EVAL_STRIDE=64 \
QUANT_PRESET=front2_back1_8_middle6 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
```
