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

## Current Implemented Stack

- [x] 11L×512d, MLP 3x, GQA (8h/4kv), RoPE, relu², U-Net skips
- [x] SmearGate + BigramHash(10240×128) — zero-init, learnable scale, improved hash
- [x] XSA (Exclusive Self Attention) on last 3 layers — zero-param eval gain
- [x] int6 per-row quant + zstd-22 + front-heavy preset (`front3_back1_8_middle6`)
- [x] FP16 tied embedding passthrough (FP16_EMBED=1)
- [x] Sliding window eval (stride=64, compiled forward_logits, batch=64)
- [x] seq2048 training, batch 786K tokens/step
- [x] Muon WD=0.02, AdamW WD=0.01, grad clip 0.3
- [x] Orthogonal init + 1/sqrt(2L) projection scaling
- [x] SWA (checkpoint averaging during warmdown, SWA_EVERY=50)
- [x] Higher LRs (matrix=0.04, scalar=0.04, embed=0.05)
- [x] Late QAT toggle (QAT_START_FRAC, default 0.0 = always-on STE)
- [x] Flash Attention 3 (conditional import, SDPA fallback)
- [x] TTT (test-time training, full-model SGD on val data)
- [x] Depth recurrence (NUM_RECURRENCE × blocks + optional LoRA adapters)
- [ ] FA3 package installed on cloud (compiling on A6000)
- [ ] Mixed int5/int6 quant (int5 MLP, int6 attn — merged SOTA uses this)
- [ ] Larger vocab (4096/8192)
- [ ] Real 8xH100 SXM submission run (RunPod)

## Competition State (as of 2026-03-21)

**Merged SOTA**: 1.1428 BPB (thwu1 — 10L, mixed int5/int6, BigramHash(10240), SWA, Muon WD)
**Best pending**: 1.1303 BPB (PR #254 — TTT + full meta stack, 3-seed validated)
**Paid prefix banned**: Organizers ruled it out-of-scope

### The Meta Stack (every top submission uses all of these)
- 11L×512d, MLP 3x, GQA, SmearGate + BigramHash, OrthoInit
- Int6 (or mixed int5/int6) + zstd-22 + FP16 embedding
- Muon WD 0.03-0.04, SWA, sliding window stride=64, FA3

### Frontier Differentiators (what separates top from pack)
- **TTT (full-weight SGD)** — ~0.01 BPB eval-time gain (PR #254)
- **XSA (last 3 layers)** — ~0.002 BPB, zero params (PR #265)
- **Mixed int5/int6** — int5 MLP + int6 attn, fits more layers (merged SOTA)
- **BigramHash(10240)** — larger hash table (merged SOTA)
- **Depth recurrence** — 5 unique blocks looped for 11 effective depth (PR #268, pending)
- **EMA weight averaging** — untapped by most submissions (PR #274)

## Cloud Experiment Plan

**Budget**: ~$10 Modal remaining, considering Latitude/ThunderCompute for dedicated GPU.

**Run queue** (ordered):

**Platform**: Thunder Compute A100 production ($1.79/hr). Final submission on RunPod 8xH100 SXM.
**Launch**: `RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501`

### 1. Full-stack 5-min baseline with XSA + BigramHash(10240)
```bash
RUN_ID=xsa_bigram10k_5m NUM_LAYERS=11 EVAL_STRIDE=64 \
QUANT_PRESET=front3_back1_8_middle6 TTT_ENABLED=0 SWA_EVERY=50 \
MAX_WALLCLOCK_SECONDS=300 python3 sota_train_gpt.py
```

### 2. Depth recurrence: 5 blocks × 3 loops = 15 virtual layers
```bash
RUN_ID=recur_5x3_lora8 NUM_LAYERS=5 NUM_RECURRENCE=3 LORA_RANK=8 \
EVAL_STRIDE=64 QUANT_PRESET=front3_back1_8_middle6 SWA_EVERY=50 \
MAX_WALLCLOCK_SECONDS=300 python3 sota_train_gpt.py
```

### 3. Late QAT (70% activation) vs always-on
```bash
RUN_ID=lateqat_70_5m NUM_LAYERS=11 QAT_START_FRAC=0.7 \
EVAL_STRIDE=64 QUANT_PRESET=front3_back1_8_middle6 SWA_EVERY=50 \
MAX_WALLCLOCK_SECONDS=300 python3 sota_train_gpt.py
```

### 4. TTT on properly trained model (10-min train first)
```bash
RUN_ID=ttt_tuned_10m NUM_LAYERS=11 EVAL_STRIDE=64 \
QUANT_PRESET=front3_back1_8_middle6 TTT_ENABLED=1 TTT_LR=3e-4 \
TTT_MAX_SECONDS=200 SWA_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
python3 sota_train_gpt.py
```

### 5. Final submission (RunPod 8xH100 SXM)
```bash
torchrun --standalone --nproc_per_node=8 sota_train_gpt.py
# Best config from experiments above
```
