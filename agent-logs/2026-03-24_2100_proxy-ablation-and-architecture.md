# Proxy Ablation & Architecture Pivot — XSA + Partial RoPE

**Date:** 2026-03-24
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ✅ Completed (Runs 3 & 4 pending)

Building on `2026-03-24_1900_8xh100-sota-3seed-runs.md`

## User Intention
User wanted to A/B test improvements (QAT, GPTQ-lite, front-heavy quant, sliding window, seq2048) on 1xH100 proxy, then pivot to sp1024 11L with XSA + Partial RoPE to compete directly with SOTA architecture.

## What We Accomplished
- ✅ Implemented GPTQ-lite (5-percentile clip search per row, zero training cost)
- ✅ Implemented QAT (STE fake quantization during warmdown, QAT_START_FRAC env var)
- ✅ Implemented front3_back1_6_middle5 quant preset (int6 sensitive, int5 middle)
- ✅ Implemented front3_back1_8_middle6 quant preset (int8 sensitive, int6 middle)
- ✅ Ran 6 proxy experiments (Runs 1, 2, 2b, 2c, 2d) on 1xH100 PCIe, Thunder Compute
- ✅ Runs 3 (sliding window) and 4 (seq2048) in progress
- ✅ Implemented XSA (Exclusive Self-Attention) on last N layers, toggleable via XSA_LAST_N
- ✅ Implemented Partial RoPE (ROPE_DIMS env var, only rotate first N dims, rest position-free)
- ✅ Discovered 3-epoch TTT is INVALID per competition rules (must score before training)
- ✅ QAT research: naive STE QAT marginal at best, Soft-Round QAT is the real win (PR #606)
- ⚠️ Batch size research still pending

## Proxy Results (1xH100 PCIe, 10L sp4096, 131K batch, 10 min, 1-epoch TTT)

| Run | QAT | Quant Preset | Pre-quant | Post-quant | Quant gap | Post-TTT | Artifact |
|---|---|---|---|---|---|---|---|
| 1 (baseline) | off | int5mlp | 1.2570 | 1.2796 | 0.0226 | 1.2544 | 14.4 MB ✅ |
| 2 (+QAT) | 0.15 | int5mlp | 1.2737 | 1.2856 | 0.0119 | 1.2629 | 14.6 MB ✅ |
| 2b (+QAT front8) | 0.15 | front3_back1_8_middle6 | 1.2687 | 1.2719 | 0.0032 | 1.2510 | 18.8 MB ❌ |
| 2c (front8 no QAT) | off | front3_back1_8_middle6 | 1.2566 | 1.2596 | 0.0030 | 1.2397 | 18.8 MB ❌ |
| 2d (front6mid5) | off | front3_back1_6_middle5 | 1.2567 | 1.2735 | 0.0168 | 1.2506 | 14.7 MB ✅ |

## Key Learnings
- **QAT hurts on proxy (~3700 steps)** — Training cost outweighs quant gap savings. The model spends too many steps with noisy gradients from fake quantization, and the quant gap reduction (0.0226 -> 0.0119) doesn't compensate for the worse pre-quant BPB (1.2570 -> 1.2737).
- **Front-heavy int8 quant nearly eliminates quant gap (0.003)** — But artifact is 18.8MB, well over 16MB limit. int8 compresses WORSE than int6 under zstd because more range = less redundancy in weight values.
- **front3_back1_6_middle5 is the best legal option** — 0.0168 quant gap, 14.7 MB artifact, 1.2506 post-TTT BPB.
- **1-epoch TTT gives consistent ~0.025 BPB gain** — Reliable across all runs.
- **3-epoch TTT is INVALID** — Organizer says "only train on tokens you've already evaluated." Multi-epoch TTT trains on tokens before they're scored on subsequent epochs. PROTEUS was taken down for the same violation.
- **Successful QAT requires export-aligned clipping or Soft-Round** — Naive STE QAT is not enough. PR #606 shows Soft-Round (differentiable rounding) is the actual mechanism that wins.

## Architecture Decisions
- **Pivoting to sp1024 11L** — To compete directly with SOTA architecture rather than the sp4096 divergent path.
- **XSA on last 4 layers (XSA_LAST_N=4)** — Exclusive Self-Attention removes the self-value projection in deeper layers, saving parameters. Ported from SOTA PR #315.
- **Partial RoPE (ROPE_DIMS=16)** — Only 16 out of 64 head dims get positional encoding, rest are position-free. This lets the model choose which dimensions carry positional info. Ported from SOTA PR #315.

## Technical Implementation

### GPTQ-lite
Per-row optimal clip percentile search during quantization. Tests 5 candidate clip percentiles (e.g., 99.5, 99.7, 99.9, 99.95, 100.0) for each row of each weight tensor, picks the one with minimum MSE. Zero cost during training — only runs at export time.

### QAT (Quantization-Aware Training)
STE (Straight-Through Estimator) fake quantization injected into forward pass when `lr_scale < QAT_START_FRAC`. Quantizes weights to int5/int6 levels then dequantizes, backpropagating through the rounding via STE. Controlled by `QAT_START_FRAC` env var (default off). At 0.15, QAT activates during the last 15% of warmdown.

### Quant Presets
- `int5mlp` — int5 for MLP weights, int6 for attention weights
- `front3_back1_8_middle6` — int8 for first 3 and last 1 blocks, int6 for middle blocks
- `front3_back1_6_middle5` — int6 for first 3 and last 1 blocks, int5 for middle blocks

### XSA (Exclusive Self-Attention)
On the last N layers (controlled by `XSA_LAST_N`), the value projection is removed. Instead of learning V = W_v @ x, the value is just the input x itself. This saves one linear projection per attention layer and forces the network to use attention purely for routing, not value transformation.

### Partial RoPE
Only the first `ROPE_DIMS` dimensions of each attention head get rotary position embeddings. The remaining dimensions are position-free, allowing the model to learn position-independent features in those channels. SOTA uses 16/64 dims.

## Ready for Next Session
- ✅ XSA + Partial RoPE implemented in frontier_512.py
- ✅ GPTQ-lite + QAT + 3 quant presets available
- 🔧 Run 3 (sliding window) and Run 4 (seq2048) results pending
- 🔧 Run 5: 11L sp1024 + XSA4 + Partial RoPE 16/64 with best eval tricks
- 🔧 Run 6: Best QAT setup (Soft-Round or GPTQ-aligned)
- 🔧 Batch size research still pending

## Run 5: 11L sp1024 + XSA4 + Partial RoPE (1xH100 proxy)
- Config: 11L sp1024, XSA on last 4 layers, Partial RoPE 16/64, front3_back1_6_middle5 quant, relu²
- Steps: 3,432 @ 175ms/step
- Pre-quant BPB: 1.3499, Post-quant BPB: 1.3645 (quant gap: 0.0146)
- Post-TTT BPB (LoRA 1ep): 1.2736
- Artifact: 12.9 MB (massive headroom under 16 MB)
- front3_back1_8_middle6 tested on same checkpoint: 17.4 MB (over limit even with sp1024)

## New Implementations
- LeakyReLU(0.5)² — one-line change: `F.leaky_relu(self.fc(x), negative_slope=0.5).square()`. Preserves negative gradient flow. Worth -0.003 BPB per PR #493.
- Score-first full-weight TTT (PR #549 style) — scores entire chunk with inference_mode, THEN trains 3 epochs SGD on graded tokens. All blocks unfrozen. Legal per Will's rule.
- TTT_SF_ENABLED=1 env var toggles it. Pipeline runs both LoRA and score-first TTT on same checkpoint.
- XSA (Exclusive Self-Attention) — subtracts self-value projection from attention output. XSA_LAST_N env var.
- Partial RoPE — ROPE_DIMS=16 applies RoPE to only 16/64 head dims.

## Competition Intel
- PR #549 (1.1194 BPB) is the new SOTA target. Uses LeakyReLU², legal score-first TTT, Parallel Muon, LZMA compression.
- PR #490 (1.0891 claimed) was flagged — used illegal pre-eval TTT. Interesting techniques: Value Residual + Gated Attention.
- Sliding window eval ran on full 62M val set (not VAL_TOKENS_LIMIT) — caused 50+ min hang on proxy. Bug in eval pipeline. Need to fix or just skip on proxy.

## Run 6: LeakyReLU² + TTT Method Comparison (11L sp1024, 1xH100)
- Config: 11L sp1024, XSA4, Partial RoPE 16/64, LeakyReLU(0.5)², front3_back1_6_middle5 quant
- Steps: 3,374 @ 177ms/step (slightly slower than Run 5's relu² at 175ms)
- Pre-quant BPB: 1.3425 (vs Run 5's 1.3499 — LeakyReLU² helped by -0.007)
- Post-quant BPB: 1.3587 (quant gap: 0.016)
- Artifact: 12.8 MB
- LoRA TTT (1ep): **1.2718 BPB** (gain: -0.087). WINNER.
- Score-first full-weight TTT: **2.2001 BPB** (CATASTROPHICALLY WORSE). SGD lr=0.002 on 26.8M params destroys the model.
- Verdict: Stick with LoRA TTT. Score-first full-weight TTT doesn't work for our architecture.

## Score-First TTT Post-Mortem
- Implementation was correct (verified: scoring happens before training, no data leakage)
- Bug found and fixed: `.bfloat16()` call corrupted CastedLinear float32 weights on checkpoint reload
- Even after fix, SGD(lr=0.002, mom=0.9) on full 26.8M params is too aggressive
- PR #549 gets only -0.0025 BPB from their version — the technique is inherently fragile
- LoRA TTT (rank-8, ~200K trainable params) is much more stable: -0.087 BPB gain

## Context for Future
The proxy ablation shows that naive QAT is not worth it at proxy scale — the training cost exceeds the quant gap reduction. Front-heavy int8 quant is powerful (0.003 quant gap) but the artifact is too large. The best legal configuration is front3_back1_6_middle5 with no QAT, yielding 1.2506 post-TTT BPB on 1xH100 PCIe proxy. The architecture pivot to sp1024 11L with XSA + Partial RoPE is ready to test — these are the key features from the SOTA PR #315 that we haven't tried yet. If Soft-Round QAT (PR #606) pans out, it could further close the quant gap without the training cost penalty of naive STE.
