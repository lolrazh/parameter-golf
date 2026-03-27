# Diffusion LM — Speed Optimizations, Full Pipeline, and TTT

**Date:** 2026-03-27
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ✅ Completed (ready for H100 proxy run)

Building on `2026-03-27_1500_diffusion-lm-experiment.md`

## User Intention
User wanted to take the diffusion LM built in the previous session and prepare it for an actual H100 run. This meant: (1) identifying and fixing speed bottlenecks that would make the model unrunnable within the 10-minute competition window, (2) validating the complete end-to-end pipeline locally on MLX (train → quantize → dequant roundtrip → eval), and (3) implementing all the eval-time tricks from the GPT pipeline — specifically TTT (test-time training) — so the diffusion model gets the same 3-stage BPB evaluation (pre-quant, post-quant, post-TTT).

## What We Accomplished
- ✅ **Identified 5 critical speed bottlenecks** — Full analysis of train_diffusion.py: eval context window OOM, eval forward pass count (8M FP for 1M val tokens), self-conditioning 50% overhead, wasteful full-sequence CE with block masking, bidirectional 2x FLOPs
- ✅ **Fixed 3 critical config issues** — `SELF_COND_PROB=0` (25% training speedup), `EVAL_T_SAMPLES=32→8` (4x eval speedup), `EVAL_CONTEXT_LEN=0→2048` (OOM prevention)
- ✅ **FLOP breakdown: AdaLN is the dominant overhead** — Quantified per-layer costs. AdaLN conditioning (512→2048 matmul/layer) accounts for 67% of the 55% overhead vs GPT. Bidirectional attention is only 33%. Predicted H100 PCIe proxy step time: 250-350ms.
- ✅ **Fixed stale `_sample_noise` reference** — Warmup code at line 1560 still used old function name. Would have crashed on H100.
- ✅ **Wired up int8+zlib quantization roundtrip in MLX** — GPT-matching pipeline: quantize → pickle → zlib → save → load → decompress → dequantize → reload weights → re-eval. Artifact size: **13.95 MB** (under 16MB limit).
- ✅ **Full pipeline validated on MLX** — Train (50 steps) → pre-quant NELBO (17.81 BPB) → int8+zlib (13.95 MB) → post-quant NELBO (19.56 BPB) → TTT (19.01 BPB). All stages work end-to-end.
- ✅ **Implemented diffusion TTT** — Score-first recipe: process val in chunks, score each chunk with block NELBO, then fine-tune with block diffusion CE loss (same as training loss). SGD with cosine LR across chunks. Freezes first N blocks. Reduces BPB by 0.55 on toy data.
- ✅ **Updated defaults across all 4 files** — `train_diffusion.py`, `train_diffusion_mlx.py`, `run_diffusion.sh`, `DIFFUSION_CONFIG.md` all synced.

## Technical Implementation

### Speed Optimizations Applied
| Change | Impact | Files |
|--------|--------|-------|
| `SELF_COND_PROB=0` | 25% fewer FLOPs in training | All 4 files |
| `EVAL_T_SAMPLES=8` | 4x faster eval | All 4 files |
| `EVAL_CONTEXT_LEN=2048` | Prevents OOM (was unbounded) | All 4 files |

### Quantization Roundtrip (newly wired in MLX)
1. `quantize_state_dict_int8()` — per-row int8 for 2D tensors, per-tensor for others
2. `pickle.dumps()` + `zlib.compress(level=9)`
3. Save to disk as `.int8.ptz`
4. Load → decompress → unpickle → `dequantize_state_dict_int8()`
5. `model.update(tree_unflatten(...))` — reload dequantized weights
6. Re-eval with block NELBO

### TTT Implementation (`eval_block_nelbo_bpb_ttt`)
- **Scoring phase**: Block NELBO on each chunk's blocks (identical to `eval_block_nelbo_bpb`)
- **Training phase**: Reshape chunk into sequences → sample block noise → `nn.value_and_grad` → `mlx.optimizers.SGD` with momentum 0.9
- **LR schedule**: Cosine decay across chunks: `lr * 0.5 * (1 + cos(π * chunk_idx / num_chunks))`
- **Freezing**: `model.blocks[i].freeze()` for first N blocks
- **Legal**: Score-first — every token scored BEFORE any weight update that could use it

### FLOP Analysis (Diffusion vs GPT per layer)
| Component | GPT | Diffusion | Delta |
|-----------|-----|-----------|-------|
| QKV projections | 537M | 537M | 0 |
| Attention (causal vs full) | 537M | 1074M | +537M |
| Output projection | 268M | 268M | 0 |
| MLP (3x) | 1610M | 1610M | 0 |
| **AdaLN** | 0 | **1074M** | **+1074M** |
| **Total** | **2952M** | **4563M** | **+55%** |

**Files Modified:**
- `train_diffusion_mlx.py` — Added TTT hyperparameters, `eval_block_nelbo_bpb_ttt()`, quant roundtrip in `main()`, fixed `_sample_noise` reference, updated defaults
- `train_diffusion.py` — Updated defaults (self_cond_prob=0, eval_t_samples=8, eval_context_len=2048)
- `run_diffusion.sh` — Updated all three config values
- `DIFFUSION_CONFIG.md` — Updated config table to match

## Bugs & Issues Encountered
1. **EVAL_CONTEXT_LEN=0 (unlimited) would OOM on H100** — With 1M val tokens, the last block tries to process the entire val set in one forward pass. Context grows linearly with block position.
   - **Fix:** Cap at `EVAL_CONTEXT_LEN=2048` (match training seq_len). This is the correct setting — model was never trained on longer contexts.

2. **Eval forward pass count was 8M for 1M val tokens** — 250K blocks × 32 MC samples = 8M forward passes ≈ 2.2 hours on 1xH100.
   - **Fix:** Reduced `EVAL_T_SAMPLES` from 32 to 8 (4x speedup, unbiased MC — just noisier). Context cap also helps by bounding per-block sequence length.

3. **Stale `_sample_noise` reference in warmup code** — Line 1560 still used old function name after rename to `_sample_noise_uniform`.
   - **Fix:** Updated to `_sample_noise_uniform`.

4. **MLX `model.trainable_parameters()` returns nested dict, not flat tuples** — TTT code tried `for _, p in model.trainable_parameters()` which fails with "too many values to unpack".
   - **Fix:** Wrap with `tree_flatten()`: `for _, p in tree_flatten(model.trainable_parameters())`

5. **Previous TTT test crashed silently** — Grep pipe in command ate the Python traceback. Process appeared to die without error.
   - **Fix:** Ran without grep pipe to capture full stderr. Found the `trainable_parameters()` bug above.

## Key Learnings
- **AdaLN is the dominant overhead, not bidirectional attention** — The `[512→2048]` conditioning matmul per layer costs 2x more than doubling the attention matrix. If squeezing more steps, AdaLN is the optimization target (share across layers, reduce outputs from 4 to 2).
- **EVAL_CONTEXT_LEN must match train seq_len** — Unlimited context is NOT better. The model has never seen sequences longer than `train_seq_len`. RoPE encodings, attention patterns, and AdaLN conditioning are all untrained for longer contexts. Capping at 2048 is architecturally correct.
- **TTT transfers directly from AR to diffusion** — Despite the agent's initial assessment that TTT is "fundamentally incompatible" with diffusion, it works identically. The only change: use block diffusion CE loss instead of next-token CE loss for fine-tuning. Scoring uses block NELBO. Same score-first recipe.
- **N-gram cache is feasible but harder** — During block NELBO, the model's logits ARE probability distributions over the vocabulary (conditioned on noise level). Blending with n-gram probabilities is mathematically valid but the benefit is unclear for diffusion where logits vary by noise level.
- **MLX eval on 4K tokens takes ~5 minutes per NELBO pass** — Block NELBO is inherently expensive: 1024 blocks × 8 MC samples × ~42ms per forward pass. On H100 this should be ~100x faster.

## Architecture Decisions
- **SELF_COND_PROB=0 over 0.5** — At the current stage, training step count matters more than per-step quality. Self-conditioning adds ~25% overhead (50% chance of 2 forward passes). Re-enable once step budget is confirmed on H100.
- **EVAL_T_SAMPLES=8 over 32** — 4x fewer MC samples is unbiased (same expected value), just noisier. At eval time on H100, the compute savings far outweigh the estimation noise. Can increase for final submission runs.
- **MLX uses int8+zlib, PyTorch uses mixed int6+lzma** — Different quant recipes because the PyTorch script has the more sophisticated `mixed_quantize_int6` with per-layer sensitivity analysis. MLX int8 is sufficient for local testing.
- **TTT uses SGD (not Muon/Adam)** — Matching the GPT TTT recipe. SGD with momentum is simpler, stable for fine-tuning, and doesn't require maintaining optimizer state across chunks.

## Experiment Results

| # | Run ID | Pipeline Stage | BPB | Notes |
|---|--------|---------------|-----|-------|
| d7 | diffusion_full_pipeline | Pre-quant NELBO | 17.92 | 50 steps, 4K batch |
| d7 | diffusion_full_pipeline | Post-quant NELBO | 19.56 | +1.65 BPB quant cost (toy scale) |
| d8 | diffusion_ttt_v2 | Pre-quant NELBO | 17.81 | 50 steps, 4K batch |
| d8 | diffusion_ttt_v2 | Post-quant NELBO | 19.56 | +1.75 BPB quant cost |
| d8 | diffusion_ttt_v2 | Post-TTT NELBO | **19.01** | **-0.55 BPB from TTT** |

## Ready for Next Session
- ✅ **Full pipeline validated** — Train → EMA/SWA → pre-quant NELBO → int8+zlib → dequant → post-quant NELBO → TTT → post-TTT NELBO
- ✅ **Artifact fits** — 13.95 MB (under 16MB limit)
- ✅ **Speed optimizations applied** — Self-cond off, eval_t_samples=8, context_len=2048
- ✅ **TTT works for diffusion** — 0.55 BPB improvement on toy data
- ✅ **run_diffusion.sh ready** — Proxy (1xH100) and prod (8xH100) configs
- 🔧 **H100 proxy smoke test** — `./run_diffusion.sh --proxy` on 1xH100
- 🔧 **Sparse output head** — Only compute logits for block positions (99.8% savings on output head during training). Not implemented yet.
- 🔧 **Batched MC samples in eval** — Process multiple noise levels in one forward pass. Not implemented yet.
- 🔧 **N-gram cache for diffusion** — Feasible but not implemented.
- 🔧 **Quant recipe investigation** — Current int8 may not be optimal. Mixed int6/int5 with layer-sensitivity analysis needs testing on diffusion architecture.

## Context for Future
This session bridged the gap from "diffusion model works on MLX" to "full competition pipeline validated end-to-end." The BPB numbers (17-19) are meaningless at this scale — they're from 50 steps on 4K batch. BD3-LM shows diffusion converges to within ~18% of AR at scale, targeting ~1.4 BPB on 8xH100. The key finding is that AdaLN is the dominant speed overhead (67% of the extra cost vs GPT), and TTT transfers directly from AR to diffusion with a simple loss function swap. Next step: spin up H100 and run `./run_diffusion.sh --proxy` for the first real BPB numbers.
