# Preset-Aligned QAT, Muon Fixes, Neural Cache Implementation

**Date:** 2026-03-21
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ⚠️ Partial — neural cache too slow, needs vectorization
**Building on:** `2026-03-21_0400_algorithmic-experiments.md`

## User Intention
User wanted to execute the structured experiment queue from `gpu_queue_overlooked.md` — starting with Branch A (preset-aligned QAT), Branch D (token-class calibration), and Branch C (neural cache). Focus shifted to implementing novel algorithmic changes that transfer to 8xH100 competition hardware, not hyperparameter tuning.

## What We Accomplished
- ✅ **Preset-aligned QAT (Branch A1)** — STE fake-quant now matches export quant levels per-layer. 12 CastedLinear layers marked int8 for boundary blocks. Result: -0.0014 BPB improvement (1.3895 → 1.3881). Transferable.
- ✅ **Batched Muon bug fixes** — Fixed CUDA graph double-update NaN bug (graph capture step ran _batched_step AND _graph_step_impl). Fixed `batched_newton_schulz` global scope error. Batched Muon without CUDA graph works: 133ms vs 136ms baseline.
- ✅ **Token-class calibration (Branch D1)** — Implemented and tested. Found per-class temp differences (1-byte: T=0.90, 3+ byte: T=1.06) but no overall BPB improvement. Per-class signals cancel when weighted by byte contribution.
- ⚠️ **Neural cache implementation** — `eval_val_with_cache()` and `forward_hidden_and_logits()` added. Code works (checkpoint loads, model runs) but eval is impractical: per-token sequential loop over 62M val tokens is too slow. Also hit nohup output buffering issues (zero-byte output files). Needs vectorized implementation to be usable.
- ✅ **Eval-only script** — `eval_cache.py` created for running eval experiments without retraining. Loads quantized checkpoint directly.
- ✅ **Dynamic compile support** — Added `COMPILE_DYNAMIC` env var for `torch.compile(dynamic=True)`. Tested with seq curriculum: 219ms/step (too slow vs 133ms static).
- ❌ **CUDA graph Muon** — Still produces NaN. Double-update fix wasn't sufficient; deeper capture issue remains. Disabled by default (`MUON_USE_CUDA_GRAPH=0`).
- ❌ **Seq curriculum (3 attempts)** — Dead. torch.compile recompilation (48s), no-compile too slow (298ms), dynamic compile overhead (219ms). All three paths fail.

## Technical Implementation

### Preset-Aligned QAT
`apply_qat_preset_alignment(model, num_layers)` called after model construction. Iterates through named modules, marks CastedLinear layers with `_qat_range` attribute matching the export quant preset. CastedLinear.forward reads `_qat_range` instead of hardcoded `INT6_QUANT_RANGE`.

### Neural Cache
`forward_hidden_and_logits()` returns (hidden_states, logits) from the model. `eval_val_with_cache()` processes sequences one at a time, building a cache of (normalized_hidden, target_token) pairs. At each position, computes cosine similarity with all cached states, builds a cache probability distribution, and blends with model probs: `P = (1-λ)P_model + λP_cache`.

### Batched Muon Fixes
- Double-update: Changed `if self._step_count <= 2` to `if self._step_count <= 1` for warmup, separate `if self._graph is None` for capture
- Global scope: Added `batched_newton_schulz` to `global` declaration in `main()`
- Pre-computed reverse index: `_param_to_group_idx` dict for O(1) lookup instead of O(n) `.index()`

**Files Modified:**
- `sota_train_gpt.py` — Added: preset-aligned QAT (`apply_qat_preset_alignment`, `_qat_range`), neural cache (`eval_val_with_cache`, `forward_hidden_and_logits`), token-class calibration (`calibrate_token_class_temps`), `COMPILE_DYNAMIC` env var, Muon bug fixes
- `results.tsv` — algo_008 through algo_014 logged

## Bugs & Issues Encountered
1. **CUDA graph double-update NaN** — Graph capture executes the step (that's how CUDA graphs work), so running `_batched_step` first = double weight update on step 2
   - **Fix:** Skip `_batched_step` on capture step. Still NaN though — deeper issue with graph replay and gradient buffer management
2. **Port EADDRINUSE on remote** — Stale python3 processes hold port 29501 after SSH drops
   - **Workaround:** `fuser -k 29501/tcp` and use different ports (29502, 29503)
3. **`batched_newton_schulz` UnboundLocalError** — Function defined at module level but `torch.compile()` reassignment in `main()` needed `global` declaration
   - **Fix:** Added to `global` statement
4. **`reduce-overhead` compile + RoPE cache** — RoPE cache regeneration on seq_len change conflicts with internal CUDA graphs used by `reduce-overhead` mode
   - **No fix:** Fundamental incompatibility with dynamic shapes in this compile mode

## Key Learnings
- **Preset-aligned QAT is a real win.** Training STE should match export quant precision. Boundary blocks that get int8 at export should train with int8 fake-quant, not int6.
- **Token-class calibration doesn't help.** Despite finding real per-class temperature differences, they cancel out when weighted by byte contribution to BPB. The softcap already calibrates well.
- **CUDA graphs + optimizers = complex.** The graph must BE the step (not follow a normal step). Gradient buffers must persist (`set_to_none=False`). Dynamic hyperparams need `.fill_()` not Python scalars. Many subtle traps.
- **Batched Newton-Schulz is the safe Muon speedup.** 4.8x faster than per-param NS, no CUDA graph complexity. Clean ~3ms savings per step.
- **Seq curriculum is fundamentally blocked by torch.compile.** All three approaches (static recompile, no-compile, dynamic compile) fail on performance. Would need a shape-invariant curriculum design (e.g., padding short sequences to full length).

## Architecture Decisions
- **Disabled CUDA graph by default** — The batched approach alone gives most of the speedup (29ms → ~5ms) without the fragility. CUDA graph adds ~1.3ms more but has NaN issues.
- **Neural cache processes sequences one-by-one** — Necessary because cache is built incrementally within each sequence. Slow but eval-only, acceptable.
- **Per-class temps not applied in final eval** — Calibration was diagnostic only. Would need per-token logit modification in eval_val to actually apply, but the signal is too weak to justify.

## Bugs & Issues (addendum)
5. **nohup output buffering** — `nohup python3 script.py > file.out 2>&1 &` produces 0-byte output files because Python buffers stdout. Fix: use `PYTHONUNBUFFERED=1` or `python3 -u`.
6. **Neural cache too slow** — Per-token sequential loop over 62M val tokens (30K sequences × 2048 positions, each doing dot-product against all previous positions) is O(T^2) per sequence. Needs vectorization: pre-compute all hidden states, then use `torch.tril(H @ H.T)` for all-pairs similarity in one shot.

## Ready for Next Session
- ✅ **Preset-aligned QAT** — Working, integrated, confirmed -0.0014 BPB
- ✅ **Eval-only script** — `eval_cache.py` loads checkpoints without retraining
- 🔧 **Neural cache** — Code exists but needs vectorized implementation (O(T^2) → batched matmul)
- 🔧 **PPM-C mixing** — Not yet implemented (byte-level, complex)
- 🔧 **Trigram expert (C1)** — Not yet implemented
- 🔧 **CUDA graph Muon** — Still broken, low priority
- 🔧 **Line count** — Script at 2149 lines, needs trimming to 1500 for submission
- ✅ **GPU instance** — 69.19.136.6:32581, checkpoint `final_model.int6.ptz` saved from best run

## Context for Future
Branch A1 (preset-aligned QAT) is the only confirmed algorithmic win so far (-0.0014 BPB). Branch D1 (token-class calib) negative. Branch C3 (neural cache) blocked by implementation speed. Next: either vectorize neural cache, implement trigram expert (C1), or move to 8xH100 submission run with current best stack. The eval-only script (`eval_cache.py`) enables fast iteration on eval-time ideas without retraining.
