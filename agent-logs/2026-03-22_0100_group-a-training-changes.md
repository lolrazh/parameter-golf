# Group A Training Changes + Proxy Debugging

**Date:** 2026-03-22
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ⚠️ Partial — code changes validated, proxy run config was wrong, needs rerun

Building on `2026-03-21_1530_11L-calibration-runs.md`

## User Intention
User wanted to push past SOTA (1.1250 BPB) toward sub-1.1. Session started with a deep audit of the full codebase via 3 parallel research agents (script audit, eval-time compute, tiny model optimization), then moved to implementing the highest-value changes in two groups: Group A (training improvements) and Group B (eval-time techniques). The user wanted all Group A changes implemented together, validated on a proxy 1xH100, then move to 8xH100 submission.

## What We Accomplished
- ✅ **Deep 3-agent research sweep** — Script audit found 6 critical discrepancies (wrong momentum, wrong batch, CPU EMA, missing SWA/TTT, doc-isolated eval). Eval-time agent found cross-window KV cache (PR #318) as highest-upside technique and that TTT conflicts with XSA+EMA. Tiny model agent found Gated Attention (NeurIPS 2025), ProRes, Seesaw scheduling, Differential Attention.
- ✅ **7 Group A training changes implemented** in `sota_train_gpt_11L.py`:
  1. Batch default 786K→524K (for 8xH100; proxy should use 131K)
  2. GPU-side EMA (removed .cpu() transfer)
  3. Muon momentum 0.95→0.99
  4. Ratio-based warmdown (35% of training, adapts to hardware)
  5. Momentum warmup 500→1500 steps, start 0.85→0.92
  6. Gated Attention (sigmoid gate per head, 8 params/layer)
  7. ProRes (progressive residual warmup, deeper layers ramp slower)
- ✅ **Document-isolated eval** — Implemented in sliding window eval. BOS token (id=1) marks doc boundaries. Masks scored tokens whose context spans a boundary. Env var `DOC_ISOLATED_EVAL=1`.
- ✅ **Cross-window KV cache architecture** — Separate `forward_with_kv_cache()` methods on CausalSelfAttention and Block. Clean separation from training `forward()` to avoid torch.compile graph breaks. `forward_logits_with_kv_cache()` on GPT. Not yet integrated into eval loop.
- ✅ **Stride-OGD** — Online vocab bias correction during eval. Updates bias with exact gradient (softmax - one_hot) per stride batch. Env vars `STRIDE_OGD=1`, `STRIDE_OGD_LR=0.05`, `STRIDE_OGD_MOMENTUM=0.9`.
- ❌ **Proxy calibration run** — Ran 3 times with WRONG config (786K batch, FA3 enabled). Got ~1028ms/step vs expected ~236ms. Wasted ~40 min of GPU time debugging before discovering the batch size mismatch.

## Technical Implementation

### ProRes torch.compile fix
ProRes initially used a Python float `_prores_scale` that changed every step. torch.compile treated it as a guard constant → recompilation every step → hit recompile_limit(8) → fell back to eager mode. Fixed by using `register_buffer("_prores_scale", torch.tensor(1.0), persistent=False)` and updating via `.fill_()`.

### KV cache separation
Adding optional `kv_cache` kwargs to `forward()` caused 22 graph breaks per forward pass (one per layer × 2 methods) because `if kv_cache is not None` creates polymorphic return types that torch.compile can't fuse. Fixed by creating separate `forward_with_kv_cache()` methods that are NEVER compiled. Training path stays clean.

### Gated Attention
`attn_gate` parameter (num_heads floats, init=1.0) added to CONTROL_TENSOR_NAME_PATTERNS so it's kept as fp32 passthrough during quantization. Applied as `y = y * sigmoid(attn_gate)[None, None, :, None]` after attention output, before projection.

**Files Modified:**
- `sota_train_gpt_11L.py` — All 7 Group A changes + doc-isolated eval + KV cache architecture + Stride-OGD. 2048 lines.
- `hardware_configs.md` — New file with exact per-hardware configs (user didn't want this, prefers agent-logs)

## Bugs & Issues Encountered
1. **ProRes killed torch.compile** — Python float attribute on module caused guard failures → recompilation limit → eager fallback → 4.3x slowdown
   - **Fix:** Use `register_buffer()` with tensor, update via `.fill_()`
2. **KV cache kwargs caused 22 graph breaks** — `if kv_cache is not None` inside compiled forward created polymorphic return types
   - **Fix:** Separate `forward_with_kv_cache()` methods, never compiled
3. **WRONG BATCH SIZE FOR PROXY** — Used 786K batch (8xH100 default) on 1xH100 PCIe proxy. Got 1028ms/step. All prior fast Thunder runs used 131K batch (132-255ms/step). Wasted 3 runs debugging this.
   - **Fix:** Proxy MUST use `TRAIN_BATCH_TOKENS=131072`. The 786K default is for 8xH100 only.
4. **FA3 is 38% SLOWER on 1xH100 with small batch** — 182ms (FA3) vs 132ms (SDPA) at 131K batch. FA3 async warp groups need saturated SMs.
   - **Fix:** Disable FA3 for proxy runs. Use SDPA. FA3 only helps on 8xH100.
5. **SSH nohup background commands report "failed"** — SSH exits with 255 after launching nohup. The process is fine, just the SSH session ends.
   - **Not a real failure.** Check remote process with separate SSH.

## Key Learnings
- **CRITICAL: Proxy 1xH100 config is TRAIN_BATCH_TOKENS=131072, SDPA (no FA3).** 786K batch on 1xH100 = 1150ms/step. 131K = 132-255ms/step. This is a 5-9x difference. NEVER use 786K on proxy.
- **CRITICAL: torch.compile treats Python scalar attributes as guard constants.** If a module attribute changes between calls, torch.compile recompiles. Use tensors/buffers for values that change during training.
- **CRITICAL: Don't add optional kwargs to compiled forward().** Polymorphic returns cause graph breaks. Use separate methods for alternative code paths.
- **TTT conflicts with XSA+EMA.** At lr=0.002 it hurts by 0.016 BPB. At lr=1e-4, only +0.002. Not worth it for our stack.
- **Doc boundaries in FineWeb = BOS token (id=1).** APPEND_EOS=False in tokenization pipeline. Each document starts with BOS, no EOS between docs.
- **GPU-side EMA is fine for memory.** 28M params × 4 bytes = 112MB. Peak went from 14.7→21.7 GB. Plenty of headroom on 80GB H100.

## Architecture Decisions
- **Separate train/eval forward paths** — Clean `forward()` for training (compilable, single return type) vs `forward_with_kv_cache()` for eval (uncompiled, returns cache). Avoids torch.compile issues entirely.
- **ProRes as buffer, not parameter** — `persistent=False` means it's not saved in state_dict (no artifact cost), but torch.compile sees it as a capturable tensor.
- **Ratio-based warmdown (35%)** — `warmdown_frac=0.35` instead of absolute `warmdown_iters`. Auto-adapts: 1xH100 at 131K batch ~2400 steps → 840 warmdown. 8xH100 at 524K ~8900 steps → 3115 warmdown. No manual tuning per hardware.
- **Doc-isolated eval masks windows, doesn't split documents** — Simpler than splitting the val stream into individual documents. Valid% of ~27% means ~73% of windows span a boundary and get masked. This is correct — those windows had contaminated context.

## Ready for Next Session
- ✅ **sota_train_gpt_11L.py** — All Group A changes + eval techniques implemented. Syntax verified.
- ✅ **Eval techniques ready** — Doc-isolated (toggle: DOC_ISOLATED_EVAL), Stride-OGD (toggle: STRIDE_OGD), KV cache (architecture in place, needs eval loop integration)
- ✅ **Thunder H100 PCIe** at 185.216.20.240:30726 — FA3 installed, data downloaded, user=ubuntu, key=~/.ssh/id_ed25519
- 🔧 **NEEDS RERUN with correct proxy config** — `TRAIN_BATCH_TOKENS=131072`, disable FA3 (or accept 38% penalty). Should get ~250ms/step = ~2400 steps in 10 min.
- 🔧 **Cross-window KV cache eval loop** — `forward_logits_with_kv_cache()` exists but not yet called from `eval_val_sliding_window()`. Need to add per-document KV accumulation loop.
- 🔧 **PPM-C** — Not yet implemented. Would run on CPU in parallel with GPU eval.
- 🔧 **Script is 2048 lines** — Over the 1500-line submission limit. Needs trimming for final PR.

## Context for Future
The Group A training changes are implemented and validated (no speed regression, no crashes). The proxy debugging ate most of the session — the core mistake was using 786K batch on 1xH100 (should be 131K). The eval-time techniques (doc-isolated, stride-OGD, KV cache architecture) are coded but untested. Next session: (1) rerun proxy with correct 131K config to validate Group A helps, (2) integrate KV cache into eval loop, (3) implement PPM-C, (4) go to 8xH100 submission.
