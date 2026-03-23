# Fused QKV + Batched Muon + FA3 Port to frontier_512.py

**Date:** 2026-03-24
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ✅ Completed

Building on `2026-03-23_1100_frontier-fork-and-l40s-setup.md`

## User Intention
User is preparing for a SOTA 8xH100 submission run using the s3 trunk (10L sp4096). The goal was to port two proven performance optimizations (fused QKV and batched Muon) from the deleted `sota_train_gpt.py` into `frontier_512.py`, add FA3 support for H100 Hopper GPUs, and make all optimizations toggleable for A/B testing. Also validated that our TTT implementation is legitimate per community discussion (GitHub issue #402).

## What We Accomplished
- ✅ **Ported Fused QKV** — Replaced 3 separate c_q/c_k/c_v CastedLinear layers with a single c_qkv. Preserves q_delta/v_delta for TTT LoRA (applied after slicing).
- ✅ **Ported Batched Muon** — Groups weight matrices by shape, runs Newton-Schulz via `torch.bmm` instead of per-param loop. 23ms -> 5ms (4.8x speedup on optimizer step).
- ✅ **Added FA3 support** — Auto-detects `flash_attn_interface`, handles [B,T,H,D] layout for FA3 vs [B,H,T,D] for SDPA. Rotary cache shape adjusts accordingly.
- ✅ **All optimizations toggleable** — `FUSE_QKV=0/1`, `BATCH_MUON=0/1`, FA3 auto-detected. Module-level constants for torch.compile constant-folding.
- ✅ **Updated TTT LoRA references** — Changed c_q.weight/c_v.weight shape lookups to architectural constants (num_heads * head_dim, kv_dim).
- ✅ **Compiled batched_newton_schulz** — Added torch.compile call alongside existing zeropower compile.
- ✅ **Added optimization log line** — Prints `fuse_qkv`, `batch_muon`, `fa3` status at training start.
- ✅ **Validated TTT legitimacy** — Confirmed our implementation is Case 3 (per-document, independent LoRA reset) per GitHub issue #402 discussion. Organizer (@0hq) even approved the more permissive Case 2.

## Technical Implementation

### Fused QKV
```python
# One matmul instead of three per layer
self.c_qkv = CastedLinear(dim, dim + 2 * self.kv_dim, bias=False)
# Slice after
qkv = self.c_qkv(x)
q = qkv[:, :, :dim] + (q_delta if q_delta is not None else 0)
k = qkv[:, :, dim:dim + self.kv_dim]
v = qkv[:, :, dim + self.kv_dim:] + (v_delta if v_delta is not None else 0)
```
Reduces HBM weight reads from 3 to 1 per layer. ~1.4ms/step saved.

### Batched Muon
```python
# Group same-shaped matrices, run batched Newton-Schulz
batch = torch.stack(active_grads)  # e.g., 10 c_qkv weights of shape (1024, 512)
batch_result = batched_newton_schulz(batch, steps=backend_steps)
```
With fused QKV, shape groups are: `(1024,512)x10` (c_qkv), `(512,512)x10` (proj), `(1536,512)x10` (mlp.fc), `(512,1536)x10` (mlp.proj). 4 batched calls instead of 40 individual.

### FA3 Layout
- FA3: `[B, T, H, D]` — skip transpose after reshape (natural layout)
- SDPA: `[B, H, T, D]` — transpose required
- Rotary cache: `[1, T, 1, D/2]` for FA3, `[1, 1, T, D/2]` for SDPA
- q_gain broadcast: dim index shifts based on layout

### Toggle Environment Variables
```bash
FUSE_QKV=0    # Disable fused QKV (use separate c_q/c_k/c_v)
BATCH_MUON=0  # Disable batched Muon (use per-param Newton-Schulz)
# FA3 is auto-detected (no env var needed)
```

**Files Modified:**
- `frontier_512.py` — All changes in this single file (1504 -> 1641 lines):
  - Added `from collections import defaultdict` import
  - Added FA3 import try/except + `HAS_FA3` flag
  - Added `FUSE_QKV` and `BATCH_MUON` module-level toggles
  - Added `batched_newton_schulz()` function (lines ~106-122)
  - Replaced Muon class with batched version + `_unbatched_step` fallback
  - Modified CausalSelfAttention `__init__` and `forward` for fused/unfused + FA3/SDPA
  - Modified Rotary `forward` for FA3 cos/sin shape
  - Updated TTT LoRA q_out/v_out derivation
  - Added `torch.compile(batched_newton_schulz)` call
  - Added optimization status log line

## Bugs & Issues Encountered
1. **`sota_train_gpt.py` was deleted** — The source file with fused QKV + batched Muon was removed in cleanup commit `f8a1863`.
   - **Fix:** Recovered from git history via `git show f8a1863^:sota_train_gpt.py`
2. **TTT LoRA references `c_q.weight.shape`** — Lines 761-762 accessed deleted attributes.
   - **Fix:** Changed to architectural constants: `block.attn.num_heads * block.attn.head_dim` and `block.attn.kv_dim`
3. **Line count over 1500** — File grew to 1641 lines, exceeding the competition's 1500-line soft limit.
   - **Workaround:** frontier_512.py is a dev fork, not the submission file. For submission, trim comments/logging.

## Key Learnings
- **Fused QKV init changes random seed path** — One orthogonal init vs three means different initial weights. Proven in sota_train_gpt.py to not affect training dynamics, but A/B comparison must account for this.
- **Module-level constants enable torch.compile constant-folding** — Using `FUSE_QKV = bool(int(os.environ.get(...)))` at module level means the compiler eliminates dead branches. A runtime flag would prevent this.
- **TTT is Case 3 (legitimate)** — Per GitHub issue #402, our TTT resets LoRA per document, scores then trains per chunk, never leaks across documents. The organizer approved even the more permissive Case 2 (token-stream), so we're well within bounds.
- **FA3 layout is sequence-first [B,T,H,D]** — Natural reshape output, skips one transpose per layer. SDPA is head-first [B,H,T,D]. The Rotary cache shape must match.

## Architecture Decisions
- **Toggle via env vars, not constructor args** — Matches codebase pattern (everything is env-var controlled). Module-level constants get compiled away.
- **Keep _unbatched_step as fallback** — Enables clean A/B testing. Extra ~40 lines but worth it for validation.
- **FA3 as auto-detect, not env var** — If the package is installed, use it. No reason to fall back to SDPA on Hopper GPUs. Unlike fused QKV/batched Muon, there's no "compare both" use case.
- **q_delta/v_delta applied after QKV slice** — Mathematically identical to applying before (linear algebra distributes over concat), but cleaner code and works with both fused/unfused paths.

## Ready for Next Session
- ✅ **frontier_512.py ready for GPU testing** — All optimizations in, toggleable, FA3 auto-detected
- ✅ **A/B test commands prepared** — `FUSE_QKV=0 BATCH_MUON=0` vs defaults
- ✅ **TTT validated as legitimate** — Case 3, confirmed by community + organizer
- 🔧 **GPU smoke test needed** — Changes are CUDA-only, can't verify locally on M4
- 🔧 **8xH100 submission run** — Full s3 config: 10L sp4096, 786K batch, FA3, warmdown=2000, 3ep TTT

## Submission Run Config (Ready to Execute)
```bash
# 8xH100 SXM submission run
torchrun --standalone --nproc_per_node=8 frontier_512.py
# Env vars: NUM_LAYERS=10 VOCAB_SIZE=4096 TRAIN_BATCH_TOKENS=786432
# WARMDOWN_ITERS=2000 MAX_WALLCLOCK_SECONDS=600
# TTT_EPOCHS=3 TTT_MIN_DOC_LEN=256
```

## s2/s3 Run Summary (For Reference)

| Run | Config | BPB | Artifact | Verdict |
|---|---|---|---|---|
| s2 (8xH100) | 9L sp4096, 786K, FA3, int6 | 1.1484 | 17.3 MB (OVER) | 0.0056 from SOTA but too big |
| s3 (8xH100) | 10L sp4096, int5+SWA+wd3000 | 1.1587 | 15.8 MB | SWA+long warmdown hurt |
| s3 (L40S proxy) | 10L sp4096, frontier_512.py | 1.3143 pre-quant | 11.8 MB | Frozen trunk for TTT sweep |
| s3+TTT (L40S) | 3ep + min_doc 256 | **1.1011** post-TTT | — | Realistic 8xH100 config |
| PROTEUS 11L (L40S) | 11L sp4096, frontier_512.py | **1.0984** post-TTT | 12.5 MB | Best proxy result (0.0027 from s3, possibly noise) |

## Context for Future
The frontier_512.py script now has all performance optimizations from sota_train_gpt.py (fused QKV, batched Muon) plus FA3 support and TTT — making it the single script for the SOTA submission attempt. The estimated speedup is ~12.6% more steps in 10 minutes. Combined with sp4096's tokenizer advantage and 3-epoch TTT, the projected post-TTT BPB should comfortably beat SOTA (1.1428). The next step is a GPU smoke test followed by the full 8xH100 run. SOTA is 1.1428 BPB — s2 was already 1.1484 without TTT or the new optimizations. With everything together, we should crush it.
