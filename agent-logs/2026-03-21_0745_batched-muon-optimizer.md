# Batched Muon Optimizer with CUDA Graph Capture

**Date:** 2026-03-21
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ✅ Completed

## User Intention
User wanted to explore whether rewriting the training stack in a lower-level language (Rust/C) could speed up training for the parameter-golf competition. Through profiling, we discovered the real bottleneck wasn't Python dispatch overhead but the Muon optimizer's per-matrix Newton-Schulz loop. The session pivoted from "rewrite everything in Rust" to "surgically optimize the actual bottleneck" — batching Newton-Schulz and capturing it as a CUDA Graph.

## What We Accomplished
- ✅ **Profiling harness** — Built `profile_training.py` that breaks down step time into data loading, forward, backward, optimizer, grad clip, and sync. Ran on H100 PCIe with real training config.
- ✅ **Identified the real bottleneck** — Profiling showed Python dispatch is only ~4% overhead (torch.compile handles it). The Muon optimizer was 14.4% of step time (28.88ms), constant regardless of batch size.
- ✅ **Batched Newton-Schulz** — Grouped 55 weight matrices into 5 shape groups, replaced individual `zeropower_via_newtonschulz5` calls with batched `torch.bmm` operations. 4.8x speedup (23ms → 5ms).
- ✅ **CUDA Graph capture** — Captured the entire optimizer step as a CUDA Graph with dynamic LR/momentum/WD via GPU scalar tensors. Additional 1.3ms saving (5ms → 3.65ms). Total 6.5x speedup.
- ✅ **Integration into sota_train_gpt.py** — Drop-in replacement of the Muon class. Controllable via `MUON_USE_CUDA_GRAPH` env var. Falls back to batched-only (no graph) when disabled.
- ✅ **Correctness verification** — Numerical parity confirmed across warmup, capture, and replay phases. Max diff 0.0017 (well within bf16 precision).
- ✅ **End-to-end profiling** — Optimizer went from 28.88ms (14.4%) to 3.90ms (2.2%) in real training. Total step time improved 10.4% (200.9ms → 180.0ms). ~346 more steps in 10 minutes.

## Technical Implementation

### Batched Newton-Schulz
Instead of iterating over 55 weight matrices individually:
```python
# OLD: 55 iterations, 55 separate Newton-Schulz calls
for i, p in enumerate(params):
    g = zeropower_via_newtonschulz5(p.grad, steps=5)  # per-matrix
```
We group by shape and use `torch.bmm`:
```python
# NEW: 5 shape groups, 5 batched calls
batch = torch.stack(same_shape_grads)  # [B, rows, cols]
result = batched_newton_schulz(batch, steps=5)  # via torch.bmm
```

Shape groups in the 9L×512d model:
- `(512, 512)` ×18 — c_q + proj across layers
- `(256, 512)` ×18 — c_k + c_v across layers
- `(1536, 512)` ×9 — mlp.fc across layers
- `(512, 1536)` ×9 — mlp.proj across layers
- `(512, 128)` ×1 — bigram_hash_proj (singleton, no batching)

### CUDA Graph Capture
- Steps 1-2: Normal batched execution (warmup, allocate all buffers)
- Step 2 end: Pre-allocate static buffers, capture CUDA graph on side stream
- Steps 3+: Update scalar tensors (LR, WD factor, momentum), replay graph
- Dynamic values stored as GPU scalar tensors — graph reads from fixed addresses, values updated before replay

### Key Design Decision: Uncompiled NS in Graph Path
`torch.compile` on Newton-Schulz conflicts with CUDA graph capture (JIT compilation triggers `cuda.synchronize`, illegal during capture). Solution: save uncompiled function references at module level (`_zeropower_orig`, `_batched_ns_orig`) before any compilation. Graph path uses originals; non-graph path uses compiled versions.

**Files Modified:**
- `sota_train_gpt.py` — Replaced Muon class with batched+CUDA-graph version. Added `batched_newton_schulz` function, `_zeropower_orig`/`_batched_ns_orig` references, `MUON_USE_CUDA_GRAPH` env var. Changed `zero_grad(set_to_none=False)` for Muon only.
- `profile_training.py` — New file. Training profiler with fine-grained step timing.
- `profile_muon_gpu.py` — New file. Standalone Muon optimizer benchmark.
- `test_muon_batched.py` — New file. Correctness + shape grouping + CPU overhead tests.
- `test_muon_cudagraph.py` — New file. CUDA Graph correctness + 3-way GPU benchmark.

## Bugs & Issues Encountered
1. **`float(tensor)` during CUDA graph capture** — GPU→CPU sync is illegal during capture. The nesterov line `p.grad.add_(buf, alpha=float(momentum_tensor))` triggered this.
   - **Fix:** Replace with explicit multiply `p.grad.add_(buf * momentum_tensor)`.
2. **torch.compile + CUDA graph capture conflict** — Compiled Newton-Schulz triggers Triton JIT compilation during graph capture, which calls `cuda.synchronize`.
   - **Fix:** Save uncompiled references at module level before `torch.compile` runs. Graph path always uses uncompiled originals. Compilation still applied for non-graph fallback path.
3. **Profiler compiling NS independently** — `profile_training.py` had its own `torch.compile(T.zeropower_via_newtonschulz5)` call that bypassed the MUON_USE_CUDA_GRAPH guard.
   - **Fix:** Conditioned profiler's compile call on `T.MUON_USE_CUDA_GRAPH`.

## Key Learnings
- **Python dispatch is NOT the bottleneck for well-compiled models.** torch.compile gives 3.3x forward speedup, reducing dispatch overhead to ~2%. A full C/CUDA rewrite would gain <5% on top. The "rewrite in Rust" intuition was wrong for this model+batch combination.
- **Optimizer cost is constant regardless of batch size.** Muon's Newton-Schulz operates on weight matrices (fixed size), not gradients×batch. On competition hardware (8xH100) where per-GPU compute is small, the optimizer can be 25%+ of step time.
- **Newton-Schulz backend_steps=3 (reduced from 5) makes convergence worse.** User tried this previously — the optimizer time savings are not worth the convergence quality loss.
- **torch.bmm is the key to batching heterogeneous-but-same-shape operations.** Grouping by shape and using batched matmul eliminates Python loop overhead AND keeps GPU SMs fully utilized across all matrices simultaneously.
- **CUDA Graphs require set_to_none=False for zero_grad.** Gradient buffers must persist at fixed memory addresses for graph replay. Only applies to the Muon optimizer (Adam optimizers can still use set_to_none=True).

## Architecture Decisions
- **Batched + CUDA Graph (not either/or)** — Batching reduces the number of operations (55→5), CUDA Graph eliminates remaining Python dispatch. They stack: 23ms → 5ms → 3.65ms.
- **Env var control (MUON_USE_CUDA_GRAPH)** — CUDA Graphs add complexity (pre-allocation, set_to_none constraints). The env var allows disabling for debugging or hardware compatibility. Defaults to ON.
- **Uncompiled originals over conditional compilation** — Storing `_zeropower_orig` at module level is more robust than checking `MUON_USE_CUDA_GRAPH` at compile time, because other code (profiler, tests) might compile the functions independently.

## Ready for Next Session
- ✅ **Optimized Muon in sota_train_gpt.py** — Drop-in, tested, profiled. Optimizer is 2.2% of step time.
- ✅ **Profiling tools** — `profile_training.py`, `profile_muon_gpu.py`, `test_muon_batched.py`, `test_muon_cudagraph.py` all on the GPU machine.
- 🔧 **Verify training loss curve** — Another agent is running a training comparison now. Need to confirm loss matches old optimizer.
- 🔧 **Test on 8xH100 SXM (competition hardware)** — The optimizer savings should be proportionally larger there. Need a RunPod run.
- 🔧 **Competition ideas still pending** — Mixed int5/int6 quant, OptRot rotation, PPM-C context mixing were discussed but not implemented.

## Context for Future
This session started as "should we rewrite PyTorch in Rust?" and ended with a surgical 6.5x optimizer speedup that required zero language changes — just smarter batching and CUDA graph capture. The profiling-first approach was critical: it revealed the actual bottleneck (Muon optimizer, not Python dispatch) and prevented a multi-week rewrite that would have gained <5%. The optimizer is now effectively solved (~4ms, 2.2% of step). Future optimization efforts should focus on algorithmic improvements (mixed quant, eval-time techniques) rather than infrastructure speed.
