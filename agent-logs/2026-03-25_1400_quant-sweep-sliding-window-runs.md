# Quant Sweep, Sliding Window Debug & Runs 7-10

**Date:** 2026-03-25
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ⚠️ Partial (Run 10 in progress)

Building on `2026-03-24_2100_proxy-ablation-and-architecture.md`

## User Intention
User wanted to (1) fix a checkpoint loading bug in run_ttt.py, (2) systematically evaluate quantization presets and compression methods to find the best artifact size/quality tradeoff, (3) validate XSA=11 vs XSA=4, (4) get sliding window eval working correctly, and (5) launch an entropy-regularized QAT run. The overarching goal is finalizing the best 11L sp1024 configuration for an 8xH100 submission.

## What We Accomplished
- ✅ **Fixed checkpoint loading bug in run_ttt.py** - Missing `rope_dims` and `xsa_last_n` params caused model to initialize with wrong architecture, producing loss 12-13 instead of ~2
- ✅ **Comprehensive quant sweep** - Tested 6 presets x 3 compression methods on existing checkpoint
- ✅ **Run 7: XSA=11 (all layers) ablation** - Confirmed XSA=4 is optimal; XSA=11 is 0.021 BPB worse post-TTT
- ✅ **Run 9: Sliding window eval** - Discovered and fixed catastrophic NTK-aware RoPE scaling bug
- ✅ **DuQuant (Hadamard rotation) evaluation** - Tested and rejected; hurts compression
- ✅ **Run 10 launched** - Entropy-regularized QAT + fixed sliding window (in progress)
- ✅ **Best config confirmed** - 11L sp1024, XSA=4, ROPE_DIMS=16, LeakyReLU squared, front3_back1_6_middle5, zstd-22, LoRA TTT 1ep, sliding window stride=64 seq=1024

## Quant Sweep Results

6 presets tested with zlib, zstd-22, and LZMA compression:

| Preset | zstd-22 (MB) | zlib (MB) | LZMA (MB) | Verdict |
|--------|-------------|-----------|-----------|---------|
| int5mlp | 13.02 | 13.42 | 13.22 | Best compression |
| uniform_int6 | 15.34 | 15.75 | 15.60 | Tight fit |
| front3_back1_6_middle5 | 13.46 | 13.86 | 13.62 | Our default |
| front3_back1_7_middle5 | 13.97 | 14.39 | 14.12 | New middle ground |
| front3_back1_7_middle6 | 16.29 | 16.73 | 16.52 | OVER 16MB |
| front3_back1_8_middle6 | 17.54 | 18.01 | 17.74 | OVER 16MB |

Key finding: **zstd-22 beats LZMA on 5 out of 6 presets.** Only exception is marginal.

## Run Results

### Run 7: XSA=11 (All Layers)
- Config: 11L sp1024, XSA_LAST_N=11, ROPE_DIMS=16, LeakyReLU squared, front3_back1_7_middle5
- Steps: 3,228 @ 185.9ms/step
- Pre-quant: 1.3757, Post-quant: 1.3934, Post-TTT: 1.2715
- Artifact: 13.83 MB
- Verdict: **WORSE** than XSA=4 (Run 6) by 0.021 BPB post-TTT. XSA on all layers adds 8ms/step, loses 146 steps. XSA=4 stays default.

### Run 9: Sliding Window + Best Config
- Config: 11L sp1024, XSA4, PartialRoPE, LeakyReLU squared, front6mid5, EVAL_STRIDE=64
- Steps: 3,406 @ 176.2ms/step
- Pre-quant: 1.3503, Post-quant: 1.3672, Post-TTT: 1.2711
- Artifact: 12.8 MB
- Sliding window at EVAL_SEQ_LEN=2048: **2.1082 BPB** (catastrophically broken)
- Sliding window at seq_len=1024, stride=64: **1.2667 BPB** (standard was 1.3008, gain: -0.034)

## Bugs & Issues Encountered
1. **Checkpoint loading missing architecture params** - `run_ttt.py` did not pass `rope_dims` and `xsa_last_n` when loading checkpoints, causing the model to initialize with default architecture instead of the trained one. Loss was 12-13 instead of ~2.
   - **Fix:** Added the missing params to checkpoint loading code.

2. **NTK-aware RoPE scaling catastrophically breaks at eval_seq_len > train_seq_len** - When model trained at seq_len=1024 is evaluated with sliding window at EVAL_SEQ_LEN=2048, NTK rescaling changes RoPE frequencies for ALL positions (including 0-1023), not just the extended ones. This corrupted every single position encoding, producing 2.1082 BPB.
   - **Root cause:** NTK-aware scaling modifies the base frequency based on the ratio of eval_seq_len / train_seq_len. When this ratio > 1, ALL frequencies change, meaning even positions that were within the original training range get different encodings than what the model learned.
   - **Fix:** Use train_seq_len (1024) for sliding window evaluation, NOT eval_seq_len (2048). SOTA PRs use stride=64 with eval_seq_len=2048 because they TRAIN at 2048.

3. **DuQuant (Hadamard rotation before quantization) made things worse** - MSE increased by 0.4%, artifact grew by 293-549 KB.
   - **Root cause:** GPTQ-lite already handles outliers effectively. Hadamard rotation spreads weight values more uniformly, which actually hurts zstd compression (less redundancy in byte patterns).
   - **Resolution:** Rejected DuQuant entirely.

## Key Learnings
- **NTK RoPE scaling is global, not local** - It does not just "extend" the context window; it changes frequencies at ALL positions. You cannot use it to evaluate at 2x training length without retraining. This is a fundamental property of how NTK scaling works (it modifies the base theta, which affects every position).
- **zstd-22 is the best compression for quantized weights** - Beats LZMA on almost all presets. The structured redundancy in quantized int5/int6 weights is better exploited by zstd's dictionary-based approach.
- **Hadamard rotation hurts compressed artifact size** - Rotation distributes outliers across channels, reducing the compressibility of weight tensors. When you already have GPTQ-lite handling outliers at quantization time, rotation just adds entropy.
- **XSA diminishing returns beyond 4 layers** - XSA=4 lets deeper layers specialize in routing while early layers still transform values. XSA=11 forces all layers into routing-only mode, which is too restrictive.
- **Sliding window is a free -0.034 BPB gain** - But ONLY when eval seq_len matches train seq_len. The stride=64 setting (from SOTA PRs) is confirmed optimal.

## Architecture Decisions
- **XSA=4 over XSA=11** - The first 7 layers need value projections to transform representations. Only the last 4 benefit from the XSA parameter savings + routing-only attention pattern.
- **Sliding window at train seq_len, not extended** - Unlike SOTA PRs that train at 2048 and eval at 2048, we train at 1024 so our sliding window must use 1024. The stride=64 overlap still captures cross-boundary dependencies.
- **zstd-22 as default compressor** - Replaces LZMA for artifact packaging. Consistent wins across preset configs.
- **front3_back1_6_middle5 remains default quant preset** - Best tradeoff: 13.46 MB (3.5 MB headroom), 0.017 quant gap, proven across multiple runs.

## Ready for Next Session
- ✅ **Best config locked in** - 11L sp1024, XSA=4, ROPE_DIMS=16, LeakyReLU squared, front3_back1_6_middle5, zstd-22, LoRA TTT 1ep, sliding window stride=64 seq=1024
- ✅ **Quant presets fully mapped** - Know exactly which fit under 16MB and which don't
- ✅ **Sliding window fix validated** - -0.034 BPB free gain confirmed
- 🔧 **Run 10 in progress** - Entropy-regularized QAT + fixed sliding window. Results pending.
- 🔧 **8xH100 submission run needed** - Best proxy config needs validation at full scale with 10-min training budget

## Context for Future
This session resolved two critical bugs (checkpoint loading, NTK RoPE) and systematically eliminated several dead ends (XSA=11, DuQuant, extended-context sliding window). The best proxy configuration is now fully locked: 11L sp1024 with XSA=4, Partial RoPE, LeakyReLU squared, front3_back1_6_middle5 quant, zstd-22, sliding window at stride=64/seq=1024, and LoRA TTT. Run 10 (entropy-regularized QAT) is the last algorithmic experiment before scaling to 8xH100 for a submission attempt. The sliding window fix alone is worth -0.034 BPB, which could be decisive at the competition margin.
