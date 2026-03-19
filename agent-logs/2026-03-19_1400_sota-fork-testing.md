# SOTA Fork + Architecture Swap Testing

**Date:** 2026-03-19
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

## User Intention
User wants to stop reimplementing known techniques and instead fork the current SOTA (PR #65, 1.1630 BPB), then test whether swapping in 6L×640d architecture improves it. Goal is to combine the competition's best engineering (int6 QAT, sliding window, Muon tuning) with our novel architecture finding (wider+shallower).

## What We Accomplished
- ✅ **Downloaded PR #65 SOTA code** — `sota_train_gpt.py` (1323 lines), the current #1 standard submission
- ✅ **Full diff analysis** — Mapped every change: MLP 3x, int6+QAT via STE in CastedLinear, sliding window stride=64, Muon momentum=0.99, LR=0.02, warmdown=3000, zstd-22 compression
- ✅ **Implemented sliding window eval** in our train_gpt.py (forward_logits + eval_val_sliding)
- ✅ **5-min H100 baseline recorded** — val_bpb=1.3738, post-quant=1.3766, 856 steps
- ✅ **Modal script updated** — added `--use-sota` flag, zstandard dep, mount for sota_train_gpt.py
- 🔄 **Running SOTA reproduction** — 2-min on 1xH100, in progress

## Technical Implementation
**SOTA code key differences from baseline:**
- CastedLinear: adds int6 fake-quant STE during training (lines 638-649)
- int6 quantization: 31-level per-row on block weights, int8 on embeddings
- zstd-22 compression instead of zlib-9
- Sliding window eval with stride=64
- MLP_MULT=3, MATRIX_LR=0.02, MUON_MOMENTUM=0.99, WARMDOWN_ITERS=3000

**Files Modified:**
- `train_gpt.py` — added forward_logits(), eval_val_sliding(), EVAL_STRIDE/EVAL_SEQ_LEN params
- `train_modal.py` — added use_sota flag, zstandard dep, sota script mount
- `sota_train_gpt.py` — downloaded from PR #65 fork (aquariouseworkman)

## Key Learnings
- **SOTA didn't change architecture at all** — same 9L×512d. All gains from training recipe + quantization + eval strategy.
- **Our 6L×640d is genuinely novel** — no one in the competition has tried wider+shallower.
- **Fork and extend > reimplement** — user correctly identified this as the efficient strategy.

## Ready for Next Session
- 🔧 **Compare SOTA vs SOTA+6L×640d** — two 2-min runs on H100
- 🔧 **Track artifact sizes** — need to log compressed model size in experiments
- 🔧 **If 6L×640d wins: full 10-min run** on H100 for real BPB number

## Context for Future
We're at the critical test: does our novel architecture (6L×640d) improve on the SOTA recipe? If yes, we have a genuine submission. If no, we need a different angle. The SOTA's gains are all from engineering (QAT, sliding window, compression) not architecture — so there's room for architecture innovation to stack on top.
