# Competitive Intel Discovery + Phase 2 Transition

**Date:** 2026-03-19
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

## User Intention
User wants to compete seriously in Parameter Golf — not just learn, but win. After exhausting local M4 architecture experiments (21 runs), pivoted to studying what actual competition leaders are doing (issue #83, top PRs). Discovered the winning strategy is stacking orthogonal techniques, not finding one magic architecture. Now transitioning to Phase 2: implement proven competition techniques in train_gpt.py and test on Modal cloud GPUs.

## What We Accomplished
- ✅ **Phase 1 complete: 31 local experiments** — Found 6L×640d as optimal shape. Tested LR sweep, softcap, smear, value embeddings, depth recurrence, LAWA, SwiGLU, MLP 3x, grad clip, Muon tuning, head counts, drop MLP. Only shape change (6L×640d) gave meaningful improvement.
- ✅ **Competitive intelligence gathered** — Scraped issue #83 and top PRs (#65, #70, #61, #77, #78). Mapped the full winning technique stack.
- ✅ **competitive-intel.md created** — Dense reference of what works, with BPB deltas and PR sources.
- ✅ **research.md overhauled** — Phase 1 marked done, Phase 2 plan with implementation order.
- ✅ **Modal pipeline verified** — train_modal.py working on H100, data cached in image.
- ✅ **H100 baselines** — 2-min runs confirming 6L×640d (1.5705 BPB) beats 9L×512d (1.5903 BPB).

## Key Discoveries
1. **Sliding window eval = -0.034 BPB for FREE** — no training change, no artifact cost
2. **int6 quantization saves 4MB** — enables MLP 3x, bigger vocab, more layers
3. **QAT reduces int6 penalty from +0.048 to +0.001 BPB** — fake quant during training
4. **Muon momentum=0.99 + LR=0.02** — all top PRs use this recipe
5. **Train seq2048 > seq4096** — more steps, same BPB with sliding window (PR #61 insight)
6. **Vocab 8192 tokenizer available** — huggingface.com/sproos/parameter-golf-tokenizers

## Bugs & Issues
1. **Value embedding zero-init** — initialized scale to 0, meaning value embeddings were disabled. Fixed to 0.1, still didn't help (batch size issue, not init).
2. **MLX array has no .copy()** — LAWA snapshot needed `mx.array(v)` instead of `v.copy()`.
3. **Modal Mount API deprecated** — `Image.add_local_file(copy=True)` needed for build-step files.

## Key Learnings
- **Local M4 experiments have a ceiling** — good for architecture ranking, bad for quantization/eval tricks/optimizer tuning. All remaining gains require real GPU scale.
- **The competition is won by STACKING techniques, not finding one breakthrough** — top PRs combine 5-6 independent -0.005 to -0.034 improvements.
- **int6 quantization is the gate** — it frees 4MB that enables architecture changes. Without it, MLP 3x doesn't fit.
- **We were wrong about MLP 3x** — it didn't help locally but gives -0.019 BPB at full scale with int6.

## Ready for Next Session
- ✅ **research.md** has full Phase 2 plan with implementation order
- ✅ **competitive-intel.md** has all winning techniques documented
- 🔧 **Sliding window eval** — first implementation target, -0.034 BPB free
- 🔧 **5-min H100 baseline** — need to establish before testing techniques
- 🔧 **Muon tuning** — env vars from PR #70 ready to test
- 🔧 **int6 + QAT** — biggest engineering task, enables MLP 3x

## Context for Future
Phase 1 (local architecture exploration) is complete. Phase 2 is engineering: implement sliding window eval, int6 quantization, QAT, and Muon tuning in train_gpt.py. Test on Modal (L4 at $0.02/run or H100 at $0.15-0.27/run). $29.55 Modal budget remaining. Current best config: 6L×640d. Current leaderboard SOTA: 1.1630 BPB. Target: beat baseline 1.2244 first, then push toward SOTA.
