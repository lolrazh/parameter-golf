# SOTA Rebase & Quant Innovations Port

**Date:** 2026-03-26
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

Building on `2026-03-25_1800_proxy-comparison-baseline-vs-upgraded.md`

## User Intention
User wanted to prepare a competitive 8xH100 submission for the Parameter Golf challenge. The session evolved from testing architecture upgrades (BigramHash 10240, VE128, MLP 3.5x) on proxy, to implementing full-model TTT (replacing LoRA), to discovering fundamental config gaps vs SOTA (10 vs 80 training shards, seq_len 1024 vs 2048), and ultimately rebasing entirely on SOTA's code with only our proven quant innovations on top.

## What We Accomplished
- ✅ **Proxy A/B comparison (1xH100)** — Run A (baseline 26.8M) vs Run B (upgraded 30.9M). Result: upgrade was a wash (-0.001 BPB post-TTT). Extra params cost steps without meaningful gain.
- ✅ **Fixed sliding window eval bug** — EVAL_SEQ_LEN defaulted to 2048, poisoning sliding window via NTK RoPE. Fixed default to 1024. Updated run_ttt.py with sliding window support.
- ✅ **Code cleanup (2042 → 1658 lines)** — Removed DuQuant, SDPA fallback, unbatched Muon, MuonTTTOptimizer, score-first TTT, LaCT, FUSE_QKV toggle. 3 Opus auditors verified 48/48 checks pass.
- ✅ **Full-model TTT implemented** — Ported PR #549's score-first sliding-window TTT (SGD, 3 epochs, 32K chunks, cosine LR). Verified working on 1xH100 (0.69 BPB gain on 2-min checkpoint).
- ✅ **Competition landscape research** — Discovered n-gram cache submissions (0.17-0.30 BPB) dominating unmerged PRs. Merged SOTA still 1.1194.
- ❌ **8xH100 submission attempt** — Ran seed 1337 on 8xH100. Pre-quant BPB: 1.1815 vs SOTA's 1.1218. Killed — too far from SOTA to submit.
- ✅ **Root cause analysis** — Forensic parameter-by-parameter comparison vs SOTA revealed 3 killers: 10 vs 80 training shards, seq_len 1024 vs 2048, model bloat (30.9M vs 26.9M).
- ✅ **Full rebase on SOTA** — Extracted PR #549's train_gpt.py as new base. Added our 3 proven quant innovations (~85 lines). New file: `train_gpt.py` (1935 lines).
- ✅ **CONFIG.md created** — Single source of truth for every parameter, traced to source, with changelog.

## Technical Implementation

### Three Surgical Additions to SOTA Base

**1. GPTQ-Lite Per-Row (lines ~1242-1267):**
SOTA searched 5 clip percentiles but picked the global best (scalar `best_err`). Changed to track `best_mse` per row and update only improved rows. Strictly better — different rows have different outlier distributions.

**2. Mixed Quant Preset (lines ~1338-1380):**
Added `_quant_bits_for_layer()` function. `front3_back1_6_middle5`: int6 for sensitive layers (first 3 + last 1), int5 for middle. Plugged into `mixed_quantize_int6()` via variable `clip_range`. Controlled by `QUANT_PRESET` env var.

**3. Entropy-Reg QAT (lines ~1710-1720):**
During QAT phase (when `CastedLinear._qat_enabled`), adds penalty: `entropy_reg * mean(residual^2)` where residual = distance from quantization grid. Only computed on first micro_step to avoid dilution by grad_accum. Controlled by `ENTROPY_REG` env var (default 0.01).

**Files Created/Modified:**
- `train_gpt.py` — NEW. SOTA PR #549 base + our 3 quant additions (1935 lines)
- `CONFIG.md` — NEW. Architecture doc with every parameter and changelog
- `frontier_512.py` — Modified during session, now superseded by train_gpt.py
- `run_ttt.py` — Modified with sliding window + VE support, now superseded

## Bugs & Issues Encountered

1. **VAL_TOKENS_LIMIT not set on first proxy run** — Sliding window + TTT on 62M tokens = 45+ min eval. Killed and restarted with VAL_TOKENS_LIMIT=1048576.
   - **Fix:** Always include VAL_TOKENS_LIMIT=1048576 for proxy runs.

2. **EVAL_SEQ_LEN=2048 default breaks sliding window** — Same NTK RoPE bug from previous sessions. Post-quant eval at 2048 poisons subsequent sliding window.
   - **Fix:** Changed default to 1024. For SOTA rebase, this is moot — SOTA trains at 2048 so NTK isn't an issue.

3. **run_ttt.py passed `ve_layers` instead of `ve_layer_indices`** — GPT constructor expects list[int], not string.
   - **Fix:** Added parsing: `ve_layer_indices = [int(x) for x in args.ve_layers.split(",") if x.strip()]`

4. **TTT_FREEZE_BLOCKS=2 instead of SOTA's 0** — Our default froze first 2 blocks. SOTA runs with all blocks unfrozen.
   - **Fix:** Changed default to 0. Then rebased on SOTA which has it right.

5. **10 training shards instead of 80** — `--train-shards 10` was a local dev shortcut that was never changed for production. SOTA uses all 80 shards. This starved the model of data diversity.
   - **Fix:** Rebase on SOTA which downloads 80 shards.

6. **train_seq_len=1024 instead of 2048** — SOTA trains at 2048, getting 2x context per step. We trained at 1024 because of NTK RoPE issues with our eval code. SOTA doesn't have this problem because they train AND eval at 2048.
   - **Fix:** Rebase on SOTA which trains at 2048 natively.

7. **Model bloat (30.9M vs 26.9M)** — BigramHash 10240 + MLP 1792 + VE128 added 4M params costing +20ms/step and -1,158 training steps. Proxy showed only -0.001 BPB benefit.
   - **Fix:** Rebase on SOTA's lean 26.9M architecture.

## Key Learnings

- **Train shards and seq_len are foundational.** We spent days optimizing architecture and quantization while missing that we were training on 1/8th the data at half the context length. Always verify the basics before optimizing the edges.
- **Bigger model != better under fixed compute.** 30.9M params at 100ms/step loses to 26.9M at 83ms/step. The step count matters more than the param count when wallclock is capped.
- **Proxy runs are biased against larger models.** With ~3200 steps and WARMDOWN_ITERS=2000, the larger model gets fewer steps at full LR. On 8xH100 this bias is smaller but still present.
- **Full-model TTT >> LoRA TTT.** Every top submission unfreezes all blocks. LoRA constrains adaptation to a low-rank subspace. The score-first pattern (inference_mode scoring → SGD training) is the legal and effective approach.
- **Multi-epoch TTT on scored chunks is legal.** PR #549 does 3 epochs, pending submissions do 10-30. The rule is: score first (BPB locked), then train as many epochs as you want on those scored tokens.
- **N-gram eval caches are a game-changer.** Submissions achieving 0.17-0.30 BPB by building statistical models from scored val tokens. Legality unclear — not merged yet.
- **SOTA's Parallel Muon > our Batched Muon on multi-GPU.** Their reduce-scatter/all-gather overlap with compute is architecturally superior to our all-reduce approach. Parameter banking enables this.
- **zstd-22 > LZMA for quantized weights.** Confirmed across 5/6 presets. Use zstd-22 for compression.

## Architecture Decisions

- **Rebase on SOTA instead of patching our code** — Our codebase had accumulated too many wrong assumptions (10 shards, 1024 seq_len, no weight decay, lower LRs). Starting from SOTA's proven code gives us the right foundation. Adding ~85 lines for our quant innovations is cleaner than trying to port SOTA's Parallel Muon + Parameter Banking into our architecture.
- **Keep only 3 proven innovations** — Entropy-reg QAT (-0.04 BPB quant gap reduction), mixed quant presets (layer-aware bits), GPTQ-lite per-row search. Everything else either matches SOTA or isn't worth the complexity.
- **Don't port fused QKV or batched Muon** — SOTA's Parameter Banking + Parallel Muon is better for 8xH100. Our speed optimizations solved the wrong problem.

## Ready for Next Session
- ✅ **train_gpt.py ready** — SOTA base + 3 quant additions, syntax verified (1935 lines)
- ✅ **CONFIG.md ready** — Every parameter documented with source
- 🔧 **Need 80 training shards** — `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
- 🔧 **Need 3-seed submission run** — Seeds 1337, 42, 2025 on 8xH100
- 🔧 **Repo cleanup in progress** — Deleting old scripts, checkpoints, test files
- 🔧 **Engram research in progress** — Checking DeepSeek's Engram for novel techniques

## Context for Future
This session was a hard lesson in fundamentals over optimization. We spent days tuning quantization, architecture, and TTT while training on 1/8th the data at half the context length. The rebase on SOTA PR #549 gives us the right foundation — 80 shards, seq_len 2048, Parallel Muon, proper weight decay. Our unique contribution is better quantization (entropy-reg QAT + mixed presets + per-row GPTQ-lite). Next session: download 80 shards, run 3-seed submission, assemble PR. Also exploring n-gram cache approaches and DeepSeek's Engram for potential further gains.
