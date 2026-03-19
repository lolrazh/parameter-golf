# Parameter Golf: Initial Research & Experiment Infrastructure

**Date:** 2026-03-19
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

## User Intention
User wants to compete in OpenAI's Parameter Golf challenge as a learning exercise in AI research methodology. The core goal is to learn how ML research works — forming hypotheses, running experiments, interpreting results — while building a competitive small language model (17M params, 16MB compressed, 10 min on 8xH100). They want to start local (M4 Air) and scale to cloud GPUs strategically.

## What We Accomplished
- ✅ **Full codebase walkthrough** - Explained the baseline model architecture (9L×512d GPT with GQA, U-Net skips, relu², Muon+Adam optimizer, tied embeddings), the BPB metric, and the competition rules
- ✅ **Fast local experiment pipeline** - Built a 2-5 minute feedback loop on M4 Air with `VAL_TOKENS_LIMIT`, `SKIP_SERIALIZATION`, optimized batch/seq settings. Reduced val eval from 60K batches to 1 batch.
- ✅ **Experiment ledger** (`experiments.md`) - Tracking every experiment with run ID, change, val_loss, delta, verdict, and notes
- ✅ **Research checklist** (`research.md`) - Comprehensive prioritized attack vector list with ★ ratings
- ✅ **LR sweep (6 experiments)** - Swept 0.25× to 4× on M4. Found 3× optimal locally (val_loss=4.0579). Confirmed LR doesn't transfer across batch sizes.
- ✅ **Architecture experiments (4 local + 3 cloud)** - Tested softcap=15 (noise), smear module (noise), 12L×448d deeper (worse), 6L×640d wider (BEST), MLP 3x (noise)
- ✅ **Modal cloud pipeline** (`train_modal.py`) - Verified on H100. 2-min runs at ~$0.15 each. Data cached in image.
- ✅ **H100 validation** - Confirmed local M4 rankings transfer to real hardware. 6L×640d wins at both scales.
- ✅ **Frontier research synthesis** - Dispatched 4 research agents covering Qwen3.5, GLM-5, MiniMax, Kimi K2, Liquid AI, modded-nanogpt speedrun, Morph, diffusion LMs. Surfaced value embeddings, LAWA, depth recurrence as top priorities.

## Technical Implementation

**Local experiment config (M4 Air):**
```bash
ITERATIONS=200 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 TRAIN_SEQ_LEN=512
VAL_BATCH_SIZE=131072 VAL_TOKENS_LIMIT=131072 WARMUP_STEPS=0
MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 SKIP_SERIALIZATION=1
MATRIX_LR=0.12 SCALAR_LR=0.12 TIED_EMBED_LR=0.15  # (3× default)
```

**Cloud config (Modal 1xH100):**
```bash
modal run train_modal.py --run-id <name> --max-wallclock 120
# Default: 524K batch, 1M val tokens, default LRs
```

**Files Modified:**
- `train_gpt_mlx.py` - Added `VAL_TOKENS_LIMIT` env var (line 65) and `SKIP_SERIALIZATION` env var (line 1057)
- `CLAUDE.md` - Updated with local experiment config, Modal workflow
- `experiments.md` - Created experiment ledger (14 experiments logged)
- `research.md` - Created prioritized research checklist
- `train_modal.py` - Created Modal training script

## Bugs & Issues Encountered
1. **Val eval taking 60K batches on M4** - `val_batch_tokens = val_batch_size // grad_accum_steps` meant 1 seq per batch
   - **Fix:** Added `VAL_TOKENS_LIMIT` to cap val set size; set `GRAD_ACCUM_STEPS=1`
2. **Double validation pass** - Script runs quant roundtrip eval (second full val pass) after training
   - **Fix:** Added `SKIP_SERIALIZATION=1` env var to skip for local experiments
3. **Modal `Mount` deprecated** - `modal.Mount.from_local_file()` removed in SDK 1.3.5
   - **Fix:** Switched to `Image.add_local_file()` with `copy=True` for build-step files
4. **Modal `add_local_file` before `run_commands`** - Can't run build steps after mounting local files
   - **Fix:** Used `copy=True` on the data download script, left `train_gpt.py` as last mount (no copy)

## Key Learnings
- **Architecture rankings transfer across hardware** - M4 Air (200 steps, 4K batch) predicted the same width>depth ordering as H100 (340 steps, 524K batch). Local proxy experiments are reliable for architecture decisions.
- **LR values do NOT transfer across batch sizes** - 3× LR optimal locally (4K batch) but default LR is right for H100 (524K batch). Only use local LR to calibrate the local baseline.
- **Width > depth at 17M params and short training** - 6L×640d beat 9L×512d beat 12L×448d. Wider layers converge faster, use less memory, and run faster per step.
- **Most speedrun tricks don't transfer to param golf** - Softcap tuning and smear module mattered for time-to-target-loss but not for loss-at-fixed-params.
- **Val_loss is a perfect proxy for BPB** - Monotonic relationship via `BPB = (val_loss / ln2) × (tokens/bytes)`. Never need to compute BPB locally.
- **Thermal throttling on M4 Air is predictable** - Peak ~9K tok/s for first 100 steps, degrades to ~5K by step 200. Doesn't affect architecture rankings.
- **BitNet/ternary is a trap at 17M params** - Paper confirmed capacity loss too severe at this scale. Stick with post-training int8.
- **Modal image layer ordering matters** - Put frequently-changing files (train_gpt.py) LAST so data downloads stay cached.

## Architecture Decisions
- **3× LR as local baseline** - Chose 3× over 4× despite 4× having marginally better loss, because 4× caused severe thermal throttling (2× longer runs). Speed of iteration > marginal quality improvement.
- **6L×640d as current best architecture** - Wider+shallower won on all axes: faster per step (341ms vs 477ms), better loss per step, less memory (8.9 GB vs 12.2 GB). Trade-off: less compositional depth, but at 10 min training budget it doesn't matter.
- **Val subset (128K tokens locally, 1M on H100)** - Small enough for fast eval, large enough for reliable signal. Differences < 0.1 val_loss at 200 steps are noise.

## Ready for Next Session
- ✅ **Local experiment pipeline** - `train_gpt_mlx.py` with fast config, ready for more experiments
- ✅ **Cloud pipeline** - `train_modal.py` verified, ~$29.55 budget remaining
- ✅ **6L×640d as new baseline** - Best config to build upon
- ✅ **Research checklist** - Prioritized list of things to try next
- 🔧 **Value embeddings** - Top priority from speedrun research, needs ~20 lines of code
- 🔧 **Depth recurrence** - Highest-impact param-golf-specific technique, needs architecture changes
- 🔧 **LAWA checkpoint averaging** - Free generalization boost, needs implementation
- 🔧 **Push wider** - Test 5L×720d or 4L×800d to find the width ceiling

## Context for Future
This session established the full research workflow: local screening → cloud validation → ledger tracking. The winning architecture so far (6L×640d) beats the OpenAI baseline shape but we haven't yet tried the highest-impact techniques (value embeddings, depth recurrence, LAWA). The leaderboard baseline is 1.2244 BPB — our best 2-min proxy run hit 1.6024 post-quant BPB, which would improve significantly with a full 10-min run on 8xH100. The next session should focus on stacking winning techniques on the 6L×640d base before doing a full 10-min leaderboard attempt.
