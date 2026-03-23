# Frontier Fork, L40S Setup, and Baseline Comparison

**Date:** 2026-03-23
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

## User Intention
Shift from incremental technique-stacking to a frontier-fork approach: use PR #512 (PROTEUS) as the control trunk, set up cheap L40S research hardware, and begin TTT eval-time experiments. The user wants to stop "catching up" to SOTA and instead build on top of it — the entire 0.22 BPB gap between PROTEUS's base model and its final score is pure TTT, so TTT engineering is where the remaining gains live.

## What We Accomplished
- ✅ **Pulled PR #512 (PROTEUS) and PR #517 (Goldfish ML) scripts** — saved as `frontier_512.py` (1493 lines) and `frontier_517.py` (1699 lines)
- ✅ **Created TTT diff analysis** — `ttt_diff_notes.md` with exact code diff, hyperparameter comparison, ablation table, and experiment matrix
- ✅ **Set up L40S on RunPod** — torch.compile works, SDPA works, no FA3 needed. Zero code changes to frontier_512.py.
- ✅ **Validated full pipeline on L40S** — training (313ms/step), int6 quant, export, post-quant eval, LoRA TTT eval all work end-to-end
- ✅ **Downloaded sp1024 + sp4096 data** — both tokenizer variants available on the L40S
- ✅ **Head-to-head comparison: PROTEUS vs s3** — 10 min each, same hardware
- ✅ **Cleaned up repo** — removed 27+ old experimental scripts, committed
- ✅ **Updated NOTES.plan** — hardened frontier checklist with DeltaNet context
- ✅ **PROTEUS checkpoint retrained** — deleted by mistake, retrained, saved properly
- ✅ **Both checkpoints downloaded locally** — safe from spot preemption
- ✅ **Standalone run_ttt.py script** — loads exported checkpoint, runs LoRA TTT, reports BPB
- ✅ **PROTEUS TTT calibration** — post-TTT BPB: 1.2870 (TTT gain: -0.115)
- ✅ **s3 TTT complete** — post-TTT BPB: 1.2515 (TTT gain: -0.1113, 21 min)

## Baseline Results (L40S, 1xGPU, 10 min, 131K batch)

| Metric | PROTEUS 11L sp1024 | s3 10L sp4096 | Winner |
|---|---|---|---|
| Pre-quant BPB | 1.3455 | **1.3143** | s3 by 0.031 |
| Post-quant BPB | 1.3968 | **1.3626** | s3 by 0.034 |
| Quant gap | 0.0513 | 0.0483 | s3 |
| Steps | 1,887 | 1,988 | s3 (+5.3%) |
| Step time | 318ms | 302ms | s3 |
| Artifact | 10.1 MB (63%) | 11.8 MB (74%) | PROTEUS (smaller) |
| Params | 26.8M | 26.0M | s3 |

Key finding: s3 (10L sp4096) beats PROTEUS (11L sp1024) by 0.034 BPB post-quant. sp4096's 35% more bytes/token advantage is real and transfers to proxy hardware.

## TTT Calibration Results (L40S, LoRA rank-8, 2 epochs per doc)

| Metric | PROTEUS 11L sp1024 | s3 10L sp4096 |
|---|---|---|
| Post-quant BPB | 1.4022 | 1.3626 |
| **Post-TTT BPB** | **1.2870** | **1.2515** |
| **TTT gain** | **-0.1151** | **-0.1113** |
| TTT time | 1846s (31 min) | 1269s (21 min) |
| Short/long docs | 31585 / 18415 | 37673 / 12327 |

Notes:
- TTT takes ~31 min on 1xL40S vs ~6 min on 8xH100 (8× parallelism + 3.9× bandwidth)
- PROTEUS on 8xH100 gets -0.224 BPB TTT gain. L40S proxy sees -0.115 (~half) because model trained ~1900 steps vs ~7000
- sp4096 has fewer long docs (12K vs 18K) because tokens cover more bytes → documents are shorter in token space
- Spot preemption killed the first TTT attempt at batch 1135/1151. Restarted on same pod, completed second time.

## TTT Sweep (in progress)

Fast loop established: 2M token val subset, ~2 min per variant.
Running 10 structural variants on frozen s3 checkpoint.
All use s3 (10L sp4096) with DATA_PATH=fineweb10B_sp4096_small.

| # | Variant | BPB | vs baseline |
|---|---------|-----|-------------|
| v1 | baseline (r8, 2ep, const lr) | pending | — |
| v2 | rank 4 | pending | |
| v3 | rank 16 | pending | |
| v4 | 5 epochs | pending | |
| v5 | 10 epochs | pending | |
| v6 | chunk 128 | pending | |
| v7 | chunk 512 | pending | |
| v8 | context 512 | pending | |
| v9 | min_doc 256 (adapt more) | pending | |
| v10 | min_doc 2048 (adapt fewer) | pending | |

Skipped (H100-sensitive, don't transfer from proxy):
- LR sweep
- Warmdown tuning
- Batch size tuning

## L40S Hardware Profile

- GPU: NVIDIA L40S, 48GB GDDR6, 864 GB/s bandwidth
- PyTorch: 2.9.1+cu128, torch.compile works (dynamic=False, fullgraph=True)
- Attention: SDPA (no FA3 — Hopper only)
- Memory usage: ~4GB for 11L model at 131K batch (tons of headroom)
- Cost: $0.26/hr secure spot on RunPod
- Training: ~$0.043 per 10-min run
- TTT eval on full 62M val tokens: ~14 min (slow — use VAL_TOKENS_LIMIT for iteration)

## TTT Diff Summary (PR #512 vs PR #517)

Two completely different TTT approaches:

**PR #512 (PROTEUS):** Per-document LoRA TTT
- LoRA rank-8 on Q, V, LM_head per document
- Adam lr=0.01, 2-3 epochs per doc, reset between docs
- Score-then-train per chunk (256 tokens, 1024 trailing context)
- 64 docs in parallel, skip docs <1024 tokens
- ~350s on 8xH100 (fits budget)

**PR #517 (Goldfish):** Full-model global TTT with cosine decay
- All model weights, AdamW lr=0.008
- CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01) — THE 3-line innovation
- 100 epochs over entire val set
- ~1463s on 8xH100 (VIOLATES time limit)

The cosine insight: constant LR overfits to eval token positions after ~30 epochs. Cosine decay separates content learning from position memorization.

## Bugs & Issues Encountered
1. **Deleted PROTEUS checkpoint before saving** — ran `rm -rf final_model*` when cleaning up for s3 run
   - **Fix:** Retraining (~10 min). Added memory note to always save checkpoints before cleanup.
2. **SSH pipe kills nohup process** — original test run piped through `head -60` which killed the process when buffer filled
   - **Fix:** Always use `nohup ... > logfile 2>&1 &` without pipes
3. **sp4096 data not in official download script** — `cached_challenge_fineweb.py --variant sp4096` fails
   - **Fix:** Download from `sproos/parameter-golf-tokenizers` on HuggingFace
4. **SSH exit code 255 on long commands** — SSH drops during GPU-intensive operations
   - **Workaround:** Use nohup, check results with separate SSH command
5. **Python stdout buffered to file** — `nohup python3 script.py > log` produces empty log until buffer flushes
   - **Fix:** Always use `python3 -u` (unbuffered) when redirecting to log files
6. **zstandard not installed on fresh RunPod pods** — checkpoint decompression fails with `zlib.error: incorrect header check`
   - **Fix:** `pip install --break-system-packages zstandard` before running
7. **Spot preemption mid-TTT** — L40S spot instance died at batch 1135/1151 (16 batches from completion)
   - **Fix:** Downloaded checkpoints locally for safety. Restarted pod, checkpoints persisted on /workspace/ volume.
8. **run_ttt.py wastes time on redundant post-quant eval** — re-computes BPB we already know from training
   - **TODO:** Skip post-quant eval when the number is already known, or make it optional

## Key Learnings
- **frontier_512.py works on L40S with ZERO code changes** — already uses SDPA, handles world_size=1, all config via env vars
- **TTT_MIN_DOC_LEN=999999** skips TTT entirely (no docs qualify) — useful for training-only runs
- **TTT eval on 62M tokens takes ~14 min on L40S** — need VAL_TOKENS_LIMIT=1048576 (1M) for fast iteration (~15-20s)
- **TTT_BATCH_SIZE=16** for L40S (vs 64 on H100) — conservative for 48GB VRAM

## TTT Experiment Plan (Next Steps)

### Phase 1: SOTA Calibration (2 runs)
Both on the same L40S, same 10-min training:
- Run PROTEUS-style LoRA TTT on PROTEUS checkpoint → SOTA baseline post-TTT BPB
- Run same LoRA TTT on s3 checkpoint → our candidate post-TTT BPB
This tells us if sp4096's pre-TTT advantage survives TTT.

### Phase 2: TTT Sweep (on frozen s3 checkpoint, ~$0.60 for 20 tests)
Each test loads the saved checkpoint, runs TTT variant, measures BPB.

| # | Variant | What we learn |
|---|---------|---------------|
| 1 | PROTEUS defaults (r8, 2ep, const lr) | Baseline TTT |
| 2 | +cosine decay (2ep) | Does cosine help at low epochs? |
| 3 | 10ep constant lr | More adaptation |
| 4 | 10ep cosine decay | #517 insight on LoRA |
| 5 | 20ep cosine decay | Scaling TTT epochs |
| 6 | Rank 4 | Is r8 overkill? |
| 7 | Rank 16 | Is r8 too small? |
| 8 | Q+V only (no LM_head) | Where adaptation matters |
| 9 | All linear layers | More capacity |
| 10 | Chunk 128 | Finer granularity |
| 11 | Chunk 512 | Coarser granularity |
| 12 | Context 512 | Less trailing context |
| 13 | Context 2048 | More trailing context |
| 14 | Min doc 256 (adapt more docs) | Worth adapting short docs? |
| 15 | Min doc 2048 (adapt fewer docs) | Skip marginal docs |
| 16 | Per-layer LR (3x for MLP out) | #517's orthogonal finding |
| 17 | Best config + stride-OGD | Do they stack? |
| 18 | Best config + neural cache | Do they stack? |

### Speed Optimization
For Phase 2, use VAL_TOKENS_LIMIT=1048576 so each TTT test takes ~15-20s instead of 14 min. Final judging uses full 62M tokens.

## Files Modified
- `frontier_512.py` — PR #512 control script (new file, pulled from GitHub)
- `frontier_517.py` — PR #517 control script (new file, pulled from GitHub)
- `ttt_diff_notes.md` — TTT diff analysis (new file)
- `NOTES.plan` — Updated with DeltaNet context and frontier checklist
- Deleted 27+ old experimental scripts (committed)

## Files on L40S
- `/workspace/parameter-golf/frontier_512.py` — training script
- `/workspace/s3_checkpoint.pt` — s3 (10L sp4096) full-precision checkpoint
- `/workspace/s3_checkpoint.int8.ptz` — s3 exported int6+zstd artifact
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/` — 10 shards + val
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp4096/` — 10 shards + val

## Ready for Next Session
- ✅ **s3 checkpoint saved** — frozen trunk for TTT experiments
- 🔧 **PROTEUS checkpoint retraining** — in progress, ~10 min
- 🔧 **Standalone TTT eval script needed** — extract TTT from frontier_512.py for fast iteration
- ✅ **Both tokenizer datasets on L40S** — sp1024 and sp4096 ready

## Context for Future
The architecture race is commoditized — everyone converges on 11L/512d/GQA/SmearGate/BigramHash/relu²/MLP3x. The frontier is TTT pipeline engineering. PROTEUS gets 0.224 BPB from TTT alone. Our s3 trunk already beats PROTEUS pre-TTT by 0.034 BPB. The next step is replicating PROTEUS's TTT on our trunk, then sweeping TTT variants to find something better. A standalone TTT eval script is critical for fast iteration (~$0.03 per test vs ~$0.04 per full training run).
