# Full-Stack Buildout + Cloud GPU Setup

**Date:** 2026-03-21
**Status:** In progress (FA3 compiling on A6000)

## What We Did

### Code Changes (8 commits)
1. **Weight decay + orthogonal init + SWA + competitive defaults** — Muon WD, AdamW, ortho init with 1/sqrt(2L) proj scaling, SWA checkpoint averaging, SmearGate init 0.0, BigramHash zero-init + learnable scale + better hash primes. Updated defaults to 9x512, seq2048, batch 786K, higher LRs, grad clip 0.3.
2. **Flash Attention 3** — conditional import with SDPA fallback, shared q/k/v path.
3. **FP16 embedding passthrough** — tok_emb stored as fp16 instead of int8 quantized.
4. **Late QAT toggle** — QAT_START_FRAC controls when STE activates (0.0=always, 0.7=late).
5. **Compiled eval** — torch.compile on forward_logits, doubled eval batch to 64.
6. **TTT (test-time training)** — full-model SGD adaptation on val data before scoring.
7. **Depth recurrence** — NUM_RECURRENCE loops through blocks with optional LoRA adapters.
8. **XSA (Exclusive Self Attention)** — projects out self-value in deep layers, zero params. BigramHash bumped to 10240.

### Cloud GPU Setup
- **Modal**: Ran experiments #58 (best: pre 1.4874, post 1.5778 sliding BPB on SXM). Cost ~$3.50/run, not $0.27. Nearly exhausted.
- **Thunder Compute H100 PCIe (production)**: Set up, ran experiments #59-60. SSH drops killed eval processes. FA3 build killed by accident. ~50% slower than SXM.
- **Thunder Compute A6000 (prototyping)**: Current. FA3 compiling. Will snapshot when done, then spin up A100 production for experiments.

### Experiments
| # | Run ID | Platform | Steps | Pre BPB | Post BPB | Sliding BPB | Notes |
|---|--------|----------|-------|---------|----------|-------------|-------|
| 58 | fullstack_11L_5m | Modal H100 SXM | 386 | 1.4874 | 1.5959 | 1.5778 | Best result. Quant gap only +0.11 |
| 59 | fullstack_11L_ttt_v2 | Thunder H100 PCIe | 258 | 1.6839 | 1.8948 | 1.8966 | TTT didn't help (undertrained) |
| 60 | fullstack_11L_10m | Thunder H100 PCIe | 515 | 1.4038 | 1.6234 | (killed) | Best prequant. Eval killed by SSH |

### Key Learnings
- **Quant gap collapsed from +0.32 to +0.11** with always-on QAT + FP16 embed + WD + ortho init
- **TTT doesn't help undertrained models** — needs 7000+ steps to be useful
- **SWA works** when SWA_EVERY is tuned for run length (50 for short runs, 200 for long)
- **Thunder Compute PCIe is ~50% slower per step than SXM** — use for experiments, not calibration
- **Do setup/compilation on cheap instances** — FA3 compile on H100 wasted ~$1.80
- **Always use nohup+disown** — SSH drops kill foreground processes on Thunder

## Competition State (as of session end)
- **Merged SOTA**: 1.1428 BPB (int5/int6 mixed quant + BigramHash(10240))
- **Best pending**: 1.1303 BPB (TTT + full stack)
- **Paid prefix banned** by organizers
- **XSA** is new (~0.002 BPB, zero params)
- **Depth recurrence** independently proposed by another competitor (PR #268)

## Cloud Saga (lessons learned the hard way)
- **Modal**: $3.50/run (not $0.27 as estimated). Nearly exhausted.
- **Thunder H100 PCIe production**: Works with torch.compile. ~1150ms/step. SSH drops kill foreground processes. FA3 compile killed by accident ($1.80 wasted).
- **Thunder A6000 prototyping**: FA3 compile too slow on 4 vCPUs. Abandoned.
- **Thunder A100 production**: torch.compile OOMs on Triton shared memory (A100 has 166KB vs H100's 228KB). Cannot run our model with compile. 3300ms/step without compile = unusable.
- **Final choice**: Back to Thunder H100 PCIe production ($2.49/hr). Everything works there.
- **Lesson**: Always verify torch.compile works on target hardware BEFORE committing to it.

## New Features Added (late session)
- **XSA (Exclusive Self Attention)**: Zero-param eval gain on last 3 layers
- **BigramHash(10240)**: Bumped from 4096, matching merged SOTA
- **TORCH_COMPILE toggle**: Env var to disable compile for hardware compatibility
- **Compiled sliding window eval**: torch.compile on forward_logits + batch 64
- **run_experiment.sh**: Single-experiment runner for autoresearch loop
- **results.tsv**: Append-only experiment log

## Competition Update (March 21)
- Merged SOTA: 1.1428 BPB (mixed int5/int6 + BigramHash(10240))
- Best pending: 1.1303 BPB (TTT + full stack)
- Paid prefix BANNED by organizers
- New techniques: XSA, mixed int5/int6, depth recurrence (PR #268)

## Next Steps
1. Verify 5-min run on new H100 instance
2. Set up Ralph Loop autoresearch — 3-min training cycles, ~12/hr
3. Run #1: Full stack with XSA + BigramHash(10240) baseline
4. Run #2: Depth recurrence (5 blocks × 3 loops)
5. Run #3: Late QAT comparison
6. Run #4: TTT on properly trained model (10-min train)
7. Final: Best config on RunPod 8xH100 SXM
