# Algorithmic Experiments: Transferable Techniques for Parameter Golf

**Date:** 2026-03-21
**Agent:** Claude Opus 4.6 (1M context)
**Status:** ⚠️ Partial — 6 experiments run, 1 clear win, pivoting needed
**Building on:** `2026-03-21_0010_ralph-loop-autoresearch.md`

## User Intention
After the ralph loop wasted ~8 hours of H100 time on non-transferable hyperparameter tuning, user wanted algorithmic/architectural experiments that would transfer from proxy hardware (1xH100 PCIe, 3 min) to competition hardware (8xH100 SXM, 10 min). Focus on novel ideas nobody on the leaderboard has tried.

## What We Accomplished
- ✅ **Sliding window eval** — -0.022 BPB for free (1.3887 → 1.3666), competition metric, 100% transferable
- ✅ **Temperature scaling implementation** — Confirmed logit softcap already handles calibration (T=1.00 optimal). Negative result that transfers.
- ✅ **Mixed int5/int6 quantization** — Implemented and tested. Trade-off: -550KB artifact, +0.008 BPB. Useful for funding extra capacity.
- ✅ **Low-rank Q factorization** — Implemented (Q_LOW_RANK env var). Neutral result on proxy due to QAT overhead.
- ❌ **Sequence length curriculum** — Failed twice. torch.compile recompilation kills it (96s overhead); without compile, steps are 2x slower.
- ⚠️ **New GPU instance setup** — Thunder H100 PCIe at 69.19.136.6:32581, fully configured

## Technical Implementation

### Algo 001: Temperature Scaling
Added `eval_temperature` attribute to GPT model. Grid search over T=[0.85..1.15] after quant roundtrip. Result: T=1.00 is optimal because `logit_softcap = 30 * tanh(logits/30)` already constrains the logit range — it's a built-in temperature control.

### Algo 002/005: Sequence Length Curriculum
Added `SEQ_CURRICULUM_ENABLED`, `SEQ_CURRICULUM_START`, `SEQ_CURRICULUM_SWITCH_FRAC` env vars. Modifies `cur_seq_len` in the training loop based on wallclock fraction. Problem: `torch.compile(dynamic=False)` recompiles (~48s) when input shape changes. Two recompiles (warmup→short→full) = 96s of 180s wasted. Without compile, steps are ~298ms vs ~136ms compiled.

### Algo 003: Mixed Int5/Int6 Quantization
Added `INT5_MLP_ENABLED`, `INT5_QUANT_RANGE=15`, `INT5_MLP_PATTERNS`. MLP CastedLinear layers get `_int5_quant=True` flag. Both QAT STE (training) and post-quant use int5 range for MLP weights. Saves artifact space but coarser quantization costs BPB.

### Algo 004: Low-Rank Q Factorization
Added `Q_LOW_RANK` env var. When >0, replaces `c_q = CastedLinear(dim, dim)` with `c_q_down = CastedLinear(dim, rank)` + `c_q_up = CastedLinear(rank, dim)`. 5% slower on proxy because two CastedLinear calls = double QAT overhead (torch.quantile per row). On 8xH100 with bigger batches, matmul savings might outweigh QAT cost.

### Algo 006: Sliding Window Eval
Already implemented in codebase — just enabled with `EVAL_STRIDE=64`. Each scored token gets 960+ tokens of context. -0.022 BPB improvement, 481s eval time (fits in competition's 10-min eval budget).

**Files Modified:**
- `sota_train_gpt.py` — Added: temp scaling (eval_temperature + grid search), seq curriculum (SEQ_CURRICULUM_* env vars + cur_seq_len in loop), int5 MLP (INT5_MLP_ENABLED + patterns + QAT range), low-rank Q (Q_LOW_RANK + c_q_down/c_q_up)
- `results.tsv` — 6 algorithmic experiment rows (algo_baseline through algo_006)

## Bugs & Issues Encountered
1. **torch.compile recompilation with dynamic=False** — Changing input tensor shapes forces full recompile (~48s each). Makes seq curriculum impractical.
   - **No fix found.** Would need `dynamic=True` or pre-warming both shapes, neither tested.
2. **Low-rank Q slower than expected** — Two CastedLinear calls have MORE total QAT overhead (torch.quantile per row) than one. The matmul savings are swamped by QAT compute.
   - **Workaround:** Could skip QAT on the bottleneck layer (c_q_down), but not tested.
3. **SSH connections dropping** — Long-running eval (sliding window = 8 min) drops SSH.
   - **Workaround:** Check logs on remote after reconnecting.

## Key Learnings
- **Logit softcap IS temperature scaling.** `30 * tanh(logits/30)` constrains logit range and implicitly calibrates confidence. No need for separate temperature optimization.
- **Int5 MLP is a trade, not a free win.** -550KB artifact but +0.008 BPB. Use it to fund extra capacity (more layers, bigger BigramHash). The merged SOTA (#180) does exactly this.
- **torch.compile(dynamic=False) is a shape prison.** Any dynamic-shape technique (curriculum, variable-length sequences) is incompatible without massive recompilation cost. This is a fundamental constraint of the training setup.
- **Sliding window eval is free BPB.** -0.022 BPB with zero training change. Should ALWAYS be enabled for final evaluation. Competition metric.
- **QAT overhead dominates small layers.** torch.quantile per row is expensive relative to small matmuls. Low-rank factorization adds QAT calls, making it counterproductive.

## Architecture Decisions
- **Fixed baseline for all experiments** — Used our winning 6L config (warmdown=500, batch=131K, WD=0.04, XSA=2, ROPE=50K) as immutable baseline. Only ONE algorithmic change per experiment.
- **11L for transferability check** — Ran 11L baseline on new instance (1.4302 BPB) to verify our proxy matches. Algorithmic findings on 6L should transfer to 11L on 8xH100.

## Ready for Next Session
- ✅ **Sliding window eval** — Confirmed -0.022 BPB, ready to use in all final evaluations
- ✅ **Int5 MLP implementation** — Ready to combine with extra model capacity
- ✅ **Low-rank Q implementation** — Ready but needs different QAT strategy to be useful
- ✅ **New GPU instance** — 69.19.136.6:32581, data downloaded, script synced
- 🔧 **Seq curriculum needs dynamic compile** — Would need torch.compile(dynamic=True) or shape-invariant curriculum
- 🔧 **PPM-C context mixing** — Not yet attempted, complex to implement
- 🔧 **OptRot pre-quant rotation** — Not yet attempted

## Context for Future
The main transferable finding is that sliding window eval gives -0.022 BPB for free — it should always be enabled for competition submissions. The int5 MLP implementation is ready for use as an artifact-budget tool (trade quant precision for model size). The seq curriculum idea remains promising but needs a torch.compile-compatible implementation. For the next session, consider: (1) combining int5 savings with an extra layer, (2) implementing PPM-C eval-time mixing, (3) testing on actual 8xH100 setup.
