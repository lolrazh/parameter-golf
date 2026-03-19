# Modal Harness Hardening + Metric Policy Reset

**Date:** 2026-03-19
**Agent:** Codex (GPT-5)
**Status:** ✅ Complete

## User Intention
User was trying to run a clean 2-minute A/B test on 1xH100 between `9x512` and `6x640` using the SOTA fork, with sliding-window eval disabled to save cost. The real need was not another experiment idea, but a reliable experiment harness and a trustworthy short-run metric.

## What We Accomplished
- ✅ **Identified the root confusion as a single-source-of-truth failure**
  - Conversation and notes were assuming a `9x512` SOTA baseline.
  - `sota_train_gpt.py` had silently drifted to default `NUM_LAYERS=6`, `MODEL_DIM=640`.
  - `train_modal.py` was a separate wrapper with partial config forwarding.
  - The training log did not print enough effective config to catch the mismatch immediately.
- ✅ **Ran the first 2-minute Modal job and extracted the actual result**
  - Run ID: `sota_9x512_2m`
  - This was assumed to be `9x512`, but later turned out to still be using the file defaults.
  - Final lines:
    - `step:312/20000 val_bpb:2.0821`
    - `Total submission size int6+zstd-22: 3279995 bytes`
    - `final_int6_roundtrip_exact val_bpb:7.26030596`
- ✅ **Found and fixed the first harness bug**
  - `train_modal.py` originally ignored local shell env like `NUM_LAYERS=6 MODEL_DIM=640 ... modal run ...`
  - Patched the wrapper so explicit arguments are forwarded into the remote container.
- ✅ **Ran the second 2-minute Modal job**
  - Run ID: `sota_6x640_2m`
  - Final lines:
    - `step:312/20000 val_bpb:2.0818`
    - `Total submission size int6+zstd-22: 3188933 bytes`
    - `final_int6_roundtrip_exact val_bpb:7.26467695`
- ✅ **Discovered the more important root cause**
  - The first "baseline" was not a true `9x512` baseline because `sota_train_gpt.py` defaults were already `6x640`.
  - This invalidated the original "baseline vs variant" interpretation.
- ✅ **Hardened the Modal runner**
  - Rewrote `train_modal.py` to:
    - accept explicit `--overrides 'KEY=VALUE,...'`
    - support `--num-layers`, `--model-dim`, `--eval-stride`
    - reject duplicate/conflicting overrides
    - print `launch_overrides`
    - parse the remote log and verify that requested config actually landed
    - print a `=== SUMMARY ===` block with one selected metric
- ✅ **Added explicit effective-config logging to `sota_train_gpt.py`**
  - New lines in the training log:
    - `model_shape:num_layers:... model_dim:... mlp_mult:... vocab_size:...`
    - `eval_config:eval_stride:... train_seq_len:...`
- ✅ **Defined a short-run metric policy**
  - For short runs (`max_wallclock <= 180`), `metric_mode=auto` now selects the final **pre-quant** stop-time BPB.
  - For longer runs, `metric_mode=auto` selects post-quant BPB.
  - If sliding-window eval is enabled and present, `metric_mode=auto` selects final sliding-window BPB.
- ✅ **Verified the new harness remotely**
  - Smoke run: `modal_harness_smoke_9x512`
  - Confirmed the remote log showed:
    - `model_shape:num_layers:9 model_dim:512`
    - `eval_config:eval_stride:0`
  - Stopped early after verifying config to avoid unnecessary spend.

## Key Discoveries
1. **The main bug was organizational, not mathematical**
   - We violated single-source-of-truth. The assumed baseline, code defaults, wrapper behavior, and logs were not aligned.
2. **The old 2-minute comparison is not trustworthy**
   - Because the baseline run was mislabeled, the earlier architecture interpretation should not be used.
3. **Short-run post-quant BPB is a bad screening metric for this fork**
   - At 2 minutes, `final_int6_roundtrip_exact` exploded to ~`7.26 BPB` for both runs, which overwhelms any architecture signal.
   - The pre-quant stop-time BPB (~`2.082`) is much more stable for short-run screening.
4. **Sliding-window eval works on 1xH100, but is too expensive for routine dev**
   - Good for candidate confirmation, bad for cheap iteration.

## Bugs & Issues
1. **Wrapper env forwarding bug**
   - Local env overrides did not automatically reach the remote Modal container.
   - Fix: added explicit override plumbing in `train_modal.py`.
2. **Default-shape drift in `sota_train_gpt.py`**
   - Defaults were already `6x640`, not `9x512`.
   - This made the assumed baseline incorrect.
3. **Insufficient effective-config logging**
   - Earlier logs printed `model_params` but not explicit shape/eval settings.
   - Fix: added `model_shape:` and `eval_config:` log lines.

## Key Learnings
- **Reliable ML experimentation starts with config hygiene**
  - If the run does not prove what config actually executed, the result is not trustworthy.
- **One metric per stage**
  - Short runs: use pre-quant stop-time BPB.
  - Longer candidate runs: use post-quant BPB.
  - Final candidate scoring: add sliding-window eval.
- **Do not infer too much from matched parameter counts**
  - `9x512` and `6x640` can be close in parameter budget, so parameter count alone is not enough to validate the config.
- **Fail loud, not quiet**
  - A good experiment wrapper should reject conflicting overrides and verify remote reality against requested config.

## Files Modified
- `train_modal.py`
  - Rewritten as an explicit, self-checking Modal harness.
- `sota_train_gpt.py`
  - Added effective-config log lines for shape and eval settings.

## Ready for Next Session
- ✅ Harness now supports explicit, verified runs like:
```bash
modal run train_modal.py --run-id base_9x512_2m --max-wallclock 120 --overrides 'NUM_LAYERS=9,MODEL_DIM=512,EVAL_STRIDE=0'
```
```bash
modal run train_modal.py --run-id wide_6x640_2m --max-wallclock 120 --overrides 'NUM_LAYERS=6,MODEL_DIM=640,EVAL_STRIDE=0'
```
- ✅ Short-run metric policy is defined and encoded.
- 🔧 Need one fresh, clean A/B rerun using the new harness.
- 🔧 Do not use the earlier mislabeled "baseline vs variant" result for decision-making.

## Context for Future
This session was mostly about recovering trust in the experiment loop. The user was not blocked by lack of ideas; they were blocked by ambiguity about what actually ran and what number to trust. The main output is a stricter experiment harness and a clearer metric ladder. The next experiment should be boring: two explicit short runs, same seed, same wallclock, same eval mode, comparing only `9x512` vs `6x640` on the selected pre-quant metric from the new `=== SUMMARY ===` block.
