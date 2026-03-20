# Front-Heavy Quant Breakthrough + First-Principles Explanation

**Date:** 2026-03-20
**Agent:** Codex (GPT-5)
**Status:** ✅ Complete

## User Intention
User wanted to understand the quantization gains from first principles, not just trust the metric movement. The practical goal was to turn the winning mixed-quant recipe from a checkpoint-only sweep result into the main export path and verify that it still wins in the real training script.

## What We Accomplished
- ✅ **Ran a block-sensitivity probe on the current `9x512` control**
  - The probe showed the quantization damage is highly non-uniform across depth.
  - Block `0` was the largest sensitivity spike.
  - Block `1` also mattered clearly.
  - The last block helped somewhat.
  - Most middle blocks were much less sensitive.
- ✅ **Used that probe to build front-heavy mixed-precision candidates**
  - Tested:
    - `front2_8_middle6`
    - `front2_back1_8_middle6`
    - `front3_back1_8_middle6`
    - `front2_back2_8_middle6`
    - `front2_back1_attn8`
- ✅ **Found the best checkpoint-only recipe**
  - On the 1M-token sweep proxy:
    - `current`: `3.1298 BPB`
    - `front3_back1_8_middle6`: `2.1003 BPB`
  - Artifact bytes stayed comfortably under budget.
- ✅ **Integrated the winning recipe into the main training/export path**
  - Added `QUANT_PRESET` support to `sota_train_gpt.py`.
  - Supported presets now include:
    - `current`
    - `outer8_middle6`
    - `front2_8_middle6`
    - `front2_back1_8_middle6`
    - `front3_back1_8_middle6`
    - `front2_back2_8_middle6`
    - `front2_back1_attn8`
- ✅ **Found and fixed the integration bug**
  - First integrated run crashed at export time because `re` was used for preset resolution but not imported.
  - Added the missing import and reran.
- ✅ **Verified the result end-to-end in the normal trainer**
  - Run ID: `clean_9x512_5m_front3b1_eval2`
  - Final result:
    - pre-quant: `1.4444 BPB`
    - post-quant: `1.76293486 BPB`
    - quant gap: `+0.3185 BPB`
    - artifact size: `7,225,310` bytes

## Why The Gain Happened (First Principles)
The simplest mental model is:

1. **Quantization is controlled damage**
   - We replace many real-valued weights with a much smaller set of representable values.
   - That always introduces approximation error.
   - The question is not whether damage happens; it is **where damage matters most**.

2. **Not all layers matter equally**
   - A transformer is not a flat bag of weights.
   - Some layers sit at chokepoints where errors get amplified downstream.
   - Early layers are especially dangerous because every later block has to build on their corrupted activations.

3. **Middle layers can often tolerate more compression**
   - If a middle block is slightly wrong, later blocks can sometimes compensate.
   - If the first few blocks are wrong, the entire residual stream is bent off-course early and everything downstream inherits that distortion.

4. **This model is front-heavy in quant sensitivity**
   - The probe confirmed that protecting block `0` helped a lot.
   - Protecting block `1` helped too.
   - Protecting the last block also helped, but less than protecting the front.
   - That means the best use of bits is not symmetric by aesthetics; it is asymmetric by measured sensitivity.

5. **Why full-block int8 beat attention-only int8**
   - The bad `front2_back1_attn8` result showed the error source is not only inside attention.
   - MLP and/or surrounding block weights in those sensitive regions matter too.
   - So the right intervention was broader: protect whole sensitive blocks, not just one submodule family.

6. **Why the gain was so large**
   - The old recipe was spending too few bits in the worst possible places.
   - We did not need a fancy learned method to improve that; we just stopped making the biggest mistake.
   - In other words, the gain was large because the baseline allocation was structurally misaligned with where error actually hurts.

## Quant Results

### Probe Run
- Run ID: `clean_9x512_5m_probe3`
- Proxy pre-quant BPB: `1.7955`
- Key probe results:
  - `current`: `3.0401`
  - `probe_block_0_int8`: `2.6105`
  - `probe_block_1_int8`: `2.7692`
  - `probe_block_8_int8`: `2.9729`
  - middle blocks were mostly weak or noisy

### Candidate Sweep
- Run ID: `clean_9x512_5m_followup2`
- Proxy pre-quant BPB: `1.7962`
- Results:
  - `current`: `3.1298`
  - `outer8_middle6`: `2.6121`
  - `front2_8_middle6`: `2.3638`
  - `front2_back1_8_middle6`: `2.3144`
  - `front2_back2_8_middle6`: `2.2628`
  - `front3_back1_8_middle6`: `2.1003`
  - `front2_back1_attn8`: `2.9375`

### Integrated End-to-End Result
- Run ID: `clean_9x512_5m_front3b1_eval2`
- Real trainer metrics:
  - pre-quant: `1.4444`
  - post-quant: `1.7629`
  - total artifact bytes: `7,225,310`

## Key Discoveries
1. **The mixed-quant win was real**
   - It was not just a sweep proxy artifact.
   - The integrated export path preserved most of the gain.
2. **The model is not “outer-layer sensitive” in a symmetric way**
   - It is much more sensitive at the front than the back.
3. **Attention-only protection is not enough**
   - Sensitive blocks need broader protection than just attention weights.
4. **We have byte budget to spend**
   - Even the better protected recipe is still far below the `16 MB` artifact limit.
   - That means quality, not size, is the active bottleneck right now.

## Files Modified
- `sota_train_gpt.py`
  - Added `QUANT_PRESET` support for named mixed-quant export recipes.
  - Added `quant_preset:` logging.
  - Fixed missing `re` import used by preset resolution.
- `quant_sweep.py`
  - Earlier in the session, expanded to include front-heavy recipe variants for checkpoint-only sweeps.
- `experiments.md`
  - Updated with probe results, follow-up recipe comparisons, and the integrated end-to-end result.

## Ready for Next Session
- ✅ Main export path now supports the winning front-heavy preset.
- ✅ End-to-end quantization is dramatically less destructive than before.
- ✅ Bigger-model experiments are now back on the table.
- 🔧 Reasonable next options:
  - test one more structural preset such as `front4_back1` or `front3_back2`
  - revisit larger raw models now that the quantizer is no longer catastrophically bad
  - consider DWQ only as a second-stage refinement after structural mixed quant has mostly plateaued

## Context for Future
This session changed the project materially. Before this, quantization was so destructive that architecture questions were partly submerged by exporter damage. Now there is a concrete, validated mixed-quant recipe with a much smaller end-to-end penalty. The deeper lesson is that quantization is an allocation problem before it is a learning problem: first spend bits where they matter, then consider more advanced methods like DWQ to polish what remains.
