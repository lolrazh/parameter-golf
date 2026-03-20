# SmearGate + BigramHash Baseline Under Tight Budget

**Date:** 2026-03-20
**Agent:** Codex (GPT-5)
**Status:** ✅ Complete

## User Intention
User wanted to move faster toward the live frontier without wasting the last small chunk of budget on broad catch-up work. The specific decision was whether to spend one paid run on a known frontier ingredient (`SmearGate + BigramHash`) or save that money for something more novel. The practical goal became: implement the smallest honest version of the idea, run exactly one 5-minute baseline, and decide whether it earned another dollar.

## What We Accomplished
- ✅ **Confirmed the rule is artifact size, not parameter count**
  - The challenge cap is `16,000,000` bytes of `code + compressed model`, not a 16M-parameter cap.
  - This matters because it means additional learned structure is allowed as long as the compressed artifact still fits.
- ✅ **Confirmed `SmearGate + BigramHash` was not already implemented in this repo**
  - Local search found only the earlier toy `smear` experiment note.
  - No implementation of the current frontier-style bigram embedding path existed in the codebase.
- ✅ **Implemented a minimal `SmearGate + BigramHash` path in `sota_train_gpt.py`**
  - Added:
    - `USE_SMEARGATE`
    - `BIGRAM_HASH_BUCKETS`
    - `BIGRAM_HASH_DIM`
    - `SMEAR_GATE_INIT`
  - Added a hashed bigram embedding table (`4096 x 128` by default)
  - Added a projection from hashed bigram embedding into model space
  - Added a per-channel learned smear gate
  - Applied the feature only at the embedding stage, before the transformer stack
- ✅ **Wired the new parameters into training and export**
  - Bigram hash embedding weights join the token optimizer
  - Bigram projection weight joins the matrix optimizer
  - Smear gate joins the scalar optimizer
  - Quantization/export treats the new embedding-like tensor family as high-sensitivity weights
  - Logging now prints whether SmearGate is enabled and the hash-table shape
- ✅ **Spent exactly one 5-minute 1xH100 run on the new baseline**
  - Run ID: `smear_bigram_9x512_5m`
  - Config:
    - `9x512`
    - `QUANT_PRESET=front3_back1_8_middle6`
    - `USE_SMEARGATE=1`
    - `EVAL_STRIDE=0`

## Why This Was The Right Cheap Test
The right way to evaluate a new architectural idea on a tight budget is:

1. **Keep the control architecture fixed**
   - Same `9x512` shape as the current best integrated control.
2. **Keep the working quant stack fixed**
   - Same `front3_back1_8_middle6` export preset.
3. **Turn on only one new idea**
   - `SmearGate + BigramHash`.
4. **Ask one question**
   - Does this idea improve the final post-quant score enough to justify another run?

That is exactly what this session did.

## What SmearGate + BigramHash Is Doing (First Principles)
The plain transformer must learn token-pair relationships indirectly through attention. That means if token `B` behaves differently after token `A` than after token `C`, the model has to infer that through attention weights and intermediate representations.

`SmearGate + BigramHash` gives the model a cheap shortcut:

1. **BigramHash**
   - Hash the pair `(previous token, current token)` into a bucket.
   - Look up a learned embedding for that bucket.
   - Project it into model dimension.
   - This gives the model a cheap “pair identity hint” before attention does any work.

2. **SmearGate**
   - Learn a tiny gate that mixes the current token embedding with information from the previous token.
   - This injects immediate local-context structure into the residual stream before the transformer starts.

The hope is that this improves sample efficiency for common web-text token pairs and templates, especially when parameter budget is tight.

## The Run Result

### New Baseline
- Run ID: `smear_bigram_9x512_5m`
- Model params: `22,368,840`
- Final metrics:
  - pre-quant BPB: `1.4537`
  - post-quant BPB: `1.70737839`
  - quant gap: `+0.2537`
  - total artifact bytes: `7,688,931`

### Compare To Current Integrated Control
- Control run: `clean_9x512_5m_front3b1_eval2`
  - pre-quant: `1.4444`
  - post-quant: `1.7629`
  - quant gap: `+0.3185`
  - total bytes: `7,225,310`

### Interpretation
- Pre-quant got slightly worse (`1.4444 → 1.4537`)
- Post-quant got clearly better (`1.7629 → 1.7074`)
- Quant gap got meaningfully smaller (`+0.3185 → +0.2537`)

That means the new feature likely made the model **more quantization-friendly**, even though it did not improve the raw pre-quant score in this one run.

## Key Discoveries
1. **`SmearGate + BigramHash` is promising in this repo**
   - It is not dead on arrival.
   - One run was enough to show meaningful post-quant improvement.
2. **The gain is currently in robustness, not raw pre-quant quality**
   - This is still useful because the challenge scores the compressed artifact, not the raw checkpoint.
3. **The feature is cheap**
   - Parameter count only increased modestly.
   - Artifact size stayed comfortably below the 16MB cap.
4. **The previous toy “smear” local result did not rule this out**
   - The old one-token smear experiment was not the same as the current frontier idea.
   - The added bigram hash path appears to matter.

## Files Modified
- `sota_train_gpt.py`
  - Added minimal `SmearGate + BigramHash` support
  - Added embedding-side bigram hash lookup and projection
  - Added learned smear gate
  - Wired new params into optimizer groups
  - Added config logging for the new feature
- `experiments.md`
  - Added the `smear_bigram_9x512_5m` result row

## Ready for Next Session
- ✅ `SmearGate + BigramHash` is now implemented in the SOTA fork.
- ✅ One paid baseline run suggests it improves final compressed-model quality.
- ✅ This idea earned another run if budget allows.
- 🔧 The next reasonable move is to **build on top of this**, not revert it:
  - combine it with another frontier lever
  - or tune it slightly, but only if the budget supports another focused run

## Context for Future
This was a budget-disciplined session. The goal was not to prove `SmearGate + BigramHash` is the final breakthrough; it was to decide whether the idea is worth keeping in the stack. The answer is yes. The most important thing the run showed is that the feature improves the quantity that actually matters in this challenge — the post-quant final score — while keeping the artifact comfortably under budget. That is enough to justify stacking the next idea on top rather than backing away from it.
