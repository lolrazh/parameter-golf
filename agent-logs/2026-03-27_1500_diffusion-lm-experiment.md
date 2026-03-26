# Continuous Diffusion LM — From Scratch on MLX

**Date:** 2026-03-27
**Agent:** Claude Opus 4.6 (1M context)
**Status:** 🔄 Ongoing

Building on `2026-03-27_0100_pr885-submission.md`

## User Intention
User wanted to explore frontier diffusion language models as a fun experiment, independent of the Parameter Golf competition. The goal evolved from "what if we trained a diffusion LM?" to "I want the MOST diffusion-coded approach possible" — which led to implementing CDCD-style continuous diffusion (Gaussian noise on embeddings, score interpolation, ODE sampling) rather than the simpler masked diffusion (MDLM). The session was heavily educational: user wanted ELI5 explanations of every component, analogies to their existing GPT, and understanding of what came before each technique.

## What We Accomplished
- ✅ **Deep research on diffusion LM landscape** — Two parallel Opus research agents surveyed the entire field: MDLM, SEDD, LLaDA, Dream 7B, Mercury, Gemini Diffusion, CDCD, Plaid, CoDAR, Block Diffusion, CARD, Seed Diffusion, FMLM, consistency models. Organized by approach family and recency.
- ✅ **DIFFUSION_ARCH.md created** — Architecture north star doc with ELI5 explanations of every component, the full pipeline, what changed from GPT, what stayed the same, all key equations, and an improvement roadmap with implementation status.
- ✅ **train_diffusion_mlx.py built from scratch** — 1451-line continuous diffusion LM on MLX. Copied from train_gpt_mlx.py and surgically converted: bidirectional attention, AdaLN conditioning, learned L2-normalized embeddings, score interpolation, probability flow ODE sampling.
- ✅ **4 frontier improvements implemented:**
  - Self-conditioning (Analog Bits, 2022) — draft forward pass fed back as hint
  - Heun solver (Karras et al., 2022) — 2nd order ODE integration
  - Time warping (CDCD, 2022) — adaptive noise level sampling
  - CoDAR-style AR rounding decoder (CoDAR, March 2026) — contextual token decoding
- ✅ **Smoke tests passing** — Model trains (loss 7.08 → 2.97 over 2500 steps), generates text from noise (English words emerging after 50 steps), Heun + self-conditioning verified working.
- ✅ **Critical mx.compile bug found and fixed** — `mx.compile` freezes `mx.random` calls and Python `if`-branches at trace time. All randomness moved outside compiled functions.
- ✅ **Val loss divergence debugged and fixed** — Root cause: frozen random values in compiled loss function caused val to measure a single noise realization instead of the average. Fix: pass t, eps as external arguments. Val loss now tracks training (4.85 at 500 steps, down from diverging to 23).

## Technical Implementation

### Architecture (CDCD-style continuous diffusion)
- **DiffusionLM class**: Learned L2-normalized embeddings (64-dim) → Gaussian noise → input_proj (128→512, accounts for self-conditioning) → 9 bidirectional transformer blocks with AdaLN → output_head (512→1024 vocab logits) → softcap
- **AdaLN**: Each block has `adaln = CastedLinear(dim, 4*dim)` producing (attn_scale, attn_shift, mlp_scale, mlp_shift) from timestep embedding
- **Timestep encoding**: ln(t)/4 → sinusoidal → MLP(dim→dim, SiLU, dim→dim) → broadcast to [B,1,dim]
- **Score interpolation**: score = (E[x₀|z,t] - z) / t² where E[x₀] = softmax(logits) @ embedding_matrix
- **Sampling**: Probability flow ODE with Heun solver, 100-200 steps, log-spaced t schedule from t_max=300 → t_min=1

### Key design choices
- `embed_dim=64` (separate from model_dim=512) — CDCD/Plaid sweet spot
- `t_min=1.0, t_max=300.0` — tokens are discrete, need large noise to drown signal
- Input scaling: `z / sqrt(t² + 1)` maintains unit variance regardless of noise level
- Two compiled functions (with/without self-conditioning) since `if`-branches freeze in mx.compile
- All randomness (t, eps, self-cond coin flip) generated outside compiled functions via numpy

**Files Created:**
- `train_diffusion_mlx.py` — Full training + eval + sampling script (1451 lines)
- `DIFFUSION_ARCH.md` — Architecture doc with ELI5 explanations and improvement roadmap

**Files NOT Modified:**
- `train_gpt_mlx.py` — Original GPT script untouched
- `train_gpt.py` — CUDA submission script untouched

## Bugs & Issues Encountered

1. **mx.compile freezes mx.random values** — Every call to a compiled function returns the SAME random values. `mx.random.uniform(shape=(B,))` inside `mx.compile` generates values once at trace time, then replays them. Training appeared to work because different input tokens each step masked the frozen noise. Val loss diverged to 23 (3x random guessing) because same tokens + same frozen noise = meaningless fixed-point measurement.
   - **Fix:** Generate all randomness (t, eps) outside compiled functions using numpy/mx.random, pass as arguments. Requires separate compiled functions for self-cond on/off since Python `if`-branches also freeze.
   - **Verification:** Val loss now tracks training: 7.09 → 5.02 → 4.78 → 4.65 → 4.85 (healthy, slight overfitting from tiny 8-seq batch).

2. **mx.compile freezes Python if-branches** — `if float(mx.random.uniform(...)) < 0.5:` evaluates ONCE at trace time. All subsequent calls follow the same branch. Self-conditioning was either always-on or always-off per compiled function, not 50/50.
   - **Fix:** Same as above — compile two versions (with/without self-cond), choose randomly at call time from Python.

3. **Early training loss spike** — Loss spikes from 7 → 10.5 in first 7 steps before recovering. Caused by self-conditioning on a random model: the draft pass predicts garbage, which misleads the real pass.
   - **Status:** Unresolved but benign — model recovers by step 50. Could be fixed by disabling self-conditioning for the first N steps.

4. **Step time doubling over training** — 550ms → 1265ms/step on M4 Air. Partially thermal throttling, partially self-conditioning overhead (50% of steps do 2 forward passes).
   - **Status:** Accepted limitation of M4 Air thermal envelope + self-conditioning cost.

5. **No BPB metric** — Diffusion val_loss (reconstruction CE at random noise levels) is not comparable to GPT val_bpb. Need ELBO evaluation (integrate weighted CE over deterministic noise grid) for proper BPB, which requires ~32-128 forward passes per val sequence.
   - **Status:** Not implemented. Required for any competition comparison.

## Key Learnings

- **mx.compile and stochastic models don't mix** — CRITICAL: any `mx.random` call inside `mx.compile` is frozen at trace time. Any Python control flow (`if/else`) based on MLX values is also frozen. For diffusion models (which are inherently stochastic), ALL randomness must live outside the compiled boundary. This likely affects anyone using `mx.compile` with dropout, stochastic depth, or any random augmentation.
- **Continuous diffusion loss ≠ autoregressive loss** — The diffusion val_loss is an MC estimate of E_t[CE(f(z_t,t), x₀)], not a likelihood bound. It's useful for tracking training but not for computing BPB or comparing with AR models. The proper ELBO requires deterministic integration over noise levels.
- **Score interpolation is the key CDCD innovation** — Train with familiar CE loss, derive the continuous score analytically: `score = (E[x₀] - z) / t²`. No score matching loss needed. This is why CDCD works where Diffusion-LM (2022) didn't.
- **Self-conditioning is free at inference** — During ODE sampling, E[x₀] from step N naturally feeds into step N+1. The only cost is during training (50% of steps need a draft forward pass).
- **Mercury uses discrete masking, NOT continuous diffusion** — Despite initial reports, Mercury (Inception Labs) is MDLM-family, not CDCD-family. Gemini Diffusion's internals are unknown. True continuous text diffusion at scale is still mostly open research.

## Architecture Decisions

- **CDCD over MDLM** — User explicitly wanted "the most diffusion-coded" approach. CDCD uses real Gaussian noise and ODE sampling (true diffusion), while MDLM is "glorified BERT with iterative unmasking." Educational value drove this choice over practical performance.
- **embed_dim=64 (separate from model_dim=512)** — Plaid found small embedding dimensions work best for diffusion dynamics. Tokens need to be distinguishable in continuous space but not too high-dimensional (where noise dynamics are worse). 64 is the CoDAR/CDCD sweet spot.
- **No tied embeddings** — GPT ties input and output embeddings (same matrix). Diffusion can't: input is 64-dim continuous embeddings, output is 512-dim hidden states → 1024-dim logits. Different shapes, different purposes.
- **Heun over Euler as default** — 2x forward passes per step but smoother ODE trajectories. Since diffusion already does ~200 steps, the marginal cost is acceptable and the quality improvement is real.
- **Self-conditioning via two compiled functions** — Rather than trying to make the self-conditioning conditional work inside mx.compile (which is impossible due to frozen branches), we compile two versions and select at call time. Slightly more memory but correct behavior.

## Ready for Next Session
- ✅ **train_diffusion_mlx.py working** — Training, eval, and generation all functional
- ✅ **DIFFUSION_ARCH.md comprehensive** — Full architecture doc with ELI5, equations, improvement roadmap
- ✅ **Val loss bug fixed** — Randomness externalized from mx.compile
- 🔧 **ELBO evaluation needed** — For BPB comparison with GPT, need deterministic multi-sample evaluation
- 🔧 **CoDAR decoder not yet trained** — Class implemented but needs a training loop
- 🔧 **Consistency model not implemented** — Documented in arch doc as the "level 5" improvement; needs a trained teacher first
- 🔧 **Longer training run with fix** — The 2500-step run used the buggy code; need to rerun with fixed version
- 🔧 **Larger batch sizes** — Current 4096-token batches cause overfitting. Need 131K+ for meaningful training

## Context for Future
This session was a pure exploration/learning exercise — building a continuous diffusion LM from scratch to understand the field. The model trains and generates text from noise, but it's far from competitive with the GPT baseline (no BPB metric, tiny batch, ~2.5x slower per step). The real value is educational: user now understands the full CDCD pipeline (embeddings → noise → score interpolation → ODE → rounding), all 4 improvements (self-cond, Heun, time warp, CoDAR), and the frontier of the field (consistency models). The mx.compile frozen-random bug is a critical lesson for any future MLX work with stochastic models.
