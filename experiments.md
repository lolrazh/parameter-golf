# Experiment Ledger

Local MLX experiments on M4 Air. Comparing val_loss at 200 steps (SEED=1337, SEQ_LEN=512, BATCH=4096, GRAD_ACCUM=1).

**Rules**: Change ONE thing per experiment. Keep seed fixed. Differences < 0.1 val_loss = noise.

**Harness note (2026-03-19):** A later SOTA-fork `9x512 vs 6x640` 2-minute Modal comparison was invalidated because the assumed `9x512` "baseline" inherited `6x640` defaults from `sota_train_gpt.py`. Do not use that mislabeled comparison for architecture decisions; rerun with explicit overrides through `train_modal.py`. This note does **not** invalidate the H100 rows below, which came from the earlier baseline script.

| # | Run ID | Change from baseline | val_loss | delta | verdict | notes |
|---|--------|---------------------|----------|-------|---------|-------|
| 0 | baseline_v1 | (none — default config) | 4.3525 | — | baseline | 9L 512d 8h 4kv 2x_mlp, 17M params, ~102s |
| 1 | lr_0.25x | all LRs × 0.25 | 4.9135 | +0.561 | worse | way too slow to learn in 200 steps |
| 2 | lr_0.5x | all LRs × 0.5 | 4.5117 | +0.159 | worse | still too conservative |
| 3 | lr_2x | all LRs × 2.0 | 4.1910 | -0.162 | BETTER | despite early spike to 8.77, recovered and won |
| 4 | lr_1.5x | all LRs × 1.5 | 4.2450 | -0.108 | better | between baseline and 2x, as expected |
| 5 | lr_3x | all LRs × 3.0 | 4.0579 | -0.295 | BETTER | spike to 11.97! but recovered. still climbing? |
| 6 | lr_4x | all LRs × 4.0 | 3.9874 | -0.365 | BEST | spike to 15.0, recovered. brutal throttling though |
| 7 | softcap_15 | LOGIT_SOFTCAP=15 (vs 30) | 4.0863 | +0.028 | noise | no meaningful change. tighter cap ≠ better here |
| 8 | smear | smear module (gated 1-token lookback) | 4.0482 | -0.010 | noise | +12 params, no meaningful improvement |
| 9 | deep_12L_448d | 12 layers × 448d (vs 9×512) | 4.1726 | +0.115 | worse | deeper+narrower hurt. more layers = slower convergence? |
| 10 | wide_6L_640d | 6 layers × 640d (vs 9×512) | 3.9868 | -0.071 | BEST | wider+shallower wins at 200 steps. 17.9M params |
| 11 | mlp_3x | MLP_MULT=3 (vs 2) | 4.0571 | -0.001 | noise | 21.8M params (+28%), no improvement. bad param ROI |

**H100 runs (1xH100, 2 min, 524K batch, val on 1M tokens)**

| # | Run ID | Change | val_loss | val_bpb | post-quant BPB | steps | verdict |
|---|--------|--------|----------|---------|----------------|-------|---------|
| 12 | baseline_h100 | (default 9L×512d) | 2.6852 | 1.5903 | 1.6189 | 341 | baseline |
| 13 | wide_6L_640d_h100 | 6L×640d | 2.6517 | 1.5705 | 1.6024 | 352 | BEST |
| 14 | deep_12L_448d_h100 | 12L×448d | 3.0072 | 1.7810 | 1.8363 | 252 | WORST |
| 32 | baseline_5min | default 9L×512d, 5 min | 2.3195 | 1.3738 | 1.3766 | 856 | 5-min baseline |

**Width ceiling search (local M4, 3× LR baseline = 6L×640d @ 3.9868)**

| # | Run ID | Change | val_loss | delta | verdict | notes |
|---|--------|--------|----------|-------|---------|-------|
| 15 | wide_5L_720d | 5L×720d (18.9M params) | 4.0225 | +0.036 | worse | too wide, too few layers |
| 16 | wide_4L_768d | 4L×768d (17.3M params) | 4.0694 | +0.083 | worse | 4 layers not enough depth |
| 17 | val_embed_6L_640d | value embeddings on 6L×640d | 4.0020 | +0.015 | noise | +324K params, scales need more steps to learn? |
| 18 | recurse_6L_640d_2x | depth recurrence 2× on 6L×640d | 4.0885 | +0.102 | worse | 2× slower per step, double compute but same params |
| 19 | lawa_6L_640d | LAWA (5 snapshots, every 10 steps) | 4.0197 | +0.033 | noise | averaging too-noisy snapshots at 200 steps. needs longer runs |
| 20 | val_embed_fixed | val_emb_scale=0.1 init (vs 0.0) | 4.0019 | +0.015 | noise | init wasn't the issue. val_embed just doesn't help at 200 steps |
| 21 | swiglu_6L_640d | SwiGLU activation (vs relu²) | 4.0525 | +0.066 | worse | matched params, slower convergence. relu² wins at this scale |

**400-step runs (local M4, 6L×640d, 3× LR)**

| # | Run ID | Change | val_loss | delta | verdict | notes |
|---|--------|--------|----------|-------|---------|-------|
| 22 | baseline_400 | 6L×640d baseline at 400 steps | 3.7124 | — | baseline | ~5.5 min, big drop from 200-step (3.99→3.71) |
| 23 | val_embed_400 | value embeddings at 400 steps | 3.7144 | +0.002 | noise | still nothing. val_embed not helping even with more steps |

**Rapid sweep on 6L×640d (local M4, 3× LR, 200 steps, baseline = 3.9868)**

| # | Run ID | Change | val_loss | delta | verdict | notes |
|---|--------|--------|----------|-------|---------|-------|
| 24 | grad_clip | GRAD_CLIP_NORM=1.0 | 3.9972 | -0.009 | noise | marginal, leaning positive |
| 25 | muon3 | MUON_BACKEND_STEPS=3 (vs 5) | 4.1522 | +0.165 | WORSE | worse orthogonalization, not worth speed gain |
| 26 | full_mha | NUM_KV_HEADS=8 (full MHA) | 4.0212 | +0.034 | noise | extra KV params didn't help |
| 27 | heads4 | NUM_HEADS=4 head_dim=160 | 3.9907 | +0.004 | noise | halving heads barely hurts! saves KV params |
| 28 | drop_mlp0 | DROP_FIRST_MLP=1 | 4.0149 | +0.028 | noise | MLP matters even on layer 0 |
| 29 | lr_retune_2x | 6L×640d at 2× LR | 4.1071 | +0.120 | worse | 3× is right for this shape too |
| 30 | lr_retune_4x | 6L×640d at 4× LR | 4.0076 | +0.021 | noise | 3× still best |
| 31 | muon_mom_099 | MUON_MOMENTUM=0.99 (vs 0.95) | 4.0014 | +0.015 | noise | slightly more aggressive, no help |
