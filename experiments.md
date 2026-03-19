# Experiment Ledger

Local MLX experiments on M4 Air. Comparing val_loss at 200 steps (SEED=1337, SEQ_LEN=512, BATCH=4096, GRAD_ACCUM=1).

**Rules**: Change ONE thing per experiment. Keep seed fixed. Differences < 0.1 val_loss = noise.

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
