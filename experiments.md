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

**SOTA-fork quant sweeps (1xH100, 5 min train, 1M-token sweep proxy on same checkpoint)**

**Rule**: Compare quant-sweep rows only against each other. The sweep uses a fast proxy eval and is not directly comparable to the trainer's final full-val BPB.

| # | Run ID | Recipe / change | proxy prequant BPB | proxy post-quant BPB | delta | total bytes | verdict |
|---|--------|-----------------|--------------------|----------------------|-------|-------------|---------|
| 33 | clean_9x512_5m_quant2 | current int6 export | 1.7956 | 3.0332 | +1.2376 | 5,033,018 | baseline |
| 34 | clean_9x512_5m_quant2 | fp16_tok_emb | 1.7956 | 3.0332 | +1.2376 | 5,466,490 | no gain |
| 35 | clean_9x512_5m_quant2 | attn8_mlp6 | 1.7956 | 2.7745 | +0.9789 | 6,814,427 | better |
| 36 | clean_9x512_5m_quant2 | outer8_middle6 | 1.7956 | 2.5734 | +0.7778 | 6,194,293 | BEST |
| 37 | clean_9x512_5m_quant2 | fp16_tok_emb_attn8 | 1.7956 | 2.7740 | +0.9784 | 7,254,647 | better, worse than outer8 |

**Quant sensitivity probe (1xH100, 5 min train, 1M-token sweep proxy on same checkpoint)**

**Rule**: These rows are for ranking which blocks deserve extra bits. Compare within this probe run only.

| # | Run ID | Recipe / change | proxy prequant BPB | proxy post-quant BPB | delta | total bytes | verdict |
|---|--------|-----------------|--------------------|----------------------|-------|-------------|---------|
| 38 | clean_9x512_5m_probe3 | current int6 export | 1.7955 | 3.0401 | +1.2446 | 4,945,700 | baseline |
| 39 | clean_9x512_5m_probe3 | outer8_middle6 | 1.7955 | 2.5603 | +0.7647 | 6,017,292 | best coarse recipe |
| 40 | clean_9x512_5m_probe3 | probe_block_0_int8 | 1.7955 | 2.6105 | +0.8150 | 5,554,656 | MOST sensitive block |
| 41 | clean_9x512_5m_probe3 | probe_block_1_int8 | 1.7955 | 2.7692 | +0.9736 | 5,518,118 | second-most sensitive |
| 42 | clean_9x512_5m_probe3 | probe_block_8_int8 | 1.7955 | 2.9729 | +1.1774 | 5,487,122 | small gain |
| 43 | clean_9x512_5m_probe3 | probe_block_2_int8 .. probe_block_7_int8 | 1.7955 | 2.9095 .. 3.0324 | +1.1140 .. +1.2368 | ~5.45M .. 5.57M | mostly weak / noisy |

**Front-heavy mixed quant follow-up (1xH100, 5 min train, 1M-token sweep proxy on same checkpoint)**

| # | Run ID | Recipe / change | proxy prequant BPB | proxy post-quant BPB | delta | total bytes | verdict |
|---|--------|-----------------|--------------------|----------------------|-------|-------------|---------|
| 44 | clean_9x512_5m_followup1 | current int6 export | 1.7945 | 3.0692 | +1.2747 | 4,946,288 | baseline |
| 45 | clean_9x512_5m_followup1 | outer8_middle6 | 1.7945 | 2.6002 | +0.8056 | 6,045,820 | solid |
| 46 | clean_9x512_5m_followup1 | front2_8_middle6 | 1.7945 | 2.3373 | +0.5427 | 6,094,980 | BETTER |
| 47 | clean_9x512_5m_followup1 | front2_back1_8_middle6 | 1.7945 | 2.2886 | +0.4940 | 6,711,561 | BEST |

**Expanded front-heavy mixed quant follow-up (1xH100, 5 min train, 1M-token sweep proxy on same checkpoint)**

| # | Run ID | Recipe / change | proxy prequant BPB | proxy post-quant BPB | delta | total bytes | verdict |
|---|--------|-----------------|--------------------|----------------------|-------|-------------|---------|
| 48 | clean_9x512_5m_followup2 | current int6 export | 1.7962 | 3.1298 | +1.3335 | 4,996,226 | baseline |
| 49 | clean_9x512_5m_followup2 | outer8_middle6 | 1.7962 | 2.6121 | +0.8159 | 6,096,011 | okay |
| 50 | clean_9x512_5m_followup2 | front2_8_middle6 | 1.7962 | 2.3638 | +0.5675 | 6,166,558 | better |
| 51 | clean_9x512_5m_followup2 | front2_back1_8_middle6 | 1.7962 | 2.3144 | +0.5182 | 6,669,190 | strong |
| 52 | clean_9x512_5m_followup2 | front3_back1_8_middle6 | 1.7962 | 2.1003 | +0.3041 | 7,335,031 | BEST |
| 53 | clean_9x512_5m_followup2 | front2_back2_8_middle6 | 1.7962 | 2.2628 | +0.4666 | 7,247,510 | better |
| 54 | clean_9x512_5m_followup2 | front2_back1_attn8 | 1.7962 | 2.9375 | +1.1412 | 5,601,427 | bad, attention-only not enough |

**Integrated end-to-end quant preset eval (1xH100, 5 min train, normal trainer final eval)**

| # | Run ID | Change | prequant BPB | post-quant BPB | quant gap | total bytes | verdict |
|---|--------|--------|--------------|----------------|-----------|-------------|---------|
| 55 | clean_9x512_5m_front3b1_eval2 | QUANT_PRESET=front3_back1_8_middle6 | 1.4444 | 1.7629 | +0.3185 | 7,225,310 | BIG WIN |

**Aggressive larger-model proxy (1xH100, 5 min train, sliding-window attempt)**

| # | Run ID | Change | prequant BPB | post-quant BPB | sliding BPB | total bytes | verdict |
|---|--------|--------|--------------|----------------|-------------|-------------|---------|
| 56 | frontier_11x640_5m_sw64 | 11L×640d, QUANT_PRESET=front3_back1_8_middle6, stride=64 | 1.6784 | 2.5008 | timeout | 9,064,054 | worse at 5 min; fits budget, undertrained |

**SmearGate + BigramHash baseline (1xH100, 5 min train, normal trainer final eval)**

| # | Run ID | Change | prequant BPB | post-quant BPB | quant gap | total bytes | verdict |
|---|--------|--------|--------------|----------------|-----------|-------------|---------|
| 57 | smear_bigram_9x512_5m | USE_SMEARGATE=1, BIGRAM_HASH=4096x128, QUANT_PRESET=front3_back1_8_middle6 | 1.4537 | 1.7074 | +0.2537 | 7,688,931 | promising: post-quant better, prequant slightly worse |

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
