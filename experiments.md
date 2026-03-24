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

**Full-stack competitive build (1xH100, 5 min train, sliding window eval stride=64)**

| # | Run ID | Change | prequant BPB | post-quant BPB | sliding BPB | quant gap | steps | total bytes | verdict |
|---|--------|--------|--------------|----------------|-------------|-----------|-------|-------------|---------|
| 58 | fullstack_11L_5m | 11L, WD, ortho init, SWA(broken), SmearGate(0.0), seq2048, batch786K, FP16 embed, QAT always-on | 1.4874 | 1.5959 | 1.5778 | +0.1085 | 386 | 9,875,401 | BIG WIN on quant gap; undertrained (779ms/step); SWA needs fix |
| 59 | fullstack_11L_ttt_v2 | same + SWA(5 snap) + TTT(1811 steps) on Thunder H100 PCIe | 1.6839 | 1.8948 | 1.8966 | +0.2109 | 258 | 8,460,774 | PCIe ~50% slower; TTT didn't help (model too undertrained); SWA worked |
| 60 | fullstack_11L_10m | 10 min train, no TTT, SWA(10 snap), Thunder H100 PCIe | 1.4038 | 1.6234 | (killed) | +0.2196 | 515 | 10,646,376 | best prequant yet; sliding eval killed by SSH drop |

**Ralph Loop: Autonomous hyperparameter sweep (1xH100 PCIe Thunder, 3 min)**

**WARNING**: These are proxy-hardware results. Hyperparameter VALUES don't transfer to 8xH100 10-min, but qualitative insights do. See `agent-logs/2026-03-21_0010_ralph-loop-autoresearch.md`.

Best proxy config found: 6L×512d, WARMDOWN_ITERS=500, BATCH=131K, WD=0.04, XSA=2, ROPE=50K

| # | Run ID | Change | steps | prequant BPB | postquant BPB | step_avg_ms | verdict |
|---|--------|--------|-------|--------------|---------------|-------------|---------|
| r01 | ralph_001 | baseline 11L (broken warmdown) | 153 | 2.0488 | 2.9142 | 1181 | LR at 12.7% entire run |
| r02 | ralph_002 | WARMDOWN_ITERS=50 | 153 | 1.6551 | 1.6854 | 1180 | MASSIVE fix: +1.23 BPB |
| r04 | ralph_004 | BATCH=262K | 409 | 1.4626 | 1.4707 | 440 | crossed 1.50 target |
| r09 | ralph_009 | BATCH=131K | 707 | 1.4559 | 1.4599 | 255 | sweet spot |
| r12 | ralph_012 | WARMDOWN=250 | 701 | 1.4271 | 1.4333 | 257 | optimal warmdown |
| r14 | ralph_014 | ROPE_BASE=50K | 698 | 1.4220 | 1.4281 | 258 | best 11L proxy |
| r25 | ralph_025 | 9L | 865 | 1.4099 | 1.4128 | 208 | fewer layers = more steps |
| r27 | ralph_027 | 7L | 1123 | 1.3895 | 1.3928 | 160 | throughput > capacity |
| r30 | ralph_030 | 6L | 1321 | 1.3861 | 1.3892 | 136 | best proxy config |

Key negative results: EMA hurt (r20: quant gap 0.155), high momentum hurt (r15), lower LRs hurt postquant (r08), small batch too noisy (r16), SmearGate off marginal (r17), delayed QAT hurt (r18).

**Algorithmic experiments (1xH100 PCIe Thunder, 3 min, TRANSFERABLE)**

| # | Run ID | Algorithmic Change | postquant BPB | sliding BPB | delta vs baseline | verdict |
|---|--------|--------------------|---------------|-------------|-------------------|---------|
| a01 | algo_001 | Temperature scaling (grid search T) | 1.3896 | — | 0.000 | No effect: softcap = built-in temp |
| a02 | algo_002 | Seq curriculum 512→2048 (compiled) | 1.5642 | — | +0.175 | FAILED: 96s compile overhead |
| a03 | algo_003 | Mixed int5 MLP / int6 attn | 1.3967 | — | +0.008 | Trade-off: -550KB artifact, +0.008 BPB |
| a04 | algo_004 | Low-rank Q (512→192→512) | 1.3915 | — | +0.002 | Neutral: QAT overhead > matmul savings |
| a05 | algo_005 | Seq curriculum (no compile) | 1.5219 | — | +0.133 | FAILED: compile essential for speed |
| a06 | algo_006 | Sliding window eval (stride=64) | 1.3887 | 1.3666 | -0.022 | FREE WIN: competition metric |
| a07 | algo_007 | Int5 MLP + 12L (extra depth from int5 savings) | 1.4535 | — | +0.064 | Artifact 12MB (fits!), needs 8xH100 to validate (proxy undertrained at 646 steps) |
| a08 | algo_008 | Batched Muon + CUDA graph | nan | — | — | NaN: double-update bug during graph capture |
| a10 | algo_010 | Batched Muon (no CUDA graph) | 1.3895 | — | +0.000 | Works! 133ms vs 136ms, +33 steps, numerically identical |
| a12 | algo_012 | Dynamic compile + seq curriculum | 1.5764 | — | +0.187 | FAILED: dynamic compile 219ms/step overhead |
| **a13** | **algo_013** | **Preset-aligned QAT (Branch A1)** | **1.3881** | — | **-0.001** | **WIN: STE matches export quant levels, best postquant** |

**Infrastructure + architecture experiments (1xH100 PCIe Thunder)**

| # | Run ID | Change | layers | steps | step_ms | prequant BPB | postquant BPB | sliding BPB | quant gap | verdict |
|---|--------|--------|--------|-------|---------|--------------|---------------|-------------|-----------|---------|
| a10 | algo_010 | Batched Muon (no CUDA graph) | 6 | 1356 | 133 | 1.3861 | 1.3895 | — | +0.003 | +33 steps over baseline, identical BPB |
| a11 | algo_011 | + Partial RoPE 16/64 + LN Scale + EMA + fused QKV | 6 | 1291 | 140 | 1.3974 | 1.4276 | 1.4039 | +0.030 | WORSE at 6L: new features hurt shallow models |
| v2-9L | fullstack_v2 | Same as a11 but 9L | 9 | 872 | 206 | 1.4313 | 1.4975 | — | +0.066 | WORSE: too few steps for 9L on proxy |
| **v2-6L-10m** | **tenmin_6L_control_v2** | **ralph_030 recipe, 10 min** | **6** | **4537** | **132** | **1.3019** | **1.3077** | **1.2851** | **+0.006** | **BEST: 1.2851 sliding on 1xH100 PCIe** |
| v2-11L-10m | tenmin_11L_fullstack_v2 | Partial RoPE + LN Scale + EMA, 10 min | 11 | 2378 | 252 | 1.3278 | 1.3382 | 1.3141 | +0.010 | 6L wins by 0.029 BPB — not enough steps for 11L depth to help on 1xPCIe |

**Big swing experiments (1xH100 PCIe Thunder, 5 min)**

| # | Run ID | Experiment | layers | steps | step_ms | prequant BPB | postquant BPB | sliding BPB | artifact MB | verdict |
|---|--------|-----------|--------|-------|---------|--------------|---------------|-------------|-------------|---------|
| moe | moe_6L_5m | MoE 4 experts × 384 hidden, top-2 routing | 6 | — | — | — | — | — | — | CRASHED: DDP unused params (need find_unused_parameters=True) |
| **vocab** | **vocab4096_6L_5m** | **sp4096 tokenizer, int8 embed, 3.34 bytes/tok** | **6** | **2169** | **138** | **1.3178** | **1.3219** | **1.3054** | **11.7** | **BIG WIN: 1.3054 in 5 min vs sp1024's 1.2851 in 10 min. Half the time, similar BPB. 4.3MB headroom.** |
| ens | ensemble_test | Two models (6L+4L) in one artifact | — | — | — | — | — | — | — | CRASHED: OOM — two compiled models exceed 80GB |
| a14 | algo_014 | Token-class calibration (Branch D1) | 1.3891 | — | +0.000 | No effect: per-class temps cancel in aggregate BPB |
| a15 | algo_015 | Neural cache eval (Branch C3) | 🔄 | — | — | RUNNING on GPU |

**8xH100 SXM submission runs (RunPod spot, FA3, front3_back1_8_middle6 quant)**

| # | Run ID | Config | layers | steps | step_ms | prequant BPB | postquant BPB | sliding BPB | artifact MB | verdict |
|---|--------|--------|--------|-------|---------|--------------|---------------|-------------|-------------|---------|
| **s1** | **submission_6L_sp4096** | **sp4096, 786K batch, FA3, front3_back1 quant** | **6** | **~10200** | **59** | **1.1914** | **1.1985** | **1.1818** | **13.6** | **HUGE: +0.103 BPB vs 1xPCIe best. 2.4MB headroom.** |
| **s2** | **submission_9L_sp4096** | **sp4096, 786K batch, FA3, front3_back1 quant** | **9** | **7249** | **82** | **—** | **1.1654** | **1.1484** | **17.3 (OVER)** | **0.0056 from SOTA! But 1.3MB over 16MB. Needs int5 MLP + zstd.** |
| s3 | sub_10L_sp4096_int5_swa | sp4096, int5 MLP, SWA, warmdown=3000, zstd | 10 | 🔄 | ~85 | 🔄 | 🔄 | 🔄 | 🔄 | RUNNING — the SOTA attempt |

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

**L40S proxy runs (1xL40S, 10 min, 131K batch, SDPA)**

| # | Run ID | Config | steps | step_ms | prequant BPB | postquant BPB | post-TTT BPB | artifact MB | verdict |
|---|--------|--------|-------|---------|--------------|---------------|--------------|-------------|---------|
| L40S-proteus | L40S_proteus | 11L sp1024, frontier_512.py | 1887 | 318 | 1.3455 | 1.3968 | 1.2870 | 10.1 | PROTEUS control on L40S |
| L40S-s3 | L40S_s3 | 10L sp4096, frontier_512.py | 1988 | 302 | 1.3143 | 1.3626 | 1.2515 | 11.8 | s3 wins by 0.036 post-TTT |
| L40S-11L-sp4096 | L40S_proteus_11L_sp4096 | 11L sp4096, frontier_512.py | 1847 | 325 | 1.3158 | 1.3741 | 1.0984 | 12.5 | WINNER — best post-TTT BPB, proxy checkpoint not needed — recipe is the deliverable |
| ttt-3ep-md256 | ttt_3ep_md256 | 3ep + min_doc 256, s3 checkpoint | — | — | — | — | 1.1011 | — | realistic 8xH100 config |
| ttt-11L-3ep-md256 | ttt_11L_3ep_md256 | 3ep + min_doc 256, PROTEUS 11L sp4096 | — | — | — | — | 1.0984 | — | best TTT result |

**TTT sweep (1xL40S, 2M token val subset, frozen s3 checkpoint)**

| # | Run ID | Config | BPB | vs baseline | time | verdict |
|---|--------|--------|-----|-------------|------|---------|
| ttt-v1 | ttt_v1 | r8, 2ep, const lr (baseline) | 1.2645 | — | 120s | baseline |
| ttt-v2 | ttt_v2 | rank 4 | 1.2721 | +0.008 | 121s | worse |
| ttt-v3 | ttt_v3 | rank 16 | 1.2687 | +0.004 | 121s | worse |
| ttt-v4 | ttt_v4 | 5 epochs | 1.0331 | -0.231 | 277s | massive gain |
| ttt-v5 | ttt_v5 | 10 epochs | 0.8386 | -0.426 | 539s | likely overfitting on 2M subset |
| ttt-v6 | ttt_v6 | chunk 128 | 1.2610 | -0.003 | 222s | noise |
| ttt-v7 | ttt_v7 | chunk 512 | 1.2837 | +0.019 | 42s | worse |
| ttt-v8 | ttt_v8 | context 512 | 1.2641 | -0.000 | 45s | noise |
| ttt-v9 | ttt_v9 | min_doc 256 | 1.2394 | -0.025 | 72s | free improvement |
| ttt-v10 | ttt_v10 | min_doc 2048 | 1.3028 | +0.038 | 60s | worse |
| ttt-c1 | ttt_c1 | 5ep + cosine | 1.3158 | +0.051 | 139s | cosine hurts |
| ttt-c2 | ttt_c2 | 10ep + cosine | 1.2934 | +0.029 | 260s | cosine hurts |

**1xH100 PCIe optimization benchmark (Thunder Compute, 10L sp1024, 131K batch, 30 steps)**

| # | Run ID | Config | step_ms (stable) | step_avg | speedup | verdict |
|---|--------|--------|------------------|----------|---------|---------|
| opt-1 | baseline_1xh100 | Separate QKV + Unbatched Muon + SDPA | ~187ms | 189.4ms | — | baseline |
| opt-2 | fused_batched_sdpa | Fused QKV + Batched Muon + SDPA | ~170ms | 172.6ms | 9.0% | batched Muon is the big win |
| opt-3 | fused_batched_fa3 | Fused QKV + Batched Muon + FA3 | ~153ms | 154.3ms | 18.5% | FA3 now faster with fused QKV layout |

**8xH100 SXM submission runs (frontier_512.py, fused QKV, batched Muon, FA3, int5 MLP)**

| # | Run ID | Config | seed | steps | step_ms | prequant BPB | postquant BPB | post-TTT BPB | artifact bytes | verdict |
|---|--------|--------|------|-------|---------|--------------|---------------|--------------|----------------|---------|
| s4-trial | s4_trial_int6 | 10L sp4096, int6 uniform, wd=2000, 3ep TTT | 1337 | 6,130 | 97.8 | 1.1610 | 1.1736 | **0.9531** | 16,718,251 (OVER) | Proved 0.95 BPB achievable, artifact too big |
| **s4-42** | **s4_seed42** | **10L sp4096, int5 MLP, wd=2000, 3ep TTT** | **42** | **8,238** | **72.8** | **1.1585** | **1.1995** | **0.9636** | **14,728,565** | **VALID** |
| **s4-1337** | **s4_seed1337** | **10L sp4096, int5 MLP, wd=2000, 3ep TTT** | **1337** | **8,244** | **72.8** | **1.1543** | **1.1944** | **0.9598** | **14,762,244** | **VALID — best seed** |
| **s4-2024** | **s4_seed2024** | **10L sp4096, int5 MLP, wd=2000, 3ep TTT** | **2024** | **8,239** | **72.8** | **1.1557** | **1.1955** | **0.9623** | **14,672,017** | **VALID** |
| | | **Mean (3 seeds)** | | **8,240** | | **1.1562** | **1.1965** | **0.9619** | | **SOTA: 1.1428. We beat by 0.181 BPB.** |
