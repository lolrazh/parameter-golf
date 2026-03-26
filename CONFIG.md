# Submission Config

Single source of truth. Every parameter, every change, every reason.

## Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| num_layers | 11 | SOTA PR #549 |
| model_dim | 512 | SOTA PR #549 |
| num_heads | 8 | SOTA PR #549 |
| num_kv_heads | 4 | SOTA PR #549 |
| vocab_size | 1024 | sp1024 tokenizer |
| mlp_mult | 3.0 | SOTA PR #549 |
| tie_embeddings | True | SOTA PR #549 |
| logit_softcap | 30.0 | SOTA PR #549 |
| rope_base | 10000 | SOTA PR #549 |
| rope_dims | 16 | Partial RoPE (SOTA PR #315) |
| xsa_last_n | 4 | SOTA PR #549 |
| bigram_buckets | 2048 | SOTA PR #549 |
| bigram_dim | 128 | SOTA PR #549 |
| ve_enabled | True | SOTA PR #549 |
| ve_dim | 128 | SOTA PR #549 |
| ve_layers | 9,10 | SOTA PR #549 |
| ln_scale | True | SOTA PR #549 |
| activation | LeakyReLU(0.9)^2 | Sweep in issue #140, +0.013 BPB over 0.5 |

## Training

| Parameter | Value | Source |
|-----------|-------|--------|
| train_seq_len | 2048 | SOTA PR #549 |
| train_batch_tokens | 786432 | SOTA PR #549 (8xH100) |
| train_shards | **80** | SOTA PR #549 (ALL shards) |
| iterations | 20000 | cap, wallclock stops first |
| max_wallclock_seconds | 600 | 10 min |
| warmup_steps | 20 | SOTA PR #549 |
| warmdown_iters | 3500 | SOTA PR #549 |
| seed | 1337, 42, 2025 | 3-seed submission |

## Optimizer

| Parameter | Value | Source |
|-----------|-------|--------|
| optimizer | Parallel Muon + Adam | SOTA PR #549 |
| matrix_lr | 0.025 | SOTA PR #549 |
| scalar_lr | 0.025 | SOTA PR #549 |
| tied_embed_lr | 0.035 | SOTA PR #549 |
| embed_lr | 0.6 | SOTA PR #549 |
| muon_momentum | 0.99 | SOTA PR #549 |
| muon_backend_steps | 5 | SOTA PR #549 |
| muon_wd | 0.04 | SOTA PR #549 |
| adam_wd | 0.04 | SOTA PR #549 |
| beta1 | 0.9 | SOTA PR #549 |
| beta2 | 0.95 | SOTA PR #549 |
| grad_clip_norm | 0.3 | SOTA PR #549 |

## Weight Averaging

| Parameter | Value | Source |
|-----------|-------|--------|
| ema_decay | 0.997 | SOTA PR #549 |
| swa_enabled | True | SOTA PR #549 |
| swa_every | 50 | SOTA PR #549 |

## Quantization (OUR additions on top of SOTA)

| Parameter | Value | Source |
|-----------|-------|--------|
| quant_preset | front3_back1_6_middle5 | Ours (experiments 44-55) |
| gptq_lite | True (per-row) | Ours (experiment a13) |
| entropy_reg | 0.01 | Ours (run10, halves quant gap) |
| qat_start_frac | 0.15 | Ours (run10) |
| compression | zstd-22 | Ours (beats LZMA on 5/6 presets) |

## Eval / TTT

| Parameter | Value | Source |
|-----------|-------|--------|
| eval_stride | 64 | SOTA PR #549 |
| eval_seq_len | 2048 | SOTA PR #549 (trains at 2048, so no NTK issue) |
| ttt_lr | 0.002 | SOTA PR #549 |
| ttt_epochs | 3 | SOTA PR #549 |
| ttt_chunk_tokens | 32768 | SOTA PR #549 |
| ttt_freeze_blocks | 0 | SOTA PR #549 (all blocks unfrozen) |
| ttt_momentum | 0.9 | SOTA PR #549 |
| ttt_grad_clip | 1.0 | SOTA PR #549 |

## Changelog

| Date | Change | Why |
|------|--------|-----|
| 2026-03-26 | Rebased entirely on SOTA PR #549 | Our foundation was broken: 10 shards (not 80), seq_len 1024 (not 2048), no weight decay, lower LRs. Start from proven baseline. |
| 2026-03-26 | Added entropy-reg QAT (0.01 reg, 0.15 start) | Halves quant gap from 0.017 to 0.009. Proven in run10. |
| 2026-03-26 | Added mixed quant preset front3_back1_6_middle5 | int6 for sensitive layers (first 3 + last 1), int5 for middle. Proven in experiments 44-55. |
| 2026-03-26 | Added GPTQ-lite per-row clip search | Per-row optimal percentile beats global search. 5 candidates per row. |
| 2026-03-26 | Switched compression to zstd-22 | Beats LZMA on 5/6 quant presets. Proven in quant sweep. |
| 2026-03-26 | LeakyReLU slope 0.5 → 0.9 | Controlled sweep in issue #140 shows monotonic improvement. 0.9 beats 0.5 by 0.013 BPB. One-line change. |
