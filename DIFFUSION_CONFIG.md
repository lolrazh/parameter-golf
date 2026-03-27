# Diffusion LM Config

Single source of truth for the continuous diffusion language model (CDCD + BD3-LM hybrid).

## Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| num_layers | 11 | Matched to GPT prod config |
| model_dim | 512 | Matched to GPT prod config |
| num_heads | 8 | Matched to GPT prod config |
| num_kv_heads | 4 | GQA, matched to GPT |
| vocab_size | 1024 | sp1024 tokenizer |
| mlp_mult | 3.0 | Matched to GPT prod config |
| activation | LeakyReLU(0.9)^2 | Proven +0.013 BPB over relu^2 in GPT sweeps |
| logit_softcap | 30.0 | Matched to GPT |
| rope_base | 10000 | Matched to GPT |
| rope_dims | 16 | Partial RoPE, matched to GPT |
| attention | **Bidirectional** | No causal mask. FA3 with causal=False. |
| conditioning | **Per-position AdaLN** | Each position gets (scale, shift) from its noise level |
| embed_dim | 64 | Diffusion embedding dimension (CDCD/Plaid sweet spot) |
| tie_embeddings | False | Separate diff_emb (64d) and output_head (512→1024) |
| parameter_banks | qo, kv, mlp_up, mlp_down, **adaln** | 5 banks for Parallel Muon |

## Diffusion Schedule

| Parameter | Value | Notes |
|-----------|-------|-------|
| t_min | 1.0 | Min noise. Tokens distinguishable at sigma=1 |
| t_max | 300.0 | Max noise. Pure Gaussian, no signal |
| t_sampling | log-uniform | ln(t) ~ U(ln(t_min), ln(t_max)) |
| self_cond_prob | 0.0 | Disabled for speed. Re-enable on H100 once step budget is known. |
| score_temp | 0.5 | Temperature on logits during ODE sampling |

## Block Diffusion Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| train_block_size | 4 | Noise 4 tokens, keep rest clean. 0 = uniform noise (legacy). |
| block_mask | per-position | Loss only on noisy block positions |
| per_position_t | [B, L] | Context positions: t=0 (clean). Block: t=sampled. |

## Training

| Parameter | Value | Source |
|-----------|-------|--------|
| train_seq_len | 2048 (prod), 1024 (proxy) | Matched to GPT |
| train_batch_tokens | 786432 (prod), 131072 (proxy) | Matched to GPT |
| train_shards | 80 | All FineWeb shards |
| iterations | 20000 | Cap, wallclock stops first |
| max_wallclock_seconds | 600 | 10 min |
| warmup_steps | 20 | Compile/allocation priming |
| warmdown_iters | 3500 (prod), 1500 (proxy) | Linear LR decay |

## Optimizer

| Parameter | Value | Source |
|-----------|-------|--------|
| optimizer | Parallel Muon + AdamW | Same as GPT |
| matrix_lr | 0.025 | For 5 banks (qo, kv, mlp_up, mlp_down, adaln) |
| scalar_lr | 0.025 | Block scales, q_gain, skip_weights |
| embed_lr | 0.05 | diff_emb, input_proj, output_head, time_fc1, time_fc2 |
| muon_momentum | 0.99 | Matched to GPT |
| muon_backend_steps | 5 | Newton-Schulz iterations |
| muon_wd | 0.04 | Weight decay on banks |
| adam_wd | 0.04 | Weight decay on non-bank params |
| beta1/beta2 | 0.9 / 0.95 | Matched to GPT |
| grad_clip_norm | 0.3 | Global gradient clipping |

## Weight Averaging

| Parameter | Value | Source |
|-----------|-------|--------|
| ema_decay | 0.997 | Matched to GPT |
| swa_enabled | True | Matched to GPT |
| swa_every | 50 | Matched to GPT |

## Quantization

| Parameter | Value | Source |
|-----------|-------|--------|
| quant_preset | front3_back1_6_middle5 | int6 for sensitive layers, int5 for middle |
| gptq_lite | True (per-row) | Per-row optimal percentile search |
| entropy_reg | 0.01 | QAT entropy regularization |
| late_qat_threshold | 0.15 | Enable QAT when LR scale < 0.15 |
| compression | lzma preset=6 | Matched to GPT |

## Evaluation (Block NELBO → BPB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| eval_block_size | 4 | BD3-LM style. L'=4 gives ~18% gap to AR at scale. L'=1 = exact AR. |
| eval_t_samples | 8 | MC noise samples per block. 8 = unbiased, just noisier than 32. |
| eval_context_len | 2048 | Match train seq_len. Must be bounded or OOM on long val sets. |
| scoring_method | Block NELBO | Valid upper bound on NLL. Converges to AR NLL as L'→1. |

## ODE Sampling (Text Generation)

| Parameter | Value | Notes |
|-----------|-------|-------|
| sample_steps | 200 | ODE integration steps (noise → text) |
| solver | heun | 2nd order. 2x forward passes per step but smoother trajectory. |
| sample_len | 256 | Default generation length |
| score_temp | 0.5 | Temperature on logits (sharper = more confident) |

## Scripts

| Script | What It Does |
|--------|-------------|
| `train_diffusion.py` | PyTorch/CUDA training (DDP, Parallel Muon, FA3). Runs on H100. |
| `train_diffusion_mlx.py` | MLX training (Apple Silicon). Local dev/experiments. |
| `run_diffusion.sh --proxy` | 1xH100, 131K batch, 120s. Fast iteration. |
| `run_diffusion.sh --prod` | 8xH100, 786K batch, 600s. Submission runs. |

## Ideas To Explore

- **Reduce eval_block_size to 1** — exact AR-equivalent BPB, but L forward passes per seq. Worth trying once model is trained.
- **Consistency distillation** — compress 200 ODE steps to 8-16 steps. Needs trained teacher first (Song et al. 2023, CDLM Nov 2025).
- **CoDAR rounding decoder** — 2-layer AR decoder cross-attends to diffusion output for better token selection. Class implemented, needs training. (CoDAR, March 2026)
- **Time warping** — adaptive noise sampling. Implemented (TIME_WARP=1), untested at scale.
- **N-gram cache at eval** — blend n-gram predictions with diffusion during block NELBO scoring. Would need adaptation from GPT's cache.
- **TTT for diffusion** — test-time training on val chunks. Score via block NELBO, train on chunk, re-score. Mechanics transfer directly from GPT.
- **Larger embed_dim** — 64 is Plaid's finding, but 128 or 256 might work better with our architecture. Ablation needed.
- **Self-conditioning warmup** — disable self-cond for first N steps (avoids early loss spike from garbage draft). SELF_COND_START_STEP env var.
- **Block size curriculum** — start with large blocks (L'=64, faster training), shrink to L'=4 for tighter scoring. Like noise schedule but for block size.

## Changelog

| Date | Change | Why |
|------|--------|-----|
| 2026-03-27 | Created CDCD continuous diffusion LM on MLX | Exploration: build the most "diffusion-coded" text model possible |
| 2026-03-27 | Added self-conditioning, Heun solver, time warping, CoDAR decoder | 4 frontier improvements from CDCD/Plaid/Karras/CoDAR papers |
| 2026-03-27 | Found and fixed mx.compile frozen-random bug | mx.compile freezes mx.random values AND Python if-branches at trace time |
| 2026-03-27 | Upgraded to per-position AdaLN + block diffusion training | Required for fair BPB scoring via block NELBO |
| 2026-03-27 | Implemented block NELBO → BPB evaluation (BD3-LM style) | First BPB number: 18.11 (50 steps, 4K batch — scaling problem) |
| 2026-03-27 | Updated defaults: 11L, MLP 3x, LeakyReLU(0.9)², partial RoPE | Match GPT prod config for fair comparison |
| 2026-03-27 | Ported to PyTorch/CUDA (train_diffusion.py) | 8xH100 ready. DDP + Parallel Muon + FA3 + adaln_bank |
| 2026-03-27 | Created run_diffusion.sh | Proxy (1xH100) + prod (8xH100) configs |
