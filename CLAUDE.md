# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

**OpenAI Parameter Golf** — a competition to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Evaluated by compression on FineWeb validation set using bits-per-byte (BPB), which is tokenizer-agnostic.

This is an L(N) optimization problem: minimize loss given a fixed parameter count, unconstrained by data, compute, steps, or architecture.

## Setup

- Two training scripts: `train_gpt.py` (PyTorch/CUDA) and `train_gpt_mlx.py` (MLX/Apple Silicon)
- Local dev on MLX (Apple M4, 24 GB unified memory, 120 GB/s bandwidth)
- Cloud runs on RunPod (8xH100 SXM for leaderboard submissions)
- Data download: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
- Data lands in `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`

## Running

### Local MLX experiment (fast feedback loop)
```bash
source .venv/bin/activate
RUN_ID=experiment_name \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=4096 \
GRAD_ACCUM_STEPS=1 \
TRAIN_SEQ_LEN=512 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
VAL_TOKENS_LIMIT=131072 \
WARMUP_STEPS=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
SKIP_SERIALIZATION=1 \
python3 train_gpt_mlx.py
```
- **Why these values**: 4K batch + seq_len 512 = fast steps on M4 Air.
  VAL_TOKENS_LIMIT=131072 evaluates on ~128K val tokens (256 sequences) instead of 62M — near-instant.
  SKIP_SERIALIZATION=1 skips int8 quantization roundtrip (saves a second full validation pass).
  WARMUP_STEPS=0 skips compile warmup (~30s saved). 200 iters avoids early volatility.
- **Metric**: Use `val_loss` for local comparison (monotonically related to BPB).
- **Noise**: Differences < 0.1 in val_loss at 200 steps are likely noise. Keep seed fixed (default 1337).
- **Thermal**: M4 Air throttles after ~5-10s peak. Runs should finish in 2-5 min total.

### Cloud via Modal (1xH100, ~$0.11 per 2-min experiment)
```bash
# Minimal verify run (~10 steps, ~$0.01)
modal run train_modal.py --run-id verify --iterations 10 --max-wallclock 30 --val-tokens-limit 32768

# 2-min baseline (~340 steps, ~$0.11)
modal run train_modal.py --run-id baseline_h100 --max-wallclock 120

# 2-min experiment with custom config
modal run train_modal.py --run-id wide_6L --max-wallclock 120 --num-layers 6 --model-dim 640
```
- Data is cached in the Modal image (downloaded at build time).
- Default batch size: 524K tokens (real H100 batch, not local toy batch).
- Default val: 1M tokens (fast but reliable on H100).
- Use `--max-wallclock 120` for experiments, `--max-wallclock 600` for final runs.

### Cloud direct (RunPod, 8xH100 — leaderboard submission only)
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Structure

- `train_gpt.py` — PyTorch training script (CUDA, DDP, torch.compile). Hard cap: 1500 lines.
- `train_gpt_mlx.py` — MLX training script (Apple Silicon). Hard cap: 1500 lines.
- `data/` — Dataset download scripts, tokenizer specs
- `records/track_10min_16mb/` — Leaderboard submissions (10 min on 8xH100)
- `records/track_non_record_16mb/` — Non-record / unlimited compute submissions

## Architecture (Baseline)

- 9 transformer blocks, 512 dim, 8 attention heads, 4 KV heads (GQA)
- Tied embeddings (input = output projection), 1024 vocab, 1024 seq len
- relu^2 MLP (2x expansion), RMSNorm (no learnable weight), RoPE
- U-Net skip connections: encoder half stores activations, decoder half adds them back with learned `skip_weights`
- Logit softcap: `30 * tanh(logits / 30)` — prevents logit blow-up
- Optimizer: Muon (matrix params) + Adam (embeddings, scalars)

## Key Metrics

- **val_bpb** (bits per byte) — the competition metric, tokenizer-agnostic compression quality
- **val_loss** — standard cross-entropy in nats, related but not identical to BPB
- **Artifact size** — code bytes + compressed model bytes, must be < 16,000,000 bytes (decimal)
- Baseline: 1.2244 val_bpb (10 min), 1.2074 val_bpb (4 hours, non-record)

## Submission Requirements

1. Beat existing SOTA by >= 0.005 nats (with p < 0.01 significance)
2. Run reproducibly in < 10 min on 8xH100s
3. PR adds folder to `/records/` with: README.md, submission.json, train_gpt.py, train.log

## Hardware

- **Local**: Apple M4, 24 GB unified memory, 120 GB/s memory bandwidth
  - MLX training is memory-bandwidth bound — no software trick bypasses this
  - `caffeinate -dims` prevents Mac sleep during long runs
  - `mx.compile` is already active in the training script
- **Cloud**: 8xH100 SXM (80 GB VRAM each), NVLink, ~$20/hr on RunPod

## MLX-Specific Knowledge

- `mx.compile` captures full model state (including non-trainable arrays like RoPE buffers) — compiling only trainable params throws "uncaptured inputs"
- `mlx_max_microbatch_tokens` (default 8192) controls peak memory on Mac
- Warmup steps prime compile/allocation paths without updating parameters
- `mx.clear_cache()` (not `mx.metal.clear_cache()` which is deprecated)
- Gradient accumulation is manual: accumulate flat grads, scale, then step

## Quantization (Artifact Compression)

- Model is int8 quantized + zlib compressed for the 16MB artifact
- Per-row int8 scales for 2D tensors (better than per-tensor)
- Small tensors (< 65,536 elements) kept as fp16 passthrough
- Control tensors (attn_scale, mlp_scale, resid_mix, q_gain, skip_weights) kept as fp32
- Dequantized back to full precision for evaluation — quantization cost is the BPB delta

## Teaching / Communication Style

- Explain concepts with analogies before showing code
- Walk through code chunk by chunk
- The user has built models from scratch (MNIST MLP, character-level GPT, full transformers)
- The user has deep LoRA fine-tuning and quantization experience
- Keep it conversational — pause for questions between steps
