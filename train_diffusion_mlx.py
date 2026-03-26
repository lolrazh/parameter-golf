#!/usr/bin/env python3
"""
Continuous Diffusion Language Model (MLX) — CDCD-style on Apple Silicon.

Based on CDCD (Dieleman et al., 2022) and Plaid (Gulrajani & Hashimoto, 2023).
Tokens are embedded into continuous space, corrupted with Gaussian noise, and
a bidirectional transformer learns to predict the original tokens from the noisy
embeddings. Sampling follows the probability flow ODE using score interpolation.

See DIFFUSION_ARCH.md for the full architecture doc.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Continuous Diffusion LM (CDCD-style):
# - 9 bidirectional transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - Learned 64-dim L2-normalized token embeddings
# - Gaussian noise in embedding space, score interpolation for sampling
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 512))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_tokens_limit: int = int(os.environ.get("VAL_TOKENS_LIMIT", 0))

    # Model architecture.
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Diffusion-specific.
    embed_dim: int = int(os.environ.get("EMBED_DIM", 64))       # continuous embedding dimension
    t_min: float = float(os.environ.get("T_MIN", 1.0))          # min noise level (tokens still distinguishable)
    t_max: float = float(os.environ.get("T_MAX", 300.0))        # max noise level (pure noise)
    self_cond_prob: float = float(os.environ.get("SELF_COND_PROB", 0.5))  # self-conditioning probability
    score_temp: float = float(os.environ.get("SCORE_TEMP", 0.5))          # temperature during sampling
    sample_steps: int = int(os.environ.get("SAMPLE_STEPS", 200))          # ODE solver steps
    sample_len: int = int(os.environ.get("SAMPLE_LEN", 256))              # generation length

    # Optimizer.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    embed_lr: float = float(os.environ.get("EMBED_LR", 0.05))   # diffusion embedding + output head
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# TIME WARPING: Adaptive noise level sampling
# ==============================================================================
# Instead of sampling t uniformly in log-space, concentrate training on noise
# levels where the model struggles most. Periodically measure L(t) across noise
# levels, fit a CDF, then sample via inverse CDF transform.

class TimeWarp:
    """Adaptive noise level sampler (CDCD, Dieleman et al. 2022)."""

    def __init__(self, t_min: float, t_max: float, num_bins: int = 50, update_every: int = 100):
        self.t_min = t_min
        self.t_max = t_max
        self.num_bins = num_bins
        self.update_every = update_every
        # Start with uniform log-spaced bins
        self.bin_edges = np.linspace(math.log(t_min), math.log(t_max), num_bins + 1, dtype=np.float32)
        self.bin_weights = np.ones(num_bins, dtype=np.float32) / num_bins  # uniform initially
        self._step_count = 0

    def sample_t(self, batch_size: int) -> mx.array:
        """Sample noise levels using current warped distribution."""
        # Pick a bin according to weights, then sample uniformly within that bin
        bins = np.random.choice(self.num_bins, size=batch_size, p=self.bin_weights)
        lo = self.bin_edges[bins]
        hi = self.bin_edges[bins + 1]
        ln_t = lo + np.random.uniform(size=batch_size).astype(np.float32) * (hi - lo)
        return mx.exp(mx.array(ln_t))

    def maybe_update(self, model: "DiffusionLM", sample_tokens: mx.array) -> bool:
        """Measure L(t) and update warping weights. Returns True if updated."""
        self._step_count += 1
        if self._step_count % self.update_every != 0:
            return False

        # Measure loss at each bin center
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        losses = np.zeros(self.num_bins, dtype=np.float32)
        e0 = model.get_embeddings(sample_tokens)

        for i, ln_t in enumerate(bin_centers):
            t = mx.array([math.exp(float(ln_t))])
            eps = mx.random.normal(e0.shape)
            z_t = e0 + t[:, None, None] * eps
            logits = model(z_t, t)
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, model.vocab_size).astype(mx.float32),
                sample_tokens.reshape(-1),
                reduction="mean",
            )
            mx.eval(ce)
            losses[i] = float(ce.item())

        # Weight bins by loss (harder noise levels get more training)
        losses = np.maximum(losses, 1e-8)
        self.bin_weights = losses / losses.sum()
        return True


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> mx.array:
        """Return a batch of token sequences for diffusion (no shift — targets = inputs)."""
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable)
        return mx.array(chunk.reshape(-1, seq_len), dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


def sinusoidal_embedding(t: mx.array, dim: int) -> mx.array:
    """Timestep → vector via sinusoidal encoding (same idea as RoPE, but for time).
    t: [B] → out: [B, dim]. Uses CDCD convention: input is ln(sigma)/4."""
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(COMPUTE_DTYPE)


class BidirectionalAttention(nn.Module):
    # Same as CausalSelfAttention but with FULL attention (no causal mask).
    # Every position attends to every other position — left AND right.
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        # THE KEY CHANGE: no mask="causal" — full bidirectional attention
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)  # relu²


class DiffusionBlock(nn.Module):
    # Transformer block with AdaLN conditioning and bidirectional attention.
    #
    # AdaLN = Adaptive Layer Norm. Instead of static norm(x), each block does:
    #   (1 + scale(t)) * norm(x) + shift(t)
    # where scale(t) and shift(t) come from the timestep embedding.
    # This lets the model behave differently at high noise vs low noise.
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn = BidirectionalAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        # AdaLN: timestep → (attn_scale, attn_shift, mlp_scale, mlp_shift)
        self.adaln = CastedLinear(dim, 4 * dim)
        # Residual scaling (same as GPT baseline — learnable per-channel gate)
        self.attn_gate = mx.ones((dim,), dtype=mx.float32)
        self.mlp_gate = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array, t_emb: mx.array) -> mx.array:
        # t_emb: [B, 1, dim] — timestep conditioning vector
        adaln_out = self.adaln(t_emb.astype(COMPUTE_DTYPE))       # [B, 1, 4*dim]
        a_s, a_b, m_s, m_b = mx.split(adaln_out, 4, axis=-1)     # each [B, 1, dim]

        # Attention with AdaLN-conditioned norm
        h = (1 + a_s) * rms_norm(x) + a_b
        x = x + self.attn_gate.astype(x.dtype)[None, None, :] * self.attn(h)

        # MLP with AdaLN-conditioned norm
        h = (1 + m_s) * rms_norm(x) + m_b
        x = x + self.mlp_gate.astype(x.dtype)[None, None, :] * self.mlp(h)
        return x


class DiffusionLM(nn.Module):
    # Continuous diffusion language model (CDCD-style).
    #
    # Pipeline:
    #   tokens → learned embedding (L2-norm) → add Gaussian noise → scale →
    #   input_proj → transformer (bidirectional + AdaLN) → output_head → logits
    #
    # Training: cross-entropy on predicted tokens given noisy embeddings.
    # Sampling: score interpolation → probability flow ODE (Euler steps).
    def __init__(self, vocab_size: int, num_layers: int, dim: int, embed_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 t_min: float, t_max: float, self_cond_prob: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_dim = embed_dim
        self.logit_softcap = logit_softcap
        self.t_min = t_min
        self.t_max = t_max
        self._self_cond_prob = self_cond_prob
        self.time_warp: TimeWarp | None = None  # set externally to enable adaptive sampling

        # --- Diffusion embedding: tokens → continuous L2-normalized vectors ---
        # This is the "GPS coordinate" for each word in continuous space.
        # L2-normalized and scaled by sqrt(embed_dim) so each component has ~unit variance.
        self.diff_emb = nn.Embedding(vocab_size, embed_dim)
        self.diff_emb.weight = mx.random.normal(
            self.diff_emb.weight.shape, dtype=mx.float32
        ) * 0.01

        # --- Input projection: 2*embed_dim → model_dim ---
        # Always takes (noisy_embed, self_cond_or_zeros) concatenated.
        # Self-conditioning: previous E[x₀] estimate, or zeros if not available.
        self.input_proj = CastedLinear(2 * embed_dim, dim)

        # --- Timestep encoding: scalar t → [dim] vector ---
        # ln(t)/4 → sinusoidal → MLP. Tells each block "how noisy is the input?"
        self.time_fc1 = CastedLinear(dim, dim)
        self.time_fc2 = CastedLinear(dim, dim)

        # --- Transformer blocks (bidirectional + AdaLN) ---
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            DiffusionBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]

        # Zero-init output projections (same trick as GPT — starts as identity)
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

        # --- Output: model_dim → vocab logits ---
        # NOT tied with diff_emb (different dimensions: 512 vs 64).
        self.output_norm_fn = rms_norm
        self.output_head = CastedLinear(dim, vocab_size)

    def get_embeddings(self, token_ids: mx.array) -> mx.array:
        """Look up and L2-normalize embeddings. [B, L] → [B, L, embed_dim]."""
        e = self.diff_emb(token_ids).astype(mx.float32)
        norms = mx.sqrt(mx.sum(e * e, axis=-1, keepdims=True) + 1e-8)
        return e / norms * math.sqrt(self.embed_dim)

    def get_embedding_matrix(self) -> mx.array:
        """L2-normalized embedding matrix for score interpolation. → [V, embed_dim]."""
        e = self.diff_emb.weight.astype(mx.float32)
        norms = mx.sqrt(mx.sum(e * e, axis=-1, keepdims=True) + 1e-8)
        return e / norms * math.sqrt(self.embed_dim)

    def encode_timestep(self, t: mx.array) -> mx.array:
        """Noise level → conditioning vector. [B] → [B, 1, dim]."""
        t_input = mx.log(t + 1e-8) / 4.0  # CDCD convention
        t_emb = sinusoidal_embedding(t_input, self.dim)       # [B, dim]
        t_emb = nn.silu(self.time_fc1(t_emb))
        t_emb = self.time_fc2(t_emb)
        return t_emb[:, None, :]  # [B, 1, dim] for broadcasting over sequence

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, z_t: mx.array, t: mx.array, self_cond: mx.array | None = None) -> mx.array:
        """
        Forward pass: noisy embeddings → token logits.
        z_t:       [B, L, embed_dim] — noisy continuous embeddings
        t:         [B] — noise levels (sigma values)
        self_cond: [B, L, embed_dim] or None — previous E[x₀] estimate for self-conditioning
        Returns: [B, L, vocab_size] logits
        """
        # Scale input to maintain unit variance regardless of noise level.
        t_broad = t[:, None, None]  # [B, 1, 1]
        z_scaled = z_t / mx.sqrt(t_broad * t_broad + 1)

        # Concatenate with self-conditioning (zeros if None)
        if self_cond is None:
            self_cond = mx.zeros_like(z_scaled)
        z_cat = mx.concatenate([z_scaled, self_cond], axis=-1)  # [B, L, 2*embed_dim]

        # Project from 2*embed_dim (128) to model_dim (512)
        x = self.input_proj(z_cat.astype(COMPUTE_DTYPE))

        # Timestep conditioning
        t_emb = self.encode_timestep(t)  # [B, 1, dim]

        # Transformer with U-Net skip connections
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, t_emb)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, t_emb)

        # Output head → vocab logits
        x = self.output_norm_fn(x)
        logits = self.output_head(x.astype(COMPUTE_DTYPE))
        return self.softcap(logits)

    def loss(self, token_ids: mx.array, t: mx.array, eps: mx.array,
             do_self_cond: bool = False) -> mx.array:
        """
        Diffusion training loss: add noise to embeddings, predict original tokens.

        ALL randomness is passed in from outside (not generated here) because
        mx.compile freezes mx.random calls — they return the SAME values every call.
        The caller must generate fresh t, eps, and the self-cond coin flip.

        token_ids:    [B, L] — clean token sequences
        t:            [B] — noise levels (pre-sampled outside compiled fn)
        eps:          [B, L, embed_dim] — Gaussian noise (pre-sampled outside)
        do_self_cond: whether to run draft pass for self-conditioning
        Returns: scalar cross-entropy loss
        """
        # 1. Embed tokens into continuous space
        e0 = self.get_embeddings(token_ids)  # [B, L, embed_dim]

        # 2. Add Gaussian noise: z_t = embedding + t * epsilon
        z_t = e0 + t[:, None, None] * eps  # [B, L, embed_dim]

        # 3. Self-conditioning (caller decides whether to do it)
        self_cond = None
        if do_self_cond:
            draft_logits = mx.stop_gradient(self(z_t, t, None))
            draft_probs = mx.softmax(draft_logits.astype(mx.float32), axis=-1)
            self_cond = mx.stop_gradient(draft_probs @ self.get_embedding_matrix())

        # 4. Real forward pass
        logits = self(z_t, t, self_cond=self_cond)

        # 5. Cross-entropy loss on ALL positions
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.vocab_size).astype(mx.float32),
            token_ids.reshape(-1),
            reduction="mean",
        )

# ==============================================================================
# CoDAR: Contextual Autoregressive Rounding Decoder
# ==============================================================================
# Instead of pointwise argmax (which ignores token-token dependencies),
# a small AR decoder cross-attends to the diffusion output and decodes
# tokens left-to-right. "cat sat" instead of "car mat."

class CrossAttention(nn.Module):
    """Decoder attends to diffusion encoder output (cross-attention)."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.proj = CastedLinear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        """x: decoder hidden [B,L,D]. context: diffusion output [B,L,D]."""
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(context).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(context).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)  # no mask: full cross-attn
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class RoundingDecoderBlock(nn.Module):
    """One block of the AR rounding decoder: causal self-attn + cross-attn to diffusion + MLP."""
    def __init__(self, dim: int, num_heads: int, mlp_mult: int):
        super().__init__()
        # Causal self-attention (AR decoding)
        self.self_attn = BidirectionalAttention(dim, num_heads, num_heads, rope_base=10000.0, qk_gain_init=1.0)
        # Cross-attention to diffusion output
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        # Causal self-attention (with causal mask for AR)
        bsz, seqlen, dim = x.shape
        q = self.self_attn.c_q(rms_norm(x)).reshape(bsz, seqlen, self.self_attn.num_heads, self.self_attn.head_dim).transpose(0, 2, 1, 3)
        k = self.self_attn.c_k(rms_norm(x)).reshape(bsz, seqlen, self.self_attn.num_kv_heads, self.self_attn.head_dim).transpose(0, 2, 1, 3)
        v = self.self_attn.c_v(rms_norm(x)).reshape(bsz, seqlen, self.self_attn.num_kv_heads, self.self_attn.head_dim).transpose(0, 2, 1, 3)
        q = self.self_attn.rope(q.astype(COMPUTE_DTYPE))
        k = self.self_attn.rope(k.astype(COMPUTE_DTYPE))
        sa_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.self_attn.scale, mask="causal")
        sa_out = sa_out.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        x = x + self.self_attn.proj(sa_out)
        # Cross-attention to diffusion context
        x = x + self.cross_attn(rms_norm(x), context)
        # MLP
        x = x + self.mlp(rms_norm(x))
        return x


class RoundingDecoder(nn.Module):
    """
    Small AR transformer that takes diffusion output and decodes tokens left-to-right.
    Cross-attends to the continuous diffusion states, capturing token-token dependencies
    that pointwise argmax misses.
    """
    def __init__(self, vocab_size: int, dim: int, num_layers: int = 2,
                 num_heads: int = 8, mlp_mult: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = [RoundingDecoderBlock(dim, num_heads, mlp_mult) for _ in range(num_layers)]
        self.output_head = CastedLinear(dim, vocab_size)
        # Zero-init output projections
        for b in self.blocks:
            b.self_attn.proj.weight = mx.zeros_like(b.self_attn.proj.weight)
            b.cross_attn.proj.weight = mx.zeros_like(b.cross_attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

    def __call__(self, target_ids: mx.array, context: mx.array) -> mx.array:
        """
        target_ids: [B, L] — shifted token IDs (teacher forcing)
        context:    [B, L, dim] — diffusion model's final hidden states
        Returns:    [B, L, vocab_size] logits
        """
        x = self.tok_emb(target_ids).astype(COMPUTE_DTYPE)
        for block in self.blocks:
            x = block(x, context)
        return self.output_head(rms_norm(x))

    def loss(self, token_ids: mx.array, context: mx.array) -> mx.array:
        """Teacher-forced CE loss. Context is the diffusion output."""
        # Input: [BOS, tok1, tok2, ...], Target: [tok1, tok2, tok3, ...]
        # For simplicity, use the same tokens shifted: input = tokens[:-1], target = tokens[1:]
        # But since context has L positions matching L tokens, we predict all positions
        # given the context, using teacher forcing with left-shifted tokens.
        B, L = token_ids.shape
        # Prepend a zero token as BOS
        bos = mx.zeros((B, 1), dtype=mx.int32)
        input_ids = mx.concatenate([bos, token_ids[:, :-1]], axis=1)  # [B, L]
        logits = self(input_ids, context)
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.vocab_size).astype(mx.float32),
            token_ids.reshape(-1),
            reduction="mean",
        )

    def decode(self, context: mx.array) -> mx.array:
        """Autoregressive decoding given diffusion context. Returns [B, L] token IDs."""
        B, L, _ = context.shape
        tokens = mx.zeros((B, L), dtype=mx.int32)
        for pos in range(L):
            logits = self(tokens, context)  # [B, L, V]
            next_token = mx.argmax(logits[:, pos, :], axis=-1)  # [B]
            if pos < L - 1:
                tokens = tokens.at[:, pos + 1].add(next_token)  # type: ignore
            # Store the decoded token
            tokens_list = tokens.tolist()
            for b in range(B):
                tokens_list[b][pos] = int(next_token[b].item())
            tokens = mx.array(tokens_list, dtype=mx.int32)
            mx.eval(tokens)
        return tokens


# ==============================================================================
# SAMPLING: Generate text from noise via probability flow ODE
# ==============================================================================

def _score_interpolation(
    model: DiffusionLM, z: mx.array, t_batch: mx.array,
    emb_matrix: mx.array, temperature: float,
    self_cond: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """One forward pass → score via interpolation. Returns (score, E[x₀])."""
    logits = model(z, t_batch, self_cond=self_cond)
    probs = mx.softmax((logits / temperature).astype(mx.float32), axis=-1)
    e_x0 = probs @ emb_matrix  # expected clean embedding
    t_sq = t_batch[:, None, None] ** 2
    score = (e_x0 - z) / t_sq
    return score, e_x0


def sample_text(
    model: DiffusionLM,
    sp: "spm.SentencePieceProcessor",
    num_samples: int = 1,
    seq_len: int = 256,
    num_steps: int = 200,
    temperature: float = 0.5,
    solver: str = "heun",
    rounding_decoder: RoundingDecoder | None = None,
) -> list[str]:
    """
    Generate text by denoising from pure Gaussian noise.

    Solvers:
      - "euler": 1 forward pass per step. Simple, fast.
      - "heun":  2 forward passes per step. Look, tentative step, look again, average.
                 Same compute as Euler with 2x steps, but smoother ODE trajectory.
    """
    embed_dim = model.embed_dim
    t_min, t_max = model.t_min, model.t_max

    # Initialize from pure noise (scaled by t_max)
    z = mx.random.normal((num_samples, seq_len, embed_dim)) * t_max

    # Build log-spaced timestep schedule: t_max → t_min
    log_ts = mx.array(
        np.linspace(math.log(t_max), math.log(t_min), num_steps + 1, dtype=np.float32)
    )
    ts = mx.exp(log_ts)

    # Get embedding matrix for score interpolation
    emb_matrix = model.get_embedding_matrix()  # [V, embed_dim]
    self_cond = None  # self-conditioning: starts as None, updated each step

    for i in range(num_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_now  # negative (moving from high noise to low noise)
        t_batch_now = mx.broadcast_to(t_now, (num_samples,))

        # Score at current position
        score_now, e_x0 = _score_interpolation(
            model, z, t_batch_now, emb_matrix, temperature, self_cond,
        )
        # drift = -t × score  (the ODE right-hand side)
        d1 = -t_now * score_now

        if solver == "heun" and i < num_steps - 1:
            # Heun: take tentative step, evaluate score there, average
            z_tentative = z + d1 * dt
            t_batch_next = mx.broadcast_to(t_next, (num_samples,))
            score_next, _ = _score_interpolation(
                model, z_tentative, t_batch_next, emb_matrix, temperature, e_x0,
            )
            d2 = -t_next * score_next
            z = z + 0.5 * (d1 + d2) * dt
        else:
            # Euler: just go
            z = z + d1 * dt

        # Self-conditioning: feed E[x₀] to the next step
        self_cond = e_x0
        mx.eval(z)

    # Final rounding: get the diffusion model's final hidden states + logits
    t_final = mx.broadcast_to(ts[-1], (num_samples,))
    final_logits = model(z, t_final, self_cond=self_cond)

    if rounding_decoder is not None:
        # CoDAR: contextual AR rounding using the diffusion output as context
        # Get the final hidden states (before output head) by re-running the model
        # For simplicity, use the logits as context (the decoder will learn to use them)
        context = model.output_norm_fn(
            model.input_proj(
                mx.concatenate([
                    z / mx.sqrt(ts[-1] ** 2 + 1),
                    self_cond if self_cond is not None else mx.zeros_like(z),
                ], axis=-1).astype(COMPUTE_DTYPE)
            )
        )
        token_ids = rounding_decoder.decode(context)
    else:
        # Simple argmax rounding
        token_ids = mx.argmax(final_logits, axis=-1)

    mx.eval(token_ids)
    texts = []
    for i in range(num_samples):
        ids = token_ids[i].tolist()
        texts.append(sp.decode(ids))
    return texts


# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # Three optimizer groups (same structure as GPT, different param names):
    # - Diffusion embedding + output head + timestep MLP: Adam with embed_lr
    # - Block matrices (2D weights): Muon (orthogonalized SGD)
    # - Block scalars + skip weights: Adam with scalar_lr
    def __init__(self, model: DiffusionLM, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))

        # Embedding group: diff_emb, input_proj, output_head, time MLP
        self.embed_keys = [
            k for k in params
            if k.startswith(("diff_emb.", "input_proj.", "output_head.", "time_fc"))
        ]
        # Matrix group: 2D block weights (attn q/k/v/proj, mlp, adaln)
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        # Scalar group: skip_weights + 1D block params (attn_gate, mlp_gate, q_gain)
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights"
            or (k.startswith("blocks.") and (p.ndim < 2
                or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)))
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.embed_lr, betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )

    def step(self, model: DiffusionLM, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        # Muon for block matrices
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        # Adam for embeddings + projections + timestep MLP
        self.adam_embed.learning_rate = self.args.embed_lr * lr_mul
        embed_grads = {k: grads[k] for k in self.embed_keys}
        embed_params = {k: params[k] for k in self.embed_keys}
        updated.update(self.adam_embed.apply_gradients(embed_grads, embed_params))

        # Adam for scalars
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    # For diffusion: no shift-by-1 needed, just align to seq_len
    usable = (tokens.size // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[:usable]


def _sample_noise(B: int, L: int, embed_dim: int, t_min: float, t_max: float,
                   time_warp: "TimeWarp | None" = None) -> tuple[mx.array, mx.array]:
    """Sample noise level and Gaussian noise OUTSIDE compiled functions."""
    if time_warp is not None:
        t = time_warp.sample_t(B)
    else:
        ln_t = np.random.uniform(
            math.log(t_min), math.log(t_max), size=B
        ).astype(np.float32)
        t = mx.array(np.exp(ln_t))
    eps = mx.random.normal((B, L, embed_dim))
    return t, eps


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad_sc,
    compiled_loss_and_grad_no_sc,
    model: DiffusionLM,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        token_ids = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        B = token_ids.shape[0]
        t, eps = _sample_noise(B, args.train_seq_len, args.embed_dim,
                               args.t_min, args.t_max, model.time_warp)
        # Self-conditioning: 50% of the time
        if np.random.random() < args.self_cond_prob:
            loss, grads = compiled_loss_and_grad_sc(token_ids, t, eps)
        else:
            loss, grads = compiled_loss_and_grad_no_sc(token_ids, t, eps)
        scale = float(token_ids.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss_no_sc,
    val_tokens: np.ndarray,
    embed_dim: int,
    log_fn: Callable[[str], None] | None = None,
) -> float:
    # Validation: compute average diffusion loss on val set.
    # Uses NO self-conditioning (deterministic eval) and FRESH random noise per batch
    # (generated outside mx.compile to avoid the frozen-random bug).
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = max(val_batch_tokens // args.train_seq_len, 1)
    total_seqs = val_tokens.size // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len
        chunk = val_tokens[raw_start:raw_end]
        token_ids = mx.array(chunk.reshape(-1, args.train_seq_len), dtype=mx.int32)
        B = token_ids.shape[0]
        # Fresh noise generated OUTSIDE compiled function
        t, eps = _sample_noise(B, args.train_seq_len, embed_dim, args.t_min, args.t_max)
        chunk_token_count = float(token_ids.size)
        batch_loss = compiled_loss_no_sc(token_ids, t, eps).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        total_tokens += chunk_token_count
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    return total_loss_sum / total_tokens

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path, args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    if args.val_tokens_limit > 0:
        limit = (args.val_tokens_limit // args.train_seq_len) * args.train_seq_len
        val_tokens = val_tokens[:limit]
        log(f"val_tokens_limit:{args.val_tokens_limit} actual_val_tokens:{val_tokens.size}")

    # ==============================================================================
    # MODEL + OPTIMIZER
    # ==============================================================================
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = DiffusionLM(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        t_min=args.t_min,
        t_max=args.t_max,
        self_cond_prob=args.self_cond_prob,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED FUNCTIONS (MLX)
    # ==============================================================================
    # CRITICAL: mx.compile freezes mx.random calls and Python if-branches.
    # All randomness (t, eps, self-cond coin flip) MUST be generated outside
    # and passed as arguments. We compile TWO versions of loss: with and without
    # self-conditioning (since the if-branch is frozen at trace time).
    compiled_loss_no_sc = mx.compile(
        lambda ids, t, eps: model.loss(ids, t, eps, do_self_cond=False),
        inputs=model.state, outputs=model.state,
    )
    compiled_loss_sc = mx.compile(
        lambda ids, t, eps: model.loss(ids, t, eps, do_self_cond=True),
        inputs=model.state, outputs=model.state,
    )
    compiled_loss_and_grad_no_sc = mx.compile(
        nn.value_and_grad(model, lambda ids, t, eps: model.loss(ids, t, eps, do_self_cond=False)),
        inputs=model.state, outputs=model.state,
    )
    compiled_loss_and_grad_sc = mx.compile(
        nn.value_and_grad(model, lambda ids, t, eps: model.loss(ids, t, eps, do_self_cond=True)),
        inputs=model.state, outputs=model.state,
    )

    # Time warping: adaptive noise level sampling (enable with TIME_WARP=1)
    if os.environ.get("TIME_WARP", "0") == "1":
        tw_update_every = int(os.environ.get("TW_UPDATE_EVERY", 100))
        model.time_warp = TimeWarp(args.t_min, args.t_max, update_every=tw_update_every)
        log(f"time_warp:enabled update_every:{tw_update_every}")

    # Log config
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"model:DiffusionLM(CDCD) params:{n_params} vocab:{args.vocab_size} "
        f"layers:{args.num_layers} dim:{args.model_dim} embed_dim:{args.embed_dim} "
        f"heads:{args.num_heads} kv_heads:{args.num_kv_heads} seq_len:{args.train_seq_len}")
    log(f"diffusion: t_min:{args.t_min} t_max:{args.t_max} "
        f"sample_steps:{args.sample_steps} score_temp:{args.score_temp}")
    log(f"optimizer:muon+adam muon_params:{len(opt.matrix_keys)} "
        f"embed_params:{len(opt.embed_keys)} scalar_params:{len(opt.scalar_keys)}")
    log(f"training: iters:{args.iterations} batch_tokens:{args.train_batch_tokens} "
        f"grad_accum:{args.grad_accum_steps} warmup:{args.warmup_steps}")

    # ==============================================================================
    # WARMUP (prime compile paths)
    # ==============================================================================
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(
                    args, train_loader, compiled_loss_and_grad_sc,
                    compiled_loss_and_grad_no_sc, model)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime eval graph
        warm_seqs = min(
            args.val_batch_size // (args.grad_accum_steps * args.train_seq_len),
            val_tokens.size // args.train_seq_len,
        )
        if warm_seqs > 0:
            warm_chunk = val_tokens[:warm_seqs * args.train_seq_len]
            warm_ids = mx.array(warm_chunk.reshape(-1, args.train_seq_len), dtype=mx.int32)
            t, eps = _sample_noise(warm_ids.shape[0], args.train_seq_len, args.embed_dim,
                                   args.t_min, args.t_max)
            mx.eval(compiled_loss_no_sc(warm_ids, t, eps))
            mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss = eval_val(args, compiled_loss_no_sc, val_tokens, args.embed_dim, log_fn=log)
            if step % 25 == 0 or last_step:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(
                args, train_loader, compiled_loss_and_grad_sc,
                compiled_loss_and_grad_no_sc, model)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        # Time warp: periodically measure L(t) and update sampling weights
        if model.time_warp is not None:
            # Use a small sample of val tokens for L(t) measurement
            tw_sample = mx.array(
                val_tokens[:args.train_seq_len].reshape(1, args.train_seq_len), dtype=mx.int32
            )
            model.time_warp.maybe_update(model, tw_sample)

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # SERIALIZATION
    # ==============================================================================
    if os.environ.get("SKIP_SERIALIZATION", "0") == "1":
        log("skip_serialization:1 — skipping model save")
    else:
        out_path = out_dir / f"{args.run_id}_diffusion_model.npz"
        flat_state = {k: v for k, v in tree_flatten(model.state)}
        mx.savez(str(out_path), **flat_state)
        log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    # ==============================================================================
    # GENERATION DEMO: Text from noise
    # ==============================================================================
    if os.environ.get("SKIP_GENERATION", "0") != "1":
        log("=" * 60)
        log("GENERATING TEXT FROM NOISE")
        log("=" * 60)
        gen_t0 = time.perf_counter()
        texts = sample_text(
            model, sp,
            num_samples=2,
            seq_len=args.sample_len,
            num_steps=args.sample_steps,
            temperature=args.score_temp,
        )
        gen_ms = 1000.0 * (time.perf_counter() - gen_t0)
        for i, text in enumerate(texts):
            log(f"--- sample {i+1} ({gen_ms:.0f}ms) ---")
            log(text[:500])  # first 500 chars
        log("=" * 60)


if __name__ == "__main__":
    main()
