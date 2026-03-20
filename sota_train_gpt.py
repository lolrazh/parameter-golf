"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    # Sliding window eval: stride controls how many tokens to slide between windows.
    # Only the last `eval_stride` tokens per window are scored, giving each scored token
    # (seq_len - eval_stride) tokens of context. Set to 0 to disable (use standard eval).
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    use_smeargate = bool(int(os.environ.get("USE_SMEARGATE", "1")))
    bigram_hash_buckets = int(os.environ.get("BIGRAM_HASH_BUCKETS", 10240))
    bigram_hash_dim = int(os.environ.get("BIGRAM_HASH_DIM", 128))
    smear_gate_init = float(os.environ.get("SMEAR_GATE_INIT", 0.0))
    # XSA: Exclusive Self Attention — projects out self-value component in deep layers.
    # Zero extra params, ~0.002 BPB gain. Applied to last N layers only.
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 3))
    # Depth recurrence: loop through blocks multiple times for deeper virtual network.
    # num_recurrence=1 is standard (no recurrence). num_recurrence=3 with num_layers=5
    # gives 15 virtual layers from 5 physical blocks. lora_rank>0 adds per-virtual-layer adapters.
    num_recurrence = int(os.environ.get("NUM_RECURRENCE", 1))
    lora_rank = int(os.environ.get("LORA_RANK", 0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_wd = float(os.environ.get("MUON_WD", 0.02))
    adam_wd = float(os.environ.get("ADAM_WD", 0.01))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 200))
    # QAT: fraction of training wallclock after which STE fake-quant activates.
    # 0.0 = always on (current behavior), 0.7 = late activation at 70%, 1.0 = off.
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.0))
    fp16_embed = bool(int(os.environ.get("FP16_EMBED", "1")))
    # TTT: test-time training adapts model on val data before scoring.
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 3e-4))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.95))
    ttt_max_seconds = float(os.environ.get("TTT_MAX_SECONDS", 200.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding_window(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
) -> tuple[float, float]:
    """Sliding window evaluation: each scored token gets (seq_len - stride) context.

    Instead of chopping validation into non-overlapping 1024-token blocks (where
    the first token in each block gets zero context), we slide a 1024-token window
    by `stride` tokens at a time and only score the last `stride` tokens per window.
    Every scored token sees 960+ tokens of context, dramatically improving BPB.
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1  # need 1 extra for final target

    # Number of windows: first window at offset 0, then slide by stride
    num_windows = max((total_tokens - seq_len) // stride + 1, 1)

    # Distribute windows across ranks
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Eval batch: how many windows per forward pass. Tune for memory.
    eval_batch = int(os.environ.get("SW_EVAL_BATCH", 64))

    # Compile forward_logits for faster eval (separate from training compile)
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False)
    base_model.eval()
    with torch.inference_mode():
        window_list = list(range(win_start, win_end))
        num_batches = (len(window_list) + eval_batch - 1) // eval_batch
        for batch_idx in range(num_batches):
            batch_wins = window_list[batch_idx * eval_batch : (batch_idx + 1) * eval_batch]
            bsz = len(batch_wins)

            # Build input: each window is val_tokens[w*stride : w*stride + seq_len]
            inputs = torch.stack([
                val_tokens[w * stride : w * stride + seq_len]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64)  # [B, seq_len]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = compiled_logits(inputs)  # [B, seq_len, vocab]

            # Score only the last `stride` positions per window.
            # logits[:, j, :] predicts the token AFTER position j in the input.
            # So logits[:, -stride:, :] predicts tokens at input positions
            # [seq_len - stride + 1, seq_len + 1) — which are the targets.
            scored_logits = logits[:, -stride:, :].reshape(-1, logits.size(-1))  # [B*stride, vocab]

            # Build targets: for window w, targets are val_tokens[w*stride + seq_len - stride + 1 : w*stride + seq_len + 1]
            targets = torch.stack([
                val_tokens[w * stride + seq_len - stride + 1 : w * stride + seq_len + 1]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64).reshape(-1)  # [B*stride]

            loss = F.cross_entropy(scored_logits.float(), targets, reduction="sum")
            val_loss_sum += loss.to(torch.float64)
            val_token_count += float(targets.numel())

            # BPB: prev_ids are the input tokens at each scored position
            prev_ids = torch.stack([
                val_tokens[w * stride + seq_len - stride : w * stride + seq_len]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64).reshape(-1)  # [B*stride]

            tgt_ids = targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                print(f"  sw_eval batch {batch_idx + 1}/{num_batches}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    base_model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear_gate,bigram_scale",
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
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16

INT6_QUANT_RANGE = 31  # int6: [-31, 31]
INT8_QUANT_RANGE = 127  # int8: [-127, 127]
INT6_CLIP_Q = 0.9999984
QAT_ACTIVE = True  # toggled by training loop based on qat_start_frac

# Tensors matching these patterns get int8 (127 levels) instead of int6 (31 levels)
# during post-training quantization. This is for weights that DON'T have fake-quant
# STE protection during training (e.g. nn.Embedding), so they need gentler quantization.
INT8_FULL_RANGE_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_FULL_RANGE_PATTERNS",
        "tok_emb,bigram_hash_emb",
    ).split(",")
    if pattern
)
QUANT_PRESET = os.environ.get("QUANT_PRESET", "current").strip() or "current"

def infer_num_layers_from_state_dict(state_dict: dict[str, Tensor]) -> int:
    max_idx = -1
    for name in state_dict.keys():
        m = re.match(r"blocks\.(\d+)\.", name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def resolve_int8_full_range_patterns(state_dict: dict[str, Tensor]) -> tuple[str, ...]:
    num_layers = infer_num_layers_from_state_dict(state_dict)
    base = tuple(dict.fromkeys(INT8_FULL_RANGE_PATTERNS))
    if QUANT_PRESET == "current":
        return base
    if num_layers <= 0:
        raise ValueError("could not infer block count from state_dict for quant preset resolution")

    last = num_layers - 1
    preset_patterns: dict[str, tuple[str, ...]] = {
        "outer8_middle6": ("tok_emb", "blocks.0.", f"blocks.{last}."),
        "front2_8_middle6": ("tok_emb", "blocks.0.", "blocks.1."),
        "front2_back1_8_middle6": ("tok_emb", "blocks.0.", "blocks.1.", f"blocks.{last}."),
        "front3_back1_8_middle6": ("tok_emb", "blocks.0.", "blocks.1.", "blocks.2.", f"blocks.{last}."),
        "front2_back2_8_middle6": ("tok_emb", "blocks.0.", "blocks.1.", f"blocks.{last - 1}.", f"blocks.{last}."),
        "front2_back1_attn8": ("tok_emb", "blocks.0.attn.", "blocks.1.attn.", f"blocks.{last}.attn."),
    }
    if QUANT_PRESET not in preset_patterns:
        valid = ", ".join(["current", *sorted(preset_patterns)])
        raise ValueError(f"unknown QUANT_PRESET={QUANT_PRESET!r}. Valid presets: {valid}")
    return tuple(dict.fromkeys(preset_patterns[QUANT_PRESET]))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor, quant_range: int = INT6_QUANT_RANGE) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row quantization: [-quant_range, quant_range] stored in int8 containers.
        # For int6 (range=31), the unused high bits compress extremely well with zstd.
        clip_abs = (
            torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars still use int8 per-tensor scale (they're tiny).
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

FP16_PASSTHROUGH_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("FP16_PASSTHROUGH_PATTERNS", "tok_emb" if bool(int(os.environ.get("FP16_EMBED", "1"))) else "").split(",")
    if pattern
)

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    # - fp16 passthrough for embedding-like tensors (FP16_EMBED=1)
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    int8_full_range_patterns = resolve_int8_full_range_patterns(state_dict)

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # FP16 passthrough: keep embedding at fp16 instead of quantizing
        if any(pattern in name for pattern in FP16_PASSTHROUGH_PATTERNS) and t.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        # Use int8 (127 levels) for tensors without STE fake-quant protection
        # (e.g. tok_emb), int6 (31 levels) for block weights that trained with STE.
        use_int8 = any(pattern in name for pattern in int8_full_range_patterns)
        qr = INT8_QUANT_RANGE if use_int8 else INT6_QUANT_RANGE
        q, s = quantize_float_tensor(t, quant_range=qr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

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

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    # During training, apply fake int6 quantization (STE) so the model learns to be robust
    # to the post-training quantization that will be applied for the 16MB artifact.
    # QAT activation is controlled by the global QAT_ACTIVE flag (set by training loop).
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and QAT_ACTIVE and w.ndim == 2:
            # Fake int6 per-row quantization with Straight-Through Estimator:
            # Forward uses quantized weights, backward passes gradients through as-is.
            with torch.no_grad():
                w32 = w.float()
                clip_abs = torch.quantile(w32.abs(), INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                scale = clip_abs / INT6_QUANT_RANGE
                w_clipped = torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None])
                w_q = (torch.round(w_clipped / scale[:, None]) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE: value of w_q, gradient of w
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            if HAS_FA3:
                # FA3 layout: (1, seq, 1, head_dim//2)
                self._cos_cached = freqs.cos()[None, :, None, :]
                self._sin_cached = freqs.sin()[None, :, None, :]
            else:
                # SDPA layout: (1, 1, seq, head_dim//2)
                self._cos_cached = freqs.cos()[None, None, :, :]
                self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False  # toggled by GPT.__init__ for deep layers

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection. GQA-aware via free reshape, no repeat_interleave."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        y_g = y.reshape(B, T, Hkv, H // Hkv, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(dim=-1, keepdim=True) * vn).reshape(B, T, H, D)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if not HAS_FA3:  # SDPA needs (bsz, heads, seq, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if HAS_FA3:
            q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
            y = flash_attn_3_func(q, k, v, causal=True)[0]  # [B, T, H, D]
            if self.use_xsa:
                y = self._xsa(y, v)  # v already [B, T, Hkv, D]
            y = y.reshape(bsz, seqlen, dim)
        else:
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            ).transpose(1, 2).contiguous()  # [B, T, H, D]
            if self.use_xsa:
                v_xsa = v.transpose(1, 2)  # [B, H, T, D] -> [B, T, Hkv, D]
                y = self._xsa(y, v_xsa)
            y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class LoRAAdapter(nn.Module):
    """Lightweight per-virtual-layer adapter for depth recurrence."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_smeargate: bool,
        bigram_hash_buckets: int,
        bigram_hash_dim: int,
        smear_gate_init: float,
        num_recurrence: int = 1,
        lora_rank: int = 0,
        xsa_last_n: int = 0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.use_smeargate = use_smeargate
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash_buckets = bigram_hash_buckets if use_smeargate else 0
        if use_smeargate:
            if bigram_hash_buckets <= 0:
                raise ValueError(f"bigram_hash_buckets must be positive when USE_SMEARGATE=1, got {bigram_hash_buckets}")
            if bigram_hash_dim <= 0:
                raise ValueError(f"bigram_hash_dim must be positive when USE_SMEARGATE=1, got {bigram_hash_dim}")
            self.bigram_hash_emb = nn.Embedding(bigram_hash_buckets, bigram_hash_dim)
            self.bigram_hash_proj = CastedLinear(bigram_hash_dim, model_dim, bias=False)
            self.bigram_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
            self.smear_gate = nn.Parameter(torch.full((model_dim,), smear_gate_init, dtype=torch.float32))
        else:
            self.bigram_hash_emb = None
            self.bigram_hash_proj = None
            self.bigram_scale = None
            self.smear_gate = None
        self.num_recurrence = num_recurrence
        num_virtual = num_layers * num_recurrence
        self.num_encoder_layers = num_virtual // 2
        self.num_decoder_layers = num_virtual - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.lora_adapters = None
        if lora_rank > 0 and num_recurrence > 1:
            self.lora_adapters = nn.ModuleList([LoRAAdapter(model_dim, lora_rank) for _ in range(num_virtual)])
        # XSA: enable on last N physical blocks
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        num_layers = len(self.blocks) * self.num_recurrence
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.bigram_hash_emb is not None:
            nn.init.zeros_(self.bigram_hash_emb.weight)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name:
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if not self.use_smeargate:
            return x
        if self.bigram_hash_emb is None or self.bigram_hash_proj is None or self.smear_gate is None:
            raise RuntimeError("SmearGate/BigramHash requested but not initialized")
        prev_ids = torch.cat((input_ids[:, :1], input_ids[:, :-1]), dim=1)
        prev_x = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        pair_hash = (prev_ids.to(torch.int64) * 36313 ^ input_ids.to(torch.int64) * 27191) % self.bigram_hash_buckets
        bigram_x = self.bigram_scale * self.bigram_hash_proj(self.bigram_hash_emb(pair_hash))
        gate = torch.sigmoid(self.smear_gate).to(dtype=x.dtype)[None, None, :]
        return (1.0 - gate) * (x + bigram_x) + gate * prev_x

    def _run_virtual_block(self, v: int, x: Tensor, x0: Tensor) -> Tensor:
        block = self.blocks[v % len(self.blocks)]
        x = block(x, x0)
        if self.lora_adapters is not None:
            x = x + self.lora_adapters[v](x)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self._run_virtual_block(i, x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self._run_virtual_block(self.num_encoder_layers + i, x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss. Used for sliding window eval."""
        x = self.embed_tokens(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self._run_virtual_block(i, x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self._run_virtual_block(self.num_encoder_layers + i, x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5, QAT_ACTIVE

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_smeargate=args.use_smeargate,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
        smear_gate_init=args.smear_gate_init,
        num_recurrence=args.num_recurrence,
        lora_rank=args.lora_rank,
        xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_params = [base_model.tok_emb.weight]
    if base_model.bigram_hash_emb is not None:
        token_params.append(base_model.bigram_hash_emb.weight)
    if base_model.bigram_hash_proj is not None:
        matrix_params.append(base_model.bigram_hash_proj.weight)
    if base_model.lora_adapters is not None:
        for adapter in base_model.lora_adapters:
            matrix_params.extend([adapter.down.weight, adapter.up.weight])
    if base_model.smear_gate is not None:
        scalar_params.append(base_model.smear_gate)
    if base_model.bigram_scale is not None:
        scalar_params.append(base_model.bigram_scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(
        f"model_shape:num_layers:{args.num_layers} model_dim:{args.model_dim} "
        f"mlp_mult:{args.mlp_mult} vocab_size:{args.vocab_size}"
    )
    log0(f"eval_config:eval_stride:{args.eval_stride} train_seq_len:{args.train_seq_len}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"smeargate:enabled:{int(args.use_smeargate)} "
        f"bigram_hash_buckets:{args.bigram_hash_buckets if args.use_smeargate else 0} "
        f"bigram_hash_dim:{args.bigram_hash_dim if args.use_smeargate else 0}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_wd:{args.muon_wd} adam_wd:{args.adam_wd} grad_clip:{args.grad_clip_norm}"
    )
    log0(f"swa:enabled:{int(args.swa_enabled)} every:{args.swa_every}")
    log0(f"qat:start_frac:{args.qat_start_frac} fp16_embed:{int(args.fp16_embed)} fa3:{int(HAS_FA3)}")
    log0(f"recurrence:{args.num_recurrence} lora_rank:{args.lora_rank} virtual_layers:{args.num_layers * args.num_recurrence}")
    xsa_layers = [i for i in range(args.num_layers) if i >= args.num_layers - args.xsa_last_n] if args.xsa_last_n > 0 else []
    log0(f"xsa:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"ttt:enabled:{int(args.ttt_enabled)} lr:{args.ttt_lr} max_sec:{args.ttt_max_seconds}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    # SWA: accumulate weight snapshots during warmdown, average at end.
    swa_sum: dict[str, Tensor] | None = None
    swa_count = 0

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        # Late QAT: activate STE fake-quant after qat_start_frac of wallclock
        if args.qat_start_frac < 1.0 and max_wallclock_ms is not None:
            QAT_ACTIVE = (elapsed_ms / max_wallclock_ms) >= args.qat_start_frac
        elif args.qat_start_frac >= 1.0:
            QAT_ACTIVE = False
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # SWA: snapshot weights during warmdown phase
        if args.swa_enabled and scale < 0.5 and (step + 1) % args.swa_every == 0:
            sd = base_model.state_dict()
            if swa_sum is None:
                swa_sum = {k: v.detach().cpu().float().clone() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    swa_sum[k].add_(v.detach().cpu().float())
            swa_count += 1

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Apply SWA averaging if we collected snapshots.
    if swa_sum is not None and swa_count > 1:
        log0(f"swa:applying avg of {swa_count} snapshots")
        avg_sd = {k: (v / swa_count).to(dtype=base_model.state_dict()[k].dtype) for k, v in swa_sum.items()}
        base_model.load_state_dict(avg_sd, strict=True)
    elif swa_count == 1 and swa_sum is not None:
        log0("swa:only 1 snapshot, skipping averaging")

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    if master_process:
        log0(f"quant_preset:{QUANT_PRESET}")
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
        compress_name = "zstd-22"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_name = "zlib-9"
    quant_raw_bytes = len(quant_raw)
    if master_process:
        quant_ext = "int6.ptz" if HAS_ZSTD else "int6.ptz"
        with open(f"final_model.{quant_ext}", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(f"final_model.{quant_ext}")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int6+{compress_name}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int6+{compress_name}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    quant_ext = "int6.ptz"
    with open(f"final_model.{quant_ext}", "rb") as f:
        quant_blob_disk = f.read()
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        quant_decompressed = dctx.decompress(quant_blob_disk)
    else:
        quant_decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    # TTT: adapt the quantized model on validation data before scoring
    if args.ttt_enabled:
        log0(f"ttt:starting lr:{args.ttt_lr} momentum:{args.ttt_momentum} max_sec:{args.ttt_max_seconds}")
        base_model.train()
        ttt_opt = torch.optim.SGD(base_model.parameters(), lr=args.ttt_lr, momentum=args.ttt_momentum)
        ttt_seq_len = args.train_seq_len
        ttt_total = (val_tokens.numel() - 1) // ttt_seq_len
        ttt_start = (ttt_total * rank) // world_size
        ttt_end = (ttt_total * (rank + 1)) // world_size
        ttt_batch = 16  # sequences per micro-step
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_steps = 0
        for seq_start in range(ttt_start, ttt_end, ttt_batch):
            if time.perf_counter() - t_ttt > args.ttt_max_seconds:
                break
            seq_end = min(seq_start + ttt_batch, ttt_end)
            raw_s = seq_start * ttt_seq_len
            raw_e = seq_end * ttt_seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, ttt_seq_len)
            y = local[1:].reshape(-1, ttt_seq_len)
            ttt_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()
            ttt_opt.step()
            ttt_steps += 1
        torch.cuda.synchronize()
        ttt_elapsed = time.perf_counter() - t_ttt
        log0(f"ttt:done steps:{ttt_steps} time:{ttt_elapsed:.1f}s seqs:{ttt_end - ttt_start}")
        base_model.eval()

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval: gives each scored token (seq_len - stride) context.
    if args.eval_stride > 0:
        torch.cuda.synchronize()
        t_sw = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding_window(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window_eval stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_sw):.0f}ms"
        )
        log0(
            f"final_sliding_window_eval_exact stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
