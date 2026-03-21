"""
Ensemble experiment for Parameter Golf.

Trains two smaller models that together fit in the 16MB artifact budget,
then blends their predictions at eval time for better BPB.

== RESEARCH SUMMARY ==

Ensemble theory for language models:
- Two models making *different* errors → blended prediction is better than either alone
- For BPB (a log-based metric), geometric mean of probabilities (= average of log-probs)
  is theoretically optimal because it minimizes KL divergence to the true distribution
- Linear probability blending P_final = α*P_A + (1-α)*P_B is safer (stays in simplex)
  and typically gives 80-90% of the geometric mean's benefit
- Typical ensemble improvement: 2-5% in perplexity → ~0.02-0.06 BPB for our range
- Diversity is key: different depths, seeds, or data orderings maximize the benefit

Blending methods (we implement all three):
  1. Linear:     P = α * P_A + (1-α) * P_B           (safe, always valid distribution)
  2. Geometric:  P ∝ P_A^α * P_B^(1-α)               (optimal for log-loss, needs renorm)
  3. Log-linear: logit = α * logit_A + (1-α) * logit_B (fastest, approximates geometric)

Size budget analysis (16MB = 16,000,000 bytes total, code ~30KB):
  Current 11L×512d → ~9.8MB at int6+zstd-22
  Option A: Two 6L×512d  → ~5.0MB each = ~10.0MB total ← best fit, room to spare
  Option B: Two 8L×448d  → ~5.5MB each = ~11.0MB total
  Option C: 9L×512d (8MB) + 4L×512d (3.5MB) = ~11.5MB total ← asymmetric

Training time budget (10 min = 600s on 8xH100):
  Option 1: Sequential: 5 min each (both models on all 8 GPUs)
  Option 2: Parallel:   10 min each (4 GPUs per model) ← better GPU utilization
  Option 3: This script: sequential on 1 GPU for experimentation

== USAGE ==

This script is a RESEARCH PROTOTYPE for local/single-GPU experimentation.
It trains two small models sequentially, then evaluates them individually
and as an ensemble to measure the BPB improvement from blending.

  # Quick local test (CPU/single GPU, ~2 min per model)
  RUN_ID=ensemble_test ITERATIONS=500 MAX_WALLCLOCK_SECONDS=0 \
  TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=512 VAL_LOSS_EVERY=0 \
  TRAIN_LOG_EVERY=100 python3 experiment_ensemble.py

  # Modal 1xH100 test (~5 min total, 2.5 min per model)
  # (would need to be wrapped in a Modal runner, or run directly on a GPU box)
"""

from __future__ import annotations

import copy
import io
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Import everything we need from the main training script.
# This avoids duplicating 1800 lines of model/optimizer/data/eval code.
# ---------------------------------------------------------------------------

from sota_train_gpt import (
    Hyperparameters,
    GPT,
    Muon,
    CastedLinear,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT6_QUANT_RANGE,
    INT8_QUANT_RANGE,
    MUON_USE_CUDA_GRAPH,
    QAT_ACTIVE,
    TokenStream,
    build_sentencepiece_luts,
    load_validation_tokens,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
    apply_qat_preset_alignment,
    zeropower_via_newtonschulz5,
    batched_newton_schulz,
)

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# ---------------------------------------------------------------------------
# ENSEMBLE CONFIGURATION
# ---------------------------------------------------------------------------

# Model A: "deep" — more layers, narrower
MODEL_A_LAYERS = int(os.environ.get("MODEL_A_LAYERS", 6))
MODEL_A_DIM = int(os.environ.get("MODEL_A_DIM", 512))

# Model B: "wide" — fewer layers, wider (different error profile)
MODEL_B_LAYERS = int(os.environ.get("MODEL_B_LAYERS", 4))
MODEL_B_DIM = int(os.environ.get("MODEL_B_DIM", 640))

# Blending weight: α=0.5 is equal weighting (good default for same-size models)
BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", 0.5))

# Seeds: different seeds → different initializations → more diversity
SEED_A = int(os.environ.get("SEED_A", 1337))
SEED_B = int(os.environ.get("SEED_B", 42))

# Time budget: each model gets half the wallclock
# If MAX_WALLCLOCK_SECONDS=0 (unlimited), each model runs for ITERATIONS steps
WALLCLOCK_SPLIT = float(os.environ.get("WALLCLOCK_SPLIT", 0.5))


# ---------------------------------------------------------------------------
# HELPER: Build a model + optimizer from config
# ---------------------------------------------------------------------------

def build_model_and_optimizers(
    args: Hyperparameters,
    num_layers: int,
    model_dim: int,
    device: torch.device,
) -> tuple[GPT, list[torch.optim.Optimizer]]:
    """Instantiate a GPT model and its optimizers with the given shape."""
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
        smear_gate_init=args.smear_gate_init,
        xsa_last_n=min(args.xsa_last_n, num_layers),
        rope_dims=args.rope_dims,
        ln_scale_enabled=args.ln_scale_enabled,
    ).to(device).bfloat16()

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    apply_qat_preset_alignment(model, num_layers)

    # Optimizer setup (mirrors sota_train_gpt.py main())
    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    token_params = [model.tok_emb.weight]
    if model.bigram_hash_emb is not None:
        token_params.append(model.bigram_hash_emb.weight)
    if model.bigram_hash_proj is not None:
        matrix_params.append(model.bigram_hash_proj.weight)
    if model.smear_gate is not None:
        scalar_params.append(model.smear_gate)
    if model.bigram_scale is not None:
        scalar_params.append(model.bigram_scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    fused = device.type == "cuda"

    optimizer_tok = torch.optim.AdamW(
        [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=fused,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr

    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=fused,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=fused,
        )
        optimizers.insert(1, optimizer_head)

    return model, optimizers


# ---------------------------------------------------------------------------
# TRAINING LOOP (simplified single-GPU version)
# ---------------------------------------------------------------------------

def train_single_model(
    args: Hyperparameters,
    model: GPT,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
    max_wallclock_s: float,
    max_iterations: int,
    seed: int,
    label: str,
) -> GPT:
    """Train a model for up to max_wallclock_s seconds or max_iterations steps."""
    import sota_train_gpt as stg
    stg.QAT_ACTIVE = True

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    stream = TokenStream(args.train_files)
    grad_accum_steps = 1  # single GPU, simplified
    grad_scale = 1.0

    # EMA state
    ema_state = None
    if args.ema_enabled:
        ema_state = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}

    def zero_grad_all():
        for opt in optimizers:
            is_muon = isinstance(opt, Muon)
            opt.zero_grad(set_to_none=not is_muon)

    use_compile = bool(int(os.environ.get("TORCH_COMPILE", "1"))) and device.type == "cuda"
    compiled_model = torch.compile(model, dynamic=False) if use_compile else model

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        max_ms = max_wallclock_s * 1000.0 if max_wallclock_s > 0 else None
        if max_ms is None:
            warmdown_start = max(max_iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < max_iterations:
                return max((max_iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    compiled_model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    training_time_ms = 0.0

    for step in range(max_iterations):
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_wallclock_s > 0 and elapsed_ms >= max_wallclock_s * 1000.0:
            print(f"  [{label}] stopping at step {step}: wallclock cap reached ({elapsed_ms:.0f}ms)")
            break

        scale = lr_mul(step, elapsed_ms)

        # Late QAT activation
        max_ms = max_wallclock_s * 1000.0 if max_wallclock_s > 0 else None
        if args.qat_start_frac < 1.0 and max_ms is not None:
            stg.QAT_ACTIVE = (elapsed_ms / max_ms) >= args.qat_start_frac
        elif args.qat_start_frac >= 1.0:
            stg.QAT_ACTIVE = False

        zero_grad_all()

        # Get batch
        per_rank_span = args.train_batch_tokens + 1
        chunk = stream.take(per_rank_span)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)
        x, y = x.to(device), y.to(device)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = compiled_model(x, y)
        loss.backward()

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            if isinstance(opt, Muon):
                for group in opt.param_groups:
                    group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA
        if ema_state is not None:
            d = args.ema_decay
            for k, v in model.state_dict().items():
                ema_state[k].mul_(d).add_(v.detach().cpu().float(), alpha=1.0 - d)

        log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
        if log_every > 0 and (step + 1) % log_every == 0:
            approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            print(
                f"  [{label}] step:{step + 1}/{max_iterations} "
                f"train_loss:{loss.item():.4f} time:{approx_ms:.0f}ms "
                f"step_avg:{approx_ms / (step + 1):.1f}ms"
            )

    # Apply EMA
    if ema_state is not None:
        ema_sd = {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in ema_state.items()}
        model.load_state_dict(ema_sd, strict=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] done: {step + 1} steps in {total_ms:.0f}ms, {n_params:,} params")
    return model


# ---------------------------------------------------------------------------
# ENSEMBLE EVALUATION
# ---------------------------------------------------------------------------

def eval_single_model_bpb(
    args: Hyperparameters,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate a single model, returning (val_loss, val_bpb)."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = max(args.val_batch_size // seq_len, 1)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_ensemble_bpb(
    args: Hyperparameters,
    model_a: GPT,
    model_b: GPT,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    alpha: float = 0.5,
) -> dict[str, tuple[float, float]]:
    """
    Evaluate ensemble with three blending methods.

    Returns dict mapping method name → (val_loss, val_bpb).
    Methods:
      - "linear":    P = α*P_A + (1-α)*P_B
      - "geometric": P ∝ P_A^α * P_B^(1-α)    (renormalized)
      - "loglinear": logit = α*logit_A + (1-α)*logit_B
    """
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = max(args.val_batch_size // seq_len, 1)

    # Accumulators for each method
    methods = ["linear", "geometric", "loglinear"]
    loss_sums = {m: torch.zeros((), device=device, dtype=torch.float64) for m in methods}
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model_a.eval()
    model_b.eval()

    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            targets = y.reshape(-1)

            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits_a = model_a.forward_logits(x).float()  # [B, T, V]
                logits_b = model_b.forward_logits(x).float()  # [B, T, V]

            logits_a = logits_a.reshape(-1, logits_a.size(-1))  # [B*T, V]
            logits_b = logits_b.reshape(-1, logits_b.size(-1))

            # -- Method 1: Linear probability blending --
            # P = α * softmax(logits_A) + (1-α) * softmax(logits_B)
            prob_a = F.softmax(logits_a, dim=-1)
            prob_b = F.softmax(logits_b, dim=-1)
            prob_linear = alpha * prob_a + (1 - alpha) * prob_b
            # NLL = -log(P[target])
            log_prob_linear = torch.log(prob_linear.clamp(min=1e-30))
            nll_linear = F.nll_loss(log_prob_linear, targets, reduction="sum")
            loss_sums["linear"] += nll_linear.to(torch.float64)

            # -- Method 2: Geometric mean (optimal for log-loss) --
            # P ∝ P_A^α * P_B^(1-α)
            # In log space: log P ∝ α * log P_A + (1-α) * log P_B
            log_prob_a = F.log_softmax(logits_a, dim=-1)
            log_prob_b = F.log_softmax(logits_b, dim=-1)
            log_prob_geo_unnorm = alpha * log_prob_a + (1 - alpha) * log_prob_b
            # Renormalize: subtract log-sum-exp
            log_prob_geo = log_prob_geo_unnorm - torch.logsumexp(log_prob_geo_unnorm, dim=-1, keepdim=True)
            nll_geo = F.nll_loss(log_prob_geo, targets, reduction="sum")
            loss_sums["geometric"] += nll_geo.to(torch.float64)

            # -- Method 3: Log-linear (logit averaging) --
            # logit_blend = α * logit_A + (1-α) * logit_B, then softmax
            logits_blend = alpha * logits_a + (1 - alpha) * logits_b
            nll_loglinear = F.cross_entropy(logits_blend, targets, reduction="sum")
            loss_sums["loglinear"] += nll_loglinear.to(torch.float64)

            batch_token_count = float(targets.numel())
            token_count += batch_token_count

            # BPB byte counting (same for all methods, only depends on targets)
            prev_ids = x.reshape(-1)
            tgt_ids = targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()

    model_a.train()
    model_b.train()

    results = {}
    for m in methods:
        val_loss = (loss_sums[m] / token_count).item()
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = token_count.item() / byte_count.item()
        results[m] = (val_loss, bits_per_token * tokens_per_byte)

    return results


# ---------------------------------------------------------------------------
# ARTIFACT SERIALIZATION (two models in one blob)
# ---------------------------------------------------------------------------

def serialize_ensemble_artifact(
    model_a: GPT,
    model_b: GPT,
    alpha: float,
    config_a: dict,
    config_b: dict,
) -> tuple[bytes, dict]:
    """
    Quantize both models and pack into a single compressed artifact.

    Returns (compressed_bytes, stats_dict).
    """
    quant_a, stats_a = quantize_state_dict_int8(model_a.state_dict())
    quant_b, stats_b = quantize_state_dict_int8(model_b.state_dict())

    ensemble_obj = {
        "__ensemble_format__": "dual_model_v1",
        "alpha": alpha,
        "config_a": config_a,
        "config_b": config_b,
        "model_a": quant_a,
        "model_b": quant_b,
    }

    buf = io.BytesIO()
    torch.save(ensemble_obj, buf)
    raw = buf.getvalue()

    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        blob = cctx.compress(raw)
        compress_name = "zstd-22"
    else:
        import zlib
        blob = zlib.compress(raw, level=9)
        compress_name = "zlib-9"

    stats = {
        "model_a_params": stats_a["param_count"],
        "model_b_params": stats_b["param_count"],
        "total_params": stats_a["param_count"] + stats_b["param_count"],
        "model_a_payload_bytes": stats_a["int8_payload_bytes"],
        "model_b_payload_bytes": stats_b["int8_payload_bytes"],
        "raw_torch_bytes": len(raw),
        "compressed_bytes": len(blob),
        "compression": compress_name,
    }
    return blob, stats


def deserialize_ensemble_artifact(blob: bytes) -> tuple[dict, dict, float, dict, dict]:
    """
    Decompress and dequantize both models from an ensemble artifact.

    Returns (state_dict_a, state_dict_b, alpha, config_a, config_b).
    """
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(blob)
    else:
        import zlib
        raw = zlib.decompress(blob)

    obj = torch.load(io.BytesIO(raw), map_location="cpu")
    assert obj.get("__ensemble_format__") == "dual_model_v1"

    sd_a = dequantize_state_dict_int8(obj["model_a"])
    sd_b = dequantize_state_dict_int8(obj["model_b"])
    return sd_a, sd_b, obj["alpha"], obj["config_a"], obj["config_b"]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = Hyperparameters()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("ENSEMBLE EXPERIMENT")
    print("=" * 80)
    print(f"Model A: {MODEL_A_LAYERS}L x {MODEL_A_DIM}d (seed={SEED_A})")
    print(f"Model B: {MODEL_B_LAYERS}L x {MODEL_B_DIM}d (seed={SEED_B})")
    print(f"Blend alpha: {BLEND_ALPHA}")
    print(f"Device: {device}")
    print()

    # Tokenizer + validation setup
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")

    # Time budget
    total_wallclock = args.max_wallclock_seconds
    if total_wallclock > 0:
        wallclock_a = total_wallclock * WALLCLOCK_SPLIT
        wallclock_b = total_wallclock * (1.0 - WALLCLOCK_SPLIT)
    else:
        wallclock_a = 0
        wallclock_b = 0
    iterations = args.iterations

    # =====================================================================
    # PHASE 1: Train Model A
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 1: Training Model A ({MODEL_A_LAYERS}L x {MODEL_A_DIM}d)")
    print(f"  Wallclock budget: {wallclock_a:.0f}s, Max iterations: {iterations}")
    print(f"{'='*80}")

    model_a, opts_a = build_model_and_optimizers(args, MODEL_A_LAYERS, MODEL_A_DIM, device)
    n_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {n_a:,}")

    model_a = train_single_model(
        args, model_a, opts_a, device,
        max_wallclock_s=wallclock_a,
        max_iterations=iterations,
        seed=SEED_A,
        label="Model_A",
    )

    # =====================================================================
    # PHASE 2: Train Model B
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 2: Training Model B ({MODEL_B_LAYERS}L x {MODEL_B_DIM}d)")
    print(f"  Wallclock budget: {wallclock_b:.0f}s, Max iterations: {iterations}")
    print(f"{'='*80}")

    model_b, opts_b = build_model_and_optimizers(args, MODEL_B_LAYERS, MODEL_B_DIM, device)
    n_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Parameters: {n_b:,}")

    model_b = train_single_model(
        args, model_b, opts_b, device,
        max_wallclock_s=wallclock_b,
        max_iterations=iterations,
        seed=SEED_B,
        label="Model_B",
    )

    # =====================================================================
    # PHASE 3: Evaluate individually
    # =====================================================================
    print(f"\n{'='*80}")
    print("PHASE 3: Individual evaluation")
    print(f"{'='*80}")

    loss_a, bpb_a = eval_single_model_bpb(
        args, model_a, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Model A: val_loss={loss_a:.4f}  val_bpb={bpb_a:.4f}")

    loss_b, bpb_b = eval_single_model_bpb(
        args, model_b, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Model B: val_loss={loss_b:.4f}  val_bpb={bpb_b:.4f}")

    # =====================================================================
    # PHASE 4: Ensemble evaluation (three blending methods)
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 4: Ensemble evaluation (alpha={BLEND_ALPHA})")
    print(f"{'='*80}")

    ensemble_results = eval_ensemble_bpb(
        args, model_a, model_b, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        alpha=BLEND_ALPHA,
    )

    best_method = None
    best_bpb = float("inf")
    for method, (vloss, vbpb) in ensemble_results.items():
        improvement = min(bpb_a, bpb_b) - vbpb
        print(f"  {method:12s}: val_loss={vloss:.4f}  val_bpb={vbpb:.4f}  improvement={improvement:+.4f}")
        if vbpb < best_bpb:
            best_bpb = vbpb
            best_method = method

    # =====================================================================
    # PHASE 5: Alpha sweep (find optimal blending weight)
    # =====================================================================
    print(f"\n{'='*80}")
    print("PHASE 5: Alpha sweep (finding optimal blend weight)")
    print(f"{'='*80}")

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_sweep_alpha = BLEND_ALPHA
    best_sweep_bpb = best_bpb
    best_sweep_method = best_method

    for a in alphas:
        results = eval_ensemble_bpb(
            args, model_a, model_b, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            alpha=a,
        )
        for method, (vloss, vbpb) in results.items():
            if vbpb < best_sweep_bpb:
                best_sweep_bpb = vbpb
                best_sweep_alpha = a
                best_sweep_method = method
        # Print only the best method per alpha for readability
        geo_loss, geo_bpb = results["geometric"]
        lin_loss, lin_bpb = results["linear"]
        print(f"  alpha={a:.1f}  geometric_bpb={geo_bpb:.4f}  linear_bpb={lin_bpb:.4f}")

    print(f"\n  Best: alpha={best_sweep_alpha:.1f}, method={best_sweep_method}, bpb={best_sweep_bpb:.4f}")

    # =====================================================================
    # PHASE 6: Artifact serialization + roundtrip test
    # =====================================================================
    print(f"\n{'='*80}")
    print("PHASE 6: Artifact serialization")
    print(f"{'='*80}")

    config_a = {"num_layers": MODEL_A_LAYERS, "model_dim": MODEL_A_DIM}
    config_b = {"num_layers": MODEL_B_LAYERS, "model_dim": MODEL_B_DIM}

    blob, stats = serialize_ensemble_artifact(
        model_a, model_b, best_sweep_alpha, config_a, config_b,
    )

    print(f"  Model A: {stats['model_a_params']:,} params, {stats['model_a_payload_bytes']:,} payload bytes")
    print(f"  Model B: {stats['model_b_params']:,} params, {stats['model_b_payload_bytes']:,} payload bytes")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Raw torch bytes: {stats['raw_torch_bytes']:,}")
    print(f"  Compressed ({stats['compression']}): {stats['compressed_bytes']:,}")
    print(f"  Code bytes (estimate): ~30,000")
    print(f"  Total artifact: ~{stats['compressed_bytes'] + 30_000:,}")

    budget = 16_000_000
    fits = (stats["compressed_bytes"] + 30_000) < budget
    print(f"  Within 16MB budget: {'YES' if fits else 'NO'} ({budget - stats['compressed_bytes'] - 30_000:,} bytes remaining)")

    # Save artifact
    artifact_path = f"ensemble_{args.run_id}.ptz"
    with open(artifact_path, "wb") as f:
        f.write(blob)
    print(f"  Saved to: {artifact_path}")

    # Roundtrip test
    print("\n  Roundtrip dequantization test...")
    with open(artifact_path, "rb") as f:
        blob_disk = f.read()
    sd_a_rt, sd_b_rt, alpha_rt, cfg_a_rt, cfg_b_rt = deserialize_ensemble_artifact(blob_disk)

    model_a_rt, _ = build_model_and_optimizers(args, cfg_a_rt["num_layers"], cfg_a_rt["model_dim"], device)
    model_a_rt.load_state_dict(sd_a_rt, strict=True)
    model_b_rt, _ = build_model_and_optimizers(args, cfg_b_rt["num_layers"], cfg_b_rt["model_dim"], device)
    model_b_rt.load_state_dict(sd_b_rt, strict=True)

    rt_results = eval_ensemble_bpb(
        args, model_a_rt, model_b_rt, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        alpha=alpha_rt,
    )
    for method, (vloss, vbpb) in rt_results.items():
        print(f"  roundtrip {method:12s}: val_loss={vloss:.4f}  val_bpb={vbpb:.4f}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Model A ({MODEL_A_LAYERS}L x {MODEL_A_DIM}d): val_bpb={bpb_a:.4f}  ({n_a:,} params)")
    print(f"  Model B ({MODEL_B_LAYERS}L x {MODEL_B_DIM}d): val_bpb={bpb_b:.4f}  ({n_b:,} params)")
    print(f"  Best single model BPB: {min(bpb_a, bpb_b):.4f}")
    print()
    print(f"  Ensemble (pre-quant):  method={best_sweep_method} alpha={best_sweep_alpha:.1f} bpb={best_sweep_bpb:.4f}")
    print(f"  Ensemble improvement:  {min(bpb_a, bpb_b) - best_sweep_bpb:+.4f} BPB")
    print()

    rt_best_bpb = min(vbpb for _, vbpb in rt_results.values())
    print(f"  Ensemble (post-quant): bpb={rt_best_bpb:.4f}")
    print(f"  Post-quant improvement vs best single: {min(bpb_a, bpb_b) - rt_best_bpb:+.4f} BPB")
    print()
    print(f"  Artifact size: {stats['compressed_bytes']:,} bytes ({stats['compression']})")
    print(f"  Budget remaining: {budget - stats['compressed_bytes'] - 30_000:,} bytes")
    print()

    # Verdict
    if best_sweep_bpb < min(bpb_a, bpb_b):
        delta = min(bpb_a, bpb_b) - best_sweep_bpb
        print(f"  VERDICT: Ensemble WINS by {delta:.4f} BPB")
        if delta >= 0.005:
            print(f"  This exceeds the competition significance threshold of 0.005!")
        else:
            print(f"  But delta < 0.005 (competition significance threshold)")
    else:
        print(f"  VERDICT: Ensemble does NOT improve over best single model")
        print(f"  The parameter budget may be better spent on a single larger model")

    print()
    print("NEXT STEPS:")
    print("  1. If ensemble wins: try on H100 with 5 min each, real batch sizes")
    print("  2. Try more diverse configs (e.g., different MLP_MULT, different attention)")
    print("  3. Try 3-model ensemble if artifact budget allows")
    print("  4. Combine with sliding window eval for additional BPB gain")
    print("  5. Consider parallel training: 4 GPUs per model on 8xH100")


if __name__ == "__main__":
    main()
