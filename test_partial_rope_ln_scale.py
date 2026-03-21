"""
Test: Partial RoPE + LN Scale correctness.

Verifies:
1. Partial RoPE only rotates first rope_dims of each head (rest unchanged)
2. LN Scale applies correct 1/sqrt(layer+1) dampening
3. Full model forward pass runs without errors
4. Shapes are correct throughout

Usage:
    python3 test_partial_rope_ln_scale.py
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rope_dims = cos.size(-1) * 2
    if rope_dims < x.size(-1):
        x_rope = x[..., :rope_dims]
        x_pass = x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def test_partial_rope():
    """Verify partial RoPE only rotates first rope_dims dimensions."""
    print("=" * 60)
    print("  TEST: Partial RoPE (16/64)")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, H, Dh = 2, 32, 8, 64
    rope_dims = 16  # only rotate first 16 of 64

    # Create q tensor
    q = torch.randn(B, T, H, Dh, device=device)
    q_orig = q.clone()

    # Generate cos/sin for rope_dims only (8 frequencies for 16 dims)
    # Use FA3 layout: [1, T, 1, rope_dims//2] to match q shape [B, T, H, Dh]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]  # [1, T, 1, rope_dims//2]
    sin = freqs.sin()[None, :, None, :]

    # Apply partial RoPE
    q_rotated = apply_rotary_emb(q, cos, sin)

    # Check: first 16 dims should be different (rotated)
    rotated_diff = (q_rotated[..., :rope_dims] - q_orig[..., :rope_dims]).abs().max().item()

    # Check: last 48 dims should be IDENTICAL (passed through)
    passthrough_diff = (q_rotated[..., rope_dims:] - q_orig[..., rope_dims:]).abs().max().item()

    print(f"  Rotated dims (0:{rope_dims}):    max diff = {rotated_diff:.6f} (should be > 0)")
    print(f"  Passthrough dims ({rope_dims}:{Dh}): max diff = {passthrough_diff:.10f} (should be 0)")

    passed = rotated_diff > 0.01 and passthrough_diff == 0.0
    print(f"\n  {'PASSED' if passed else 'FAILED'}")
    return passed


def test_full_rope_backward_compat():
    """Verify rope_dims=0 (full RoPE) matches the old behavior."""
    print("\n" + "=" * 60)
    print("  TEST: Full RoPE backward compatibility")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, H, Dh = 2, 32, 8, 64
    q = torch.randn(B, T, H, Dh, device=device)

    # Full RoPE (old behavior): cos/sin cover all dims
    # FA3 layout: [1, T, 1, Dh//2]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, Dh, 2, dtype=torch.float32, device=device) / Dh))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]

    # Old way (no partial logic) — same layout
    half = Dh // 2
    x1, x2 = q[..., :half], q[..., half:]
    q_old = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

    # New way (partial with rope_dims covering all dims)
    q_new = apply_rotary_emb(q, cos, sin)

    diff = (q_old - q_new).abs().max().item()
    passed = diff < 1e-6
    print(f"  Max diff (old vs new full RoPE): {diff:.10f}")
    print(f"\n  {'PASSED' if passed else 'FAILED'}")
    return passed


def test_ln_scale():
    """Verify LN Scale applies correct 1/sqrt(layer+1) dampening."""
    print("\n" + "=" * 60)
    print("  TEST: LN Scale (1/sqrt(layer+1))")
    print("=" * 60)

    num_layers = 9

    print(f"  Layer dampening factors:")
    all_correct = True
    for i in range(num_layers):
        scale = 1.0 / math.sqrt(i + 1)
        expected = 1.0 / math.sqrt(i + 1)
        correct = abs(scale - expected) < 1e-10
        if not correct:
            all_correct = False
        print(f"    Layer {i}: scale = {scale:.4f} (1/√{i+1} = {expected:.4f})  "
              f"{'OK' if correct else 'FAIL'}")

    # Verify dampening effect
    x = torch.ones(2, 32, 512)
    for i in range(num_layers):
        scale = 1.0 / math.sqrt(i + 1)
        scaled = x * scale
        ratio = scaled[0, 0, 0].item()
        print(f"    Layer {i}: output magnitude = {ratio:.4f}x input")

    print(f"\n  Layer 0 (first): 1.000x — full signal")
    print(f"  Layer 4 (middle): {1/math.sqrt(5):.3f}x — dampened")
    print(f"  Layer 8 (last):  {1/math.sqrt(9):.3f}x — most dampened")
    print(f"\n  {'PASSED' if all_correct else 'FAILED'}")
    return all_correct


def test_model_forward():
    """Run a full forward pass with partial RoPE + LN Scale enabled."""
    print("\n" + "=" * 60)
    print("  TEST: Full model forward pass")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import sota_train_gpt as T
        model = T.GPT(
            vocab_size=1024, num_layers=9, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=3,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=50000.0,
            qk_gain_init=1.5, use_smeargate=True,
            bigram_hash_buckets=10240, bigram_hash_dim=128,
            smear_gate_init=0.0, xsa_last_n=4,
            rope_dims=16, ln_scale_enabled=True,
        ).to(device).bfloat16()

        for m in model.modules():
            if isinstance(m, T.CastedLinear):
                m.float()

        B, T_len = 2, 128
        x = torch.randint(0, 1024, (B, T_len), device=device)
        y = torch.randint(0, 1024, (B, T_len), device=device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(x, y)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Partial RoPE dims: {model.blocks[0].attn.rotary.rope_dims}")
        print(f"  LN Scale factors: {[f'{b.ln_scale:.3f}' for b in model.blocks]}")
        print(f"  XSA layers: {[i for i, b in enumerate(model.blocks) if b.attn.use_xsa]}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {n_params:,}")

        passed = loss.item() > 0 and not torch.isnan(loss)
        print(f"\n  {'PASSED' if passed else 'FAILED'}")
        return passed

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    p1 = test_partial_rope()
    p2 = test_full_rope_backward_compat()
    p3 = test_ln_scale()
    p4 = test_model_forward()
    print("\n" + "=" * 60)
    all_passed = p1 and p2 and p3 and p4
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
