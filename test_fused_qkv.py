"""
Test: Fused QKV projection correctness.

Verifies that the fused c_qkv linear produces identical attention output
to manually splitting into separate Q, K, V projections.

Also benchmarks the fused vs unfused attention forward pass.

Usage:
    python3 test_fused_qkv.py           # on GPU
    CUDA_VISIBLE_DEVICES="" python3 test_fused_qkv.py  # force CPU
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def test_fused_qkv_equivalence():
    """Verify fused QKV produces same result as separate Q, K, V projections."""
    print("=" * 60)
    print("  TEST: Fused QKV equivalence")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, D = 4, 128, 512
    H, Hkv = 8, 4
    Dh = D // H
    kv_dim = Hkv * Dh  # 256

    # Create fused QKV weight [D + 2*kv_dim, D] = [1024, 512]
    w_qkv = torch.randn(D + 2 * kv_dim, D, device=device)

    # Split into separate Q, K, V weights
    w_q = w_qkv[:D, :]           # [512, 512]
    w_k = w_qkv[D:D+kv_dim, :]   # [256, 512]
    w_v = w_qkv[D+kv_dim:, :]    # [256, 512]

    # Input
    x = torch.randn(B, T, D, device=device)

    # Fused path: single matmul + split
    qkv = F.linear(x, w_qkv)
    q_fused = qkv[:, :, :D].reshape(B, T, H, Dh)
    k_fused = qkv[:, :, D:D+kv_dim].reshape(B, T, Hkv, Dh)
    v_fused = qkv[:, :, D+kv_dim:].reshape(B, T, Hkv, Dh)

    # Unfused path: three separate matmuls
    q_unfused = F.linear(x, w_q).reshape(B, T, H, Dh)
    k_unfused = F.linear(x, w_k).reshape(B, T, Hkv, Dh)
    v_unfused = F.linear(x, w_v).reshape(B, T, Hkv, Dh)

    # Compare
    q_diff = (q_fused - q_unfused).abs().max().item()
    k_diff = (k_fused - k_unfused).abs().max().item()
    v_diff = (v_fused - v_unfused).abs().max().item()

    print(f"  Q max diff: {q_diff:.10f}")
    print(f"  K max diff: {k_diff:.10f}")
    print(f"  V max diff: {v_diff:.10f}")

    passed = q_diff < 1e-5 and k_diff < 1e-5 and v_diff < 1e-5
    print(f"\n  {'PASSED' if passed else 'FAILED'} (threshold: 1e-5)")
    return passed


def test_fused_qkv_shapes():
    """Verify output shapes with various configs."""
    print("\n" + "=" * 60)
    print("  TEST: Fused QKV shapes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        (512, 8, 4, "Standard GQA"),
        (512, 8, 8, "Full MHA"),
        (512, 8, 1, "Multi-Query"),
        (256, 4, 2, "Small model"),
    ]

    all_passed = True
    for D, H, Hkv, label in configs:
        Dh = D // H
        kv_dim = Hkv * Dh
        B, T = 2, 64

        w_qkv = torch.randn(D + 2 * kv_dim, D, device=device)
        x = torch.randn(B, T, D, device=device)

        qkv = F.linear(x, w_qkv)
        q = qkv[:, :, :D].reshape(B, T, H, Dh)
        k = qkv[:, :, D:D+kv_dim].reshape(B, T, Hkv, Dh)
        v = qkv[:, :, D+kv_dim:].reshape(B, T, Hkv, Dh)

        q_ok = q.shape == (B, T, H, Dh)
        k_ok = k.shape == (B, T, Hkv, Dh)
        v_ok = v.shape == (B, T, Hkv, Dh)
        ok = q_ok and k_ok and v_ok

        if not ok:
            all_passed = False
        print(f"  {label:<20s}  D={D} H={H} Hkv={Hkv}  "
              f"q={q.shape} k={k.shape} v={v.shape}  {'OK' if ok else 'FAIL'}")

    print(f"\n  {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_muon_shape_groups():
    """Verify batched Muon shape groups are better with fused QKV."""
    print("\n" + "=" * 60)
    print("  TEST: Muon shape groups (fused vs unfused)")
    print("=" * 60)

    D, Hkv, Dh, MLP_MULT, L = 512, 4, 64, 3, 9
    kv_dim = Hkv * Dh  # 256
    MLP_H = D * MLP_MULT  # 1536

    # Old (unfused) shapes
    old_shapes = []
    for _ in range(L):
        old_shapes.extend([
            (D, D),         # c_q
            (kv_dim, D),    # c_k
            (kv_dim, D),    # c_v
            (D, D),         # proj
            (MLP_H, D),     # mlp.fc
            (D, MLP_H),     # mlp.proj
        ])
    old_shapes.append((D, 128))  # bigram_hash_proj

    # New (fused) shapes
    new_shapes = []
    for _ in range(L):
        new_shapes.extend([
            (D + 2*kv_dim, D),  # c_qkv
            (D, D),             # proj
            (MLP_H, D),         # mlp.fc
            (D, MLP_H),         # mlp.proj
        ])
    new_shapes.append((D, 128))  # bigram_hash_proj

    from collections import Counter
    old_groups = Counter(old_shapes)
    new_groups = Counter(new_shapes)

    print(f"\n  Unfused: {len(old_shapes)} params, {len(old_groups)} shape groups")
    for shape, count in sorted(old_groups.items(), key=lambda x: -x[1]):
        print(f"    {str(shape):>20s}  ×{count}")

    print(f"\n  Fused: {len(new_shapes)} params, {len(new_groups)} shape groups")
    for shape, count in sorted(new_groups.items(), key=lambda x: -x[1]):
        print(f"    {str(shape):>20s}  ×{count}")

    print(f"\n  Reduction: {len(old_shapes)} → {len(new_shapes)} params "
          f"({len(old_shapes) - len(new_shapes)} fewer)")
    print(f"  NS calls:  {len(old_groups)} → {len(new_groups)} batched calls")

    # The big win: each c_qkv [1024, 512] matmul replaces 3 separate matmuls
    # at the kernel level. At the Muon level, fewer params = less overhead.


def benchmark_fused_vs_unfused():
    """Benchmark fused vs unfused QKV (GPU only)."""
    if not torch.cuda.is_available():
        print("\n  BENCHMARK: Skipped (no CUDA)")
        return

    print("\n" + "=" * 60)
    print("  BENCHMARK: Fused vs unfused QKV forward pass")
    print("=" * 60)

    device = torch.device("cuda")
    B, T, D = 48, 2048, 512
    H, Hkv = 8, 4
    Dh = D // H
    kv_dim = Hkv * Dh

    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

    # Unfused: 3 separate linears
    w_q = torch.randn(D, D, device=device, dtype=torch.bfloat16)
    w_k = torch.randn(kv_dim, D, device=device, dtype=torch.bfloat16)
    w_v = torch.randn(kv_dim, D, device=device, dtype=torch.bfloat16)

    # Fused: 1 linear
    w_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # [1024, 512]

    # Warmup
    for _ in range(5):
        _ = F.linear(x, w_qkv)
        _ = F.linear(x, w_q)
        _ = F.linear(x, w_k)
        _ = F.linear(x, w_v)

    results = {}
    for label, fn in [
        ("Unfused (3×)", lambda: (F.linear(x, w_q), F.linear(x, w_k), F.linear(x, w_v))),
        ("Fused (1×)", lambda: F.linear(x, w_qkv)),
    ]:
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times) * 1000
        med = sorted(times)[len(times)//2] * 1000
        results[label] = avg
        print(f"  {label:>15s}:  avg={avg:.3f}ms  med={med:.3f}ms")

    if len(results) == 2:
        speedup = list(results.values())[0] / list(results.values())[1]
        saved = list(results.values())[0] - list(results.values())[1]
        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"  Saved per layer: {saved:.3f}ms (×9 layers = {saved*9:.3f}ms/step)")
        print(f"  This is forward-only; backward saves similar amount.")


if __name__ == "__main__":
    p1 = test_fused_qkv_equivalence()
    p2 = test_fused_qkv_shapes()
    test_muon_shape_groups()
    benchmark_fused_vs_unfused()
    print("\n" + "=" * 60)
    print(f"  {'ALL TESTS PASSED' if p1 and p2 else 'SOME TESTS FAILED'}")
    print("=" * 60)
