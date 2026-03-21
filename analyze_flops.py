"""
Theoretical FLOP and bandwidth analysis for parameter-golf model.

No GPU needed — pure math. Calculates:
1. Exact FLOPs per forward pass
2. Memory bytes moved per operation
3. Arithmetic intensity (FLOPs/byte) — determines compute vs memory bound
4. Theoretical minimum time on H100 PCIe and H100 SXM
5. Efficiency = measured / theoretical

Usage:
    python3 analyze_flops.py
"""

from __future__ import annotations


def analyze():
    # ─── Model Config ───
    B = 48          # batch size per micro-step (98304 tokens / 2048 seq_len)
    T = 2048        # sequence length
    D = 512         # model dimension
    H = 8           # attention heads
    Hkv = 4         # KV heads (GQA)
    Dh = D // H     # head dim = 64
    L = 9           # layers (physical blocks)
    V = 1024        # vocab size
    MLP_MULT = 3    # MLP expansion
    MLP_H = D * MLP_MULT  # 1536
    BIGRAM_HASH = 10240
    BIGRAM_DIM = 128

    # ─── Hardware Specs ───
    # H100 PCIe: 51 TFLOPS bf16, 2.0 TB/s bandwidth
    # H100 SXM:  ~990 TFLOPS bf16 (with sparsity: 1979), 3.35 TB/s bandwidth
    h100_pcie_tflops = 51.0     # tensor core bf16
    h100_pcie_bw_tbs = 2.0      # TB/s HBM bandwidth
    h100_sxm_tflops = 990.0     # tensor core bf16
    h100_sxm_bw_tbs = 3.35

    # Actually, for non-sparse bf16 matmul:
    # H100 PCIe: 756 TFLOPS (tensor core, dense bf16)
    # H100 SXM: 989 TFLOPS (tensor core, dense bf16)
    # The 51 TFLOPS number is FP32. Let me use bf16 tensor core numbers.
    h100_pcie_tflops = 756.0
    h100_sxm_tflops = 989.0

    print("=" * 70)
    print("  THEORETICAL FLOP & BANDWIDTH ANALYSIS")
    print("=" * 70)
    print(f"  Model: {L}L × {D}d, {H} heads, {Hkv} KV heads, MLP {MLP_MULT}x")
    print(f"  Batch: {B} seqs × {T} tokens = {B*T:,} tokens/micro-step")
    print()

    # ─── Helper ───
    # matmul [M, K] × [K, N] = 2*M*K*N FLOPs (multiply + add)
    def matmul_flops(M, K, N):
        return 2 * M * K * N

    # matmul bytes: read both inputs + write output (bf16 = 2 bytes each)
    def matmul_bytes(M, K, N):
        return 2 * (M * K + K * N + M * N)  # bf16

    total_fwd_flops = 0
    total_fwd_bytes = 0
    ops = []

    def add_op(name, flops, bytes_moved):
        nonlocal total_fwd_flops, total_fwd_bytes
        total_fwd_flops += flops
        total_fwd_bytes += bytes_moved
        intensity = flops / bytes_moved if bytes_moved > 0 else float('inf')
        ops.append((name, flops, bytes_moved, intensity))

    # ─── Embedding ───
    # tok_emb: lookup B*T embeddings of dim D
    emb_bytes = B * T * D * 2 + V * D * 2  # read weight + write output
    add_op("tok_emb lookup", 0, emb_bytes)

    # BigramHash: lookup + proj
    bigram_lookup_bytes = B * T * BIGRAM_DIM * 2 + BIGRAM_HASH * BIGRAM_DIM * 2
    add_op("bigram_hash lookup", 0, bigram_lookup_bytes)

    # bigram_hash_proj: [B*T, 128] × [128, 512]
    bhp_flops = matmul_flops(B * T, BIGRAM_DIM, D)
    bhp_bytes = matmul_bytes(B * T, BIGRAM_DIM, D)
    add_op("bigram_hash_proj", bhp_flops, bhp_bytes)

    # SmearGate mixing: ~5 elementwise ops on [B, T, D]
    gate_bytes = 5 * B * T * D * 2 * 2  # read + write per op
    add_op("smeargate mixing", B * T * D * 5, gate_bytes)

    # Initial RMSNorm
    rms_bytes = B * T * D * 2 * 2  # read + write
    add_op("initial rms_norm", B * T * D * 2, rms_bytes)

    # ─── Per Layer ───
    for layer in range(L):
        prefix = f"L{layer}"

        # resid_mix: 2 elementwise muls + add on [B, T, D]
        mix_bytes = B * T * D * 2 * 3 * 2  # 3 reads (x, x0, mix) + 1 write, ×2 for two mixes
        add_op(f"{prefix} resid_mix", B * T * D * 3, mix_bytes)

        # attn_norm (RMSNorm)
        add_op(f"{prefix} attn_norm", B * T * D * 2, B * T * D * 2 * 2)

        # c_q: [B*T, D] × [D, D] → [B*T, D]
        add_op(f"{prefix} c_q", matmul_flops(B*T, D, D), matmul_bytes(B*T, D, D))

        # c_k: [B*T, D] × [D, Hkv*Dh] → [B*T, Hkv*Dh]
        kv_dim = Hkv * Dh
        add_op(f"{prefix} c_k", matmul_flops(B*T, D, kv_dim), matmul_bytes(B*T, D, kv_dim))

        # c_v: same as c_k
        add_op(f"{prefix} c_v", matmul_flops(B*T, D, kv_dim), matmul_bytes(B*T, D, kv_dim))

        # QK RMSNorm (on q and k separately)
        q_rms_bytes = B * T * D * 2 * 2  # q norm
        k_rms_bytes = B * T * kv_dim * 2 * 2  # k norm
        add_op(f"{prefix} qk_norm", B * T * (D + kv_dim) * 2, q_rms_bytes + k_rms_bytes)

        # RoPE: ~6 ops per position (cos, sin, mul, add for each half)
        rope_bytes = B * T * D * 2 * 4  # q rope: read cos,sin,q, write q
        rope_bytes += B * T * kv_dim * 2 * 4  # k rope
        add_op(f"{prefix} rope", B * T * (D + kv_dim) * 6, rope_bytes)

        # Attention: Q×K^T + softmax + ×V
        # Q: [B, H, T, Dh], K: [B, Hkv, T, Dh], V: [B, Hkv, T, Dh]
        # With GQA, each head group shares KV. Total compute same as full attention.
        # QK^T: [B, H, T, Dh] × [B, H, Dh, T] → [B, H, T, T]  (after GQA expansion)
        attn_qk_flops = B * H * matmul_flops(T, Dh, T) // (B)  # per-head matmul
        attn_qk_flops = B * H * 2 * T * Dh * T  # correct: B*H * 2*T*Dh*T

        # Softmax: ~5 ops per element of [B, H, T, T] (max, sub, exp, sum, div)
        softmax_flops = B * H * T * T * 5
        # But causal mask means only T*(T+1)/2 elements
        softmax_flops = B * H * (T * (T + 1) // 2) * 5

        # Attn × V: [B, H, T, T] × [B, H, T, Dh] → [B, H, T, Dh]
        attn_v_flops = B * H * 2 * T * T * Dh

        # Flash Attention handles all this fused — memory is O(B*H*T*Dh), not O(B*H*T*T)
        # Flash attn memory: read Q,K,V + write O = 4 * B * T * D * 2
        flash_bytes = 4 * B * T * D * 2

        total_attn_flops = attn_qk_flops + softmax_flops + attn_v_flops
        add_op(f"{prefix} attention", total_attn_flops, flash_bytes)

        # XSA (last 3 layers): normalize V, dot product, subtract projection
        if layer >= L - 3:
            xsa_flops = B * T * D * 10  # normalize + dot + sub
            xsa_bytes = B * T * D * 2 * 4
            add_op(f"{prefix} xsa", xsa_flops, xsa_bytes)

        # proj: [B*T, D] × [D, D]
        add_op(f"{prefix} proj", matmul_flops(B*T, D, D), matmul_bytes(B*T, D, D))

        # attn_scale + residual add
        add_op(f"{prefix} attn_residual", B * T * D * 2, B * T * D * 2 * 3)

        # mlp_norm
        add_op(f"{prefix} mlp_norm", B * T * D * 2, B * T * D * 2 * 2)

        # mlp.fc: [B*T, D] × [D, MLP_H]
        add_op(f"{prefix} mlp.fc", matmul_flops(B*T, D, MLP_H), matmul_bytes(B*T, D, MLP_H))

        # relu + square: 2 elementwise ops
        add_op(f"{prefix} relu²", B * T * MLP_H * 2, B * T * MLP_H * 2 * 2)

        # mlp.proj: [B*T, MLP_H] × [MLP_H, D]
        add_op(f"{prefix} mlp.proj", matmul_flops(B*T, MLP_H, D), matmul_bytes(B*T, MLP_H, D))

        # mlp_scale + residual add
        add_op(f"{prefix} mlp_residual", B * T * D * 2, B * T * D * 2 * 3)

    # ─── Skip connections (U-Net) ───
    # 4 skip weight multiplies + adds during decoder half
    skip_bytes = 4 * B * T * D * 2 * 3
    add_op("u-net skips", 4 * B * T * D * 2, skip_bytes)

    # ─── Final ───
    add_op("final_norm", B * T * D * 2, B * T * D * 2 * 2)

    # lm_head (tied embedding): [B*T, D] × [D, V]
    add_op("lm_head", matmul_flops(B*T, D, V), matmul_bytes(B*T, D, V))

    # softcap: tanh(x/30)*30 — 3 ops
    add_op("softcap", B * T * V * 3, B * T * V * 2 * 2)

    # cross_entropy: log_softmax + nll
    add_op("cross_entropy", B * T * V * 5, B * T * V * 2 * 2)

    # ─── QAT overhead (per CastedLinear, during training) ───
    # quantile, clamp, round, scale — ~8 ops per weight matrix
    num_casted_linears = L * 6 + 1  # 6 per layer + bigram_hash_proj
    # QAT operates on weight matrices, not activations, so it's cheap
    qat_flops = 0
    for shape_desc, count, rows, cols in [
        ("attn Q/proj", L * 2, D, D),
        ("attn K/V", L * 2, Hkv * Dh, D),
        ("mlp fc", L, MLP_H, D),
        ("mlp proj", L, D, MLP_H),
        ("bigram proj", 1, D, BIGRAM_DIM),
    ]:
        qat_flops += count * rows * cols * 8  # ~8 ops for fake quant
    add_op("QAT fake quant", qat_flops, qat_flops * 2)  # rough bytes estimate

    # ─── Print Results ───
    print(f"\n{'─'*70}")
    print(f"  {'Operation':<30s} {'GFLOPs':>10s} {'GB moved':>10s} {'Intensity':>12s} {'Bound':>10s}")
    print(f"{'─'*70}")

    # Crossover intensity: above this = compute bound, below = memory bound
    # H100 PCIe: 756 TFLOPS / 2.0 TB/s = 378 FLOPS/byte
    # H100 SXM: 989 TFLOPS / 3.35 TB/s = 295 FLOPS/byte
    crossover_pcie = h100_pcie_tflops * 1e3 / (h100_pcie_bw_tbs * 1e3)  # FLOPS/byte
    crossover_sxm = h100_sxm_tflops * 1e3 / (h100_sxm_bw_tbs * 1e3)

    for name, flops, bytes_moved, intensity in ops:
        gflops = flops / 1e9
        gb = bytes_moved / 1e9
        bound = "COMPUTE" if intensity > crossover_sxm else "MEMORY"
        if gflops < 0.001 and gb < 0.001:
            continue  # skip trivial ops
        # Only print per-layer detail for layer 0, summarize rest
        if name.startswith("L") and not name.startswith("L0"):
            continue
        print(f"  {name:<30s} {gflops:>10.3f} {gb:>10.4f} {intensity:>10.1f} F/B {bound:>10s}")

    # Summarize per-layer
    layer_flops = sum(f for n, f, b, i in ops if n.startswith("L0"))
    layer_bytes = sum(b for n, f, b, i in ops if n.startswith("L0"))
    print(f"  {'(× 9 layers total)':<30s} {layer_flops*L/1e9:>10.3f} {layer_bytes*L/1e9:>10.4f}")

    print(f"{'─'*70}")
    print(f"  {'TOTAL FORWARD':<30s} {total_fwd_flops/1e9:>10.3f} {total_fwd_bytes/1e9:>10.4f}")

    # Backward ≈ 2x forward FLOPs (gradients for each matmul = same FLOPs)
    total_bwd_flops = total_fwd_flops * 2
    total_bwd_bytes = total_fwd_bytes * 2  # rough estimate

    print(f"  {'TOTAL BACKWARD (~2x)':<30s} {total_bwd_flops/1e9:>10.3f} {total_bwd_bytes/1e9:>10.4f}")
    total_flops = total_fwd_flops + total_bwd_flops
    total_bytes = total_fwd_bytes + total_bwd_bytes
    print(f"  {'TOTAL FWD+BWD':<30s} {total_flops/1e9:>10.3f} {total_bytes/1e9:>10.4f}")

    # ─── Theoretical Minimum Times ───
    print(f"\n{'='*70}")
    print(f"  THEORETICAL MINIMUM STEP TIME")
    print(f"{'='*70}")

    # Compute-bound time: total_flops / peak_flops
    compute_time_pcie = total_flops / (h100_pcie_tflops * 1e12) * 1000  # ms
    compute_time_sxm = total_flops / (h100_sxm_tflops * 1e12) * 1000

    # Memory-bound time: total_bytes / peak_bandwidth
    memory_time_pcie = total_bytes / (h100_pcie_bw_tbs * 1e12) * 1000
    memory_time_sxm = total_bytes / (h100_sxm_bw_tbs * 1e12) * 1000

    # Theoretical minimum is max(compute, memory) — you're bottlenecked by whichever is slower
    theo_min_pcie = max(compute_time_pcie, memory_time_pcie)
    theo_min_sxm = max(compute_time_sxm, memory_time_sxm)

    print(f"\n  H100 PCIe ({h100_pcie_tflops:.0f} TFLOPS bf16, {h100_pcie_bw_tbs} TB/s):")
    print(f"    Compute-bound floor:  {compute_time_pcie:8.2f} ms")
    print(f"    Memory-bound floor:   {memory_time_pcie:8.2f} ms")
    print(f"    Theoretical minimum:  {theo_min_pcie:8.2f} ms  "
          f"({'compute' if compute_time_pcie > memory_time_pcie else 'memory'}-bound)")
    print(f"    Measured (fwd+bwd):   {171.82:8.2f} ms")
    print(f"    Efficiency:           {theo_min_pcie / 171.82 * 100:7.1f}%")

    print(f"\n  H100 SXM ({h100_sxm_tflops:.0f} TFLOPS bf16, {h100_sxm_bw_tbs} TB/s):")
    print(f"    Compute-bound floor:  {compute_time_sxm:8.2f} ms")
    print(f"    Memory-bound floor:   {memory_time_sxm:8.2f} ms")
    print(f"    Theoretical minimum:  {theo_min_sxm:8.2f} ms  "
          f"({'compute' if compute_time_sxm > memory_time_sxm else 'memory'}-bound)")

    # ─── Per-op Analysis: What's compute vs memory bound? ───
    print(f"\n{'='*70}")
    print(f"  OPERATION CLASSIFICATION (H100 SXM)")
    print(f"{'='*70}")
    print(f"  Crossover intensity: {crossover_sxm:.0f} FLOPS/byte")
    print(f"  Above = compute-bound (speed up with faster math)")
    print(f"  Below = memory-bound (speed up with fewer bytes moved)")
    print()

    compute_bound_flops = 0
    memory_bound_flops = 0
    compute_bound_bytes = 0
    memory_bound_bytes = 0

    for name, flops, bytes_moved, intensity in ops:
        if intensity > crossover_sxm:
            compute_bound_flops += flops
            compute_bound_bytes += bytes_moved
        else:
            memory_bound_flops += flops
            memory_bound_bytes += bytes_moved

    print(f"  Compute-bound ops: {compute_bound_flops/1e9:8.1f} GFLOPS "
          f"({compute_bound_flops/total_fwd_flops*100:4.1f}% of total)")
    print(f"  Memory-bound ops:  {memory_bound_flops/1e9:8.1f} GFLOPS "
          f"({memory_bound_flops/total_fwd_flops*100:4.1f}% of total)")
    print()
    print(f"  Compute-bound bytes: {compute_bound_bytes/1e9:8.3f} GB")
    print(f"  Memory-bound bytes:  {memory_bound_bytes/1e9:8.3f} GB")

    # ─── Attention Cost Breakdown ───
    print(f"\n{'='*70}")
    print(f"  ATTENTION COST ANALYSIS")
    print(f"{'='*70}")
    attn_flops_total = sum(f for n, f, b, i in ops if "attention" in n)
    matmul_flops_total = sum(f for n, f, b, i in ops
                            if any(x in n for x in ["c_q", "c_k", "c_v", "proj", "mlp", "lm_head", "bigram"]))
    other_flops = total_fwd_flops - attn_flops_total - matmul_flops_total

    print(f"  Attention (QK+softmax+V): {attn_flops_total/1e9:8.1f} GFLOPS "
          f"({attn_flops_total/total_fwd_flops*100:4.1f}%)")
    print(f"  Linear projections:       {matmul_flops_total/1e9:8.1f} GFLOPS "
          f"({matmul_flops_total/total_fwd_flops*100:4.1f}%)")
    print(f"  Other (norms, activations):{other_flops/1e9:8.1f} GFLOPS "
          f"({other_flops/total_fwd_flops*100:4.1f}%)")

    print(f"\n  Attention FLOPs scale as O(B*H*T²*Dh) — quadratic in seq_len!")
    print(f"  At T={T}, attention is {'dominant' if attn_flops_total > matmul_flops_total else 'secondary'}")
    print(f"  At T=1024, attention would be ~{attn_flops_total * (1024/T)**2 / 1e9:.1f} GFLOPS "
          f"(vs {attn_flops_total/1e9:.1f} at T={T})")


if __name__ == "__main__":
    analyze()
