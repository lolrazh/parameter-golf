"""
Test + benchmark: Batched Muon optimizer vs original Muon.

Runs entirely on CPU (no GPU needed). Verifies:
1. Numerical correctness: batched output matches original to bf16 precision
2. Python overhead: counts kernel launches / function calls
3. Wall-clock comparison (CPU-only, not representative of GPU speed)

Usage:
    python test_muon_batched.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch import Tensor

# ─────────────────────────────────────────────────
# ORIGINAL Muon (copied from sota_train_gpt.py)
# ─────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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


class OriginalMuon(torch.optim.Optimizer):
    """Exact copy of the current Muon optimizer from sota_train_gpt.py."""

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
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
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            curr = 0
            for p in params:
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ─────────────────────────────────────────────────
# BATCHED Muon (the optimization)
# ─────────────────────────────────────────────────

def batched_newton_schulz(G_batch: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalization on a batch of matrices simultaneously.

    G_batch: [B, rows, cols] — all matrices must have the same shape.
    Returns: [B, rows, cols] — orthogonalized matrices.

    Key difference from the original: instead of calling newton_schulz
    once per matrix (B separate calls), we process all B matrices in
    one shot using torch.bmm (batched matrix multiply).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G_batch.bfloat16()

    # Per-matrix Frobenius norm (not across the batch!)
    # Original does: X /= X.norm() + eps (Frobenius norm of the single matrix)
    # Batched: compute norm per matrix, divide each independently
    norms = X.flatten(1).norm(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    X = X / (norms + eps)

    # Transpose if tall (rows > cols) — all matrices in batch have same shape
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)

    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))       # [B, d, d]
        B = b * A + c * torch.bmm(A, A)            # [B, d, d]
        X = a * X + torch.bmm(B, X)                # [B, d, cols]

    if transposed:
        X = X.transpose(1, 2)
    return X


class BatchedMuon(torch.optim.Optimizer):
    """Muon optimizer with batched Newton-Schulz across same-shaped matrices.

    Instead of looping over each parameter and calling newton_schulz
    individually (N separate calls), we:
    1. Group parameters by shape
    2. Stack each group into a [B, rows, cols] batch
    3. Run ONE batched newton_schulz per group
    4. Unstack back to individual updates

    This reduces Python loop iterations from ~55 to ~5 shape groups.
    """

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._shape_groups_built = False

    def _build_shape_groups(self, params: list) -> None:
        """Group parameter indices by shape (done once, cached)."""
        self._shape_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, p in enumerate(params):
            self._shape_to_indices[tuple(p.shape)].append(i)
        self._shape_groups_built = True

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            if not self._shape_groups_built:
                self._build_shape_groups(params)

            # Ensure all momentum buffers exist
            for p in params:
                if p.grad is not None:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)

            # ── Phase 1: momentum + nesterov (still per-param, but cheap) ──
            grads_after_momentum: list[Tensor | None] = [None] * len(params)
            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p]["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                grads_after_momentum[i] = g

            # ── Phase 2: batched Newton-Schulz per shape group ──
            ns_results: list[Tensor | None] = [None] * len(params)
            for shape, indices in self._shape_to_indices.items():
                # Collect grads that exist for this shape group
                active = [(idx, grads_after_momentum[idx]) for idx in indices
                          if grads_after_momentum[idx] is not None]
                if not active:
                    continue

                active_indices, active_grads = zip(*active)

                if len(active_grads) == 1:
                    # Single matrix — use original (no batching overhead)
                    result = zeropower_via_newtonschulz5(active_grads[0], steps=backend_steps)
                    ns_results[active_indices[0]] = result
                else:
                    # Stack into batch and process all at once
                    batch = torch.stack(list(active_grads))  # [B, rows, cols]
                    batch_result = batched_newton_schulz(batch, steps=backend_steps)
                    for j, idx in enumerate(active_indices):
                        ns_results[idx] = batch_result[j]

            # ── Phase 3: scale correction + pack into flat buffer ──
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if ns_results[i] is not None:
                    g = ns_results[i]
                    g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            # ── Phase 4: weight decay + apply updates ──
            curr = 0
            for p in params:
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ─────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────

def make_test_params(num_layers: int = 9, dim: int = 512, num_kv_heads: int = 4,
                     head_dim: int = 64, mlp_mult: int = 3):
    """Create parameter tensors matching the real model's Muon param shapes."""
    params = []
    names = []
    kv_dim = num_kv_heads * head_dim
    hidden = mlp_mult * dim

    for layer in range(num_layers):
        # Attention: c_q, c_k, c_v, proj
        params.append(torch.randn(dim, dim))    # c_q.weight
        names.append(f"block.{layer}.attn.c_q")
        params.append(torch.randn(kv_dim, dim)) # c_k.weight
        names.append(f"block.{layer}.attn.c_k")
        params.append(torch.randn(kv_dim, dim)) # c_v.weight
        names.append(f"block.{layer}.attn.c_v")
        params.append(torch.randn(dim, dim))    # proj.weight
        names.append(f"block.{layer}.attn.proj")
        # MLP: fc, proj
        params.append(torch.randn(hidden, dim)) # mlp.fc.weight
        names.append(f"block.{layer}.mlp.fc")
        params.append(torch.randn(dim, hidden)) # mlp.proj.weight
        names.append(f"block.{layer}.mlp.proj")

    # bigram_hash_proj
    params.append(torch.randn(dim, 128))
    names.append("bigram_hash_proj")

    return params, names


def test_correctness():
    """Verify batched Muon produces identical results to original Muon."""
    print("=" * 70)
    print("  TEST: Numerical correctness")
    print("=" * 70)

    torch.manual_seed(42)
    params_orig, names = make_test_params()
    # Deep copy for the batched version (same starting weights)
    params_batched = [p.clone() for p in params_orig]

    # Make them all nn.Parameter and require_grad
    params_orig = [torch.nn.Parameter(p.float()) for p in params_orig]
    params_batched = [torch.nn.Parameter(p.float()) for p in params_batched]

    lr = 0.04
    momentum = 0.95
    backend_steps = 5
    wd = 0.02

    opt_orig = OriginalMuon(params_orig, lr=lr, momentum=momentum,
                            backend_steps=backend_steps, weight_decay=wd)
    opt_batched = BatchedMuon(params_batched, lr=lr, momentum=momentum,
                              backend_steps=backend_steps, weight_decay=wd)

    num_steps = 5
    max_diff_all = 0.0
    all_passed = True

    for step in range(num_steps):
        # Generate identical random gradients for both
        torch.manual_seed(100 + step)
        for p in params_orig:
            p.grad = torch.randn_like(p)
        torch.manual_seed(100 + step)
        for p in params_batched:
            p.grad = torch.randn_like(p)

        opt_orig.step()
        opt_batched.step()

        # Compare all parameters
        step_max_diff = 0.0
        for i, (po, pb) in enumerate(zip(params_orig, params_batched)):
            diff = (po.data - pb.data).abs().max().item()
            step_max_diff = max(step_max_diff, diff)
            if diff > 1e-2:  # bf16 has ~1e-2 precision at these magnitudes
                print(f"  FAIL step {step} param {i} ({names[i]}): "
                      f"max_diff={diff:.6f}")
                all_passed = False

        max_diff_all = max(max_diff_all, step_max_diff)
        print(f"  Step {step+1}: max_diff = {step_max_diff:.8f}"
              f"  {'OK' if step_max_diff < 1e-2 else 'FAIL'}")

    print(f"\n  Overall max diff across {num_steps} steps: {max_diff_all:.8f}")

    # bf16 has 8 mantissa bits → precision ~1/256 ≈ 0.004 at magnitude ~1.0
    # Newton-Schulz in bf16 accumulates some error, so we allow up to 0.01
    threshold = 0.01
    if max_diff_all < threshold:
        print(f"  PASSED (threshold: {threshold})")
    else:
        print(f"  FAILED (threshold: {threshold})")
        all_passed = False

    return all_passed


def test_shape_grouping():
    """Verify shape groups are built correctly."""
    print("\n" + "=" * 70)
    print("  TEST: Shape grouping")
    print("=" * 70)

    params, names = make_test_params()
    params = [torch.nn.Parameter(p.float()) for p in params]

    opt = BatchedMuon(params, lr=0.04, momentum=0.95, backend_steps=5)
    opt._build_shape_groups(params)

    print(f"  Total parameters: {len(params)}")
    print(f"  Shape groups: {len(opt._shape_to_indices)}")
    print()

    for shape, indices in sorted(opt._shape_to_indices.items(), key=lambda x: -len(x[1])):
        example_names = [names[i] for i in indices[:3]]
        more = f" + {len(indices)-3} more" if len(indices) > 3 else ""
        print(f"  {str(shape):>20s}  ×{len(indices):2d}  "
              f"({', '.join(example_names)}{more})")

    # Original: 55 Python loop iterations with individual NS calls
    # Batched: 5 groups with batched NS calls
    print(f"\n  Original: {len(params)} individual Newton-Schulz calls")
    print(f"  Batched:  {len(opt._shape_to_indices)} batched Newton-Schulz calls")
    print(f"  Reduction: {len(params)}→{len(opt._shape_to_indices)} "
          f"({len(params)/len(opt._shape_to_indices):.1f}x fewer Python iterations)")


def benchmark_overhead():
    """Compare wall-clock time of both optimizers (CPU, measures Python overhead)."""
    print("\n" + "=" * 70)
    print("  BENCHMARK: Python overhead comparison (CPU)")
    print("=" * 70)
    print("  NOTE: This measures Python-level overhead, not GPU speed.")
    print("  The actual GPU speedup will be larger because batched bmm")
    print("  keeps the GPU busier (less idle time between kernel launches).\n")

    torch.manual_seed(42)
    num_trials = 3  # fewer trials since CPU Newton-Schulz is slow

    for label, OptClass in [("Original", OriginalMuon), ("Batched", BatchedMuon)]:
        params, _ = make_test_params()
        params = [torch.nn.Parameter(p.float()) for p in params]
        opt = OptClass(params, lr=0.04, momentum=0.95, backend_steps=5, weight_decay=0.02)

        # Warm up (build shape groups, allocate momentum buffers)
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()

        times = []
        for trial in range(num_trials):
            for p in params:
                p.grad = torch.randn_like(p)
            t0 = time.perf_counter()
            opt.step()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times)
        print(f"  {label:>10s}: {avg*1000:8.2f} ms/step "
              f"(min={min(times)*1000:.2f}, max={max(times)*1000:.2f})")

    print("\n  On GPU, the batched version will show larger gains because:")
    print("  - torch.bmm on [9, 512, 512] fully utilizes GPU SMs")
    print("  - No idle gaps between per-matrix Newton-Schulz calls")
    print("  - Fewer kernel launches = less CUDA launch overhead")


if __name__ == "__main__":
    passed = test_correctness()
    test_shape_grouping()
    benchmark_overhead()

    print("\n" + "=" * 70)
    if passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — check output above")
    print("=" * 70)
