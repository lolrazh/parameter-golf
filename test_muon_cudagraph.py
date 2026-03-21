"""
Test + benchmark: Batched Muon with CUDA Graph capture.

Stacks two optimizations:
1. Batched Newton-Schulz (55 calls → 5 batched calls)
2. CUDA Graph replay (eliminates remaining Python dispatch + launch overhead)

Usage:
    # On GPU machine:
    python test_muon_cudagraph.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch
from torch import Tensor

from test_muon_batched import (
    OriginalMuon,
    BatchedMuon,
    batched_newton_schulz,
    zeropower_via_newtonschulz5,
    make_test_params,
)


class CUDAGraphBatchedMuon(torch.optim.Optimizer):
    """Batched Muon with CUDA Graph capture for near-zero Python overhead.

    How it works:
    1. Steps 1-2: Normal batched execution (warmup, allocate all buffers)
    2. Step 3: Capture the ENTIRE optimizer step as a CUDA graph
    3. Steps 4+: Copy gradients into static buffers, replay the graph

    Key trick: LR, momentum, and WD are stored as GPU scalar tensors.
    The graph reads from their memory addresses, so updating the tensors
    before replay changes the effective values without re-capture.
    """

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._step_count = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        self._shape_groups_built = False

        # Static scalar tensors for dynamic values (updated before replay)
        self._lr_tensor: Tensor | None = None
        self._wd_factor_tensor: Tensor | None = None
        self._momentum_tensor: Tensor | None = None

        # Static buffers (pre-allocated during warmup)
        self._static_grad_batches: dict[tuple[int, ...], Tensor] = {}
        self._static_ns_outputs: dict[tuple[int, ...], Tensor] = {}
        self._static_updates_flat: Tensor | None = None
        self._shape_to_indices: dict[tuple[int, ...], list[int]] = {}

    def _build_shape_groups(self, params: list) -> None:
        self._shape_to_indices = defaultdict(list)
        for i, p in enumerate(params):
            self._shape_to_indices[tuple(p.shape)].append(i)
        self._shape_groups_built = True

    def _warmup_step(self, params, group):
        """Normal batched step (used for warmup before graph capture)."""
        lr = group["lr"]
        momentum = group["momentum"]
        backend_steps = group["backend_steps"]
        nesterov = group["nesterov"]
        wd = group["weight_decay"]

        if not self._shape_groups_built:
            self._build_shape_groups(params)

        # Ensure all momentum buffers + grads exist
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.data)
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)

        # Momentum + nesterov
        grads: list[Tensor] = []
        for p in params:
            g = p.grad
            buf = self.state[p]["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if nesterov:
                g = g.add(buf, alpha=momentum)
            grads.append(g)

        # Batched Newton-Schulz
        ns_results: list[Tensor | None] = [None] * len(params)
        for shape, indices in self._shape_to_indices.items():
            active_grads = [grads[i] for i in indices]
            if len(active_grads) == 1:
                ns_results[indices[0]] = zeropower_via_newtonschulz5(
                    active_grads[0], steps=backend_steps)
            else:
                batch = torch.stack(active_grads)
                batch_result = batched_newton_schulz(batch, steps=backend_steps)
                for j, idx in enumerate(indices):
                    ns_results[idx] = batch_result[j]

        # Pack, apply WD + update
        total_numel = sum(int(p.numel()) for p in params)
        updates_flat = torch.zeros(total_numel, device=params[0].device, dtype=torch.bfloat16)
        curr = 0
        for i, p in enumerate(params):
            if ns_results[i] is not None:
                g = ns_results[i] * max(1, ns_results[i].size(0) / ns_results[i].size(1)) ** 0.5
                updates_flat[curr:curr + p.numel()] = g.reshape(-1)
            curr += p.numel()

        curr = 0
        for p in params:
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)
            g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
            p.add_(g, alpha=-lr)
            curr += p.numel()

    def _pre_allocate(self, params, group):
        """Pre-allocate all static buffers for CUDA graph capture."""
        device = params[0].device

        # Scalar tensors for dynamic values
        self._lr_tensor = torch.tensor(group["lr"], device=device, dtype=torch.float32)
        self._wd_factor_tensor = torch.tensor(
            1.0 - group["lr"] * group["weight_decay"], device=device, dtype=torch.float32)
        self._momentum_tensor = torch.tensor(
            group["momentum"], device=device, dtype=torch.float32)

        # Batch buffers per shape group
        for shape, indices in self._shape_to_indices.items():
            B = len(indices)
            self._static_grad_batches[shape] = torch.empty(
                B, *shape, device=device, dtype=torch.bfloat16)
            self._static_ns_outputs[shape] = torch.empty(
                B, *shape, device=device, dtype=torch.bfloat16)

        # Flat update buffer
        total_numel = sum(int(p.numel()) for p in params)
        self._static_updates_flat = torch.zeros(
            total_numel, device=device, dtype=torch.bfloat16)

        # Pre-compute scale corrections per shape (constant)
        self._scale_corrections: dict[tuple[int, ...], float] = {}
        for shape in self._shape_to_indices:
            rows, cols = shape
            self._scale_corrections[shape] = max(1, rows / cols) ** 0.5

    def _graph_step_impl(self, params, group):
        """The step function executed during CUDA graph capture.
        Uses ONLY pre-allocated buffers — no new allocations allowed."""
        nesterov = group["nesterov"]
        backend_steps = group["backend_steps"]

        # Phase 1: Momentum + nesterov (in-place on existing grad buffers)
        # NOTE: Can't use alpha=float(tensor) during graph capture (GPU→CPU sync).
        # Instead, multiply explicitly: grad += momentum * buf
        for p in params:
            buf = self.state[p]["momentum_buffer"]
            buf.mul_(self._momentum_tensor).add_(p.grad)
            if nesterov:
                p.grad.add_(buf * self._momentum_tensor)

        # Phase 2: Batched Newton-Schulz
        for shape, indices in self._shape_to_indices.items():
            batch_buf = self._static_grad_batches[shape]
            # Copy grads into batch buffer (no allocation, just copy)
            for j, idx in enumerate(indices):
                batch_buf[j].copy_(params[idx].grad)

            # In-place Newton-Schulz
            if len(indices) == 1:
                result = zeropower_via_newtonschulz5(batch_buf[0], steps=backend_steps)
                self._static_ns_outputs[shape][0].copy_(result)
            else:
                result = batched_newton_schulz(batch_buf, steps=backend_steps)
                self._static_ns_outputs[shape].copy_(result)

        # Phase 3: Scale + pack into flat buffer
        self._static_updates_flat.zero_()
        curr = 0
        for i, p in enumerate(params):
            shape = tuple(p.shape)
            idx_in_group = self._shape_to_indices[shape].index(i)
            g = self._static_ns_outputs[shape][idx_in_group]
            g = g * self._scale_corrections[shape]
            self._static_updates_flat[curr:curr + p.numel()] = g.reshape(-1)
            curr += p.numel()

        # Phase 4: Weight decay + apply
        curr = 0
        for p in params:
            p.data.mul_(self._wd_factor_tensor)
            g = self._static_updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
            p.data.add_(g * (-self._lr_tensor))
            curr += p.numel()

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue

            if self._step_count <= 2:
                # Warmup: normal batched step
                self._warmup_step(params, group)
                if self._step_count == 2:
                    self._pre_allocate(params, group)
                    # Ensure grads exist (set_to_none=False from here on)
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p.data)
                    # Capture CUDA graph
                    self._graph = torch.cuda.CUDAGraph()
                    # Side stream for capture
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        with torch.cuda.graph(self._graph, stream=s):
                            self._graph_step_impl(params, group)
                    torch.cuda.current_stream().wait_stream(s)
                return

            # Update dynamic scalar tensors before replay
            self._lr_tensor.fill_(group["lr"])
            self._wd_factor_tensor.fill_(1.0 - group["lr"] * group["weight_decay"])
            self._momentum_tensor.fill_(group["momentum"])

            # Replay the captured graph
            self._graph.replay()

    def set_lr(self, lr: float, wd: float = 0.0):
        """Update LR and WD for next replay (called by training loop)."""
        for group in self.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd


def bench_all():
    """Benchmark Original vs Batched vs CUDAGraph Muon on GPU."""
    if not torch.cuda.is_available():
        print("No CUDA")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    results = {}

    for label, OptClass in [("Original", OriginalMuon),
                             ("Batched", BatchedMuon),
                             ("CUDA Graph", CUDAGraphBatchedMuon)]:
        torch.manual_seed(42)
        params, _ = make_test_params()
        params = [torch.nn.Parameter(p.float().to(device)) for p in params]
        opt = OptClass(params, lr=0.04, momentum=0.95,
                       backend_steps=5, weight_decay=0.02)

        # Warmup (build groups, allocate buffers, capture graph)
        for warmup in range(5):
            for p in params:
                if p.grad is None:
                    p.grad = torch.randn_like(p)
                else:
                    p.grad.copy_(torch.randn_like(p))
            opt.step()

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for trial in range(30):
            for p in params:
                if p.grad is None:
                    p.grad = torch.randn_like(p)
                else:
                    p.grad.copy_(torch.randn_like(p))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times) * 1000
        med = sorted(times)[len(times) // 2] * 1000
        mn = min(times) * 1000
        results[label] = avg
        print(f"  {label:>12s}:  avg={avg:7.2f}ms  "
              f"med={med:7.2f}ms  min={mn:7.2f}ms")

    print()
    if "Original" in results and "CUDA Graph" in results:
        speedup = results["Original"] / results["CUDA Graph"]
        saved = results["Original"] - results["CUDA Graph"]
        print(f"  Total speedup (Original → CUDA Graph): {speedup:.1f}x")
        print(f"  Time saved per step: {saved:.1f}ms")
        print(f"  At 200ms/step: {saved / 200 * 100:.1f}% total improvement")


def test_correctness():
    """Verify CUDA Graph Muon matches Original across multiple steps."""
    if not torch.cuda.is_available():
        print("No CUDA")
        return False

    device = torch.device("cuda")
    print("=" * 60)
    print("  Correctness: CUDAGraphBatchedMuon vs OriginalMuon")
    print("=" * 60)

    torch.manual_seed(42)
    params_orig, names = make_test_params()
    params_graph = [p.clone() for p in params_orig]

    params_orig = [torch.nn.Parameter(p.float().to(device)) for p in params_orig]
    params_graph = [torch.nn.Parameter(p.float().to(device)) for p in params_graph]

    opt_orig = OriginalMuon(params_orig, lr=0.04, momentum=0.95,
                            backend_steps=5, weight_decay=0.02)
    opt_graph = CUDAGraphBatchedMuon(params_graph, lr=0.04, momentum=0.95,
                                      backend_steps=5, weight_decay=0.02)

    all_passed = True
    for step in range(8):  # extra steps to test past graph capture
        torch.manual_seed(200 + step)
        for p in params_orig:
            p.grad = torch.randn_like(p)
        torch.manual_seed(200 + step)
        for p in params_graph:
            if p.grad is None:
                p.grad = torch.randn_like(p)
            else:
                p.grad.copy_(torch.randn(p.shape, device=device, dtype=p.dtype,
                                          generator=torch.Generator(device).manual_seed(200 + step)))

        # Use same random grads
        torch.manual_seed(200 + step)
        grads = [torch.randn_like(p) for p in params_orig]
        for i, p in enumerate(params_orig):
            p.grad = grads[i].clone()
        for i, p in enumerate(params_graph):
            if p.grad is None:
                p.grad = grads[i].clone()
            else:
                p.grad.copy_(grads[i])

        opt_orig.step()
        opt_graph.step()

        max_diff = 0.0
        for i, (po, pg) in enumerate(zip(params_orig, params_graph)):
            diff = (po.data - pg.data).abs().max().item()
            max_diff = max(max_diff, diff)

        status = "OK" if max_diff < 0.02 else "FAIL"
        if max_diff >= 0.02:
            all_passed = False
        phase = "warmup" if step < 2 else ("capture" if step == 2 else "replay")
        print(f"  Step {step+1} ({phase:>7s}): max_diff = {max_diff:.6f}  {status}")

    print(f"\n  {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    passed = test_correctness()
    print()
    bench_all()
    print()
    print("=" * 60)
    print(f"  {'ALL GOOD' if passed else 'CORRECTNESS ISSUES — check above'}")
    print("=" * 60)
