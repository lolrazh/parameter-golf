"""Quick GPU benchmark: Original vs Batched Muon optimizer step time."""

from __future__ import annotations

import time
import torch
from test_muon_batched import OriginalMuon, BatchedMuon, make_test_params


def bench_gpu():
    if not torch.cuda.is_available():
        print("No CUDA — run on GPU machine")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    for label, OptClass in [("Original Muon", OriginalMuon),
                             ("Batched Muon", BatchedMuon)]:
        torch.manual_seed(42)
        params, _ = make_test_params()
        params = [torch.nn.Parameter(p.float().to(device)) for p in params]
        opt = OptClass(params, lr=0.04, momentum=0.95,
                       backend_steps=5, weight_decay=0.02)

        # Warmup (allocate buffers, JIT, etc)
        for _ in range(3):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for trial in range(20):
            for p in params:
                p.grad = torch.randn_like(p)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times) * 1000
        med = sorted(times)[len(times)//2] * 1000
        mn = min(times) * 1000
        print(f"  {label:>15s}:  avg={avg:7.2f}ms  "
              f"med={med:7.2f}ms  min={mn:7.2f}ms")

    print()
    print("  If batched is faster, integrate into sota_train_gpt.py")
    print("  Expected: 2-4x speedup on optimizer step (29ms → ~8-12ms)")


if __name__ == "__main__":
    bench_gpu()
