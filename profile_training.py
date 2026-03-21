"""
Training profiler for parameter-golf.

Runs a short training loop (default 30 steps) and produces a detailed
breakdown of where wall-clock time is spent:

  1. GPU kernel time (actual compute)
  2. CPU overhead (Python dispatch, autograd bookkeeping)
  3. Data loading time
  4. Optimizer step time (Muon Newton-Schulz, Adam, grad clip)
  5. DDP communication time (if multi-GPU)
  6. Idle/sync gaps

Usage:
  # Local quick profile (no compile warmup, small batch):
  ITERATIONS=30 WARMUP_STEPS=0 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=512 \
  VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 SKIP_SERIALIZATION=1 \
  python profile_training.py

  # GPU profile (matches real training config):
  ITERATIONS=50 WARMUP_STEPS=5 \
  python profile_training.py

  # Full NVIDIA Nsight Systems trace (produces .nsys-rep file):
  ITERATIONS=30 WARMUP_STEPS=0 TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=512 \
  nsys profile --trace=cuda,nvtx --output=training_profile \
  python profile_training.py

The script imports your model and training components from sota_train_gpt.py,
runs them with torch.profiler instrumentation, and prints a human-readable
summary showing exactly where your time goes.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time

# Force short run defaults if not set
os.environ.setdefault("ITERATIONS", "50")
os.environ.setdefault("WARMUP_STEPS", "3")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "0")
os.environ.setdefault("TRAIN_LOG_EVERY", "999999")
os.environ.setdefault("RUN_ID", "profile_run")

import torch
import torch.cuda

# We need to import everything from the training script
# But we'll run our own instrumented loop instead of main()
import sota_train_gpt as T


def fmt_us(us: float) -> str:
    """Format microseconds into human-readable string."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:.2f}ms"
    return f"{us:.1f}μs"


def fmt_pct(part: float, total: float) -> str:
    if total == 0:
        return "  --%"
    return f"{100.0 * part / total:5.1f}%"


def profile_training():
    """Run a short training loop with detailed timing instrumentation."""
    args = T.Hyperparameters()

    # --- Setup (mirrors main() but without the full ceremony) ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for profiling. On Mac, this shows what WOULD happen.")
        print("Run this on your GPU machine for real numbers.")
        return

    import torch.distributed as dist

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = T.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = T.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
    base_model = T.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, use_smeargate=args.use_smeargate,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
        smear_gate_init=args.smear_gate_init,
        num_recurrence=args.num_recurrence, lora_rank=args.lora_rank,
        xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, T.CastedLinear):
            module.float()
    T.restore_low_dim_params_to_fp32(base_model)

    use_compile = bool(int(os.environ.get("TORCH_COMPILE", "1")))
    compile_mode = os.environ.get("COMPILE_MODE", "default")
    compiled_model = torch.compile(base_model, dynamic=False, mode=compile_mode) if use_compile else base_model

    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Build optimizers (same split as main)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in T.CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in T.CONTROL_TENSOR_NAME_PATTERNS)]
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
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True,
    )
    if not T.MUON_USE_CUDA_GRAPH:
        T.zeropower_via_newtonschulz5 = torch.compile(T.zeropower_via_newtonschulz5)
        T.batched_newton_schulz = torch.compile(T.batched_newton_schulz)
    optimizer_muon = T.Muon(matrix_params, lr=args.matrix_lr,
                            momentum=args.muon_momentum,
                            backend_steps=args.muon_backend_steps,
                            weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    train_loader = T.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    n_params = sum(p.numel() for p in base_model.parameters())

    # --- Warmup (let torch.compile do its thing) ---
    print(f"\n{'='*70}")
    print(f"  PARAMETER GOLF TRAINING PROFILER")
    print(f"{'='*70}")
    print(f"  Model: {args.num_layers}L × {args.model_dim}d, {n_params:,} params")
    print(f"  Batch: {args.train_batch_tokens:,} tokens, seq_len={args.train_seq_len}")
    print(f"  Grad accum: {grad_accum_steps} micro-steps")
    print(f"  torch.compile: {'ON' if use_compile else 'OFF'}")
    print(f"  Device: {torch.cuda.get_device_name(device)}")
    print(f"{'='*70}\n")

    warmup_steps = args.warmup_steps
    if warmup_steps > 0:
        print(f"Warming up ({warmup_steps} steps, includes torch.compile JIT)...")
        model.train()
        for ws in range(warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        # Reset data loader for clean profiling
        train_loader = T.DistributedTokenLoader(args.train_files, rank, world_size, device)
        print(f"Warmup complete.\n")

    # --- Phase 1: Manual fine-grained timing ---
    print("Phase 1: Fine-grained step timing")
    print("-" * 70)

    profile_steps = min(args.iterations - warmup_steps, 30)
    times_data_load = []
    times_forward = []
    times_backward = []
    times_optimizer = []
    times_total_step = []
    times_grad_clip = []
    times_zero_grad = []

    model.train()
    for step in range(profile_steps):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        # --- Zero grad ---
        t0 = time.perf_counter()
        zero_grad_all()
        torch.cuda.synchronize()
        t_zero_grad = time.perf_counter() - t0

        t_data_total = 0.0
        t_fwd_total = 0.0
        t_bwd_total = 0.0

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            # --- Data loading ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            torch.cuda.synchronize()
            t_data_total += time.perf_counter() - t0

            # --- Forward ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            torch.cuda.synchronize()
            t_fwd_total += time.perf_counter() - t0

            # --- Backward ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (loss * grad_scale).backward()
            torch.cuda.synchronize()
            t_bwd_total += time.perf_counter() - t0

        # --- Grad clip ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        torch.cuda.synchronize()
        t_clip = time.perf_counter() - t0

        # --- Optimizer step ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for opt in optimizers:
            opt.step()
        torch.cuda.synchronize()
        t_opt = time.perf_counter() - t0

        torch.cuda.synchronize()
        t_step_total = time.perf_counter() - t_step_start

        times_data_load.append(t_data_total)
        times_forward.append(t_fwd_total)
        times_backward.append(t_bwd_total)
        times_optimizer.append(t_opt)
        times_total_step.append(t_step_total)
        times_grad_clip.append(t_clip)
        times_zero_grad.append(t_zero_grad)

        if step < 3 or (step + 1) % 10 == 0:
            print(f"  step {step+1:3d}  total={t_step_total*1000:7.2f}ms  "
                  f"data={t_data_total*1000:6.2f}ms  fwd={t_fwd_total*1000:6.2f}ms  "
                  f"bwd={t_bwd_total*1000:6.2f}ms  opt={t_opt*1000:6.2f}ms")

    # Skip first 3 steps (compile artifacts)
    skip = min(3, profile_steps - 1)
    def avg_ms(lst):
        return 1000.0 * sum(lst[skip:]) / max(len(lst) - skip, 1)
    def med_ms(lst):
        vals = sorted(lst[skip:])
        n = len(vals)
        if n == 0: return 0.0
        return 1000.0 * vals[n // 2]

    total_avg = avg_ms(times_total_step)
    data_avg = avg_ms(times_data_load)
    fwd_avg = avg_ms(times_forward)
    bwd_avg = avg_ms(times_backward)
    opt_avg = avg_ms(times_optimizer)
    clip_avg = avg_ms(times_grad_clip)
    zg_avg = avg_ms(times_zero_grad)
    other_avg = total_avg - data_avg - fwd_avg - bwd_avg - opt_avg - clip_avg - zg_avg

    print(f"\n{'='*70}")
    print(f"  PROFILING RESULTS (avg over steps {skip+1}-{profile_steps})")
    print(f"{'='*70}")
    print(f"  Total step time:    {total_avg:8.2f} ms")
    print(f"{'─'*70}")
    print(f"  Data loading:       {data_avg:8.2f} ms  {fmt_pct(data_avg, total_avg)}")
    print(f"  Forward pass:       {fwd_avg:8.2f} ms  {fmt_pct(fwd_avg, total_avg)}")
    print(f"  Backward pass:      {bwd_avg:8.2f} ms  {fmt_pct(bwd_avg, total_avg)}")
    print(f"  Grad clipping:      {clip_avg:8.2f} ms  {fmt_pct(clip_avg, total_avg)}")
    print(f"  Optimizer step:     {opt_avg:8.2f} ms  {fmt_pct(opt_avg, total_avg)}")
    print(f"  Zero grad:          {zg_avg:8.2f} ms  {fmt_pct(zg_avg, total_avg)}")
    print(f"  Other (sync/sched): {other_avg:8.2f} ms  {fmt_pct(other_avg, total_avg)}")
    print(f"{'─'*70}")

    # Compute theoretical limits
    compute_ms = fwd_avg + bwd_avg
    overhead_ms = total_avg - compute_ms
    print(f"\n  GPU compute (fwd+bwd):  {compute_ms:8.2f} ms  {fmt_pct(compute_ms, total_avg)}")
    print(f"  Everything else:        {overhead_ms:8.2f} ms  {fmt_pct(overhead_ms, total_avg)}")
    print(f"{'─'*70}")

    steps_per_10min = int(600_000 / total_avg) if total_avg > 0 else 0
    steps_per_10min_ideal = int(600_000 / compute_ms) if compute_ms > 0 else 0
    pct_gain = 100.0 * (steps_per_10min_ideal - steps_per_10min) / max(steps_per_10min, 1)

    print(f"\n  Steps in 10 min (current):   {steps_per_10min:,}")
    print(f"  Steps in 10 min (0 overhead): {steps_per_10min_ideal:,}")
    print(f"  Potential gain from eliminating overhead: {pct_gain:.1f}% more steps")
    print(f"  Tokens in 10 min (current):   {steps_per_10min * args.train_batch_tokens:,}")
    print(f"  Tokens in 10 min (ideal):     {steps_per_10min_ideal * args.train_batch_tokens:,}")

    print(f"\n{'='*70}")
    print(f"  OPTIMIZER BREAKDOWN")
    print(f"{'='*70}")
    print(f"  Muon (Newton-Schulz + momentum + WD + allreduce):")
    print(f"    This is likely the biggest chunk of optimizer time.")
    print(f"    Newton-Schulz runs {args.muon_backend_steps} iterations of")
    print(f"    matrix orthogonalization per param group.")
    print(f"  Adam (token embeddings + scalars): fused=True")
    print(f"  Total optimizer time: {opt_avg:.2f}ms ({fmt_pct(opt_avg, total_avg)} of step)")

    # --- Phase 2: torch.profiler trace (if requested) ---
    if os.environ.get("TORCH_PROFILE", "0") == "1":
        print(f"\n{'='*70}")
        print(f"  Phase 2: torch.profiler kernel-level trace")
        print(f"{'='*70}")
        print(f"  Writing trace to ./profile_trace.json (view at chrome://tracing)")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(10):
                zero_grad_all()
                for micro_step in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y)
                    (loss * grad_scale).backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
                prof.step()

        # Print kernel-level summary
        print("\nTop 20 CUDA kernels by total time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\nTop 20 CPU operations by total time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        # Export chrome trace
        prof.export_chrome_trace("./profile_trace.json")
        print("\nChrome trace written to ./profile_trace.json")
        print("Open chrome://tracing and load the file to visualize.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    profile_training()
