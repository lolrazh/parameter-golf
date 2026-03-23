# TTT Diff Notes: PR #442 (baseline) vs PR #517 (Goldfish ML Cosine TTT)

PR #517 builds directly on PR #442's codebase. The diff between the two files
is **exactly 8 changed lines** — all in the TTT function and hyperparameters.

PR #512 (PROTEUS v7) is a completely different TTT approach (LoRA-based,
per-document, backward-looking) and is not directly comparable.

## The 3 Lines That Matter

The entire innovation is adding a cosine annealing scheduler to the existing
AdamW TTT optimizer. In PR #442, the learning rate is constant throughout all
epochs. In PR #517, it decays from `ttt_lr` to `ttt_lr * 0.01` following a
cosine schedule:

```python
# Line 1: Create scheduler (after optimizer creation)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.01
)

# Line 2: Step scheduler (after each epoch's inner loop)
scheduler.step()

# Line 3: Log current LR (for monitoring)
cur_lr = scheduler.get_last_lr()[0]
```

## Hyperparameter Differences

| Parameter    | PR #442 (baseline) | PR #517 (cosine)   | Notes                                   |
|-------------|--------------------|--------------------|------------------------------------------|
| `ttt_lr`    | 0.0005             | 0.008              | 16x higher starting LR                   |
| `ttt_epochs`| 10                 | 20 (default), 100 (submission) | 10x more epochs in final run |
| `eta_min`   | N/A (constant LR)  | ttt_lr * 0.01 = 0.00008 | Minimum LR at end of schedule   |

The submission was run with `TTT_EPOCHS=100` via environment variable override.

## Full Diff (PR #442 -> PR #517)

### Hyperparameters (lines 119-120)

```diff
-    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
-    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 10))
+    ttt_lr = float(os.environ.get("TTT_LR", 0.008))
+    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 20))
```

### ttt_adapt function (lines 1024-1075)

```diff
     ttt_params = [p for p in base_model.parameters() if p.requires_grad]
+    # PR #442: AdamW beats SGD for TTT. Cosine LR decay to prevent late-epoch overfitting.
     optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
+    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
+        optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.01
+    )

     ...  # (inner training loop is identical)

+        scheduler.step()
+
         if world_size > 1:
             ...

         elapsed = time.perf_counter() - t0
+        cur_lr = scheduler.get_last_lr()[0]
         if log_fn:
             log_fn(f"ttt_epoch:{epoch+1}/{args.ttt_epochs} "
-                   f"loss:{epoch_loss_sum.item()/max(epoch_tokens.item(),1):.4f} time:{elapsed:.1f}s")
+                   f"loss:{epoch_loss_sum.item()/max(epoch_tokens.item(),1):.4f} "
+                   f"lr:{cur_lr:.6f} time:{elapsed:.1f}s")
```

## Why It Works

With constant LR, TTT overfits to eval token positions after ~30 epochs —
sliding BPB degrades while roundtrip BPB keeps improving. The cosine decay
solves this: the model learns the content distribution during early high-LR
epochs, then the near-zero LR in late epochs prevents position memorization.

This unlocks scaling TTT to many more epochs:

| TTT Config              | Sliding BPB | Roundtrip BPB | Gap   | TTT Time |
|------------------------|-------------|---------------|-------|----------|
| 10ep constant lr (#442)| ~1.085      | ~1.100        | 0.015 | 2.5min   |
| 30ep constant lr       | 1.052       | --            | overfits | --    |
| 50ep constant lr       | 1.070       | --            | overfits | --    |
| 30ep cosine lr         | 1.018       | 1.018         | 0.000 | 7min     |
| 50ep cosine lr         | 0.993       | 0.971         | 0.022 | 12min    |
| 100ep cosine lr (#517) | 0.978       | 0.901         | 0.077 | 24min    |

## Orthogonal Finding: Per-Layer TTT LR

PR #517 also tested giving MLP output projections 3x base LR during TTT (they
have 3.4x higher quantization error). This **halves the roundtrip-sliding
overfitting gap** (0.040 vs 0.077 at matched epoch count). Orthogonal to cosine
scheduling. Not in the submitted code — documented as future work.

## PR #512 vs PR #517: Completely Different TTT Approaches

PR #512 (PROTEUS v7) uses **LoRA TTT** — a per-document, backward-looking,
chunk-level adaptation with LoRA adapters (rank 8) on Q/V projections + LM head.

| Aspect          | PR #512 (PROTEUS v7 LoRA)     | PR #517 (Goldfish full-weight) |
|----------------|-------------------------------|--------------------------------|
| What adapts    | LoRA A/B matrices (rank 8)    | All model weights              |
| Granularity    | Per-document, per-chunk       | Global (all val data at once)  |
| Optimizer      | Adam (lr=0.01)                | AdamW (lr=0.008, cosine decay) |
| Epochs         | 2                             | 100                            |
| Reset          | Per-document                  | Never (cumulative)             |
| Eval protocol  | Score-then-train per chunk    | Train all, then eval           |
| Architecture   | BatchedLinearLoRA class       | No extra parameters            |
| TTT time       | ~350s                         | ~1463s                         |
| Result (mean)  | 0.9968 BPB                    | 0.9789 BPB                    |

## TTT Hyperparameters Summary (PR #517)

```python
ttt_enabled = True       # TTT on by default
ttt_lr = 0.008           # Starting LR (16x higher than #442)
ttt_epochs = 20          # Default (overridden to 100 for submission)
ttt_momentum = 0.9       # Not used (AdamW, not SGD)
ttt_batch_seqs = 32      # Sequences per batch
ttt_freeze_blocks = 0    # No layers frozen
# Derived:
# eta_min = ttt_lr * 0.01 = 0.00008  (cosine minimum)
# T_max = ttt_epochs  (full cosine period)
# weight_decay = 0.0  (no regularization)
# grad_clip_norm = 1.0  (hardcoded)
```

## Complete ttt_adapt Function (PR #517, lines 1007-1081)

```python
def ttt_adapt(args: Hyperparameters, base_model: nn.Module, device: torch.device,
              val_tokens: Tensor, rank: int = 0, world_size: int = 1,
              log_fn=None) -> None:
    """Full-weight SGD adaptation on validation data with DDP across all GPUs."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs

    frozen_params: set[int] = set()
    if args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
                    frozen_params.add(id(p))

    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    # PR #442: AdamW beats SGD for TTT. Cosine LR decay to prevent late-epoch overfitting.
    optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.01
    )

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    base_model.train()
    t0 = time.perf_counter()

    for epoch in range(args.ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)

        for batch_start in range(my_start, my_end, batch_seqs):
            batch_end = min(batch_start + batch_seqs, my_end)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()

            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            optimizer.step()

            epoch_loss_sum += loss.detach().to(torch.float64) * y.numel()
            epoch_tokens += float(y.numel())

        scheduler.step()

        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)

        elapsed = time.perf_counter() - t0
        cur_lr = scheduler.get_last_lr()[0]
        if log_fn:
            log_fn(f"ttt_epoch:{epoch+1}/{args.ttt_epochs} "
                   f"loss:{epoch_loss_sum.item()/max(epoch_tokens.item(),1):.4f} "
                   f"lr:{cur_lr:.6f} time:{elapsed:.1f}s")

    for p in base_model.parameters():
        p.requires_grad_(True)

    if log_fn:
        log_fn(f"ttt:done elapsed={time.perf_counter()-t0:.1f}s")
```

## What to Port From #517

The cosine scheduler doesn't directly apply to #512's per-document design because
#512 only runs 2-3 epochs PER DOCUMENT (not 100 global epochs). But:

1. If we increase per-document epochs (e.g. 10-20), cosine decay becomes relevant.
2. The cosine principle could apply to the per-document Adam optimizer:
   ```python
   for epoch in range(num_epochs):
       lr_scale = 0.5 * (1 + cos(pi * epoch / num_epochs))
       for pg in opt.param_groups:
           pg['lr'] = base_lr * max(lr_scale, 0.01)
   ```
3. Or: combine both approaches -- per-document LoRA with more epochs + cosine.
4. Or: full-weight global TTT with cosine (the #517 approach) applied to our architecture.

## Experiment Matrix

These are the TTT variants to test on a frozen checkpoint:

| # | Variant                                  | What we learn                          |
|---|------------------------------------------|----------------------------------------|
| 1 | #512 as-is (LoRA r8, 2ep, const lr)     | Baseline control                       |
| 2 | #512 + cosine decay (2ep)               | Does cosine help even at 2 epochs?     |
| 3 | #512 + more epochs (10ep, const lr)     | Does more adaptation help?             |
| 4 | #512 + more epochs (10ep, cosine)       | The #517 insight applied to LoRA       |
| 5 | #517 approach (full model, 20ep cosine) | Compare LoRA vs full-model             |
| 6 | #517 approach (full model, 100ep cosine)| Full reproduction                      |
| 7 | Best of above + per-layer LR            | Per-layer LR (3x for MLP out)          |
