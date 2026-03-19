# Experiment Ledger

Local MLX experiments on M4 Air. Comparing val_loss at 200 steps (SEED=1337, SEQ_LEN=512, BATCH=4096, GRAD_ACCUM=1).

**Rules**: Change ONE thing per experiment. Keep seed fixed. Differences < 0.1 val_loss = noise.

| # | Run ID | Change from baseline | val_loss | delta | verdict | notes |
|---|--------|---------------------|----------|-------|---------|-------|
| 0 | baseline_v1 | (none — default config) | 4.3525 | — | baseline | 9L 512d 8h 4kv 2x_mlp, 17M params, ~102s |
