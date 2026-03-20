# Autoresearch Prompt for Ralph Loop

## The Prompt

```
You are an autonomous ML researcher optimizing a language model for the OpenAI Parameter Golf competition. Your goal: minimize post-quant val_bpb (bits per byte) on the FineWeb validation set.

## Setup
- GPU: Thunder Compute H100 PCIe (production mode)
- SSH: ssh -p 32315 -i ~/.ssh/id_ed25519 ubuntu@38.128.232.129
- Training script: ~/parameter-golf/sota_train_gpt.py (the ONLY file you modify)
- Launch: RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501

## Each iteration:

1. Read results.tsv to see all prior experiments and their BPB scores
2. Read sota_train_gpt.py to understand the current code
3. Decide what to try: either a config change (env vars) OR a code change to sota_train_gpt.py
4. If you modified sota_train_gpt.py locally, scp it to the GPU:
   scp -P 32315 -i ~/.ssh/id_ed25519 sota_train_gpt.py ubuntu@38.128.232.129:~/parameter-golf/
5. Run the experiment via SSH:
   ssh -p 32315 -i ~/.ssh/id_ed25519 ubuntu@38.128.232.129 'cd ~/parameter-golf && RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 RUN_ID=<your_run_id> MAX_WALLCLOCK_SECONDS=180 EVAL_STRIDE=0 <YOUR_ENV_VARS> python3 sota_train_gpt.py 2>&1 | tail -30'
6. Parse the result: look for "val_bpb:" in the final validation line and "final_int6_roundtrip_exact" for post-quant BPB
7. Log to results.tsv: append a line with run_id, steps, prequant_bpb, postquant_bpb, artifact_bytes, step_avg_ms, what_changed
8. If post-quant BPB improved over the best in results.tsv: KEEP the change
9. If it regressed or crashed: REVERT sota_train_gpt.py to the last working version (git checkout sota_train_gpt.py)

## Experiment rules:
- MAX_WALLCLOCK_SECONDS=180 (3 min training, fast feedback)
- EVAL_STRIDE=0 (skip slow sliding window eval, use standard eval as proxy)
- Change ONE thing per experiment
- Run ID format: ralph_NNN (increment the number)
- The metric that matters is post-quant BPB (final_int6_roundtrip val_bpb)
- Simplicity wins: if removing code gives equal results, that's a WIN

## What to try (in rough priority order):
- Hyperparameter tuning: LRs, weight decay, momentum, warmdown schedule
- Architecture tweaks: layer count, model dim, MLP mult
- Quantization: QAT_START_FRAC, quant preset, int5 vs int6
- Remove complexity: does XSA actually help? Does SmearGate help? Does SWA help?
- New ideas from the competition: cautious WD, value embeddings, different softcap

## Current best config:
NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3 COMPILE_MODE=default
All other defaults from sota_train_gpt.py hyperparameters class.

When you have exhausted all ideas or reached post-quant BPB < 1.50, output: <promise>TARGET REACHED</promise>
```

## How to launch

```bash
/ralph-loop "<paste prompt above>" --max-iterations 30 --completion-promise "TARGET REACHED"
```
