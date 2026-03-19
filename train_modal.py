"""
Modal script for running Parameter Golf experiments on cloud GPUs.

Usage:
    # Minimal verify run (~10 steps, ~$0.01)
    modal run train_modal.py --run-id verify --iterations 10 --val-tokens-limit 32768

    # 2-min baseline (~340 steps, ~$0.11)
    modal run train_modal.py --run-id baseline_h100 --max-wallclock 120

    # 2-min experiment with custom config
    modal run train_modal.py --run-id wide_6L --max-wallclock 120 --num-layers 6 --model-dim 640
"""

import modal

app = modal.App("parameter-golf")

# Image layers ordered for cache efficiency:
# 1. pip deps (rarely change)
# 2. data download script (rarely changes)
# 3. download data (cached unless script changes)
# 4. train_gpt.py (changes often — doesn't invalidate data cache)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch==2.6.0",
        "sentencepiece",
        "huggingface-hub",
        "setuptools",
        "typing-extensions==4.15.0",
    )
    .add_local_file("data/cached_challenge_fineweb.py", remote_path="/root/data/cached_challenge_fineweb.py", copy=True)
    .run_commands(
        "cd /root && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    )
    .add_local_file("train_gpt.py", remote_path="/root/train_gpt.py")
)


@app.function(image=image, gpu="H100", timeout=900)
def train(
    run_id: str = "test",
    max_wallclock: int = 120,
    iterations: int = 20000,
    num_layers: int = 9,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 2,
    vocab_size: int = 1024,
    train_seq_len: int = 1024,
    train_batch_tokens: int = 524288,
    logit_softcap: float = 30.0,
    matrix_lr: float = 0.04,
    scalar_lr: float = 0.04,
    embed_lr: float = 0.05,
    val_tokens_limit: int = 1000000,
    val_loss_every: int = 0,
    train_log_every: int = 50,
    rope_base: float = 10000.0,
    qk_gain_init: float = 1.5,
    warmdown_iters: int = 1200,
    grad_clip_norm: float = 0.0,
    eval_stride: int = 0,
    eval_seq_len: int = 0,
    muon_momentum_warmup_start: float = 0.85,
    muon_momentum_warmup_steps: int = 500,
    muon_momentum: float = 0.95,
):
    import os
    import subprocess

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "ITERATIONS": str(iterations),
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(num_heads),
        "NUM_KV_HEADS": str(num_kv_heads),
        "MLP_MULT": str(mlp_mult),
        "VOCAB_SIZE": str(vocab_size),
        "TRAIN_SEQ_LEN": str(train_seq_len),
        "TRAIN_BATCH_TOKENS": str(train_batch_tokens),
        "LOGIT_SOFTCAP": str(logit_softcap),
        "MATRIX_LR": str(matrix_lr),
        "SCALAR_LR": str(scalar_lr),
        "TIED_EMBED_LR": str(embed_lr),
        "VAL_TOKENS_LIMIT": str(val_tokens_limit),
        "VAL_LOSS_EVERY": str(val_loss_every),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "ROPE_BASE": str(rope_base),
        "QK_GAIN_INIT": str(qk_gain_init),
        "WARMDOWN_ITERS": str(warmdown_iters),
        "GRAD_CLIP_NORM": str(grad_clip_norm),
        "EVAL_STRIDE": str(eval_stride),
        "EVAL_SEQ_LEN": str(eval_seq_len),
        "MUON_MOMENTUM": str(muon_momentum),
        "MUON_MOMENTUM_WARMUP_START": str(muon_momentum_warmup_start),
        "MUON_MOMENTUM_WARMUP_STEPS": str(muon_momentum_warmup_steps),
        "DATA_PATH": "/root/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/root/data/tokenizers/fineweb_1024_bpe.model",
    }

    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "/root/train_gpt.py"]

    print(f"=== Parameter Golf: {run_id} ===")
    print(f"Config: {num_layers}L {model_dim}d {num_heads}h {num_kv_heads}kv {mlp_mult}x_mlp")
    print(f"Max wallclock: {max_wallclock}s, LR: matrix={matrix_lr} scalar={scalar_lr} embed={embed_lr}")
    print(f"Val tokens limit: {val_tokens_limit}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, cwd="/root")

    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        return

    # Print the log file
    log_path = f"/root/logs/{run_id}.txt"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print("\n=== RESULTS (last 30 lines) ===")
        for line in lines[-30:]:
            print(line, end="")


@app.local_entrypoint()
def main(
    run_id: str = "test",
    max_wallclock: int = 120,
    iterations: int = 20000,
    num_layers: int = 9,
    model_dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 2,
    vocab_size: int = 1024,
    train_seq_len: int = 1024,
    train_batch_tokens: int = 524288,
    logit_softcap: float = 30.0,
    matrix_lr: float = 0.04,
    scalar_lr: float = 0.04,
    embed_lr: float = 0.05,
    val_tokens_limit: int = 1000000,
    val_loss_every: int = 0,
    train_log_every: int = 50,
    rope_base: float = 10000.0,
    qk_gain_init: float = 1.5,
    warmdown_iters: int = 1200,
    grad_clip_norm: float = 0.0,
    eval_stride: int = 0,
    eval_seq_len: int = 0,
    muon_momentum_warmup_start: float = 0.85,
    muon_momentum_warmup_steps: int = 500,
    muon_momentum: float = 0.95,
):
    train.remote(
        run_id=run_id,
        max_wallclock=max_wallclock,
        iterations=iterations,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        vocab_size=vocab_size,
        train_seq_len=train_seq_len,
        train_batch_tokens=train_batch_tokens,
        logit_softcap=logit_softcap,
        matrix_lr=matrix_lr,
        scalar_lr=scalar_lr,
        embed_lr=embed_lr,
        val_tokens_limit=val_tokens_limit,
        val_loss_every=val_loss_every,
        train_log_every=train_log_every,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        warmdown_iters=warmdown_iters,
        grad_clip_norm=grad_clip_norm,
        eval_stride=eval_stride,
        eval_seq_len=eval_seq_len,
        muon_momentum=muon_momentum,
        muon_momentum_warmup_start=muon_momentum_warmup_start,
        muon_momentum_warmup_steps=muon_momentum_warmup_steps,
    )
