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
    .run_commands(
        # Download data at image build time so it's cached across runs
        "pip install huggingface-hub",
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('willdepueoai/parameter-golf', repo_type='dataset', "
        "local_dir='/data/parameter-golf', allow_patterns=['datasets/fineweb10B_sp1024/*', 'tokenizers/*', 'manifest.json'])"
        "\"",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=900,  # 15 min max (training + eval + buffer)
    mounts=[modal.Mount.from_local_file("train_gpt.py", remote_path="/root/train_gpt.py"),
            modal.Mount.from_local_file("data/cached_challenge_fineweb.py", remote_path="/root/data/cached_challenge_fineweb.py")],
)
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
):
    import os
    import subprocess

    # Set up data paths — use the cached HF download
    data_path = "/data/parameter-golf/datasets/fineweb10B_sp1024"
    tokenizer_path = "/data/parameter-golf/tokenizers/fineweb_1024_bpe.model"

    # Symlink data into expected locations if needed
    os.makedirs("/root/data/datasets", exist_ok=True)
    os.makedirs("/root/data/tokenizers", exist_ok=True)
    if not os.path.exists(f"/root/{data_path.lstrip('/')}"):
        os.symlink(data_path, "/root/data/datasets/fineweb10B_sp1024")
    tok_dest = "/root/data/tokenizers/fineweb_1024_bpe.model"
    if not os.path.exists(tok_dest):
        os.symlink(tokenizer_path, tok_dest)

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
        "DATA_PATH": "/root/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/root/data/tokenizers/fineweb_1024_bpe.model",
    }

    # Run with torchrun for single GPU (handles DDP init cleanly)
    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "/root/train_gpt.py"]

    print(f"=== Parameter Golf: {run_id} ===")
    print(f"Config: {num_layers}L {model_dim}d {num_heads}h {num_kv_heads}kv {mlp_mult}x_mlp")
    print(f"Max wallclock: {max_wallclock}s, LR: matrix={matrix_lr} scalar={scalar_lr} embed={embed_lr}")
    print(f"Val tokens limit: {val_tokens_limit}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, capture_output=False, text=True, cwd="/root")

    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        return

    # Print the log file for easy reading
    log_path = f"/root/logs/{run_id}.txt"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        # Print last 30 lines (contains val results)
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
    )
