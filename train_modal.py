"""
Modal script for Parameter Golf. Dead simple.

Usage:
    modal run train_modal.py

Change hyperparams by editing sota_train_gpt.py directly.
"""

import modal

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.6.0", "sentencepiece", "huggingface-hub", "setuptools", "typing-extensions==4.15.0", "zstandard")
    .add_local_file("data/cached_challenge_fineweb.py", remote_path="/root/data/cached_challenge_fineweb.py", copy=True)
    .run_commands("cd /root && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10")
    .add_local_file("sota_train_gpt.py", remote_path="/root/train_gpt.py")
)


@app.function(image=image, gpu="H100", timeout=900)
def train(run_id: str = "run", max_wallclock: int = 120, train_log_every: int = 100):
    import os
    import subprocess

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "DATA_PATH": "/root/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/root/data/tokenizers/fineweb_1024_bpe.model",
    }

    subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "/root/train_gpt.py"], env=env, cwd="/root")

    log_path = f"/root/logs/{run_id}.txt"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print("\n=== RESULTS ===")
        for line in lines[-20:]:
            print(line, end="")


@app.local_entrypoint()
def main(run_id: str = "run", max_wallclock: int = 120, train_log_every: int = 100):
    train.remote(run_id=run_id, max_wallclock=max_wallclock, train_log_every=train_log_every)
