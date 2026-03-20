"""
Modal runner for Parameter Golf experiments.

Examples:
    modal run train_modal.py --run-id sota_9x512_2m --max-wallclock 120 \
        --overrides 'NUM_LAYERS=9,MODEL_DIM=512,EVAL_STRIDE=0'

    modal run train_modal.py --run-id sota_6x640_2m --max-wallclock 120 \
        --overrides 'NUM_LAYERS=6,MODEL_DIM=640,EVAL_STRIDE=0'

    modal run train_modal.py --run-id clean_9x512_5m --max-wallclock 300 \
        --overrides 'NUM_LAYERS=9,MODEL_DIM=512,EVAL_STRIDE=0' --save-checkpoint

Short-run metric policy:
    - `metric_mode=auto` picks the final pre-quant validation BPB for short runs
      (`max_wallclock <= 180`) because the int6 roundtrip can be misleading early.
    - For longer runs, `metric_mode=auto` picks post-quant BPB unless sliding-window
      eval is enabled, in which case it picks the final sliding-window BPB.
"""

from __future__ import annotations

import re

import modal

app = modal.App("parameter-golf")

DEFAULT_DEV_OVERRIDES = {
    "EVAL_STRIDE": "0",
}

METRIC_MODES = {"auto", "prequant", "postquant", "sliding"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch==2.6.0",
        "sentencepiece",
        "huggingface-hub",
        "setuptools",
        "typing-extensions==4.15.0",
        "zstandard",
    )
    .add_local_file(
        "data/cached_challenge_fineweb.py",
        remote_path="/root/data/cached_challenge_fineweb.py",
        copy=True,
    )
    .run_commands("cd /root && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10")
    .add_local_file("quant_sweep.py", remote_path="/root/quant_sweep.py")
    .add_local_file("sota_train_gpt.py", remote_path="/root/sota_train_gpt.py")
    .add_local_file("sota_train_gpt.py", remote_path="/root/train_gpt.py")
)


def parse_overrides(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    text = raw.strip()
    if not text:
        return overrides
    for item in text.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"override must look like KEY=VALUE, got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"invalid override key: {key!r}")
        if not value:
            raise ValueError(f"override {key!r} is missing a value")
        overrides[key] = value
    return overrides


def merge_overrides(
    num_layers: int,
    model_dim: int,
    eval_stride: int,
    overrides: str,
) -> dict[str, str]:
    convenience: dict[str, str] = {}
    if num_layers > 0:
        convenience["NUM_LAYERS"] = str(num_layers)
    if model_dim > 0:
        convenience["MODEL_DIM"] = str(model_dim)
    if eval_stride >= 0:
        convenience["EVAL_STRIDE"] = str(eval_stride)

    parsed = parse_overrides(overrides)
    overlap = set(convenience) & set(parsed)
    if overlap:
        dupes = ", ".join(sorted(overlap))
        raise ValueError(f"duplicate overrides supplied via flags and --overrides: {dupes}")

    return {**DEFAULT_DEV_OVERRIDES, **convenience, **parsed}


def last_matching_line(lines: list[str], pattern: str) -> str | None:
    rx = re.compile(pattern)
    for line in reversed(lines):
        if rx.search(line):
            return line.strip()
    return None


def parse_kv_line(line: str | None, prefix: str) -> dict[str, str]:
    if line is None or not line.startswith(prefix):
        return {}
    values: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        values[key] = value
    return values


def extract_val_bpb(line: str | None) -> float | None:
    if line is None:
        return None
    m = re.search(r"val_bpb:([0-9.]+)", line)
    return float(m.group(1)) if m else None


def select_metric(
    metric_mode: str,
    max_wallclock: int,
    prequant_line: str | None,
    postquant_line: str | None,
    sliding_line: str | None,
) -> tuple[str, str]:
    if metric_mode not in METRIC_MODES:
        valid = ", ".join(sorted(METRIC_MODES))
        raise ValueError(f"metric_mode must be one of: {valid}")

    if metric_mode == "sliding":
        if sliding_line is None:
            raise RuntimeError("metric_mode=sliding requested, but no sliding-window metric was found in the log")
        return "sliding", sliding_line
    if metric_mode == "postquant":
        if postquant_line is None:
            raise RuntimeError("metric_mode=postquant requested, but no post-quant metric was found in the log")
        return "postquant", postquant_line
    if metric_mode == "prequant":
        if prequant_line is None:
            raise RuntimeError("metric_mode=prequant requested, but no pre-quant metric was found in the log")
        return "prequant", prequant_line

    if sliding_line is not None:
        return "sliding", sliding_line
    if max_wallclock <= 180 and prequant_line is not None:
        return "prequant", prequant_line
    if postquant_line is not None:
        return "postquant", postquant_line
    if prequant_line is not None:
        return "prequant", prequant_line
    raise RuntimeError("could not find any candidate metric in the log")


def verify_effective_config(lines: list[str], expected: dict[str, str]) -> None:
    shape = parse_kv_line(last_matching_line(lines, r"^model_shape:"), "model_shape:")
    eval_cfg = parse_kv_line(last_matching_line(lines, r"^eval_config:"), "eval_config:")
    seed_line = last_matching_line(lines, r"^seed:")

    checks = {
        "NUM_LAYERS": shape.get("num_layers"),
        "MODEL_DIM": shape.get("model_dim"),
        "EVAL_STRIDE": eval_cfg.get("eval_stride"),
    }
    if seed_line is not None and seed_line.startswith("seed:"):
        checks["SEED"] = seed_line.split(":", 1)[1].strip()

    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in checks:
            continue
        actual_value = checks[key]
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    if mismatches:
        raise RuntimeError("effective config mismatch: " + "; ".join(mismatches))


def print_summary(lines: list[str], metric_mode: str, max_wallclock: int) -> None:
    model_shape = last_matching_line(lines, r"^model_shape:")
    eval_config = last_matching_line(lines, r"^eval_config:")
    stop_line = last_matching_line(lines, r"^stopping_early:")
    prequant_line = last_matching_line(lines, r"^step:\d+/\d+ val_loss:")
    artifact_line = last_matching_line(lines, r"^Total submission size int6\+")
    postquant_line = last_matching_line(lines, r"^final_int6_roundtrip_exact ")
    sliding_line = last_matching_line(lines, r"^final_sliding_window_eval_exact ")

    metric_source, metric_line = select_metric(
        metric_mode=metric_mode,
        max_wallclock=max_wallclock,
        prequant_line=prequant_line,
        postquant_line=postquant_line,
        sliding_line=sliding_line,
    )
    selected_val_bpb = extract_val_bpb(metric_line)
    val_bpb_text = "n/a" if selected_val_bpb is None else f"{selected_val_bpb:.8f}"

    print("\n=== SUMMARY ===")
    if model_shape is not None:
        print(model_shape)
    if eval_config is not None:
        print(eval_config)
    print(f"selected_metric:mode:{metric_mode} source:{metric_source} val_bpb:{val_bpb_text}")
    if stop_line is not None:
        print(stop_line)
    if prequant_line is not None:
        print(f"prequant_stop:{prequant_line}")
    if artifact_line is not None:
        print(artifact_line)
    if postquant_line is not None:
        print(postquant_line)
    if sliding_line is not None:
        print(sliding_line)


@app.function(image=image, gpu="H100", timeout=1800)
def train(
    run_id: str = "run",
    max_wallclock: int = 120,
    train_log_every: int = 100,
    num_layers: int = 0,
    model_dim: int = 0,
    eval_stride: int = -1,
    overrides: str = "",
    metric_mode: str = "auto",
    save_checkpoint: bool = False,
    run_quant_sweep: bool = False,
    quant_recipes: str = "",
    quant_val_tokens_limit: int = 1_048_576,
    quant_probe_blocks: bool = False,
):
    import os
    import subprocess

    effective_overrides = merge_overrides(
        num_layers=num_layers,
        model_dim=model_dim,
        eval_stride=eval_stride,
        overrides=overrides,
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "RUN_ID": run_id,
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "DATA_PATH": "/root/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "/root/data/tokenizers/fineweb_1024_bpe.model",
        **effective_overrides,
    }
    override_text = ",".join(f"{k}={v}" for k, v in sorted(effective_overrides.items()))
    print(f"launch_config:run_id:{run_id} max_wallclock:{max_wallclock} metric_mode:{metric_mode}")
    print(f"launch_overrides:{override_text}")

    proc = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "/root/train_gpt.py"],
        env=env,
        cwd="/root",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"training failed with exit code {proc.returncode}")

    log_path = f"/root/logs/{run_id}.txt"
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log file not found: {log_path}")

    with open(log_path) as f:
        lines = f.readlines()

    verify_effective_config(lines, effective_overrides)
    print_summary(lines, metric_mode=metric_mode, max_wallclock=max_wallclock)
    if run_quant_sweep:
        quant_cmd = [
            "python3",
            "/root/quant_sweep.py",
            "--checkpoint-path",
            "/root/final_model.pt",
            "--run-id",
            f"{run_id}_quant",
            "--overrides",
            override_text,
            "--val-tokens-limit",
            str(quant_val_tokens_limit),
            "--output-path",
            f"/root/logs/{run_id}_quant.json",
        ]
        if quant_recipes.strip():
            quant_cmd.extend(["--recipes", quant_recipes])
        if quant_probe_blocks:
            quant_cmd.append("--probe-blocks")
        print("\n=== QUANT SWEEP ===")
        quant_proc = subprocess.run(quant_cmd, env=env, cwd="/root", check=False)
        if quant_proc.returncode != 0:
            raise RuntimeError(f"quant sweep failed with exit code {quant_proc.returncode}")
    if save_checkpoint:
        checkpoint_path = "/root/final_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            return f.read()
    return None


@app.local_entrypoint()
def main(
    run_id: str = "run",
    max_wallclock: int = 120,
    train_log_every: int = 100,
    num_layers: int = 0,
    model_dim: int = 0,
    eval_stride: int = -1,
    overrides: str = "",
    metric_mode: str = "auto",
    save_checkpoint: bool = False,
    checkpoint_dir: str = "checkpoints",
    run_quant_sweep: bool = False,
    quant_recipes: str = "",
    quant_val_tokens_limit: int = 1_048_576,
    quant_probe_blocks: bool = False,
):
    checkpoint_bytes = train.remote(
        run_id=run_id,
        max_wallclock=max_wallclock,
        train_log_every=train_log_every,
        num_layers=num_layers,
        model_dim=model_dim,
        eval_stride=eval_stride,
        overrides=overrides,
        metric_mode=metric_mode,
        save_checkpoint=save_checkpoint,
        run_quant_sweep=run_quant_sweep,
        quant_recipes=quant_recipes,
        quant_val_tokens_limit=quant_val_tokens_limit,
        quant_probe_blocks=quant_probe_blocks,
    )
    if save_checkpoint and checkpoint_bytes is not None:
        from pathlib import Path

        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.pt"
        out_path.write_bytes(checkpoint_bytes)
        print(f"saved_checkpoint:{out_path}")
