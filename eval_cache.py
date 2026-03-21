"""Eval-only: load a saved checkpoint and run neural cache eval."""
import io, math, os, sys, time, zlib
from pathlib import Path

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# Import everything from the training script
sys.path.insert(0, str(Path(__file__).parent))
from sota_train_gpt import (
    Hyperparameters, GPT, CastedLinear, RMSNorm,
    build_sentencepiece_luts, load_validation_tokens,
    dequantize_state_dict_int8, eval_val, eval_val_with_cache,
    restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS,
    INT6_QUANT_RANGE, INT8_QUANT_RANGE,
)

def main():
    args = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    print(f"val tokens: {val_tokens.numel() - 1}")

    # Build model
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, use_smeargate=args.use_smeargate,
        bigram_hash_buckets=args.bigram_hash_buckets, bigram_hash_dim=args.bigram_hash_dim,
        smear_gate_init=args.smear_gate_init, xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    # Load quantized checkpoint
    ckpt_path = os.environ.get("CHECKPOINT", "final_model.int6.ptz")
    print(f"loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        blob = f.read()
    if HAS_ZSTD:
        raw = zstd.ZstdDecompressor().decompress(blob)
    else:
        raw = zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    print("checkpoint loaded")

    # Standard eval (baseline)
    base_model.eval()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        args, base_model, 0, 1, device, 1, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"standard eval: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} time={time.perf_counter()-t0:.1f}s")

    # Neural cache eval — sweep lambda
    cache_theta = float(os.environ.get("NEURAL_CACHE_THETA", "1.0"))
    for lam in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        t0 = time.perf_counter()
        c_loss, c_bpb = eval_val_with_cache(
            args, base_model, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            cache_lambda=lam, cache_theta=cache_theta,
            log_fn=lambda msg: print(f"  {msg}"),
        )
        elapsed = time.perf_counter() - t0
        delta = c_bpb - val_bpb
        print(f"cache lambda={lam:.2f}: val_bpb={c_bpb:.4f} delta={delta:+.4f} time={elapsed:.1f}s")

if __name__ == "__main__":
    main()
