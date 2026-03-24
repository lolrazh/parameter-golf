"""Quick test: load raw checkpoint, run sliding window eval. No training."""
import sys, os, time, math
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
from frontier_512 import (
    Hyperparameters, GPT, eval_val_sliding, eval_val,
    build_sentencepiece_luts, load_validation_tokens,
    quantize_state_dict_int8, dequantize_state_dict_int8,
)

def main():
    args = Hyperparameters()
    device = torch.device("cuda")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, args.eval_seq_len))
    if args.val_tokens_limit > 0:
        val_tokens = val_tokens[:args.val_tokens_limit + 1]
    print(f"val_tokens:{val_tokens.numel()} eval_seq_len:{args.eval_seq_len} eval_stride:{args.eval_stride}")

    # Load raw checkpoint (no torch.compile involved)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "final_model_raw.pt"
    print(f"Loading {ckpt_path}...")
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    print("loaded")

    # Standard eval (seq_len=2048, no sliding window) for baseline
    t0 = time.perf_counter()
    std_loss, std_bpb = eval_val(args, model, 0, 1, device, 1, val_tokens,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                  eval_seq_len=args.eval_seq_len)
    print(f"standard_eval bpb:{std_bpb:.6f} loss:{std_loss:.4f} time:{time.perf_counter()-t0:.1f}s")

    # Sliding window eval
    if args.eval_stride > 0:
        t1 = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, model, 0, 1, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            eval_seq_len=args.eval_seq_len, eval_stride=args.eval_stride,
        )
        print(f"sliding_window bpb:{sw_bpb:.6f} loss:{sw_loss:.4f} time:{time.perf_counter()-t1:.1f}s")
        print(f"sliding_window_gain: {std_bpb - sw_bpb:.6f} BPB")

if __name__ == "__main__":
    main()
