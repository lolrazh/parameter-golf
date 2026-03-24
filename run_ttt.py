"""Standalone TTT eval: load exported checkpoint, run LoRA TTT, report BPB.
Supports SKIP_POSTQUANT_EVAL=1 to skip redundant post-quant eval."""
import sys, os, io, time, math, zlib, glob
import torch
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
from frontier_512 import (
    Hyperparameters, GPT, eval_val_ttt_lora, eval_val,
    build_sentencepiece_luts, load_validation_tokens,
    dequantize_state_dict_int8, HAVE_ZSTD,
)
if HAVE_ZSTD:
    import zstandard as zstd

def main():
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "final_model.int8.ptz")
    skip_postquant = int(os.environ.get("SKIP_POSTQUANT_EVAL", "0"))
    args = Hyperparameters()
    device = torch.device("cuda")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    print(f"val_tokens:{val_tokens.numel()} checkpoint:{checkpoint_path}")
    print(f"ttt_config: rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} epochs={args.ttt_epochs} "
          f"chunk={args.ttt_chunk_size} ctx={args.ttt_eval_seq_len} batch={args.ttt_batch_size} "
          f"min_doc={args.ttt_min_doc_len} cosine={args.ttt_cosine}")

    with open(checkpoint_path, "rb") as f:
        blob = f.read()
    if HAVE_ZSTD:
        raw = zstd.ZstdDecompressor().decompress(blob)
    else:
        raw = zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
    ).to(device)
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    print("checkpoint loaded")

    if not skip_postquant:
        t0 = time.perf_counter()
        q_loss, q_bpb = eval_val(args, model, 0, 1, device, 1, val_tokens,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                  eval_seq_len=args.eval_seq_len)
        print(f"postquant_bpb:{q_bpb:.6f} loss:{q_loss:.4f} time:{time.perf_counter()-t0:.1f}s")

    ttt_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
    ).to(device)
    ttt_model.load_state_dict(model.state_dict(), strict=True)

    t1 = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_ttt_lora(args, ttt_model, 0, 1, device,
                                           base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"ttt_bpb:{ttt_bpb:.6f} loss:{ttt_loss:.4f} time:{time.perf_counter()-t1:.1f}s")

if __name__ == "__main__":
    main()
