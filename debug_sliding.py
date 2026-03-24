"""Debug: compare forward(x,y) vs get_logits(x) + manual CE on same input."""
import sys, os, torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
from frontier_512 import (
    Hyperparameters, GPT,
    build_sentencepiece_luts, load_validation_tokens,
)

def main():
    args = Hyperparameters()
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, 2048)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
    ).to(device)
    state = torch.load("final_model_raw.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Single sequence, seq_len=2048
    x = val_tokens[:2048].to(device=device, dtype=torch.int64).unsqueeze(0)
    y = val_tokens[1:2049].to(device=device, dtype=torch.int64).unsqueeze(0)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Method 1: forward(x, y) — returns mean loss
        loss_fwd = model(x, y).float().item()

        # Method 2: get_logits(x) + manual CE
        logits = model.get_logits(x)
        logits_f = logits[0].float()
        targets_f = y[0]
        loss_manual = F.cross_entropy(logits_f, targets_f, reduction="mean").item()

        # Method 3: get_logits, but only score last 64 tokens (like sliding window)
        score_from = 2048 - 64
        suffix_logits = logits[0, score_from:].float()
        suffix_targets = y[0, score_from:]
        loss_suffix = F.cross_entropy(suffix_logits, suffix_targets, reduction="mean").item()

    print(f"forward(x,y) mean loss: {loss_fwd:.6f}")
    print(f"get_logits + full CE:    {loss_manual:.6f}")
    print(f"get_logits + last 64:    {loss_suffix:.6f}")
    print(f"diff fwd vs manual:      {abs(loss_fwd - loss_manual):.6f}")

    # Also test with seq_len=1024 to see if length matters
    x1k = val_tokens[:1024].to(device=device, dtype=torch.int64).unsqueeze(0)
    y1k = val_tokens[1:1025].to(device=device, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_1k_fwd = model(x1k, y1k).float().item()
        logits_1k = model.get_logits(x1k)
        loss_1k_manual = F.cross_entropy(logits_1k[0].float(), y1k[0], reduction="mean").item()
    print(f"\nseq_len=1024:")
    print(f"forward(x,y) mean loss: {loss_1k_fwd:.6f}")
    print(f"get_logits + full CE:    {loss_1k_manual:.6f}")
    print(f"diff: {abs(loss_1k_fwd - loss_1k_manual):.6f}")

    # Test last 64 tokens at seq_len=1024 — should be BETTER than mean
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_1k = model.get_logits(x1k)
        sf_1k = 1024 - 64
        loss_1k_suffix = F.cross_entropy(logits_1k[0, sf_1k:].float(), y1k[0, sf_1k:], reduction="mean").item()
    print(f"get_logits + last 64:    {loss_1k_suffix:.6f}")
    print(f"last64 vs mean at 1024:  {loss_1k_suffix - loss_1k_fwd:+.6f}")

if __name__ == "__main__":
    main()
