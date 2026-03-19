# Research Ideas

Attack vectors for improving BPB. Test one at a time. Check off what's done.
Priority: ★★★ = high impact + low effort, ★★ = high impact + medium effort, ★ = speculative

## LR Sweep (DONE)
- [x] 0.25x, 0.5x, 1x, 1.5x, 2x, 3x, 4x → 3x best locally (val_loss=4.0579)
- [x] Note: LR values don't transfer to H100 (different batch size). Purpose was to calibrate local baseline.

## Proven Speedrun Tricks (from modded-nanogpt, battle-tested)
- [ ] ★★★ Value embeddings: learned embed tables mixed into attention values. Near-zero FLOP cost, WR-setting.
- [ ] ★★★ Smear module: `x[t] += 0.07 * sigmoid(gate) * x[t-1]`. Bigram stats. Basically free.
- [ ] ★★★ Logit softcap 15 (not 30): modded-nanogpt tuned this down.
- [ ] ★★★ LAWA (Latest Weight Averaging): running avg of recent checkpoints. Free generalization boost.
- [ ] ★★ Drop first MLP: layer 0 doesn't need MLP. Saves params to redistribute.
- [ ] ★★ Backout mechanism: store residual at 2/3 depth, subtract learned fraction before LM head.
- [ ] ★★ Cautious weight decay: only decay when update and weights agree in direction.
- [ ] ★★ Sparse attention gate: per-head sigmoid gate on first 12 dims. Nearly free.
- [ ] ★ Bigram hash embedding: hash consecutive token pairs → separate embed table.
- [ ] ★ Multiple embedding tables: per-layer mixing weights, decouple embed from model dim.

## Model Shape (env vars only)
- [ ] ★★★ Width vs depth: deeper+narrower (12L×384d, 16L×320d) vs baseline 9L×512d
- [ ] ★★ MLP expansion: 3x, 4x (more knowledge storage per layer)
- [ ] ★★ SwiGLU activation: 3 matrices at 2.7x expansion (vs relu² at 2x). Matched params.
- [ ] ★ Full MHA: NUM_KV_HEADS=8 (remove GQA bottleneck)
- [ ] ★ Fewer KV heads: NUM_KV_HEADS=2 (save params, spend elsewhere)
- [ ] ★ More heads: NUM_HEADS=16 (finer attention, head_dim=32)
- [ ] ★ Fewer heads: NUM_HEADS=4 (Kimi showed halving heads barely hurts)
- [ ] ★ Longer seq: TRAIN_SEQ_LEN=2048 (more context, O(n²) cost)

## Architecture (code changes)
- [ ] ★★★ Depth recurrence: run N layers K times (2x depth, 1x params). Unique to param golf.
- [ ] ★★ QK-Norm: normalize Q,K before dot product (Qwen3 uses this, may replace softcap)
- [ ] ★★ MLA (Multi-Head Latent Attention): compress K/V through bottleneck (Kimi K2)
- [ ] ★★ Causal Conv1d before attention: cheap local pattern capture (Qwen3.5 DeltaNet)
- [ ] ★ Remove U-Net skips: test if they actually help at this scale
- [ ] ★ Dense skip connections: every layer gets skip from every earlier layer
- [ ] ★ Learnable RMSNorm: add weight to norm layers
- [ ] ★ MoE: multiple small MLPs, sigmoid routing (not softmax, per GLM-5)
- [ ] ★ Shared QKV projections: tie Q/K or K/V weights to save params
- [ ] ★ Multi-query attention: NUM_KV_HEADS=1 (extreme param saving)

## Optimizer / Training Recipe
- [ ] ★★ WSD schedule (warmup-stable-decay, trapezoidal). Speedrun uses this.
- [ ] ★★ Verify Muon has: weight decay, per-shape update RMS scaling, 0.2 RMS coeff
- [ ] ★ Gradient clipping: GRAD_CLIP_NORM=1.0
- [ ] ★ Muon steps: 3 vs 5 vs 7 (Newton-Schulz iterations)
- [ ] ★ Higher Muon momentum: 0.97, 0.99
- [ ] ★ Decouple LRs: different multipliers for embed vs matrix vs scalar
- [ ] ★ Longer warmdown: 2400 vs 1200 iters
- [ ] ★ β₂=0.99 (vs 0.95)
- [ ] ★ Sequence length scaling: start short, grow during training

## Quantization / Compression
- [ ] ★★ QAT: simulate int8 noise during training (reduce 0.007 BPB quant tax)
- [ ] ★ Mixed-bit: int4 for large matrices, int8 for small
- [ ] ★ Weight pruning: zero small weights for better zlib ratio
- [ ] ★ Per-channel vs per-row quantization
- [x] ✗ BitNet/ternary at 17M params: confirmed BAD (paper tested 6M-48M, capacity loss too severe)

## Tokenizer
- [ ] ★ sp2048: 2048 vocab (needs data re-download)
- [ ] ★ sp4096: 4096 vocab
- [ ] ★ byte260: raw bytes, no tokenizer (tiny embedding, many tokens)

## Eval Tricks (rules allow these)
- [ ] ★★ Longer eval context: train 1024, eval 2048+ (RoPE extrapolation)
- [ ] ★ Sliding window eval: overlapping windows for boundary predictions
- [ ] ★ Eval at multiple seq lengths, pick best

## Wild Ideas
- [ ] ★ Hybrid linear/softmax attention: 3:1 DeltaNet:Attention ratio (Qwen3.5 style)
- [ ] ★ KDA (Kimi Delta Attention): recurrent memory, eliminates position embeddings
- [ ] ★ Test-time training: fine-tune on eval context before predicting
- [ ] ★ Knowledge distillation from larger model into 16MB student
- [ ] ★ Progressive training: start small, grow model during training
- [ ] ★ Auxiliary MLM loss: BERT-style masked prediction as regularizer alongside AR loss
- [x] ✗ Diffusion LMs: wrong eval paradigm, 100-1000x more compute per example
- [x] ✗ SSM/Liquid layers: only win at long context, we're at 1024 seq len
