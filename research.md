# Research Ideas

Attack vectors for improving BPB. Test one at a time. Check off what's done.

## Model Shape (env vars only)
- [ ] Width vs depth: 6L×640d vs 12L×384d vs 9L×512d (matched params)
- [ ] MLP expansion: 3x, 4x (more knowledge storage per layer)
- [ ] Full MHA: NUM_KV_HEADS=8 (remove GQA bottleneck)
- [ ] Fewer KV heads: NUM_KV_HEADS=2 (save params, spend elsewhere)
- [ ] More heads: NUM_HEADS=16 (finer attention, head_dim=32)
- [ ] Bigger vocab: VOCAB_SIZE=2048 or 4096 (better BPB ratio but param cost)
- [ ] Longer seq: TRAIN_SEQ_LEN=2048 (more context, O(n²) cost)

## Architecture (code changes)
- [ ] Depth recurrence: run N layers K times (2x depth, 1x params)
- [ ] GLU activation: replace relu² with SwiGLU/GeGLU (3 matrices instead of 2, but better)
- [ ] Learnable RMSNorm: add weight to norm layers
- [ ] ALiBi positions: replace RoPE with ALiBi (no position params, length-generalizes)
- [ ] Remove U-Net skips: test if they actually help at this scale
- [ ] Dense skip connections: every layer gets skip from every earlier layer
- [ ] Mixture of Experts: multiple small MLPs, route tokens to top-k
- [ ] Shared QKV projections: tie Q/K or K/V weights to save params
- [ ] Multi-query attention: NUM_KV_HEADS=1 (extreme param saving)
- [ ] Different softcap values: 15, 20, 50 (vs current 30)
- [ ] Zero-init variations: different init schemes for proj layers

## Optimizer / Training Recipe
- [ ] Cosine LR schedule (vs linear warmdown)
- [ ] WSD schedule (warmup-stable-decay)
- [ ] Gradient clipping: GRAD_CLIP_NORM=1.0
- [ ] Muon steps: 3 vs 5 vs 7 (Newton-Schulz iterations)
- [ ] Higher Muon momentum: 0.97, 0.99
- [ ] Decouple LRs: different multipliers for embed vs matrix vs scalar
- [ ] Longer warmdown: 2400 vs 1200 iters
- [ ] β₂=0.99 (vs 0.95)

## Quantization / Compression
- [ ] QAT: simulate int8 noise during training
- [ ] Mixed-bit: int4 for large matrices, int8 for small
- [ ] Weight pruning: zero small weights for better zlib ratio
- [ ] Per-channel vs per-row quantization
- [ ] Distillation: train large model, distill into small quantized model

## Tokenizer
- [ ] sp2048: 2048 vocab (needs data re-download)
- [ ] sp4096: 4096 vocab
- [ ] byte260: raw bytes, no tokenizer (tiny embedding, many tokens)

## Eval Tricks (rules allow these)
- [ ] Longer eval context: train 1024, eval 2048+ (RoPE extrapolation)
- [ ] Sliding window eval: overlapping windows for boundary predictions
- [ ] Eval at multiple seq lengths, pick best

## Wild Ideas
- [ ] BitNet: ternary weights {-1, 0, 1} — nearly free to compress
- [ ] Hyena/convolution hybrid: replace some attention layers with convolutions
- [ ] State-space model (Mamba-style): replace attention entirely
- [ ] Test-time training: fine-tune on eval context before predicting
- [ ] Weight tying across non-adjacent layers (not just embed)
- [ ] Knowledge distillation from larger model into 16MB student
- [ ] Learned position interpolation for longer eval context
- [ ] Progressive training: start small, grow model during training
