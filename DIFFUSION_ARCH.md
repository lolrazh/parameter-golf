# Diffusion LM Architecture — Continuous (CDCD-style)

North star doc. If you're lost in the code, come back here.

## ELI5: What Is This?

Our GPT is a **reader** — reads left to right, guesses the next word. Never peeks ahead.

Our Diffusion LM is a **sculptor** — starts with a block of pure static (Gaussian noise), and chisels it into coherent text. Each chisel pass, the noise becomes a little more word-shaped. After enough passes: real sentences.

Think of it like tuning a radio. You start with pure static. Each step, the signal gets a little clearer. Eventually: music.

```
Step 0:  [0.73, -1.2, 0.41, ...]  [2.1, 0.03, -0.8, ...]  ...    ← pure noise
Step 50: [-0.1, 0.82, 0.21, ...]  [0.5, 1.10, -0.3, ...]  ...    ← blurry words
Step 150:[0.01, 0.99, 0.01, ...]  [0.02, 0.01, 0.98, ...]  ...    ← almost tokens
Final:   "The"                     "cat"                     ...    ← snap to words
```

The key difference from masked diffusion (MDLM/LLaDA): there are no [MASK] tokens.
Tokens live in **continuous vector space** and we add **real Gaussian noise** — the same
kind of noise used in image diffusion (DALL-E, Stable Diffusion, etc).

## The 3 Levels of "Diffusion" for Text

| Level | Approach | Noise Type | What We're Doing |
|-------|----------|-----------|-----------------|
| 1. Masked (MDLM) | Replace tokens with [MASK] | Discrete deletion | Glorified BERT |
| 2. Score (SEDD) | Scramble tokens via transition matrix | Discrete permutation | Real discrete diffusion |
| **3. Continuous (CDCD)** | **Gaussian noise on embeddings** | **Continuous Gaussian** | **This. Real diffusion.** |

## How It Works (The Full Pipeline)

### Forward Process (Corruption): "Drowning Words in Noise"

```
1. Start: "The cat sat on the mat"
2. Look up each token's learned embedding:
     "The" → [0.12, 0.87, -0.34, ...]   (64-dim vector)
     "cat" → [-0.56, 0.21, 0.91, ...]
     ...
3. Add Gaussian noise scaled by t:
     z_t = embedding + t × ε,    where ε ~ N(0, I)

   At t=1 (low noise):   embeddings are slightly fuzzy, still recognizable
   At t=300 (max noise):  embeddings are drowned in noise, unrecognizable
```

ELI5: Imagine each word is a dot on a map. Low noise = dots jitter slightly. High noise = dots are scattered across the entire map. The model learns to un-scatter them.

### The Model: "Given Noisy Embeddings, What Were The Words?"

The transformer takes noisy continuous vectors (not token IDs!) and predicts
which token each position originally was. It outputs a probability distribution
over the vocabulary for every position.

```
Input:   noisy vectors z_t  +  noise level t
Output:  logits → softmax → p(token | z_t, t) for each position
Loss:    cross-entropy between predicted tokens and actual tokens
```

### Score Interpolation: "The Bridge Between CE and Diffusion"

This is the magic trick from CDCD (Dieleman et al., 2022). We train with
cross-entropy (familiar, stable), but sample with a continuous ODE (powerful).

The score (gradient of log-probability) for Gaussian-corrupted categorical data:

```
score(z, t) = (E[x₀ | z, t] − z) / t²
```

where E[x₀ | z, t] = Σᵢ p(token_i | z, t) × embedding(token_i)

In words: take the model's predicted probabilities, compute the weighted
average of all token embeddings, subtract the current noisy vector, divide
by t². That IS the score. No score matching loss needed.

### Sampling: "Following the ODE From Noise to Text"

```
1. Start: z ~ N(0, σ_max² × I)              ← pure Gaussian noise
2. For each step (200 Euler steps):
     a. Forward pass → logits → softmax probs
     b. E[x₀] = Σ probs × embeddings         ← expected clean embedding
     c. score = (E[x₀] − z) / t²             ← score interpolation
     d. z ← z − t × score × Δt               ← Euler step along ODE
3. Final: argmax of softmax logits → tokens   ← "rounding"
```

This is the probability flow ODE: dz = −t × ∇log p(z,t) × dt

ELI5: The score is like a compass pointing from "where you are" (noisy)
toward "where you should be" (clean embedding). Each step, you walk a little
bit in that direction. After enough steps, you arrive.

## What Changed From GPT

### 1. Input: Token IDs → Continuous Vectors

```
GPT:       input_ids [23, 107, 5, ...]  →  look up embedding  →  transformer
Diffusion: embeddings + noise           →  scale by 1/√(t²+1) →  transformer
```

The model never sees token IDs directly. It sees noisy continuous vectors.
Scaling by `1/√(t²+1)` keeps the input magnitude stable regardless of noise level.

### 2. Attention: Causal → Bidirectional

```
GPT (causal):                    Diffusion (bidirectional):
  T h e c a t                      T h e c a t
T ✓ · · · · ·                   T ✓ ✓ ✓ ✓ ✓ ✓
h ✓ ✓ · · · ·                   h ✓ ✓ ✓ ✓ ✓ ✓
e ✓ ✓ ✓ · · ·        →          e ✓ ✓ ✓ ✓ ✓ ✓
c ✓ ✓ ✓ ✓ · ·                   c ✓ ✓ ✓ ✓ ✓ ✓
a ✓ ✓ ✓ ✓ ✓ ·                   a ✓ ✓ ✓ ✓ ✓ ✓
t ✓ ✓ ✓ ✓ ✓ ✓                   t ✓ ✓ ✓ ✓ ✓ ✓
```

Same as masked diffusion. The model needs context from ALL directions.

### 3. Timestep Conditioning via AdaLN

The model needs to know the noise level t. Implementation: Adaptive Layer Norm.

```
Normal RMSNorm:    norm(x)
AdaLN:             (1 + scale(t)) × norm(x) + shift(t)
```

Timestep encoding:
```
scalar t
  → ln(t)/4                                          (log-scale, CDCD convention)
  → sinusoidal encoding                               [d_model]
  → MLP (Linear → SiLU → Linear)                      [d_model]
  → per-block: Linear → split → (scale, shift)        [2 × d_model]
```

### 4. Loss: Next-Token → All-Token Cross-Entropy

```
GPT:       CE on next token only,  at clean positions
Diffusion: CE on ALL tokens,       at noisy positions
```

The model predicts what EVERY position's original token was, given the noisy input.
Loss = cross_entropy(predicted_logits, original_tokens), averaged over all positions.

### 5. Self-Conditioning (Bonus — Not In GPT)

During training, 50% of the time:
  1. Run a forward pass → get predicted embeddings E[x₀]
  2. Concatenate E[x₀] with the noisy input as additional channels
  3. Run a SECOND forward pass (only this one's gradients count)

This teaches the model to refine its own predictions iteratively.
At inference, every step feeds back the previous prediction. ~0.4 nats improvement.

## What Stayed The Same

| Component | Status | Notes |
|-----------|--------|-------|
| Transformer blocks | Same | Just bidirectional + AdaLN |
| RoPE | Same | Position info still matters |
| MLP (relu²) | Same | Position-independent, no change |
| U-Net skip connections | Same | Help diffusion models too (DiT uses them) |
| Logit softcap | Same | Prevents logit blow-up |
| Muon + Adam optimizer | Same | Matrix params still orthogonalized |

## What's Gone (Not Needed)

| Component | Why Removed |
|-----------|-------------|
| Token embedding lookup in forward | Input is continuous vectors, not IDs |
| Tied embedding output head | Output head is a separate linear → vocab logits |
| Value embeddings | No token IDs to look up during forward pass |
| Autoregressive data loader (x, y=x[1:]) | Data is (x, x) — predict same tokens, not next |

## Architecture Table

| Parameter | Value | Source |
|-----------|-------|--------|
| num_layers | 9 | Baseline GPT |
| model_dim | 512 | Baseline GPT |
| num_heads | 8 | Baseline GPT |
| num_kv_heads | 4 | Baseline GPT (GQA) |
| vocab_size | 1024 | Same tokenizer, no [MASK] needed |
| mlp_mult | 2 | Baseline GPT |
| logit_softcap | 30.0 | Baseline GPT |
| rope_base | 10000 | Baseline GPT |
| activation | relu² | Baseline GPT |
| attention | **Bidirectional** | Changed from causal |
| conditioning | **AdaLN** | New — adaptive layer norm |
| **embed_dim** | **64** | New — continuous embedding dimension (CDCD sweet spot) |
| **self_conditioning** | **True** | New — feed back predictions (CDCD/Plaid) |

## Continuous Diffusion Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| embed_dim | 64 | Learned L2-normalized embeddings. Separate from model_dim. |
| t_min | 1.0 | Min noise level (tokens distinguishable even at σ=1) |
| t_max | 300.0 | Max noise level (pure noise) |
| t_schedule | log-uniform | ln(t) ~ U(ln(t_min), ln(t_max)). Concentrates on lower noise. |
| input_scale | 1/√(t²+1) | Normalizes input magnitude regardless of noise level |
| self_cond_prob | 0.5 | Probability of using self-conditioning during training |
| score_temp | 0.5 | Temperature on logits during sampling (sharper = more confident) |

## Sampling Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_steps | 200 | Euler steps along the probability flow ODE |
| solver | euler | Can upgrade to Heun (2nd order) for better quality |
| seq_len | 256 | Generation length (configurable) |

## Training Loop (Pseudocode)

```python
for each batch of token sequences x₀:              # shape: [B, L]
    # 1. Embed tokens into continuous space
    e₀ = embed(x₀)                                 # [B, L, 64], L2-normalized

    # 2. Sample noise level
    ln_t = uniform(ln(t_min), ln(t_max))            # log-uniform
    t = exp(ln_t)                                   # scalar per sample

    # 3. Add Gaussian noise
    ε = randn_like(e₀)
    z_t = e₀ + t × ε                               # noisy embeddings

    # 4. Scale input
    z_scaled = z_t / sqrt(t² + 1)                   # unit variance

    # 5. Project to model dimension + self-conditioning
    h = input_proj(z_scaled)                        # [B, L, 64] → [B, L, 512]
    if self_conditioning and random() < 0.5:
        with no_grad():
            prev_logits = model(h, t)
            prev_probs = softmax(prev_logits)
            prev_embed = prev_probs @ embed.weight  # E[x₀] estimate
        h = input_proj(cat(z_scaled, prev_embed))   # [B, L, 128] → [B, L, 512]

    # 6. Forward pass
    logits = model(h, t)                            # [B, L, vocab_size]

    # 7. Cross-entropy loss on ALL positions
    loss = cross_entropy(logits, x₀)                # predict original tokens
    loss.backward()
```

## Sampling Loop (Pseudocode)

```python
# Initialize from pure noise
z = randn(B, L, embed_dim) * t_max                  # [B, L, 64]
prev_embed = zeros(B, L, embed_dim)                  # self-conditioning init

# Build timestep schedule: t_max → t_min in num_steps
ts = exp(linspace(ln(t_max), ln(t_min), num_steps))

for i in range(num_steps):
    t = ts[i]
    dt = ts[i+1] - ts[i] if i < num_steps-1 else 0

    # Forward pass
    z_scaled = z / sqrt(t² + 1)
    h = input_proj(cat(z_scaled, prev_embed))
    logits = model(h, t) / score_temp
    probs = softmax(logits)

    # Score interpolation
    E_x0 = probs @ embed.weight                     # expected clean embedding
    score = (E_x0 - z) / t²

    # Euler step along probability flow ODE
    z = z - t × score × dt

    # Update self-conditioning
    prev_embed = E_x0

# Final rounding
logits = model(z / sqrt(t_min² + 1), t_min)
tokens = argmax(logits, dim=-1)
return tokens
```

## The Rounding Problem

At the end of sampling, we have continuous vectors, not tokens. Solutions:

| Method | Quality | Complexity | Our Choice |
|--------|---------|-----------|-----------|
| Argmax on final logits | Good | Simple | **Yes (v1)** |
| Clamping each step | Moderate | Simple | No |
| AR decoder (CoDAR) | Best | Complex | Maybe v2 |

Argmax on the final forward pass is what CDCD/Plaid do. Good enough to start.

## Key Equations (Reference)

**Forward process:**
```
z_t = e(x₀) + t × ε,    ε ~ N(0, I)
```

**Score interpolation:**
```
∇_z log p(z,t) = (E[x₀|z,t] − z) / t²
E[x₀|z,t] = Σᵢ p(tokenᵢ|z,t) × e(tokenᵢ)
```

**Probability flow ODE:**
```
dz = −t × ∇_z log p(z,t) dt
```

**Training loss:**
```
L = E_t [ CE(f_θ(z_t, t), x₀) ]    (cross-entropy, all positions)
```

## File Map

| File | Purpose |
|------|---------|
| `train_diffusion_mlx.py` | Training + eval + sampling (MLX/Apple Silicon) |
| `DIFFUSION_ARCH.md` | This file — architecture north star |
