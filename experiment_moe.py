"""
Mixture-of-Experts experiment for parameter-golf.

Replaces the dense MLP in each transformer block with a sparse MoE layer:
  - N experts (default 4), each a smaller relu^2 MLP
  - Top-K routing (default top-2) with softmax gating
  - Auxiliary load-balancing loss to prevent expert collapse
  - Optional router z-loss for numerical stability

Design rationale (16MB artifact constraint):
  Current dense MLP: fc(512->1536) + proj(1536->512) = 1,572,864 params/block
  MoE with 4 experts of hidden=384: 4 * (512*384 + 384*512) + router(512*4) = 1,574,912 params/block
  This is param-neutral (~+2K per block), so it fits in the same 16MB budget.

  With top-2 routing, each token activates 2/4 experts = 50% of MLP params.
  The hypothesis: specialized experts learn different features, improving BPB
  at the same total param count and similar compute.

Usage:
  RUN_ID=moe_test ITERATIONS=200 python experiment_moe.py

  Key env vars:
    MOE_NUM_EXPERTS   - number of experts per layer (default: 4)
    MOE_TOP_K         - experts activated per token (default: 2)
    MOE_AUX_COEFF     - load balancing loss coefficient (default: 0.01)
    MOE_ZLOSS_COEFF   - router z-loss coefficient (default: 0.001)
    MOE_EXPERT_HIDDEN - hidden dim per expert (default: auto = mlp_mult*dim/num_experts)
    MOE_EVERY_N       - apply MoE every N layers, dense MLP otherwise (default: 1 = all MoE)
    MOE_SHARED_EXPERT - include a shared expert always active (default: 0)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from collections import defaultdict

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False


# ---- Import everything we need from the base training script ----
# We import the base script and reuse all infrastructure (optimizer, data,
# quantization, eval). Only the model architecture changes.
from sota_train_gpt import (
    Hyperparameters,
    Muon,
    MUON_USE_CUDA_GRAPH,
    CastedLinear,
    CausalSelfAttention,
    RMSNorm,
    Rotary,
    apply_rotary_emb,
    restore_low_dim_params_to_fp32,
    apply_qat_preset_alignment,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    load_validation_tokens,
    build_sentencepiece_luts,
    eval_val,
    eval_val_sliding_window,
    DistributedTokenLoader,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT5_MLP_ENABLED,
    INT6_QUANT_RANGE,
    INT8_QUANT_RANGE,
    QUANT_PRESET,
    zeropower_via_newtonschulz5,
    batched_newton_schulz,
    _zeropower_orig,
    _batched_ns_orig,
)
import sota_train_gpt


# =====================================================================
# MoE CONFIGURATION
# =====================================================================

MOE_NUM_EXPERTS = int(os.environ.get("MOE_NUM_EXPERTS", 4))
MOE_TOP_K = int(os.environ.get("MOE_TOP_K", 2))
MOE_AUX_COEFF = float(os.environ.get("MOE_AUX_COEFF", 0.01))
MOE_ZLOSS_COEFF = float(os.environ.get("MOE_ZLOSS_COEFF", 0.001))
MOE_EXPERT_HIDDEN = int(os.environ.get("MOE_EXPERT_HIDDEN", 0))  # 0 = auto
MOE_EVERY_N = int(os.environ.get("MOE_EVERY_N", 1))  # 1 = all layers are MoE
MOE_SHARED_EXPERT = bool(int(os.environ.get("MOE_SHARED_EXPERT", 0)))


# =====================================================================
# MoE MLP MODULE
# =====================================================================

class ExpertMLP(nn.Module):
    """Single expert: a small relu^2 MLP, same activation as the dense MLP."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        if INT5_MLP_ENABLED:
            self.fc._int5_quant = True
            self.proj._int5_quant = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class MoEMLP(nn.Module):
    """Mixture-of-Experts MLP layer.

    Replaces the dense MLP with N expert MLPs and a learned router.
    Each token is routed to top-K experts; outputs are combined by
    router probabilities.

    Architecture choices:
      - Softmax router (simple linear -> softmax -> top-k)
      - Optional noise injection for exploration during training
      - Auxiliary load-balancing loss (Switch Transformer style)
      - Optional router z-loss for logit stability
      - Optional shared expert (DeepSeek-style, always active)

    The forward returns (output, aux_loss) so the training loop can
    add the auxiliary loss to the main CE loss.
    """

    def __init__(
        self,
        dim: int,
        expert_hidden: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_coeff: float = 0.01,
        zloss_coeff: float = 0.001,
        use_shared_expert: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_coeff = aux_coeff
        self.zloss_coeff = zloss_coeff
        self.use_shared_expert = use_shared_expert

        # Router: linear projection from token dim to num_experts logits.
        # Small enough (dim * num_experts) that it's a rounding error for params
        # and quantizes trivially.
        self.router = nn.Linear(dim, num_experts, bias=False)
        # Initialize router near-uniform to avoid early collapse
        nn.init.normal_(self.router.weight, std=0.01)

        # Expert MLPs
        self.experts = nn.ModuleList([
            ExpertMLP(dim, expert_hidden) for _ in range(num_experts)
        ])

        # Optional shared expert (always active for every token)
        self.shared_expert: ExpertMLP | None = None
        if use_shared_expert:
            self.shared_expert = ExpertMLP(dim, expert_hidden)
            self.shared_gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            output: [batch, seq_len, dim]
            aux_loss: scalar tensor (load balancing + z-loss)
        """
        bsz, seq_len, dim = x.shape
        # Flatten to [num_tokens, dim] for routing
        x_flat = x.reshape(-1, dim)  # [T, dim] where T = bsz * seq_len
        T = x_flat.shape[0]

        # ---- Router ----
        router_logits = self.router(x_flat.float())  # [T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [T, num_experts]

        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize top-k probs to sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Compute expert outputs ----
        # For small num_experts (4-8), a simple loop is efficient enough and avoids
        # the complexity of scatter/gather. Each expert processes only its assigned
        # tokens, then results are combined.
        output = torch.zeros_like(x_flat)  # [T, dim]

        for expert_idx in range(self.num_experts):
            # Find which tokens selected this expert in their top-k
            # expert_mask: [T, top_k] boolean
            expert_mask = (top_k_indices == expert_idx)  # [T, top_k]
            # token_mask: [T] boolean - True if this token routes to expert_idx
            token_mask = expert_mask.any(dim=-1)  # [T]

            if not token_mask.any():
                continue

            # Get the routing weight for this expert (could be in any of the top_k slots)
            # For each token that selected this expert, get its weight
            # expert_mask is [T, top_k], top_k_probs is [T, top_k]
            expert_weight = (expert_mask.float() * top_k_probs).sum(dim=-1)  # [T]

            # Select tokens for this expert
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]  # [n_tokens, dim]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)  # [n_tokens, dim]

            # Weight by routing probability and scatter back
            weights = expert_weight[token_indices].unsqueeze(-1).to(expert_output.dtype)
            output[token_indices] += weights * expert_output

        # ---- Shared expert (always active) ----
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            gate = torch.sigmoid(self.shared_gate).to(dtype=shared_out.dtype)[None, :]
            output = (1.0 - gate) * output + gate * shared_out

        # ---- Auxiliary losses ----
        aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        if self.training:
            # Load balancing loss (Switch Transformer Eq. 4-6):
            #   L_balance = alpha * N * sum(f_i * P_i)
            # where f_i = fraction of tokens routed to expert i
            #       P_i = mean router probability for expert i
            with torch.no_grad():
                # f_i: fraction of tokens dispatched to each expert
                one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts)  # [T, top_k, N]
                one_hot = one_hot.float().sum(dim=1)  # [T, N] - count per token (can be >1 if top_k>1)
                tokens_per_expert = one_hot.mean(dim=0)  # [N]

            # P_i: mean router probability per expert (this has gradients!)
            prob_per_expert = router_probs.mean(dim=0)  # [N]

            balance_loss = self.aux_coeff * self.num_experts * (tokens_per_expert * prob_per_expert).sum()
            aux_loss = aux_loss + balance_loss

            # Router z-loss: penalize large logit magnitudes for stability
            if self.zloss_coeff > 0:
                z_loss = self.zloss_coeff * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
                aux_loss = aux_loss + z_loss

        output = output.reshape(bsz, seq_len, dim)
        return output, aux_loss


# =====================================================================
# MoE BLOCK (replaces Block from base script)
# =====================================================================

class MoEBlock(nn.Module):
    """Transformer block with MoE MLP instead of dense MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        aux_coeff: float,
        zloss_coeff: float,
        rope_base: float,
        qk_gain_init: float,
        use_shared_expert: bool = False,
        rope_dims: int = 0,
        ln_scale: float = 1.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         rope_dims=rope_dims)
        self.moe = MoEMLP(
            dim=dim,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            aux_coeff=aux_coeff,
            zloss_coeff=zloss_coeff,
            use_shared_expert=use_shared_expert,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = ln_scale
        self.is_moe = True  # tag for optimizer grouping

    def forward(self, x: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out, aux_loss = self.moe(self.mlp_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        if self.ln_scale != 1.0:
            x = x * self.ln_scale
        return x, aux_loss


class DenseBlock(nn.Module):
    """Standard transformer block with dense MLP (for interleaving with MoE)."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        ln_scale: float = 1.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         rope_dims=rope_dims)
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        if INT5_MLP_ENABLED:
            self.fc._int5_quant = True
            self.proj._int5_quant = True
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = ln_scale
        self.is_moe = False

    def forward(self, x: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_x = torch.relu(self.fc(self.mlp_norm(x)))
        mlp_out = self.proj(mlp_x.square())
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        if self.ln_scale != 1.0:
            x = x * self.ln_scale
        zero = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x, zero


# =====================================================================
# MoE GPT MODEL
# =====================================================================

class MoEGPT(nn.Module):
    """GPT with Mixture-of-Experts MLP layers.

    Identical to the base GPT except:
    1. MLP layers are replaced with MoE layers (every MOE_EVERY_N layers)
    2. forward() returns (loss, aux_loss) instead of just loss
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_smeargate: bool,
        bigram_hash_buckets: int,
        bigram_hash_dim: int,
        smear_gate_init: float,
        num_experts: int = 4,
        top_k: int = 2,
        aux_coeff: float = 0.01,
        zloss_coeff: float = 0.001,
        expert_hidden: int = 0,
        moe_every_n: int = 1,
        use_shared_expert: bool = False,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale_enabled: bool = True,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.eval_temperature = 1.0
        self.use_smeargate = use_smeargate
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash_buckets = bigram_hash_buckets
        self.bigram_hash_emb = nn.Embedding(bigram_hash_buckets, bigram_hash_dim)
        self.bigram_hash_proj = CastedLinear(bigram_hash_dim, model_dim, bias=False)
        self.bigram_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.smear_gate = nn.Parameter(torch.full((model_dim,), smear_gate_init, dtype=torch.float32))

        # Auto-compute expert hidden dim: same total MLP params as dense
        if expert_hidden <= 0:
            expert_hidden = (mlp_mult * model_dim) // num_experts
            # Round to multiple of 8 for efficiency
            expert_hidden = max((expert_hidden // 8) * 8, 8)

        # U-Net skip connections
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Build blocks: MoE or Dense depending on moe_every_n
        blocks: list[nn.Module] = []
        for i in range(num_layers):
            ln_scale = 1.0 / math.sqrt(i + 1) if ln_scale_enabled else 1.0
            is_moe_layer = (i % moe_every_n == 0)  # layer 0, N, 2N, ... are MoE

            if is_moe_layer:
                blocks.append(MoEBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    expert_hidden=expert_hidden,
                    num_experts=num_experts,
                    top_k=top_k,
                    aux_coeff=aux_coeff,
                    zloss_coeff=zloss_coeff,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    use_shared_expert=use_shared_expert,
                    rope_dims=rope_dims,
                    ln_scale=ln_scale,
                ))
            else:
                blocks.append(DenseBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    rope_dims=rope_dims,
                    ln_scale=ln_scale,
                ))

        self.blocks = nn.ModuleList(blocks)

        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self.expert_hidden = expert_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self._init_weights()

    def _init_weights(self) -> None:
        num_layers = len(self.blocks)
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.bigram_hash_emb is not None:
            nn.init.zeros_(self.bigram_hash_emb.weight)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip router -- already initialized with small std
                if "router" in name:
                    continue
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name:
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if not self.use_smeargate:
            return x
        if self.bigram_hash_emb is None or self.bigram_hash_proj is None or self.smear_gate is None:
            raise RuntimeError("SmearGate/BigramHash requested but not initialized")
        prev_ids = torch.cat((input_ids[:, :1], input_ids[:, :-1]), dim=1)
        prev_x = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        pair_hash = (prev_ids.to(torch.int64) * 36313 ^ input_ids.to(torch.int64) * 27191) % self.bigram_hash_buckets
        bigram_x = self.bigram_scale * self.bigram_hash_proj(self.bigram_hash_emb(pair_hash))
        gate = torch.sigmoid(self.smear_gate).to(dtype=x.dtype)[None, None, :]
        return (1.0 - gate) * (x + bigram_x) + gate * prev_x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        moe_count = 0

        for i in range(self.num_encoder_layers):
            x, aux = self.blocks[i](x, x0)
            total_aux_loss = total_aux_loss + aux
            if getattr(self.blocks[i], 'is_moe', False):
                moe_count += 1
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, aux = self.blocks[self.num_encoder_layers + i](x, x0)
            total_aux_loss = total_aux_loss + aux
            if getattr(self.blocks[self.num_encoder_layers + i], 'is_moe', False):
                moe_count += 1

        # Average aux loss across MoE layers
        if moe_count > 0:
            total_aux_loss = total_aux_loss / moe_count

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if not self.training and self.eval_temperature != 1.0:
            logits = logits / self.eval_temperature
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        # During training, add aux loss. During eval, return pure CE loss.
        if self.training:
            return ce_loss + total_aux_loss
        return ce_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss. Used for sliding window eval."""
        x = self.embed_tokens(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x, _ = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, _ = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# =====================================================================
# TRAINING (mirrors main() from sota_train_gpt.py with MoE model)
# =====================================================================

def main() -> None:
    # QAT_ACTIVE lives in sota_train_gpt module; CastedLinear.forward() reads it from there.
    # We toggle it via sota_train_gpt.QAT_ACTIVE in the training loop below.

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    if not MUON_USE_CUDA_GRAPH:
        sota_train_gpt.zeropower_via_newtonschulz5 = torch.compile(sota_train_gpt.zeropower_via_newtonschulz5)
        sota_train_gpt.batched_newton_schulz = torch.compile(sota_train_gpt.batched_newton_schulz)

    # ---- Distributed + CUDA setup ----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # ---- Tokenizer + validation setup ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # ---- MoE Model ----
    expert_hidden = MOE_EXPERT_HIDDEN
    if expert_hidden <= 0:
        expert_hidden = (args.mlp_mult * args.model_dim) // MOE_NUM_EXPERTS
        expert_hidden = max((expert_hidden // 8) * 8, 8)

    log0(f"moe_config: num_experts={MOE_NUM_EXPERTS} top_k={MOE_TOP_K} "
         f"expert_hidden={expert_hidden} aux_coeff={MOE_AUX_COEFF} "
         f"zloss_coeff={MOE_ZLOSS_COEFF} every_n={MOE_EVERY_N} "
         f"shared_expert={int(MOE_SHARED_EXPERT)}")

    base_model = MoEGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_smeargate=args.use_smeargate,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
        smear_gate_init=args.smear_gate_init,
        num_experts=MOE_NUM_EXPERTS,
        top_k=MOE_TOP_K,
        aux_coeff=MOE_AUX_COEFF,
        zloss_coeff=MOE_ZLOSS_COEFF,
        expert_hidden=expert_hidden,
        moe_every_n=MOE_EVERY_N,
        use_shared_expert=MOE_SHARED_EXPERT,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale_enabled=args.ln_scale_enabled,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    apply_qat_preset_alignment(base_model, args.num_layers)

    use_compile = bool(int(os.environ.get("TORCH_COMPILE", "1")))
    compile_mode = os.environ.get("COMPILE_MODE", "default")
    compile_dynamic = bool(int(os.environ.get("COMPILE_DYNAMIC", "0")))
    compiled_model = torch.compile(base_model, dynamic=compile_dynamic, mode=compile_mode) if use_compile else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # ---- Optimizer setup ----
    # Router weights are tiny (dim x num_experts = 512x4) and non-square, so
    # Newton-Schulz in Muon is suboptimal. Route them to Adam instead.
    # We also keep shared_gate (1D) in scalar params as usual.
    block_named_params = list(base_model.blocks.named_parameters())
    router_patterns = ("router.weight",)
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and not any(rp in name for rp in router_patterns)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        or any(rp in name for rp in router_patterns)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_params = [base_model.tok_emb.weight]
    if base_model.bigram_hash_emb is not None:
        token_params.append(base_model.bigram_hash_emb.weight)
    if base_model.bigram_hash_proj is not None:
        matrix_params.append(base_model.bigram_hash_proj.weight)
    if base_model.smear_gate is not None:
        scalar_params.append(base_model.smear_gate)
    if base_model.bigram_scale is not None:
        scalar_params.append(base_model.bigram_scale)

    # Router weights are small (dim x num_experts = 512x4 = 2048 params).
    # They're already included in matrix_params via the block_named_params sweep.
    # Muon works fine for these since it's a 2D weight.

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_moe_layers = sum(1 for b in base_model.blocks if getattr(b, 'is_moe', False))
    n_dense_layers = len(base_model.blocks) - n_moe_layers
    log0(f"model_params:{n_params}")
    log0(f"model_type:MoE num_experts:{MOE_NUM_EXPERTS} top_k:{MOE_TOP_K} "
         f"expert_hidden:{expert_hidden} moe_layers:{n_moe_layers} dense_layers:{n_dense_layers}")
    log0(
        f"model_shape:num_layers:{args.num_layers} model_dim:{args.model_dim} "
        f"mlp_mult:{args.mlp_mult} vocab_size:{args.vocab_size}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    log0(f"ema:enabled:{int(args.ema_enabled)} decay:{args.ema_decay}")

    # ---- Data loader & warmup ----
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            is_muon = isinstance(opt, Muon)
            opt.zero_grad(set_to_none=not is_muon)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- Main training loop ----
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {k: v.detach().cpu().float().clone() for k, v in base_model.state_dict().items()}

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        if args.qat_start_frac < 1.0 and max_wallclock_ms is not None:
            sota_train_gpt.QAT_ACTIVE = (elapsed_ms / max_wallclock_ms) >= args.qat_start_frac
        elif args.qat_start_frac >= 1.0:
            sota_train_gpt.QAT_ACTIVE = False

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if ema_state is not None:
            d = args.ema_decay
            for k, v in base_model.state_dict().items():
                ema_state[k].mul_(d).add_(v.detach().cpu().float(), alpha=1.0 - d)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if ema_state is not None:
        log0(f"ema:applying decay={args.ema_decay}")
        ema_sd = {k: v.to(dtype=base_model.state_dict()[k].dtype) for k, v in ema_state.items()}
        base_model.load_state_dict(ema_sd, strict=True)

    # ---- Serialization + roundtrip validation ----
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    if master_process:
        log0(f"quant_preset:{QUANT_PRESET}")
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
        compress_name = "zstd-22"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_name = "zlib-9"
    quant_raw_bytes = len(quant_raw)
    if master_process:
        quant_ext = "int6.ptz"
        with open(f"final_model.{quant_ext}", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(f"final_model.{quant_ext}")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int6+{compress_name}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int6+{compress_name}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    quant_ext = "int6.ptz"
    with open(f"final_model.{quant_ext}", "rb") as f:
        quant_blob_disk = f.read()
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        quant_decompressed = dctx.decompress(quant_blob_disk)
    else:
        quant_decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if args.eval_stride > 0:
        torch.cuda.synchronize()
        t_sw = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding_window(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window_eval stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_sw):.0f}ms"
        )
        log0(
            f"final_sliding_window_eval_exact stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
