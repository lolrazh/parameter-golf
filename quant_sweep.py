"""
Checkpoint-only quantization sweep for Parameter Golf.

This script does not retrain. It loads a saved raw checkpoint, evaluates the
pre-quant baseline once, then applies a set of post-training quantization
recipes to the same weights and reports the BPB/size tradeoff for each recipe.

Typical flow:
1. Save a raw checkpoint from Modal:
   modal run train_modal.py --run-id clean_9x512_5m --max-wallclock 300 \
       --overrides 'NUM_LAYERS=9,MODEL_DIM=512,EVAL_STRIDE=0' --save-checkpoint
2. Sweep quantization recipes on that checkpoint:
   python3 quant_sweep.py \
       --checkpoint-path checkpoints/clean_9x512_5m.pt \
       --overrides 'NUM_LAYERS=9,MODEL_DIM=512'

For cheap sweeps, the default validation budget is 1M tokens. Pass
`--val-tokens-limit 0` to score on the full validation set.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


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


@dataclass(frozen=True)
class QuantRecipe:
    name: str
    description: str
    keep_float_patterns: tuple[str, ...] = ()
    keep_fp32_patterns: tuple[str, ...] = ()
    quant_range_patterns: tuple[tuple[int, tuple[str, ...]], ...] = ()


def make_recipe_catalog(num_layers: int, control_patterns: tuple[str, ...]) -> dict[str, QuantRecipe]:
    outer_patterns = ("tok_emb", f"blocks.0.", f"blocks.{num_layers - 1}.")
    attn_patterns = ("tok_emb", ".attn.")
    catalog = {
        "current": QuantRecipe(
            name="current",
            description="Current SOTA export: int8 tok_emb, int6 block weights, fp32 control tensors.",
            keep_fp32_patterns=control_patterns,
            quant_range_patterns=((127, ("tok_emb",)),),
        ),
        "fp16_tok_emb": QuantRecipe(
            name="fp16_tok_emb",
            description="Keep tied embedding/output head in fp16, int6 block weights elsewhere.",
            keep_float_patterns=("tok_emb",),
            keep_fp32_patterns=control_patterns,
        ),
        "attn8_mlp6": QuantRecipe(
            name="attn8_mlp6",
            description="Int8 attention projections, int6 MLP weights, fp32 control tensors.",
            keep_fp32_patterns=control_patterns,
            quant_range_patterns=((127, attn_patterns),),
        ),
        "outer8_middle6": QuantRecipe(
            name="outer8_middle6",
            description="Int8 embedding plus first/last blocks, int6 middle blocks.",
            keep_fp32_patterns=control_patterns,
            quant_range_patterns=((127, outer_patterns),),
        ),
        "fp16_tok_emb_attn8": QuantRecipe(
            name="fp16_tok_emb_attn8",
            description="Fp16 tied embedding with int8 attention and int6 MLP weights.",
            keep_float_patterns=("tok_emb",),
            keep_fp32_patterns=control_patterns,
            quant_range_patterns=((127, (".attn.",)),),
        ),
    }
    return catalog


def build_probe_recipes(num_layers: int, control_patterns: tuple[str, ...]) -> list[QuantRecipe]:
    recipes: list[QuantRecipe] = []
    for idx in range(num_layers):
        recipes.append(
            QuantRecipe(
                name=f"probe_block_{idx}_int8",
                description=f"Int8 block {idx}, int6 elsewhere, tok_emb int8 baseline.",
                keep_fp32_patterns=control_patterns,
                quant_range_patterns=((127, ("tok_emb", f"blocks.{idx}.")),),
            )
        )
    return recipes


def choose_quant_range(name: str, recipe: QuantRecipe, default_range: int) -> int:
    for quant_range, patterns in recipe.quant_range_patterns:
        if any(pattern in name for pattern in patterns):
            return quant_range
    return default_range


def keep_float_tensor_recipe(
    module,
    name: str,
    tensor: torch.Tensor,
    passthrough_orig_dtypes: dict[str, str],
    recipe: QuantRecipe,
) -> torch.Tensor:
    keep_fp32_patterns = recipe.keep_fp32_patterns or module.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    if any(pattern in name for pattern in keep_fp32_patterns):
        return tensor.float().contiguous()
    if tensor.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        return tensor.to(dtype=module.INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return tensor


def quantize_state_dict_recipe(module, state_dict: dict[str, torch.Tensor], recipe: QuantRecipe) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, torch.Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += module.tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += module.tensor_nbytes(t)
            continue

        force_keep_float = any(pattern in name for pattern in recipe.keep_float_patterns)
        if force_keep_float or t.numel() <= module.INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor_recipe(module, name, t, passthrough_orig_dtypes, recipe)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += module.tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        quant_range = choose_quant_range(name, recipe, module.INT6_QUANT_RANGE)
        q, s = module.quantize_float_tensor(t, quant_range=quant_range)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += module.tensor_nbytes(q) + module.tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def serialize_quant_obj(quant_obj: dict[str, object]) -> tuple[bytes, str]:
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        return cctx.compress(quant_raw), "zstd-22"
    return zlib.compress(quant_raw, level=9), "zlib-9"


def evaluate_model(
    module,
    model: torch.nn.Module,
    eval_args: SimpleNamespace,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    grad_accum_steps: int,
) -> tuple[float, float]:
    return module.eval_val(
        eval_args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint-only quantization sweep for Parameter Golf.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to a raw PyTorch state_dict checkpoint.")
    parser.add_argument("--run-id", default=f"quant_sweep_{uuid.uuid4().hex[:8]}")
    parser.add_argument("--overrides", default="", help="Comma-separated KEY=VALUE overrides applied before importing sota_train_gpt.py.")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument(
        "--recipes",
        default="current,fp16_tok_emb,attn8_mlp6,outer8_middle6,fp16_tok_emb_attn8",
        help="Comma-separated recipe names.",
    )
    parser.add_argument("--probe-blocks", action="store_true", help="Also add one-block-at-a-time int8 probes.")
    parser.add_argument("--val-batch-size", type=int, default=524_288)
    parser.add_argument(
        "--val-tokens-limit",
        type=int,
        default=1_048_576,
        help="Validation token budget. Use 0 for the full validation set.",
    )
    parser.add_argument(
        "--eval-grad-accum-steps",
        type=int,
        default=8,
        help="Used only to mirror the eval batch partitioning from 1xH100 training.",
    )
    parser.add_argument("--code-path", default="sota_train_gpt.py", help="Code file counted toward total artifact bytes.")
    parser.add_argument("--output-path", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    for key, value in overrides.items():
        os.environ[key] = value
    os.environ.setdefault("DATA_PATH", args.data_path)
    os.environ.setdefault("TOKENIZER_PATH", args.tokenizer_path)

    import sota_train_gpt as sota

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical quant sweeps")

    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hp = sota.Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_tokens = sota.load_validation_tokens(hp.val_files, hp.train_seq_len)
    if args.val_tokens_limit > 0:
        limit = (args.val_tokens_limit // hp.train_seq_len) * hp.train_seq_len + 1
        val_tokens = val_tokens[:limit]

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = sota.build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )

    model = sota.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
    ).to(device).bfloat16()
    for submodule in model.modules():
        if isinstance(submodule, sota.CastedLinear):
            submodule.float()
    sota.restore_low_dim_params_to_fp32(model)

    checkpoint_path = Path(args.checkpoint_path)
    base_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(base_state, strict=True)

    eval_args = SimpleNamespace(train_seq_len=hp.train_seq_len, val_batch_size=args.val_batch_size)
    code_bytes = len(Path(args.code_path).read_text(encoding="utf-8").encode("utf-8"))

    print(
        f"quant_sweep_config:run_id:{args.run_id} checkpoint:{checkpoint_path} "
        f"num_layers:{hp.num_layers} model_dim:{hp.model_dim} val_tokens:{val_tokens.numel() - 1}"
    )
    pre_t0 = time.perf_counter()
    pre_val_loss, pre_val_bpb = evaluate_model(
        sota,
        model,
        eval_args,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        grad_accum_steps=args.eval_grad_accum_steps,
    )
    pre_eval_ms = 1000.0 * (time.perf_counter() - pre_t0)
    print(
        f"prequant_baseline val_loss:{pre_val_loss:.8f} val_bpb:{pre_val_bpb:.8f} "
        f"eval_time:{pre_eval_ms:.0f}ms"
    )

    catalog = make_recipe_catalog(hp.num_layers, sota.CONTROL_TENSOR_NAME_PATTERNS)
    selected_names = [name.strip() for name in args.recipes.split(",") if name.strip()]
    recipes: list[QuantRecipe] = []
    for name in selected_names:
        if name not in catalog:
            valid = ", ".join(sorted(catalog))
            raise ValueError(f"unknown recipe {name!r}. Valid recipes: {valid}")
        recipes.append(catalog[name])
    if args.probe_blocks:
        recipes.extend(build_probe_recipes(hp.num_layers, sota.CONTROL_TENSOR_NAME_PATTERNS))

    results: list[dict[str, object]] = []
    for recipe in recipes:
        t0 = time.perf_counter()
        quant_obj, quant_stats = quantize_state_dict_recipe(sota, base_state, recipe)
        quant_blob, compress_name = serialize_quant_obj(quant_obj)
        model_bytes = len(quant_blob)
        total_bytes = model_bytes + code_bytes

        dequant_state = sota.dequantize_state_dict_int8(quant_obj)
        model.load_state_dict(dequant_state, strict=True)
        q_val_loss, q_val_bpb = evaluate_model(
            sota,
            model,
            eval_args,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            grad_accum_steps=args.eval_grad_accum_steps,
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        result = {
            "recipe": recipe.name,
            "description": recipe.description,
            "val_loss": q_val_loss,
            "val_bpb": q_val_bpb,
            "delta_bpb": q_val_bpb - pre_val_bpb,
            "model_bytes": model_bytes,
            "total_bytes": total_bytes,
            "payload_bytes": quant_stats["int8_payload_bytes"],
            "compression": compress_name,
            "elapsed_ms": elapsed_ms,
        }
        results.append(result)
        print(
            f"recipe:{recipe.name} val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
            f"delta_bpb:{q_val_bpb - pre_val_bpb:.8f} model_bytes:{model_bytes} "
            f"total_bytes:{total_bytes} compression:{compress_name} elapsed_ms:{elapsed_ms:.0f}"
        )

    results.sort(key=lambda item: float(item["delta_bpb"]))
    summary = {
        "run_id": args.run_id,
        "checkpoint_path": str(checkpoint_path),
        "model_shape": {
            "num_layers": hp.num_layers,
            "model_dim": hp.model_dim,
            "mlp_mult": hp.mlp_mult,
            "num_heads": hp.num_heads,
            "num_kv_heads": hp.num_kv_heads,
            "vocab_size": hp.vocab_size,
        },
        "prequant": {
            "val_loss": pre_val_loss,
            "val_bpb": pre_val_bpb,
            "eval_ms": pre_eval_ms,
        },
        "results": results,
    }

    out_path = Path(args.output_path) if args.output_path else Path("logs") / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote_results:{out_path}")
    if results:
        best = results[0]
        print(
            f"best_recipe:{best['recipe']} val_bpb:{float(best['val_bpb']):.8f} "
            f"delta_bpb:{float(best['delta_bpb']):.8f} total_bytes:{int(best['total_bytes'])}"
        )


if __name__ == "__main__":
    main()
