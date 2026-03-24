"""Sweep quant presets + compression on an existing raw checkpoint.
Usage: python3 quant_sweep.py [--checkpoint final_model.pt] [--num-layers 11]
"""
import io, os, sys, time, lzma, zlib
import torch
sys.path.insert(0, os.path.dirname(__file__))

try:
    import zstandard as zstd
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False

from frontier_512 import quantize_state_dict_int8, _quant_bits_for_name

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "final_model.pt"
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    os.environ["NUM_LAYERS"] = str(num_layers)

    print(f"Loading {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(state_dict)} tensors")

    # Estimate code size (read frontier_512.py)
    code_path = os.path.join(os.path.dirname(__file__), "frontier_512.py")
    code_bytes = os.path.getsize(code_path) if os.path.exists(code_path) else 50000

    presets = [
        "int5mlp",
        "uniform_int6",
        "front3_back1_6_middle5",
        "front3_back1_7_middle5",
        "front3_back1_7_middle6",
        "front3_back1_8_middle6",
    ]

    print(f"\n{'Preset':<28} {'zstd-22':>12} {'lzma-9e':>12} {'zlib-9':>12} {'quant_bits':>10}")
    print("-" * 80)

    for preset in presets:
        # Show bit assignments for this preset
        sample_bits = {}
        for name in state_dict:
            if "blocks" in name and state_dict[name].ndim == 2 and state_dict[name].numel() > 65536:
                bits = _quant_bits_for_name(name, preset)
                layer_type = "sensitive" if any(f"blocks.{i}." in name for i in list(range(3)) + [num_layers - 1]) else "middle"
                sample_bits[layer_type] = bits

        os.environ["QUANT_PRESET"] = preset
        quant_obj, stats = quantize_state_dict_int8(state_dict, quant_preset=preset)

        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        raw = buf.getvalue()

        # Compress with all methods
        sizes = {}
        if HAVE_ZSTD:
            cctx = zstd.ZstdCompressor(level=22)
            sizes["zstd-22"] = len(cctx.compress(raw))

        sizes["lzma-9e"] = len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
        sizes["zlib-9"] = len(zlib.compress(raw, level=9))

        bits_str = f"s:{sample_bits.get('sensitive','?')} m:{sample_bits.get('middle','?')}"

        zstd_total = sizes.get("zstd-22", 0) + code_bytes
        lzma_total = sizes["lzma-9e"] + code_bytes
        zlib_total = sizes["zlib-9"] + code_bytes

        def fmt(total):
            marker = " OVER" if total > 16_000_000 else " ok"
            return f"{total/1e6:.2f}MB{marker}"

        print(f"{preset:<28} {fmt(zstd_total):>12} {fmt(lzma_total):>12} {fmt(zlib_total):>12} {bits_str:>10}")

    print(f"\n(code_bytes={code_bytes})")

if __name__ == "__main__":
    main()
