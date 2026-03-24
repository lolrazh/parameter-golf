"""Test DuQuant (Hadamard rotation before quantization) on a raw checkpoint.
Compares quant MSE and artifact size with/without rotation."""
import sys, os, io, time, zlib, math
import torch

sys.path.insert(0, os.path.dirname(__file__))

try:
    import zstandard as zstd
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False

from frontier_512 import (
    quantize_state_dict_int8, dequantize_state_dict_int8,
    _quant_bits_for_name, _block_hadamard_rotate,
)

def measure_quant_error(state_dict, quant_preset, label):
    """Quantize, dequantize, measure MSE per tensor and artifact size."""
    os.environ["QUANT_PRESET"] = quant_preset
    quant_obj, stats = quantize_state_dict_int8(state_dict, quant_preset=quant_preset)
    deq = dequantize_state_dict_int8(quant_obj)

    total_mse = 0.0
    total_numel = 0
    for name in deq:
        if name in state_dict:
            orig = state_dict[name].float()
            recon = deq[name].float()
            if orig.shape == recon.shape:
                mse = ((orig - recon) ** 2).mean().item()
                total_mse += mse * orig.numel()
                total_numel += orig.numel()

    avg_mse = total_mse / max(total_numel, 1)

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    if HAVE_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = len(cctx.compress(raw))
    else:
        compressed = len(zlib.compress(raw, level=9))

    code_bytes = os.path.getsize(os.path.join(os.path.dirname(__file__), "frontier_512.py"))
    total = compressed + code_bytes

    print(f"  [{label}] avg_MSE: {avg_mse:.8f}  artifact: {total/1e6:.2f}MB  raw: {len(raw)/1e6:.2f}MB")
    return avg_mse, total

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "final_model_raw.pt"
    print(f"Loading {ckpt}...")
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"Loaded {len(state_dict)} tensors")

    preset = os.environ.get("QUANT_PRESET", "front3_back1_6_middle5")

    # Test without DuQuant
    os.environ["DUQUANT"] = "0"
    # Re-import to pick up env change
    import frontier_512
    frontier_512.DUQUANT = False
    print(f"\nPreset: {preset}")
    mse_off, size_off = measure_quant_error(state_dict, preset, "no rotation")

    # Test with DuQuant
    frontier_512.DUQUANT = True
    frontier_512.DUQUANT_BLOCK = 512
    mse_on, size_on = measure_quant_error(state_dict, preset, "hadamard-512")

    frontier_512.DUQUANT_BLOCK = 128
    mse_128, size_128 = measure_quant_error(state_dict, preset, "hadamard-128")

    frontier_512.DUQUANT_BLOCK = 64
    mse_64, size_64 = measure_quant_error(state_dict, preset, "hadamard-64")

    print(f"\nMSE reduction (512): {(1 - mse_on/mse_off)*100:.2f}%")
    print(f"MSE reduction (128): {(1 - mse_128/mse_off)*100:.2f}%")
    print(f"MSE reduction (64):  {(1 - mse_64/mse_off)*100:.2f}%")
    print(f"Size delta (512): {(size_on - size_off)/1e3:.1f}KB")
    print(f"Size delta (128): {(size_128 - size_off)/1e3:.1f}KB")
    print(f"Size delta (64):  {(size_64 - size_off)/1e3:.1f}KB")

if __name__ == "__main__":
    main()
