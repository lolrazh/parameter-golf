# Quant Sweep Baseline On 9x512 Checkpoint

**Date:** 2026-03-20
**Agent:** Codex (GPT-5)
**Status:** ✅ Complete

## User Intention
User wanted to stop guessing about quantization and run a concrete checkpoint-only sweep on the current best short-run architecture candidate. The goal was to keep the model fixed, vary only the export recipe, and learn where the post-quant damage is actually coming from before attempting bigger-model training or MLX-style DWQ.

## What We Accomplished
- ✅ **Used the clean 5-minute `9x512` checkpoint as the control**
  - Architecture had already been provisionally narrowed to `9x512` on the trusted 2-minute pre-quant metric.
  - The new question was no longer architecture; it was quantization damage.
- ✅ **Extended the Modal harness to support checkpoint-only post-train quant sweeps**
  - `train_modal.py` can now train, save the raw checkpoint, and immediately run `quant_sweep.py` inside the same Modal container.
- ✅ **Ran the first combined 5-minute train + sweep job**
  - Run ID: `clean_9x512_5m_quant`
  - Training completed, but the sweep crashed before producing recipe results.
- ✅ **Found and fixed the sweep crash**
  - Root cause: `quant_sweep.py` called `torch.cuda.set_device(torch.device("cuda"))`.
  - In the Modal runtime, `set_device` required an explicit indexed device.
  - Fix: use `device_index = int(os.environ.get("LOCAL_RANK", "0"))`, then:
    - `device = torch.device("cuda", device_index)`
    - `torch.cuda.set_device(device_index)`
- ✅ **Reran the full 5-minute train + sweep successfully**
  - Run ID: `clean_9x512_5m_quant2`
  - Training completed cleanly.
  - Quant recipe sweep completed and reported a ranked result set.

## Final Training Result
- Run ID: `clean_9x512_5m_quant2`
- Effective config:
  - `NUM_LAYERS=9`
  - `MODEL_DIM=512`
  - `EVAL_STRIDE=0`
- Final trainer lines:
  - `step:783/20000 val_loss:2.4332 val_bpb:1.4411`
  - `Total submission size int6+zstd-22: 5033018 bytes`
  - `final_int6_roundtrip_exact val_loss:4.88408974 val_bpb:2.89263305`

## Quant Sweep Result
Important note: the sweep used the fast `1,048,576`-token validation proxy, not the full trainer validation pass. Absolute BPB values from the sweep should be compared **against each other**, not directly against the trainer's final full-val score.

### Sweep Baseline
- `prequant_baseline val_loss:2.99696299 val_bpb:1.79562058`

### Recipes Tested
- `current`
  - `val_bpb:3.03317159`
  - `delta_bpb:+1.23755101`
  - `total_bytes:5033018`
- `fp16_tok_emb`
  - `val_bpb:3.03317218`
  - `delta_bpb:+1.23755160`
  - `total_bytes:5466490`
- `attn8_mlp6`
  - `val_bpb:2.77448667`
  - `delta_bpb:+0.97886609`
  - `total_bytes:6814427`
- `outer8_middle6`
  - `val_bpb:2.57339223`
  - `delta_bpb:+0.77777165`
  - `total_bytes:6194293`
- `fp16_tok_emb_attn8`
  - `val_bpb:2.77397697`
  - `delta_bpb:+0.97835639`
  - `total_bytes:7254647`

### Best Recipe
- `outer8_middle6`
  - Best sweep score: `2.57339223 val_bpb`
  - Best sweep delta: `+0.77777165 BPB`
  - Total artifact bytes: `6,194,293`

## Key Discoveries
1. **The current int6 export is leaving a lot on the table**
   - On the sweep proxy metric, the best tested recipe beat the current recipe by about `0.4598 BPB`.
2. **Outer layers are much more quant-sensitive than the tied embedding**
   - Protecting the embedding in fp16 alone did essentially nothing in this fork.
   - Protecting the outer blocks with int8 helped a lot.
3. **Attention protection helps, but outer-block protection helps more**
   - `attn8_mlp6` improved over current.
   - `outer8_middle6` improved more.
4. **The artifact budget is not the immediate constraint**
   - Even the best and safer recipes tested were still far below the `16,000,000` byte cap.
   - Right now the binding problem is quality loss from quantization, not size.

## Bugs & Issues
1. **Quant sweep CUDA device bug**
   - First sweep failed before any recipe results were produced.
   - Fix was narrow and mechanical: explicit indexed CUDA device selection.
2. **Metric mismatch requires discipline**
   - The trainer’s final eval and the sweep’s proxy eval are not directly comparable.
   - Sweep numbers are valid for **ranking recipes on the same checkpoint**, not for replacing the trainer’s final number.

## Key Learnings
- **Checkpoint-only quant sweeps are the right cheap loop right now**
  - They isolate compression quality from architecture and training noise.
- **This repo’s quant pain is not primarily in the embedding**
  - The leaderboard hints about fp16 tied embeddings are not universally dominant.
  - In this specific model, outer transformer blocks appear more sensitive.
- **Mixed quant should be driven by sensitivity, not aesthetics**
  - The first useful mixed recipe came from a structural hypothesis: protect the edges, compress the middle.
- **We do not need DWQ yet to make progress**
  - A small number of targeted PTQ recipes already produced meaningful gains.
  - DWQ can come later as a finer-grained version of the same idea.

## Files Modified
- `quant_sweep.py`
  - Fixed CUDA device selection so checkpoint-only sweeps run correctly inside Modal.

## Ready for Next Session
- ✅ We now have a working checkpoint-only quant sweep loop.
- ✅ `9x512` remains the best current control architecture.
- ✅ The best tested recipe is `outer8_middle6`.
- 🔧 Next quant sweep should zoom in on outer-layer protection:
  - first 2 / last 2 blocks at int8
  - outer-block attention-only int8
  - possibly stronger compression in the middle layers
- 🔧 Do not jump to bigger-model training yet.
  - The current quantization path still has too much loss.
  - We should reduce that gap further before paying for more capacity.

## Context for Future
This session moved the project from "quantization is bad" to "this is where quantization is bad." That is a real change in state. The architecture question is currently secondary. The important new fact is that coarse sensitivity structure is visible already: outer blocks matter, the embedding alone does not, and the artifact budget is generous enough to spend extra bits where they buy quality. The next step should stay boring and local: refine mixed PTQ around the outer layers, then reconsider DWQ or QAT only if those easy wins plateau.
