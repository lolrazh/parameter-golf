# Overlooked Directions GPU Queue

This is the queue for the "directions we have not really considered yet" pass.

The key point: these are **not** all one linear stack.
Some should be tested as isolated forks from the same control, while a few are only worth stacking after an earlier result wins.

## Control

Always refresh the same control before a new branch family:

- Use the current best stable stack in `sota_train_gpt.py`
- Keep `EVAL_STRIDE=64`
- Keep the current quant/export path fixed unless the experiment is explicitly about quantization
- Use the same wallclock budget for all runs in a branch family

Suggested control run name:

```bash
control_overlooked_<date>_<budget>
```

## Queue Shape

### Branch A: Quantization-aligned training

These are the most directly actionable because they target the current mismatch between training-time fake quant and export-time mixed precision presets.

| order | run id template | depends on | stack on previous? | purpose | success signal |
|---|---|---|---|---|---|
| A0 | `ctrl_qat_align_<budget>` | none | no | fresh control for this branch | reference post-quant and sliding BPB |
| A1 | `qat_preset_align_<budget>` | code | no, fork from A0 | make training STE use the same layer-sensitive quant ranges as export | lower post-quant gap without hurting pre-quant badly |
| A2 | `qat_preset_align_fp16tok_<budget>` | A1 | maybe | check whether preset-aligned QAT changes the value of fp16 embedding passthrough | better final BPB or smaller quant gap |
| A3 | `qat_preset_align_int5mlp_<budget>` | A1 | maybe | re-evaluate int5 MLP after aligned QAT | artifact savings at lower BPB cost |

Why first:

- Your best exporter is explicitly front-heavy.
- Training STE is still mostly uniform.
- This is the cleanest place where the logs imply "the model is learning against the wrong noise."

### Branch B: Local memory without dynamic shapes

These explore the "human memory" intuition without running into the known `torch.compile(dynamic=False)` trap.

| order | run id template | depends on | stack on previous? | purpose | success signal |
|---|---|---|---|---|---|
| B0 | `ctrl_memory_<budget>` | none | no | fresh control for this branch | reference throughput and BPB |
| B1 | `fixedshape_memory_train_<budget>` | code | no, fork from B0 | short query chunk plus fixed memory state or KV carry, keeping tensor shapes static | better BPB at similar or better steps |
| B2 | `fixedshape_memory_train_sw_<budget>` | B1 | yes | confirm the memory idea still helps under sliding-window eval | better sliding BPB |

Do not stack this on top of unrelated branch wins first.
This changes the training distribution enough that it deserves its own clean control.

### Branch C: N-gram and cache style heads

These are the cheapest "predict the next token better using more immediate structure" experiments.

| order | run id template | depends on | stack on previous? | purpose | success signal |
|---|---|---|---|---|---|
| C0 | `ctrl_ngram_<budget>` | none | no | fresh control for this branch | reference BPB |
| C1 | `trigram_expert_<budget>` | code | no, fork from C0 | add a tiny hashed 2-token-context residual expert aimed at next-token logits | better post-quant BPB |
| C2 | `trigram_expert_smear_<budget>` | C1 | maybe | test whether the trigram head complements SmearGate/BigramHash instead of duplicating it | incremental gain over C1 |
| C3 | `neural_cache_eval_<budget>` | code | no, fork from C0 | eval-only interpolation with a short hidden-state or logits cache | lower sliding BPB with zero train cost |
| C4 | `neural_cache_eval_trigram_<budget>` | C1 or C3 | only if one side wins cleanly | combine the better train-time local expert with eval-time cache mixing | additive gain |

Important:

- `trigram_expert` and `neural_cache_eval` should start from the same control.
- They attack different failure modes and can be combined later, but should not be introduced together first.

### Branch D: Output-side calibration

Global temperature scaling already failed. That does **not** rule out small structured calibration.

| order | run id template | depends on | stack on previous? | purpose | success signal |
|---|---|---|---|---|---|
| D0 | `ctrl_calib_<budget>` | none | no | fresh control for this branch | reference final BPB |
| D1 | `tokenclass_calib_eval_<budget>` | code | no, fork from D0 | post-quant eval-only calibration by token class: boundary, leading-space, byte-length, maybe frequency bucket | lower final BPB with zero train cost |
| D2 | `tokenclass_calib_cache_<budget>` | D1 or C3 | only if D1 wins | combine calibration with eval-time cache mixing | additive eval-only gain |

This branch is cheap and low-risk.
If it works, it is the kind of thing that stacks well later.

### Branch E: Non-uniform capacity

Do this only after the quantization-aligned branch, because the quant probe already showed the front of the model matters more.

| order | run id template | depends on | stack on previous? | purpose | success signal |
|---|---|---|---|---|---|
| E0 | `ctrl_asymcap_<budget>` | none | no | fresh control for this branch | reference BPB and artifact size |
| E1 | `frontheavy_capacity_<budget>` | code | no, fork from E0 | spend more capacity in the first 2-3 blocks, slim the middle | better BPB at same artifact budget |
| E2 | `frontheavy_capacity_qatalign_<budget>` | A1 | yes, but only after A1 proves out | combine non-uniform capacity with preset-aligned QAT | better final compressed-model BPB |

This is the most architecture-heavy branch.
Treat it as a second-wave move, not the first experiment to queue.

## What Can Be Prepared Now

### Queueable after code lands

- Branch A can become a clean three-run mini-sweep
- Branch C can become a clean control plus two isolated forks
- Branch D is almost purely eval-side and should be easy to queue

### Should not be blindly pre-stacked

- Branch B fixed-shape memory
- Branch E front-heavy capacity

Those are real model changes, not just toggles.
They should start from a refreshed control and be judged on their own.

## Recommended Actual Order

If GPU time is scarce, run in this order:

1. A0 control
2. A1 preset-aligned QAT
3. C0 control
4. C1 trigram expert
5. C3 neural cache eval
6. D1 token-class calibration
7. B0 control
8. B1 fixed-shape memory
9. E0 control
10. E1 front-heavy capacity

## Kill Criteria

Use aggressive kill rules so the queue does not waste GPU hours:

- Kill any train-time idea that is clearly slower and not ahead on the first intermediate validation.
- Kill any eval-only idea that adds substantial eval time and gains less than roughly `0.003-0.005` BPB.
- Kill any architecture idea that worsens both steps reached and post-quant BPB.

## Bottom Line

Yes, the experiments can be made "ready," but they should be prepared as **five branch families**, not one blind stack.

The best immediate build-first target is:

1. preset-aligned QAT
2. trigram expert
3. eval-time neural cache
4. token-class calibration

Those are the ones most worth turning into runnable scripts first.
