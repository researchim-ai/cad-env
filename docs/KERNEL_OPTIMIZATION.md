# Numeric SFT and Liger investigation

The original modern SFT uses `NumericCompletionLoss` via `compute_loss_func`.
In installed TRL 1.17.1 this runs the model once for the differentiable full
logits and again under `no_grad` for fused-head metrics. The first pass also
materializes logits for ignored prompt/image positions. Effective batch eight
in the original run is microbatch one with eight accumulation passes.

`fused_numeric.py` instead consumes TRL's differentiable per-target log
probabilities from one fused-head forward. It preserves causal shifting,
completion/image masks, ASCII numeric token weight eight and division by the
accumulation-window supervised-token count. It also preserves token/entropy
diagnostics. It rejects alternate shifted labels, output-based eval, MoE and
multi-GPU; these need separate validation. Native policy evaluation is unchanged.

`train_optimized` is a separate opt-in entry point; the original trainer is not
modified. It reuses the original data audits, supervision checks, LoRA targets,
gradient guards, strict checkpoint loading and wall-time callback. A local
process-only trainer substitution enables the alternate loss. The runtime
choices and source hashes are saved in `optimized-runtime.json` after success.

Liger 0.8.5 supports Qwen3.5 dense multimodal models. Here only text RMSNorm
and SwiGLU are patched on the existing model; LoRA projections remain intact.
RoPE and Liger fused linear CE are explicitly disabled. The latter replaces
the forward and conflicts with the TRL fused-head contract; standard unweighted
CE would also not preserve our weighted objective. Package/hash are pinned in
`requirements-kernels-probe.lock`, downloaded and extracted under
`artifacts/runtimes/liger-study/package`. No active environment dependencies
were upgraded. Add that directory and the repository to `PYTHONPATH` to use
the isolated optional package.

## Evidence and scope

- Unit tests compare loss and all logits gradients, ignored prompt/image
  positions and a shared accumulation denominator. A trainer-path test checks
  that exactly one forward is called.
- `kernel_probe` constructs a tiny real Qwen3.5 VLM with hybrid attention,
  image tensors and visual/language LoRA. CPU FP32 loss matches the reference;
  GPU BF16 with Liger also passed adapter-gradient checks. This does not measure
  throughput or establish correctness for every large-model input shape.
- `kernel_benchmark` runs four bounded same-seed, same-data 4B probes on GPU 1:
  original, fused numeric, fused numeric plus Liger and the latter with
  microbatch two/accumulation four. Each processes 32 examples in four optimizer
  steps. The first step is excluded from the warm timing median. First-update
  clipped adapter gradients are stored for comparison. The complete suite has
  a 20-minute deadline; failing variants are recorded rather than retried.
- `kernel_results` requires relative loss error below 1%, overall gradient L2
  error below 3%, visual gradient L2 error below 5% and speedup above 10% before
  recommending a profile. Batch two additionally needs peak allocated memory
  at most 20.5 GiB. These are explicit BF16 pilot gates, not exact equivalence
  or full-corpus memory guarantees. Four steps are a short throughput probe.

The previous long run was stopped at approximately step 110 by user request.
Its durable checkpoint 100 contains adapters, optimizer, scheduler and RNG
state. A subsequent optimized run should resume that checkpoint with the same
4,000-row mixture and global 500-step target, preserving effective batch eight
and the remaining time budget. It must pass strict resume and gradient guards.
Faster kernels are not evidence of higher CAD success; native held-out
validation and image controls remain necessary.

Sources: [Liger implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py),
[TRL SFT documentation](https://huggingface.co/docs/trl/main/sft_trainer).
The installed source and downloaded wheel, rather than mutable main-branch
documentation, determine the behavior measured here.

## Measured 4B outcome, 2026-11-06

`artifacts/kernel-benchmark-v1/comparison.json` records the six probes. On the
same 32 training rows and effective batch eight:

- Original warm median: 10.190 s/update, peak allocated 13.271 GiB.
- Single-forward numeric loss: 7.388 s/update, 9.837 GiB; first-loss relative
  error 4.05e-7, overall gradient relative L2 1.45%, visual 1.17%. Passed.
- That loss plus Liger SwiGLU: 7.320 s/update, gradient errors 1.43% / 1.15%.
  Passed, but its additional speedup is under 1%, too small for a convincing
  practical benefit in this short probe.
- Liger RMSNorm: 6.938 s/update, gradient errors 56.9% / 74.8%. Rejected.
- Both layer kernels: 7.109 s/update, gradient errors 57.1% / 75.2%. Rejected.
- Both kernels with batch two/accumulation four: 7.479 s/update, gradient
  errors 82.6% / 112.0%. Rejected; batch two also did not improve measured speed.

These are clipped first-update gradient comparisons in BF16. Rejection means
failure of our conservative equivalence gate; it does not establish a general
Liger defect or prove that such numerical differences would worsen CAD scores.
The tiny random-model test did not reveal the large pretrained-model RMSNorm
divergence. It cannot substitute for the full-model check. The precise kernel
cause remains unresolved; RMSNorm is therefore excluded from the resumed run.

The selected profile is single-forward loss without Liger. The warm-median
speedup is 1.38x and peak allocated memory is approximately 26% lower on this
sample. Whole-run throughput includes cold start, checkpointing and variable
examples; this is not a promise of the same ratio across the entire corpus.
Optional kernels must add at least 5% over the simpler accepted profile to be
selected automatically; this is a practical threshold, not a significance test.

`recipes/visual-followup-500-fast-qwen35.json` resumes checkpoint 100 of the
stopped run, targets global step 500 and retains microbatch one/accumulation
eight. It has an 85-minute remaining optimizer ceiling. Strict adapter loading,
optimizer/scheduler/RNG resume and actual continued gradients must succeed.
State is in `artifacts/visual-followup-500-fast-v1/status.json`; successful
training is followed by the same separately bounded validation protocol.

## Second investigation and selected harder-pilot profile

`artifacts/kernel-benchmark-v2` uses 64 identical examples in eight updates.
Warm medians: original batch1 10.318 s, fused batch1 7.389 s, fused batch2 8.039 s,
fused batch4 8.272 s, fused+SwiGLU batch2 7.972 s. Larger microbatches did not
help this sample; padding and an already busy GPU mean free VRAM is not itself
evidence of more throughput. Batch4 lacked a matching full-loss batch4 gradient
reference and is not correctness-qualified, independently of its slower speed.

The gradient comparison now uses the **same microbatch size** on both sides.
Comparing batch2 kernels against batch1 mixes kernel differences with BF16
batching differences and cannot isolate the cause. Regression tests enforce
this pairing; the older cross-batch Liger result is not a batch2 correctness
conclusion. Fused batch2 and SwiGLU batch2 passed the matched reference checks.

Disabling all gradient checkpointing reduced short-step times to about 5.3–5.6 s
but OOMed on update six at approximately 22.47 GiB allocated. That incomplete
probe is rejected. The new `--checkpoint-policy linear` instead checkpoints
Gated DeltaNet/vision, retaining ordinary-attention decoder activations. It
survives repeated Trainer checkpoint-enable calls and the real longest-row
forward/backward probe. Warm median **6.971 s**, peak allocated **12.071 GiB**,
relative gradient errors **1.63% overall / 2.12% visual** against baseline.
This is about 6% above the previous fused-only throughput and 1.48x the original
warm throughput. It does not promise the same speed on longer new examples.

Liger RMSNorm was also checked locally on actual pretrained norm weights:
FP32 forward/input-gradient errors were around 1e-7 and BF16 errors up to
3.54e-5 in those samples. Thus the previous full-model divergence does **not**
establish a broken RMSNorm kernel. Deep-model BF16 effects remain a possible
explanation; an end-to-end quality A/B would be needed to accept that changed
trajectory. Its small additional speed did not justify using it here.

An official causal-conv1d 1.7.0 wheel matching Torch2.7/CUDA12/CPython3.12/ABItrue
was downloaded separately. Its BF16 local operator check passed; an FP32
backward comparison differed by about 2.6% and is unresolved. The BF16 4B
probe measured 6.995 s/update but failed our full-model gradient gate (91.9%
overall / 32.5% visual). It is excluded from the harder pilot. These checks
do not establish a general package defect or prove worse downstream quality.
Package provenance is `artifacts/runtimes/causal-conv-study/download.json`.
No active training-environment dependencies were upgraded.

`train_optimized --memory-preflight` checks the longest audited selected rows
before optimization, preserving RNG, counters and gradients. The probe has
zero optimizer updates and requires 2 GiB headroom. The actual training loop
then resets peak-memory statistics separately. The real hybrid probe had
exactly the same 2,155 supervised-token count as baseline after all eight steps,
confirming that the memory check did not pollute training diagnostics.
