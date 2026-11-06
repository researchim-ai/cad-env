# Harder visual and construction pilot — 2026-11-06

## Why change the mixture

On the completed 40-parent fresh validation, the previous best model achieved
370/640 strict successes. The overall 57.8% hides a large modality gap:
unlabelled selection was 82/320 (25.6%), labelled selection 288/320 (90%).
First-action entity correctness rose 492/640 to 536/640 while first-action
numeric correctness stayed 640/640. Premature finishes remained in 239 episodes;
247 trajectories contained historical unrequested edits (possibly later
restored). These overlapping diagnostics are not a causal error partition.
Reports are under `artifacts/hard-curriculum-v1`.

## New audited data

`hard_tasks.py` generates two native construction families:

- Counterbored plates: three coaxial through-hole/counterbore pairs and a
  separate blind hole, with distinct drilling depths and analytic union volume.
- Hollow stepped shafts: five or six coaxial stages, a central through bore
  and two small mounting holes through the bottom flange. Programs have
  16 or 18 actions and a 24-action episode budget.

The accepted construction corpus is `artifacts/hard-construction-v2`:
304 tasks, 152 physical parents, 2,788 TRAIN feedback rows. Each physical part
has text-plus-image and image-only instruction conditions sharing its split.
Parent counts are TRAIN 96 / validation 40 / TEST 16. All native teachers and
the compositional/semantic audit passed. During visual QA, v1 showed overlapping
coaxial A/D labels. The renderer now combines them into A/D once, outside the
outer circle, and a fresh v2 corpus was generated; v1 is not used in training.

`artifacts/hard-unlabelled-v1` contains 736 tasks, 92 physical parents and
2,512 TRAIN feedback rows. All tasks omit entity-index labels, require 2–6
edits, use paired mm/inches requests, four counterfactual goals and rotations
in 30-degree increments. Parent counts: TRAIN 64 / validation 20 / TEST 8.
Every native teacher and the visual information audit passed.

Parents keep natural hash splits. All previously recorded parent identities,
including sealed TEST identities, are excluded using manifest IDs alone.
Image/unit/modality variants never become independent physical samples.
This new curriculum intentionally includes longer compositions in TRAIN:
previous 5–6-edit composition categories cannot still be called unseen classes.
Prior TEST identities remain excluded and TEST is not evaluated by this pilot.
These remain synthetic restricted-tool tasks, not arbitrary CAD API mastery.

## One bounded experiment

`hard_experiment` starts from the completed 500-step visual candidate. State:
`artifacts/hard-experiment-v1/status.json`. It runs serial GPU-1 stages:

1. Full parent baseline on the new unlabelled and construction validation sets.
2. Actual parent-policy single-mark TRAIN calibration on old V6 (16×4 samples),
   then native replay and repair construction on at most eight parents. This
   existing collector does not support multi-edit RL; it is not called on the
   new multi-edit tasks. No failures are fabricated. If no repair corpus is
   available, its 200-row allocation uses old V3 instead, explicitly recorded.
3. SFT with vision/language LoRA, numeric token weight eight, LR 5e-6 after ten
   warmup steps. Prepared budgets: new unlabelled 1,000, new construction 600,
   repairs 200, prior complex 200, simple 200, old V6 200. Small sources can
   repeat rows. Effective batch eight and 300 updates give at most one epoch
   of this 2,400-row prepared mixture, not one epoch of all original corpora.
4. Before any update, forward/backward on the longest audited selected rows
   verifies memory with 2 GiB headroom. The probe restores RNG, clears gradients
   and restores loss counters/metrics; no optimizer update is made. Exactly one
   conservative memory fallback to batch one/full checkpointing is allowed
   only if this pre-optimizer check fails. Other failures are not retried.
5. Merge and compare full new validation cohorts, old 640-task fresh validation,
   simple/complex/unlabelled/labelled retention and all three image controls.

Optimizer ceiling: **45 minutes / 300 steps**, checkpoints every 100 steps.
Whole driver ceiling: **180 minutes including baseline, collection and eval**.
Deadlines preserve completed stages and checkpoints; partial benchmarks cannot
pass gates. Source SHA drift aborts the run. No automatic promotion, TEST
opening or queued follow-on training.

The chosen throughput profile is `fused_hybrid`: checkpoint Gated DeltaNet
and vision blocks, retain activations in ordinary attention decoder layers.
The matched eight-step 4B probe passed loss/gradient gates and a longest-row
memory probe. See `KERNEL_OPTIMIZATION.md` for the full speed investigation.

## Remaining limits

Novel families and unseen physical parents are progress in coverage, but do
not establish real engineering-drawing transfer, assemblies, arbitrary sketch
constraints or advanced surface operations. New-curriculum success, retention
and image dependence must be measured after training, not inferred from token
accuracy, teacher validity or a faster step time.
