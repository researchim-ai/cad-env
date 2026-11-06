# Implementation checkpoint — 2026-11-06

## Latest completed result and diagnosis

The 500-step follow-up and all its validation stages completed. Fresh 40-parent
validation: **181/640 → 370/640** strict successes, delta +29.53 pp, parent-cluster
95% CI [+23.91, +34.84] pp. Simple retention 69/69, complex 45/46, old unlabelled
42/100 and labelled 99/100. Image controls fell to 7–10%. Step300→500 added only
6/272 old-V6 successes and its CI includes zero; more identical steps are not
the primary next intervention.

Fresh validation's unlabelled slice remains only 82/320 versus labelled 288/320.
The new harder curriculum therefore focuses on unlabelled 2–6-edit tasks and
new construction compositions, not another label-heavy repeat. All 304 new
construction and 736 new visual teachers passed native and semantic audits.
See `HARD_CURRICULUM.md` for corpus identity, new drawing QA, scope and limits.

The hard pilot completed under `artifacts/hard-experiment-v1`: 300 updates in
2308.55 seconds, one epoch over 2400 prepared rows. Construction improved
18/80 → 60/80; hard visual editing 12/160 → 15/160 has no supported improvement
(parent-cluster confidence interval includes zero). Validation promotion gates
failed; TEST remains sealed. No subsequent SFT has started.

The investigation in `artifacts/training-review-v1` separates perception,
action coverage and stopping. Hard editing has zero successes for 4–6 required
edits. Final-state counts: 414 correct changes, 258 missing, 139 unwanted;
138 episodes finish unsuccessfully. Current candidate renders appear in zero
of 722 steps (dynamic JSON state is present). The 1000 new visual training rows
cover 64 designs / 250 distinct first images. Repair sampling repeats 44 rows
from four designs, rather than covering the failing multi-edit distribution.

Counterbore image-only construction gets base dimensions right 20/20 but only
3/20 complete parts; 115/140 expected pocket feature records match. This
suggests feature coverage/binding/sequencing as well as perception. Component
VQA compares raw and trained models without optimizer updates; it is a
validation diagnostic, not a CAD benchmark score. See `TRAINING_DIAGNOSIS.md`.

Perception-plan files are an untrained experimental alternative. Compiled plans
move sequencing into deterministic code, so their scores must be reported as a
different controller protocol, not evidence that the old agent improved.

## Completed pilot and bounded follow-up

Update: the first 500-step follow-up was stopped by user request at about step
110 to benchmark kernels; durable checkpoint 100 includes optimizer state.
Six short 4B probes selected a single-forward numeric loss (warm median 10.19
to 7.39 s/update, peak allocated 13.27 to 9.84 GiB). Liger RMSNorm failed the
full-model gradient-equivalence gate; SwiGLU passed but added under 1% speed.
The resumed recipe is `visual-followup-500-fast-qwen35.json`, from checkpoint
100 to global step 500 with the same data/effective batch and 85 remaining
optimizer minutes, without Liger. See `KERNEL_OPTIMIZATION.md` for evidence and
limitations. Current state is `artifacts/visual-followup-500-fast-v1/status.json`.

The 160-step visual pilot finished. V6 strict success rose from 70/272 to
107/272; simple retention stayed 69/69, complex changed 43/46 to 42/46,
unlabelled grounding 30/100 to 36/100 and labelled selection 88/100 to 92/100.
The V6 comparison still has only 17 independent parents and fails the minimum
parent-count evidence gate. Image interventions support image dependence, not
general drawing comprehension. TEST remains sealed.

The next experiment is `recipes/visual-followup-500-qwen35.json`, executed by
`cad_env.learning.visual_followup`. It starts afresh from checkpoint-500 for a
controlled comparison with the pilot, rather than stacking another adapter on
the 160-step candidate. The mixture has 4,000 rows: V6 2,400, V3 600, real
policy repairs 200, complex 400, simple 200 and FreeCAD mechanical 200. These
are prepared row budgets; small sources may repeat original rows. Effective
batch eight and 500 steps give one prepared-mixture epoch maximum.

LR is 1e-5 with 20 warmup steps and then constant. This deliberately tests
whether sustained updates help; it is not a causal isolation of data versus
schedule effects because the mixture also changed. Checkpoints every 100 steps
are retained (limit six). Optimizer time is capped at 105 minutes, with a
120-minute training-phase deadline. Native validation has a separate
120-minute deadline, no optimization or retries. Timeouts preserve completed
stages and checkpoints and cannot be reported as complete benchmarks.

The validation generator selects 40 fresh physical parents, ten per required
edit count 1–4, from naturally assigned validation splits in seed range
3,000,000+. Their 640 UI/unit variants stay outside training. The generator
requires full native acceptance and a visual/semantic audit. The run compares
the parent and final candidate there, compares the candidate with the 160-step
pilot on old V6, checks all four retention cohorts and image controls, then
evaluates saved steps 100/300 on old V6 if those checkpoints exist. Every stage
shares the bounded evaluation deadline. No TEST access or automatic promotion.

Live state and logs: `artifacts/visual-followup-500-v1/status.json` and
`training.log`. A source SHA manifest and recipe are saved before training.
The follow-up recipe/lineage tests, wall-time guard tests and previous pilot
tests passed (five tests); actual training outcomes must come from the run.

## Completed and measured

Checkpoint-500 passed full validation: FreeCAD 65/66, CadQuery 64/66, simple
retention 69/69, complex construction 43/46, unlabelled selection 30/100 and
entity-labelled selection 88/100. This is a restricted-tool synthetic benchmark,
not an official external CAD benchmark. Unlabelled visual selection remains the
clearest weakness. Detailed evidence is in `BATCHED_VALIDATION.md`.

The V5/V6 visual corpora have native/semantic audits. V6 has 2,560 episodes,
5,568 TRAIN feedback rows and 17 independent validation parents. Its 272
validation task variants are correlated across unit/UI conditions; they must
not be counted as 272 independent designs. V6 had not yet been trained on by
checkpoint-500. The old three-arm ablation remains stopped by user request.

## Current implementation: one bounded visual/repair pilot

`recipes/visual-repair-pilot-qwen35.json` starts from the merged checkpoint-500,
not the older SFT-C parent. All GPU work is serialized on physical GPU 1.

Stages:

1. Audit TRAIN inputs and measure the full V6 validation baseline.
2. Sample 32 eligible TRAIN prompts × 4 responses from the actual parent policy.
   This is data collection, not a GRPO optimizer run.
3. Independently replay incorrect but executable edits, then build repair tasks
   on up to 16 TRAIN parent boards. All four counterfactual goals share the
   damaged starting state. No errors or successes are fabricated if collection
   yields no eligible mistakes.
4. Train vision/language LoRA for at most **160 optimizer steps**, with a
   **35-minute optimizer-loop ceiling** and checkpoint saves every 40 steps.
   The 2,560-row prepared mixture receives at most 1,280 sample exposures (.5
   prepared-mixture epoch); repeated source rows are recorded as repeats.
5. Merge, reevaluate V6 under the same batch4/workers8 profile, and compare old
   simple, complex, unlabelled and labelled validation cohorts.
6. Run no-goal, blank-goal and wrong-goal controls with the candidate.

The entire driver has a **120-minute deadline including evaluation**. This is a
ceiling, not an ETA. On timeout/failure it preserves completed stages and saved
checkpoints; it neither retries nor queues another training run. TEST remains
sealed. This mixed pilot cannot isolate the causal benefit of each source; an
ablation is a subsequent step only if results justify spending more compute.

Status: `artifacts/visual-repair-pilot-v1/status.json`.
Code, configuration and SHA manifests are frozen under
`artifacts/frozen-code/visual-repair-pilot-v1`.

## New safeguards and diagnostics

- `budget.py` / `train_modern --max-train-minutes`: checks wall time at each
  optimizer-step boundary, requests a checkpoint and a graceful stop. It also
  writes `budget-progress.json` with actual optimizer steps. Loading and data
  preflight are outside this optimizer-loop measurement; a slow step may exceed
  the boundary slightly. Final manifests explicitly distinguish planned-step
  completion from a successful budget-limited stop.
- The callback passed unit tests and the pinned Qwen environment test. A real
  GPU smoke run planned three steps with a 0.06-second budget: it stopped after
  one step, saved `checkpoint-1` with optimizer state, and recorded nonzero
  language and visual gradients. It did not falsely report all three planned
  steps complete.
- `batch_evaluate --image-condition` exposes the existing native public-image
  intervention protocol for concurrent evaluation.
- `visual_diagnostics.py` reports paired before/after slices by labels, units,
  edit count and tier. These are exploratory, overlapping slices, not separate
  independent benchmarks.
- `visual_error_audit.py` diagnoses wrong entity vs wrong numeric value, use of
  an inch literal as a millimetre argument, premature finish and unrequested
  historical edits. This is private offline grading only; no labels are added
  to model inputs. Final native success is authoritative.

The focused regression suite passed 33 tests before the new visual diagnostics
and error-audit tests, which also passed. Test logs and the real GPU smoke
manifest are kept under `artifacts/`.

## Still not implemented / not demonstrated

- **Full multi-turn RL:** sampled sequential tool execution, assistant-token
  log-probability tracking and episode-level credit assignment are not wired to
  an optimizer. Existing GRPO remains a one-action contextual bandit and has no
  statistically supported broad learning gain.
- **Advanced tools in the learned agent contract:** fillet/chamfer/revolve/shell,
  arcs and multiple solids have separate native probes; they are not integrated
  into the seven-action learner or its sequential SFT corpus.
- **General FreeCAD/CadQuery/Blender API mastery:** a common restricted language
  is not arbitrary Python API or GUI use. Assemblies, general sketch constraints,
  arbitrary support-face selection and robust topological references remain.
- **Expanded Blender training:** native execution exists, but the expanded
  corpus had unresolved rejected examples; checkpoint-500 did not train on it.
  It must pass a complete native audit before inclusion.
- **Drawing transfer:** success on synthetic marker/dimension images does not
  prove engineering-drawing comprehension, hidden-feature inference or real
  image transfer. New graph families, views, sections and external tests are
  needed, with appropriately versioned holdouts.
- **External benchmarks:** the restricted-tool scores are not official
  CADBench/CADWorld numbers. A capability-complete adapter and pinned evaluation
  are still needed; unsupported tasks must not be silently dropped.

Next decisions depend on the pilot's actual error breakdown and retention,
not teacher-forced token accuracy or training loss alone.

## Actual baseline and first completed stages

The full V6 validation baseline completed: **70/272** strict successes (25.7%).
Paired UI/unit conditions share only 17 physical parents. By required edit count:
36/64 for one edit, 26/64 for two, 8/112 for three and 0/32 for four (the last
slice has only two physical parents). This strongly motivates testing sequential
completion, rather than interpreting every failure as numeric conversion.

Offline first-action diagnostics found 272/272 correct numeric values and
220/272 actions addressing one of the requested entities. There were 171
trajectories that finished before all requested values were written. Historical
unrequested edits occurred in 148 trajectories; those counts overlap and are
not a disjoint causal taxonomy. These results do not establish correctness on
unseen unit values: V6 TEST remains sealed.

Collection then produced **128 real TRAIN policy samples**. Native repair
construction accepted **60 tasks from 15 parent boards**, yielding 162 original
feedback rows, with the counterfactual nonvisual ceiling retained at .25. The
prepared pilot mixture deliberately cycles this small repair source.

Actual SFT preflight passed all 2,560 selected mixture rows (max 3,109 tokens,
149 target tokens, all image/completion/template masks checked). The pilot
entered `short_sft`. This is not yet evidence of a learned improvement.

The expanded focused regression run passed **40 tests**. The error-audit CPU
watcher saves baseline/candidate diagnostic reports automatically as each full
evaluation completes. The training remains capped; no follow-on queue was
restarted.
