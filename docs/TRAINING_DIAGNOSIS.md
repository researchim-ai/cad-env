# Training diagnosis — 2026-11-06

## Evidence and scope

No new SFT was launched during this investigation. GPU1 ran a bounded component
probe of the raw Qwen3.5-4B and the completed hard-pilot checkpoint. Both use
greedy generation, native thinking disabled, 1024 output tokens, identical
prompts and pixel budgets. Eighty questions per model cover 20 validation
parents per task family. These prompts differ from training's tool-action
contract: this is a diagnostic, not a replacement benchmark. TEST stays sealed.

Reproducible artifacts:
- `artifacts/training-review-v1/local-findings.json`: native final-state counts.
- `artifacts/training-review-v1/mixture-census.json`: actual sampled mixture.
- `artifacts/training-review-v1/component-probe-v2/status.json`: completed probe.
- `artifacts/training-review-v1/component-probe-v2/*-responses.json`: raw answers,
  parseability, exactness and possible output-cap flags.
- `cad_env/learning/perception_probe.py`: public inputs and private offline grading.

Raw → trained component exactness:
- Marker count: 18/20 → 13/20; both 20/20 parseable.
- Full marker-to-public-circle-index set: 0/20 → 0/20. Raw only 5/20
  parseable versus trained 20/20: raw/trained conditional precision is not
  a fair population comparison. Trained mean precision .623, recall .661.
- Full hole table: 20/20 → 20/20; raw answers need JSON salvage whereas
  trained answers are strict JSON. Thus table OCR is not the main explanation
  for the counterbore construction failures.
- Fixed table crop input: 6/20 → 20/20; raw only 6/20 parseable. This does not
  establish that cropping harms raw visual perception: format/cap failures
  prevent that interpretation.

Marker-count decline is a possible regression on a small, new-prompt probe;
repeat with paired variants before claiming catastrophic forgetting. Selection
failure is consistent with weak spatial binding, not proof of a capacity limit.
Construction failure despite good table extraction points toward maintaining
complete feature sets, translating rows to operations, and action sequencing.

## What published recipes actually do

[IterCAD](https://arxiv.org/html/2606.13368v1) uses the same Qwen3.5-4B:
28K trajectories, including 8K student trajectories corrected by a teacher,
then multi-turn geometry-feedback RL. Its released
[SFT script](https://github.com/KnowledgeXLab/IterCAD/blob/main/train/IterCAD_SFT.sh)
uses full tuning, three epochs and eight GPUs. This is evidence of a different
training scale and interaction loop, not a requirement to copy that budget.
Its released RL/evaluation also disable native thinking; our flag alone cannot
explain failure. Explicit reasoning and output budgets differ.

[DeciCAD](https://arxiv.org/html/2610.21755) supports semantic CAD tools,
causal next-action SFT and LoRA: the interface choice is defensible. It teaches
recovery from observed failures and online decisions, including termination.
Its data quality and harness results caution against treating more JSON or
more examples as sufficient. It uses a different model and benchmark.

[CADFusion](https://github.com/microsoft/CADFusion) alternates sequential
learning and visual-feedback preference optimization. Its instructions stress
collecting feedback on the current model's own generations. Our four-parent
single-edit repair set does not cover current multi-edit failures.

## Changes justified before another training run

1. Separate spatial binding, full feature coverage, stopping and native geometry
   metrics. Preserve strict whole-task success and image-removal controls.
2. Add current rendered CAD feedback to a controlled dataset variant; current
   hard editing supplies changing JSON state but only a static goal image.
3. Collect genuine current-student multi-edit failure histories on TRAIN,
   teacher-correct them with native verification, and supervise recovery and
   finish decisions. Keep causal history before each action.
4. Increase independent designs, feature graphs and image layouts rather than
   repeat a small set of teacher steps. Audit equivalent-valid next-action
   ordering; canonical ordering is an ablation, not an established cure.
5. Test explicit all-feature/grounding supervision against the existing
   next-action objective with the same budget and parent splits. Change one
   factor per pilot, within one epoch and the user's short-run budget.
6. Only then evaluate trajectory RL. Single-action bandit feedback does not
   reproduce multi-turn GRPO/GSPO. Check credit assignment, early stopping,
   unchanged-feature protection and exact geometry, not only global IoU.

The perception-plan prototype is untrained and unqualified. A deterministic
compiler changes the controller protocol; any score gain must be reported
separately. Neither LoRA rank, numeric-token weighting, number of steps nor
Liger kernels has been causally established as the visual failure's cause.

## Implemented next experiment

`grounding_data.py` creates and semantically audits separate TRAIN-only VQA
packages. Actual visual package: 512 rows (256 counts + 256 complete selections),
64 parent designs, 256 distinct images after exact-pixel deduplication. Actual
construction package: 48 complete hole-table targets from 48 parents. These
are auxiliary questions, not a deterministic CAD executor. Dataset hashes,
public-only prompt reconstruction, exact targets and split lineage are checked.

`train_modern --action-contract single_action_auxiliary` explicitly allows
mixing action and auxiliary-grounding exports. The default action contract
remains unchanged. The standard processor, completion-only masking, vision
adapter checks and memory preflight apply to the mixed data.

`multistep_recovery.py` collects real TRAIN student prefixes and native teacher
continuations. It intercepts a proposed finish before execution, supervises the
alternative action at that exact causal state, restores unwanted edits and
completes missing targets. It never reopens a terminated environment. With
`--current-render`, native current candidate renders are included as well.
The four-parent smoke test produced two native-successful recoveries and two
rejections because the remaining eight-step episode budget could not fit the
repairs. Rejections are retained as evidence, never as successful supervision.
The output is explicitly candidate feedback, not an approved training package;
it needs packaging/semantic audit before mixing into SFT. No broad recovery
improvement is claimed from this smoke test.

`grounding_experiment.py` runs two arms from the same hard-pilot checkpoint:
- Control: 1024 existing hard-edit action rows.
- Auxiliary: the identical first 800 sampled action rows, plus 192 visual VQA
  and 32 construction-table rows (all sampled without replacement).
- Each arm: 128 updates, effective batch 8, one prepared-mixture epoch;
  LR 5e-6, numeric weight 8, same vision LoRA and qualified fused/checkpoint
  profile. Checkpoints every 32 updates; 20 optimizer minutes per arm;
  60-minute total subprocess deadline. A stopped arm invalidates the matched
  128-update comparison; it is never silently counted as complete.
- This matches optimizer updates and examples, not supervised tokens/FLOPs or
  wall time. There are no recovery examples or observation changes in either
  arm, isolating the replacement with explicit perception supervision.
- Full 160-task native visual validation and the 80-case-per-model component
  probe compare both arms. TEST stays sealed; no automatic promotion or next
  training is permitted. Broader retention and image controls are still required
  before adopting a candidate even if this local ablation succeeds.

The experiment saves command logs, source hashes, data audits, optimizer/vision
gradient checks and durable status. Entry point (from the pinned environment):

```bash
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
  .venv-qwen35/bin/python -m cad_env.learning.grounding_experiment \
  --output artifacts/grounding-ablation-v1
```

Launch status: `artifacts/grounding-ablation-v1` started on physical GPU1.
The initial control preflight checked all 1024 rows: image tokens masked,
completion-only masks and template boundaries valid; longest row 1904 tokens.
Focused regression checks passed 22 tests, including deliberate prompt/label/
split corruption, missing auxiliary targets, and causal native-recovery logic.
Training benefit is pending the completed two-arm evaluation.

## Completed auxiliary ablation — 2026-11-07

`artifacts/grounding-ablation-v1/status.json` is complete; both arms reached
128 updates / epoch 1.0. Control optimizer runtime 987.41 s; auxiliary 839.66 s.
Both have 128/128 nonzero language and visual gradient steps. All stages
completed in approximately 51.1 minutes; TEST remains unopened, no promotion.

Identical native evaluation protocols, all 160 validation tasks / 20 parents:
control 15/160 (9.375%), auxiliary 17/160 (10.625%). Paired difference +1.25 pp,
parent-cluster 95% CI [-1.25, +4.375] pp: improvement not supported.
Invalid action rates are zero for both; truncation 21.875% versus 17.5%.

Component probe labels `raw` and `trained` denote CONTROL and AUXILIARY here,
not original pretrained Qwen versus a tuned model. Counts: 13/20 versus 17/20.
Full marked-circle index sets: 0/20 versus 0/20, all responses parseable, none
at the output cap. Mean per-case selection precision .636 versus .652; recall
.653 versus .767. Full hole tables and fixed-crop tables: 20/20 for both arms.
The small component sample suggests better counting/coverage, but spatial
binding remains inadequate and the native endpoint has no supported gain.
Training losses .312 versus .356 are across different target mixtures and
must not be interpreted as a quality ranking. Broader retention/image controls
were not run in this ablation; no general CAD improvement is claimed.

Next priority is a controlled spatial-binding intervention and audited genuine
multi-step recovery data. More identical SFT updates or an immediate RL run are
not justified by this result alone.

## Small-model suitability audit — 2026-11-07

A new paired diagnostic is complete at
`artifacts/spatial-diagnosis-v1/auxiliary/status.json` (80 questions, 20 fixed
validation parents, no optimizer update, TEST sealed). The same trained 4B
checkpoint receives four views per identical task:
- Original rotation, no IDs: 0/20 complete correct sets; mean precision .652,
  mean recall .767.
- Original rotation, IDs on ALL circles: 12/20; precision .953, recall .936.
- Zero rotation, no IDs: 0/20; precision .652, recall .769.
- Zero rotation, IDs on ALL circles: 18/20; precision .990, recall .988.

All 80 answers parse; none hits the 256-token cap. Geometry and selected subsets
are fixed across views. Original baseline pixels are regenerated and checked
against the frozen PNG. Every raster index is checked against native initial
circle coordinates/order. Label/camera interventions do not depend on selected
subsets. Label conditions add a public instruction explaining the labels; the
intervention is an observation-contract change, not just a pixel-only change.
These are component results on 20 parents, not native CAD execution scores or
proof of broad generalization. Tests cover variant identity, no task mutation
and independence of intervention settings from private selection labels.

This strengthens the hypothesis that entity correspondence, not inability to
read markers or a proven 4B capacity ceiling, limits the current unlabelled
contract. Removing rotation alone did not resolve it. Adding public entity IDs
is a legitimate assisted interface, but must be evaluated and named separately
from unassisted spatial grounding. Never relabel a native benchmark score as
improved just because its input contract changed.

### Revised priorities

1. Maintain native exact success, protected-parameter checks and parent splits.
   Add component correspondence, feature completeness, stop/revise and recovery
   scores. The hard visual family is one capability, not overall CAD competence.
2. Qualify a practical assisted harness (stable IDs on all entities, compact
   current state, consistent camera metadata, optional current renders) through
   full native rollouts. Retain the old unassisted track as a separate challenge.
3. Teach image location ↔ entity ID explicitly, including nearby negative
   entities, complete-set targets, paired counterfactuals, varied numbering and
   views. Progress from 1–2 to 4–6 edits while retaining easy and text anchors.
   Changing a set target to per-entity decisions is an ablation, not a proven cure.
4. Use an explicit TRAIN memorization/learnability gate on a tiny fixed subset.
   Failure to fit those examples means inspect preprocessing/adapter targets/
   optimization before generating a huge corpus. Passing is not generalization.
5. Package native-verified real student recoveries with causal pre-action
   prompts. Include already-correct states with finish targets and incorrect
   states with correction targets. Measure rejection/remaining-step bias: the
   current four-parent collection rejects two difficult repairs for lack of steps.
6. Pilot 1–2K newly curated rows for one epoch with a 30–45 optimizer-minute
   ceiling. Log distinct parents/images/tasks, not just row counts. Match budgets
   against a control, then check text/visual/construction retention and image
   removal or wrong-image controls before adoption. No new SFT started here.
7. Only then qualify trajectory RL: actual group reward variance, nonzero-advantage
   fraction, completion masks, on-policy log-probabilities and useful success/
   failure mixtures. Do not infer correct RL from nonzero SFT gradients. Reward
   missing/extra features and protected geometry; whole-part IoU alone is weak
   for small engineering details. Keep strict terminal validation authoritative.

### Additional primary-source grounding

[CADReview / ReCAD](https://aclanthology.org/2025.acl-long.489.pdf) explicitly
trains semantic matching, coordinate matching and localization between visual
components and program blocks before repair. This motivates correspondence
supervision; its results are not a reproduction on our 4B or our tools.

[cadrille](https://arxiv.org/html/2505.22914) uses Qwen2-VL-2B with procedural
SFT and online geometric RL. It supports studying small specialized models,
but its larger corpus, training budget and reconstruction metrics differ.

A different paper also named
[IterCAD (2608.24020)](https://arxiv.org/html/2608.24020) explicitly includes
already-correct STOP states alongside repairable REVISE states. It is distinct
from the Qwen3.5-4B IterCAD paper (2606.13368); their recipes and results must
not be conflated. Our small-model repair data should teach both decisions.

## Implemented native follow-up — 2026-11-08

`assisted_data.py` now builds observation-only variants and semantically audits
them against the frozen source. Actual validation packages contain all 160 tasks
and the same 20 parents, with original or zero camera rotation and IDs on all
circles. Reference actions, oracle actions, split identity and native FCStd/result
bytes remain unchanged. Raster indices are checked against native initial circle
ordering; audit regenerates the image pixels. Validation exports no training rows.
The TRAIN-only canonical package contains 2512 feedback rows / 512 task variants /
64 parents; answers are unchanged and public pre-action prompts are rebuilt from
the original causal state/history. This is not a diversity expansion.

`train.check_training_source` invokes the assisted semantic audit before SFT.
Twenty focused tests pass, including private-selection-independent public hints,
unchanged geometry/targets, no input mutation and TEST refusal. Loader audit
passes all 2512 actual TRAIN rows / 2050 hashed artifacts.

`assisted_experiment.py` is running on GPU1 under
`artifacts/assisted-native-v1/evaluation`: full native original-ID and canonical-ID
tracks, followed by blank-goal and wrong-goal controls on the canonical track.
All use the same frozen auxiliary checkpoint, 160 tasks, batch limit 4, eight
workers, eight native steps, no optimizer updates, total subprocess ceiling
45 minutes. Source hashes and per-stage commands/status are durable. There is no
automatic promotion or SFT queue. Native results are pending; component 18/20
must not be presented as end-to-end CAD success.

A provisional one-epoch 2000-row / 250-update / 40-optimizer-minute recipe is
saved at `recipes/assisted-sft-pilot-qwen35.json`, explicitly NOT LAUNCHED. Native
failure analysis must select the next intervention first. It preserves raw visual,
construction and simple anchors instead of replacing the whole curriculum with
assisted views. Memory/mask preflight and retention/image controls remain required.

Additional recovery limitation: every current 2–6-edit hard task has max_steps=8.
Six edits plus finish need seven actions; one wrong unmarked edit plus its
restoration would require nine. Such errors cannot always be repaired within
this budget. Increasing the CLI cap alone does not extend it (`evaluate_policy`
takes min(task max_steps, CLI cap)). A recovery-budget experiment therefore needs
an explicit separately reported task/harness variant. Do not silently enlarge
old benchmark budgets or train impossible continuations as successful examples.

## Completed native assisted qualification

All four 160-task tracks completed under `artifacts/assisted-native-v1/evaluation`.
Same auxiliary 4B checkpoint, zero optimizer updates, original eight-action
budget and native geometric/parameter/history scoring:
- Prior unassisted reference: 17/160 (10.625%).
- All-entity IDs, original camera: 57/160 (35.625%).
- All-entity IDs, canonical camera: 93/160 (58.125%).
- Canonical IDs with blank goal: 0/160.
- Canonical IDs with counterfactual wrong goal: 0/160.

This is an observation/harness gain, not weight improvement. Image controls
support dependence on the goal pixels. Exact success still fails on 67 tasks;
45 of those hit the eight-step limit. More steps may help some but cannot be
assumed to fix repeated or wrong actions. Analyze these trajectories before
launching the provisional SFT recipe. TEST remains sealed; no model promotion.

## Verified stop/repair training trial — 2026-11-08

### Failure evidence motivating the data

Offline inspection of the completed canonical-ID validation found 67 failures:
45 truncated episodes, all containing repeated identical parameter writes
(130 repeated writes total), and 22 unsuccessful finishes. Of the truncated
cases, 27 end with every checked named constraint matching its target; 28 failed
episodes pass through such a state. Parameter equality is diagnostic, not native
success: these tasks still require finish and geometric/history validation. No
oracle finish is inserted into runtime evaluation to inflate the score.

### New data actually generated

`release_data.py` proposes new seeded geometries and retains their natural hash
splits. TEST proposals are discarded before native execution/export; existing
sealed TEST task artifacts are not opened. Exclusion uses prior manifest parent
identities. Within this corpus, physical-geometry and counterfactual audits pass.
This is not a claim of global external-dataset shape deduplication.

Actual `artifacts/release-curriculum-v2`:
- 100 new TRAIN parents, 30 new validation parents, balanced across 2–6 marked
  edits; four counterfactual goals per parent and two state phases.
- 1040 native-verified tasks, zero rejected teachers. These are task/state
  variants, not 1040 independent designs or images.
- 2000 clean TRAIN feedback rows; 1583 revisit TRAIN rows.
- A shared intermediate state is reused with four different goals. Some states
  already satisfy one goal and require finish; the other goals require edits.
  Others include a partial shared edit plus a wrong dimension. All unmarked
  original diameters are explicitly specified in the public repair instruction.
- Clean tasks keep 8 steps; revisit tasks use 12 to permit restoration. Every
  compared model sees the identical budgets; these scores are separate from
  the old 160-task/8-step benchmark.
- Native shape validity, target geometry, protected parameters, feature history,
  marker pixels, public-prompt isolation, natural splits and four-way nonvisual
  input identity are audited. The resulting no-image greedy ceiling is 25%.

`release_subset.py` creates audited phase packages. `stop_contrast` retains all
four first decisions for complete-peer states (correct finish versus needed
correction), plus late finish anchors: 234 rows. `revisit_general` excludes those
same rows, so selecting both sources does not silently duplicate supervision.

`multistep_recovery --phase revisit` collected actual parent-model prefixes on
32 fresh TRAIN designs. Twenty-nine continuations reached verified native
success; three already-correct proposed finishes are retained as evidence but
not fabricated into repair examples. `recovery_export.py` packages 63 rows from
those 29 continuations. It reconstructs each prompt from pre-action native state
and past student/teacher history, checks native terminal success, rejects future
answer injection, preserves original assistant JSON versus worker float
normalization, and verifies the exact collected record. A prefix cut at six
student steps is not by itself proof that the student would subsequently fail;
these are real on-policy continuations, not 29 independently diagnosed errors.

### Controlled training and acceptance

`release_trial.py` is running at `artifacts/release-trial-v1` on physical GPU1.
`recipes/release-stop-repair-qwen35.json` records the exact plan. Both arms start
from the same frozen auxiliary checkpoint. Each has exactly 1600 unique selected
rows, no source cycling, 200 optimizer updates / one prepared-mixture epoch,
effective batch 8, checkpoints every 50 updates, LR 5e-6, vision LoRA and numeric
weight 8. Each optimizer loop is capped at 35 minutes.

Control: 1000 fresh clean rows + 600 old assisted, construction, raw visual and
simple anchor rows. Candidate: 537 common clean rows, 200 general revisit rows,
200 stop/correction contrasts, 63 actual policy continuations + the exact same
600 anchor rows. Explicit per-source seeds make shared anchors byte-identical
regardless of source ordering. Census checks normalized prompts, completions
and image hashes: both selections have 1600/1600 unique rows. This matches
updates/examples, not target-token counts or FLOPs. It tests a bundled data
recipe; ingredient-specific attribution needs later ablations.

The pipeline first measures the parent on all 240 fresh validation tasks and
construction/simple retention. It then trains and compares both arms on fresh
validation, the old assisted 160-task cohort, construction and simple tasks;
checks candidate blank/wrong goals; and computes paired parent-cluster CIs.
An independent `scripts/release_raw_retention.py` waits until GPU1 is released,
then verifies both arms on the original unlabelled 160-task cohort using the
same total deadline. It does not alter frozen trial sources. Main execution
and extra retention share a ceiling of 165 minutes after preparation.

Review qualification requires a supported fresh gain over BOTH parent and
matched-update control, point retention on old assisted/simple/construction/raw
tracks, and a substantial goal-image control gap. Point-retention checks are
pragmatic pilot guards, not statistical equivalence. There is no automatic
promotion, TEST evaluation, external benchmark claim or further training queue.
The actual training benefit remains pending the complete trial.

Thirty-five focused tests passed before launch. Additional verification before a
strong release still includes fresh blind/OOD or external tasks, repeated runs
for stability, portable evidence packaging and one final sealed TEST evaluation
after checkpoint selection. This trial alone cannot establish general CAD
competence, Blender transfer or full multi-turn GRPO quality.

### Interrupted pilot recovery — 2026-11-08

A fresh host/GPU check found no training/supervisor process and an idle GPU1,
while the candidate status still said `running`. Its last progress record was
67/200; the last durable checkpoint was step 50 (epoch 0.25). The interruption
cause is unknown. This is not a completed candidate run or a quality result.

Added `release_trial --resume` for this specific interrupted candidate stage.
It preserves the original status/source-freeze under `interruption-evidence`,
checks that all frozen sources except the orchestration module are unchanged,
and resumes optimizer/scheduler/RNG and adapter weights from checkpoint-50 into
`candidate/sft-resumed`. The completed control and parent evaluations are reused.
The 17 unsaved updates are replayed from the checkpoint; the target remains 200
logical updates / one prepared-mixture epoch. The original execution allowance
is reduced by elapsed work up to the last progress record, including lost work;
inactive downtime is excluded. The remaining shared execution budget at recovery
was 5672 seconds (94.5 minutes), including subsequent evaluation and raw retention.
No TEST tasks are opened or models promoted by this recovery.

The supervisor and retention waiter were relaunched in separate process sessions.
The retention waiter uses the recovered shared deadline. Three recovery tests
verify checkpoint selection / completed-control reuse / budget accounting,
source drift rejection, and completed-run rejection. Together with existing budget
and source-mixing tests, 9 tests passed. A live GPU/process check must accompany
future status reports: a stale `running` JSON file alone is insufficient evidence.

### Large composition and industrial-scope correction — 2026-11-08

Expanded the typed program limit to 128 actions. A native boundary test executed
128 actions successfully and a schema test rejects 129; early finish remains
invalid. New large-composition generator builds heterogeneous drilling plates,
multi-support fixtures and terraced flange stacks with 32–108 teacher actions.
All 48 native teachers passed independent analytic volume/bounds and exact-shape
checks; semantic audit passed. There are 24 independent geometry parents, not
48 independent designs. TRAIN and validation each have 12 parents / 24 tasks /
1624 feedback rows. Extreme teachers require 72, 100 or 108 actions depending on
family. Figures are paginated parameter schedules with 2–4 goal sheets. The
public prompt identifies multiple goal sheets explicitly. They are not asserted
to be industrial-standard engineering drawings; summary metadata records that
standards conformance and manufacturability are not assessed.

Largest-serialized teacher prompt per validation task was processed by the real
Qwen processor: maximum 13002 tokens including images. The old 8192 evaluation
context cannot support this cohort. New calibration uses 32768 context, 327680
pixels per image, batch 2, 4 native workers, at most 128 actions, greedy generation
and the current candidate merged checkpoint. This is a new protocol and corpus,
not a comparable replacement for previous scores. Every student call retains
strict context checking; infrastructure failures and incomplete runs are never
reported as zero capability. GPU1 calibration is bounded to 30 minutes and does
not start training. Its source hashes are frozen in the supervisor record.

47 related generator/action/learning/CLI tests passed, including real native
execution; an additional focused rerun passed 12 tests. See INDUSTRIAL_DATA.md
for the distinction between synthetic stress, planned standards-profile drawing
export, and real industrial-source evaluation.

The preceding release trial exhausted its shared execution deadline during
candidate_simple. Completed fresh/old/construction measurements remain evidence,
but the complete retention/image-ablation gate set did not finish. Do not call
that trial qualified or infer that all release gates passed. It was not restarted
or silently granted an extra execution allowance.

### Model-derived technical drawing format — 2026-11-09

Implemented `cad_env.drawings`: native TechDraw front/top/right views, a model-tied
A-A section, exact material-clipped hatch segments, native extent dimensions,
center marks on real cylindrical features, hidden-line styling, A3 first-angle
SVG/PDF/PNG pages, complete checked feature schedules, and traceable readable VLM
crops. The exporter verifies every native pad/pocket, feature dimensions, actual
support dependencies/elevations and unchanged source geometry. Unsupported
specification fields are rejected instead of silently disappearing. Source code,
CAD inputs and output hashes are preserved in each drawing bundle.

`technical-corpus-v2` is the accepted corpus: 48 tasks / 24 physical parents,
1624 TRAIN and 1624 validation feedback rows. Geometry, target actions, causal
state/history and split lineage are unchanged from `large-composition-v1`; only
the observation format is changed. Separate audits protect task and teacher
semantics and pixel assets. Rehashed completion corruption and future-state
injection are rejected by semantic checks. The TRAIN source guard passed.

CPU processor preflight measured up to 18480 tokens / 22 crops. One real GPU1
inference on the longest checked teacher-state input completed in 13.29 seconds,
returning a valid finish action; peak allocated/reserved memory was 11.70/13.04
GiB. This is not a complete CAD rollout, image-use proof or training-memory result.
54 related tests passed. PDFs of all three extreme example families were rendered
with Poppler for visual QA. No additional training was launched.

Rejected draft drawing exports/corpus v1 were preserved. A numerical comparison
in the audit originally used exact equality for floating-point native dimensions;
it now uses explicit tight numerical tolerances consistent with the native
measurement checks. The complete v2 corpus passed. Drawing standards conformance
remains unassessed, manufacturing requirements are not inferred, and real
industrial-source evaluation remains outstanding. See TECHNICAL_DRAWING_FORMAT.md.

Additional crop-reading component diagnostic: six questions on three extreme
physical designs, full and blank images. Strict JSON exactness: full 0/6, blank
0/6; the model returned single-quoted Python-style dictionaries. Explicitly
separate literal-format recovery found all six full-image field sets numerically
correct, versus zero blank-image sets. Raw responses are preserved. This supports
readability on this small component sample while exposing response-format weakness;
it is not native CAD task success. No recovery parser was added to the CAD action
or reward path. Two additional response-grader tests passed (56 related tests total).
