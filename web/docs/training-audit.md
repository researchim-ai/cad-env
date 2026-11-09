> Historical experiment report. Runtime descriptions and measured scores below refer to those runs. Current supported backends are FreeCAD and CadQuery; see the project guide and dated updates for the current implementation.

# GRPO and data scaling audit — 2026-11-05

## What the existing RL experiment actually established

The saved `vision-leap-grpo-smoke` run trained language LoRA on Qwen3.5-4B
(SFT-C parent). It sampled **48 actions on nine TRAIN physical geometries**,
made **12 optimizer updates**, and obtained 14 positive native rewards. The
vision encoder was frozen. Each policy response was one `set_parameter`, with
a fixed public `finish` appended. This is a contextual bandit; it cannot train
multi-edit planning, construction, stopping or recovery.

The independent `cad_env.learning.grpo_audit` checks the saved action against
the change between initial and reference sketch geometry, without relying on
the teacher completion. All **48/48** rewards agreed. Adapter weights changed;
all 12 logged gradient norms were nonzero. Ten reward groups had actor signal;
two all-zero groups did not. KL/optimizer effects can still update weights
when actor advantages are zero. This audit covers these saved samples only.

A near-zero scalar GRPO loss is not evidence of a missing gradient. With one
policy iteration, the forward importance ratio starts at one and normalized
advantages average to zero. Their derivatives weight different action log
probabilities and need not cancel. Zero clip fraction is also expected in this
setting. The pinned TRL 1.17.1 source, not current documentation defaults, was
inspected. Replacing GRPO with sequence-level importance weighting would not
resolve a nine-task, twelve-update budget by itself.

The original LR decayed from 1e-6 to about 8.33e-8 within twelve steps. This is
a mechanics smoke test, not a meaningful learning-budget experiment. Earlier
whole-agent validation included two-edit tasks outside the RL objective:
30→31/100 grounding and 86→87/100 labelled selection, without statistically
supported gain. Construction and retention were unchanged. No RL improvement
claim is supported by that result.

## Changes implemented

- `grpo_audit.py`: reproducible offline native reward cross-check.
- `bandit_evaluate.py`: held-out single-edit evaluation, same normalized public
  prompt, max 128 tokens, EOS union and fixed finish as training. Before/after
  runs use V3 validation: 248 tasks, 62 physical parent groups. Completed
  result: **84/248 → 87/248** (33.87% → 35.08%), paired gain +1.21 percentage
  points, cluster-bootstrap 95% CI **[0, +2.82] pp**. Improvement is not
  statistically supported. Reports: `artifacts/grpo-matched-evaluation/`.
- Calibration now explicitly uses temperature .8, top-p 1, top-k 0 and the union
  of model/tokenizer EOS IDs, matching the trainer. Previously calibration
  inherited some generation defaults. New training rejects old calibration
  reports that do not record this sampling profile.
- Every new native reward call records its sample indices, completions,
  task identities, rewards and task-group statistics in `reward-batches.jsonl`.
- GRPO exposes `--lr-scheduler-type constant` for a deliberate longer run;
  linear remains the historical default.
- SFT preprocessing fills missing public teacher-history `response` fields
  using canonical JSON of the already-visible past action. Frozen corpora
  remain intact. This removes a train/deployment observation mismatch.

## How much data we actually have

Native-audited expanded FreeCAD and CadQuery each have 316 accepted episodes,
zero rejections, and 1,786 TRAIN feedback rows. Each has **104 TRAIN, 33 validation,
21 test physical groups**. They share the same underlying designs; adding the
second engine does not double independent geometries.

The new `data_census.py` structural diagnostic found just **11** reference
feature-graph signatures per expanded mechanical engine, **3** in simple
replay, **11** in complex replay and **5** (hole-count variants of one pattern)
in visual V3. Thousands of rows therefore do not imply thousands of modeling
strategies. Signatures ignore dimensions/names and preserve dependencies;
they are not a general CAD diversity metric. Full census is saved in
`artifacts/training-data-census.json`.

The replay sources contain 2,002 simple rows / 169 TRAIN groups, 2,900 complex
rows / 146 groups, and 7,884 visual-grounding rows / 789 groups. Old visual
curricula train one/two selected holes while testing three. Existing difficult
failures therefore include a real composition gap. Old C SFT used only 1,600
sample exposures against 12,786 rows (roughly .125 epoch); it was not converged
large-data training.

The advanced `solidcad` native feature corpus covers fillet, chamfer, shell,
revolve, polygon/arc extrusion and multiple solids. Its whole-program export
is **not** currently part of the single-action SFT pipeline. A native executor
passing tests is not evidence that the model learned those operations.

Blender fixed shards still rejected two bolt-flange episodes of one TRAIN
geometry. They are not accepted training data. Blender is excluded from the
new scale experiment until native validity is fixed and the complete corpus
is re-audited. Accepted FreeCAD/CadQuery data need not wait for it.

## Concrete scale experiment

`recipes/multicad-scale-qwen35.json` starts from SFT-C, using both accepted CAD
corpora and the three replay sources. It uses vision LoRA, LR 2e-5, numeric
weight 8, batch one × accumulation eight, **2,000 steps**, 8,192 context,
327,680 pixels and at most four images. Each of five sources has an explicit
4,096-row budget: **20,480 prepared rows with repetition**, at most 12,570
distinct original rows selected. There are 16,000 training-example exposures,
roughly .78 of that prepared mixture. These are not 20,480 different designs.
Small corpora cycle; larger sources are sampled deterministically.

The driver was launched on physical GPU 1. Matched RL evaluation completed;
the driver proceeded to `baseline_freecad` before SFT.
It audits sources, measures parent validation, trains, merges, measures candidate
validation, old-task retention and goal-image removal. Physical-group paired
confidence intervals and validation gates control whether final test is opened.
Training completion and improvement must be read from the resulting reports;
launching the driver does not establish either.

The executable source is copied into `artifacts/frozen-code/multicad-scale-v1`
with SHA-256 manifest, so ongoing development cannot change this experiment.
The old Blender-blocked waiting driver was stopped before any optimizer run.
The original v1 subsequently failed during baseline evaluation; the repaired
v2 status is `artifacts/multicad-scale-experiment-v2/status.json` (details below).

## Better data, not only longer training

`cad_env.learning.visual_scale` implements a fresh V5 curriculum. Defaults:
512 IID physical boards and 32 composition-holdout boards, each with four
counterfactual marker selections and labelled/unlabelled variants: **4,352
requested episodes** before native validation. TRAIN/IID use 1–4 edits;
composition TEST uses 5–6. Millimetres and inches appear in training. Teacher
edit order is shuffled; both UI variants share the physical split. Each
condition group retains four equal-instruction/equal-initial-state alternatives.
It generates actual goal images, native teacher trajectories, feedback exports,
then runs the existing visual information-control/native audit. The 40-episode
pilot passed native grading with zero rejects and the visual audit. Its first
run exposed a factory deduplication bug: labelled and unlabelled images with
the same CAD were collapsed. The deduplication key now includes the private
goal specification while physical split lineage remains unchanged; the rerun
retained all 40 episodes. All 112 TRAIN feedback states also passed the actual Qwen3.5 processor
preflight (maximum 1,856 tokens, all completion/image/template masks checked).
Full 4,352-episode generation was launched on CPU.
New training
on three-edit examples invalidates interpreting old three-edit tests as unseen
compositions; V5 has a fresh holdout. This new corpus is not silently added to
the already-frozen scale experiment.

Next stages should use V5 only after complete native audit and processor
preflight, then mix actual on-policy repair trajectories. Increase independent
sketch/feature graphs and advanced operations after integrating their public
observation/action contract, rather than multiplying four mechanical templates.

## Next RL experiment and acceptance criteria

After selecting an improved SFT checkpoint on validation, regenerate calibration
with the matched sampler. Use the full eligible TRAIN distribution, not only
nine hand-picked mixed prompts; begin around G=8, constant LR 1–2e-6 and a
512-step budget, first checking memory and throughput. These are proposed
hyperparameters, not a validated recipe. Log mixed-group coverage, unique
geometries, entropy of actual editable choices, truncated/invalid actions,
KL and gradient norms. All-zero groups need curriculum/SFT/exploration, not
reward inflation. Keep binary native success as the primary endpoint; any
shaping needs separate reward-hacking tests and an unchanged strict evaluator.

Do not run this concurrently with SFT on GPU 1. Multi-turn RL requires a new
rollout collector that executes sampled tools, returns public native feedback,
and masks loss to assistant tokens across complete episodes. The current
forced-finish bandit must not be presented as that implementation.

Success requires matched held-out native success gain with physical-group
confidence intervals, no retention collapse, and image interventions showing
the model uses the goal. Report execution failures in the denominator. Training
reward alone, or accuracy after selecting only valid outputs, is insufficient.

## Verification

The combined relevant regression suite passed **84 tests** (three warnings),
including native teacher exports, learning-process checks, numerical loss,
visual controls, reward-audit false-positive detection and zero-signal batch
logging. Ruff and `git diff --check` passed. Historical frozen FreeCAD
controller hashes are unchanged. The new V5 processor preflight and 40-case
native pilot are separate from model training and benchmark results.

## Primary references

- [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer): group
  advantages, reward scaling and trainer instrumentation. Exact installed
  behavior was checked in pinned TRL 1.17.1 source.
- [IterCAD](https://arxiv.org/html/2606.13368v1): progressive expert SFT,
  on-policy refinement and geometry-aware RL; useful recipe principles for
  iterative CAD agents. Our typed FreeCAD tools and one-action bandit do not
  reproduce its CadQuery training system or published results.

## Research-driven quality plan — follow-up review

### Assessment of the current design

Strong foundations: actual native execution, immutable task/source provenance,
physical-parent splits, counterfactual image controls, strict protected-parameter
checks and failed rollouts in the metric denominator. These are worth keeping.

The main shortcomings are capability coverage and state distribution. Seven
portable actions are a useful controlled interface, but they do not express
arbitrary sketch constraints, face selection, non-XY placements, assemblies or
native Python workflows. Three executors consuming this interface establish
portability of a narrow policy, not mastery of three CAD applications. Existing
advanced native probes have not yet been incorporated into the model's agent
contract or training. Strong scores on generated circles and stepped shafts
cannot establish general FreeCAD competence.

Goal-image selection, dimension reading and CAD execution must be measured
separately. The counterfactual marker benchmark tests selection; dimensioned
construction drawings test a different capability. Explicit entity labels help
binding a visible feature to a tool identifier, but labelled and unlabelled
accuracy must remain separate. Multi-view renderings, occlusion, cross-sections,
hidden holes, section drawings, arbitrary coordinate frames and genuinely new
feature graphs are important missing strata. Synthetic text-heavy drawings
alone do not demonstrate transfer to engineering drawings.

### Which published approaches are useful

1. **IterCAD: progressive SFT and on-policy refinement.** Its Qwen3.5-4B recipe
   uses expert trajectories, teacher-corrected trajectories collected from its
   own policy, then geometry-aware RL. Its reported setup uses 8 A100s; it is
   not a directly transferable RTX3090 budget. Borrow the training stages and
   validation discipline first. Prefix masking is a later experiment, after we
   have full multi-turn rollouts and correct assistant-token masks. Do not
   indiscriminately mask all suffixes after the first failure: later successful
   recovery is exactly what we want to learn.
   Source: https://arxiv.org/html/2606.13368v2

2. **GIFT: reuse inference to diversify outputs and target structured mistakes.**
   It uses soft rejection sampling and rendered near-miss inputs. The useful
   transfer is spending generation compute on examples the current model gets
   wrong, rather than only more random dimensions. Its approximate IoU thresholds
   are inappropriate acceptance gates for our exact edit tasks. A safe adaptation
   pairs an incorrect CURRENT CAD state with the unchanged GOAL image and an
   explicit corrective instruction; the repaired teacher action is executed and
   strictly graded. Do not replace a precise goal by a wrong rendering while
   retaining its old exact answer without clarifying the task.
   Source: https://arxiv.org/html/2603.27448v1

3. **DAgger: train on states induced by the learned policy.** Perfect teacher
   prefixes omit wrong feature names, partial geometry, invalid tools and wrong
   but executable edits. Run the policy on TRAIN, obtain a correct action for
   each visited recoverable state, aggregate those examples, and repeat. The
   expert must solve the actual current state: replaying the original next
   oracle action after a wrong edit can teach an invalid recovery. Our existing
   on-policy repair corpus is a small first mechanism (172 feedback rows), not
   yet a broad multi-turn implementation.
   Source: https://arxiv.org/abs/1011.0686

4. **cadrille / CAD-Coder: geometric verification and multiple valid programs.**
   Both motivate feedback beyond token matching; cadrille also mines hard
   examples. Keep alternative valid feature orders/programs when they preserve
   the requested editable structure. Chamfer/IoU are useful ranking diagnostics
   but cannot replace dimensions and protected-feature checks. In our saved
   wrong-hole example geometric score was about .9903 with unchanged total
   volume: almost-perfect global similarity still meant a wrong task answer.
   Sources: https://arxiv.org/html/2505.22914v3 and
   https://arxiv.org/html/2505.19713v3

5. **DAPO: dynamic informative sampling.** Zero-variance reward groups contain
   no actor learning signal. Recalibrate difficulty as the policy changes,
   retain a coverage floor over all TRAIN task types, and cap resampling cost.
   A fixed list of nine mixed prompts is not dynamic sampling. Avoid permanently
   discarding all-zero tasks: they need SFT/repair data before RL becomes useful.
   Do not copy the paper's full-scale infrastructure into our single-GPU setup.
   Source: https://dapo-sia.github.io/

6. **VAGEN / CAD-Assistant: explicitly reason about current state and effects.**
   VAGEN motivates state estimation and next-state prediction; CAD-Assistant
   demonstrates FreeCAD tools for rendering and geometric inspection. A bounded
   auxiliary TRAIN dataset can ask which visible feature changed, its support,
   whether an operation adds/removes material, or which cross-section exposes a
   hidden feature. Targets must be derivable from the supplied public state and
   views. A learned world-model explanation is not proof of CAD correctness;
   native outcomes remain authoritative. Tool/observation changes need a new
   frozen benchmark version.
   Sources: https://vagen-ai.github.io/ and https://cadassistant.github.io/

### Prioritized experiments on physical GPU 1

**A. Finish the frozen scale-control experiment.** The initial v1 driver stopped
before optimization because SciPy was missing in the VLM evaluation environment.
SciPy 1.18.1 was installed without changing Torch/NumPy, pinned in the Qwen lock,
and an early dependency check was added. Python executable paths now preserve
venv symlinks instead of resolving them to the base interpreter. A new frozen
v2 run preserves the failed v1 report. Current status is in
`artifacts/multicad-scale-experiment-v2/status.json`. No SFT gain has yet been
established. This long run is a baseline, not evidence that scale alone suffices.

**B. Train on V5 at matched compute, then add real repairs.** Full generation
completed: **4,352/4,352 accepted, zero rejected or deduplicated**, 544 independent
physical groups, **11,104 TRAIN feedback rows**; the complete visual audit passed.
First compare replay-only vs replay+V5 from the same parent at equal update and
example budgets. Then compare V5 vs V5+on-policy repairs with the same budget.
Do not add new data silently to an already-frozen run. Recipe proportions should
be tuned on validation; an initial proposed mixture is 40% new task trajectories,
30% replay, 20% actual repair states, 10% public geometric perception/effect
examples. These are starting hypotheses, not demonstrated optimal weights.

**B2. Fix unit diversity and target binding.** V5 currently uses only
`0.625 inches` for imperial examples; learning its conversion to 15.875 mm is
not evidence of general unit conversion. A fresh next version should vary
imperial magnitudes, decimal/fraction notation and equivalent mm requests,
with disjoint value holdouts and an explicit test for unit conversion alone.
Do not alter the completed V5 in place. Treat public entity-ID overlays and a
deterministic unit-conversion/geometry-query tool as agent-interface ablations:
they can improve the product, but they change the protocol and cannot be used
to claim improvement of the unchanged raw model. Public coordinates or selected
entity bindings must come from the image/actual model, never private goals.

**C. Increase structural diversity before expanding engines.** Integrate one
advanced operation into the public agent contract at a time: revolve, fillet,
chamfer, shell, support-plane selection and general sketch constraints. For each,
require positive native examples, wrong-dimension/topology negatives, an edit
probe, sequential teacher export and a processor preflight. Sample dependency
DAGs, not just template dimensions; split at graph skeleton and parent-design
level. Keep a composition holdout built from known primitives and a separate
unseen-operation holdout. Pure extrusion/pocket tools cannot learn fillets.

**D. Use validation milestones, not only a final training loss.** In the next
version evaluate checkpoints around 400/800/1200/2000 updates, select on a
predeclared validation score and retain the test sealed. Run inference and
training serially on GPU 1. Monitor strict task success, retention and action
length, not teacher-forced numeric accuracy alone. Repeat the winning small
comparison with multiple seeds before spending a large RL budget. The currently
running frozen recipe evaluates its final checkpoint; it does not implement
milestone selection and should not be described as doing so.

**E. Multi-turn RL after a repair-capable SFT.** Collect sampled assistant actions,
execute each tool, return actual public observations, preserve log probabilities
and mask observations/images out of the policy loss. Optimize complete episodes,
including finish, retries and protected-feature preservation. Begin with short
2–4 action tasks, then expand horizons. Compare SFT-only vs SFT+RL at the same
inference budget; larger sampling alone must be a separate reported baseline.

### How to prove the improvement

Pre-register validation strata: visible feature selection, exact unit/dimension
conversion, feature dependencies, multi-edit stopping, invalid-tool recovery,
wrong-but-executable geometry recovery, and unfamiliar feature compositions.
Use the same immutable model, sampler, context, views and tool budget for both
sides of a comparison. Score full episodes and count every failure; cluster
confidence intervals by physical parent across views and engines. Preserve
parametric history and test whether a requested dimension remains editable.

For visual necessity, compare full goal against blank goal, mismatched goal,
removed goal and true text-only tasks. In some dimensioned drawings the text
already contains enough information; a no-image success there is expected, not
cheating. Use counterfactuals that keep public text/current CAD identical for
claims about image dependence. For longer horizons report success vs allowed
actions, attempts and GPU time; pass@k is not pass@1. Use an external FreeCAD
benchmark as a separate transfer study only after an adapter exposes the tools
its tasks require; do not map unsupported tasks to an easier subset and report
the result as official CADBench/CADWorld performance.

## Implemented follow-up experiments

The 2,000-step scale control has entered real SFT on GPU 1 (about 270 steps
observed at this follow-up). Its code and source mixture remain frozen.

`visual_units.py` adds V6 with fresh seed range starting at 2,000,000. It pairs
identical geometric requests in mm and inches, with and without entity labels.
TRAIN/IID use .375/.5/.625 inches; a separate value holdout uses .4375/.5625
inches and 1–4 edits. Composition holdout uses 5–6 edits with known training
values. Each case has four counterfactual selections and all UI/unit variants
share the physical split. Full generation completed: **2,560/2,560 episodes**, zero rejects/duplicates,
160 physical boards. Native and visual audits passed. TRAIN has 5,568 feedback
rows; validation 928 and test 3,232. Validation contains 17 physical boards
(68 correlated condition groups), so this ablation is exploratory; do not
count the UI/unit variants as independent designs.

The 96-episode pilot passed native grading with zero rejects/duplicates and
passed the information-control audit. It exposed a floating-point audit bug:
binary conversion of .375 inches differed from the equivalent 9.525 mm literal.
The auditor now converts decimal specifications exactly using `Decimal` before
creating geometry floats. Geometric tolerance is unchanged. Factory deduplication
also includes the instruction, preserving equivalent mm/inch task conditions
without counting them as independent geometries.

SFT now accepts `--source-row-budgets` in dataset order, mutually exclusive with
an equal `--rows-per-source` budget. Positive integer counts, unique sources and
matching budget lengths are checked. Small sources cycle explicitly; repeated
rows do not become new designs.

`recipes/curriculum-ablation-qwen35.json` defines three independent arms from
SFT-C, each with 400 updates, effective batch eight and 3,200 prepared rows:

- replay: mechanical/simple/complex/old-visual data;
- visual: replace part of replay with V6;
- repair: replace 320 V6 rows with 320 rows from the existing independently
  native-verified on-policy repair corpus (172 original feedback rows / 16
  parents, so some repeats are deliberate).

This is a controlled mixture comparison, not an assertion of optimal weights.
Repairs come from the same SFT-C policy used as parent, not from the currently
running scale model. A later round must refresh them from the selected policy.
The new runner checks native TRAIN audits, waits for the current GPU job to
finish successfully, then trains/evaluates the arms serially. It records paired
physical-group confidence intervals on V6 validation, old simple/complex/entity
retention and no-goal/blank-goal/wrong-goal interventions on the highest-validation
arm whose retention regression is at most five percentage points. If none
passes retention, no candidate is selected. TEST remains
sealed. Stage failures persist an explicit failed state; they do not count as
completed model tests. Frozen runner status:
`artifacts/curriculum-ablation-v2/status.json`. The waiting v1 was superseded
before any optimization to correct candidate selection for image controls.

A 160-row V6+repair Qwen processor preflight passed all completion/image/template
mask checks (maximum 1,851 tokens). All three additional 3,200-row per-arm preflights passed (maximum 3,167
tokens), using the audited V6 pilot to exercise budget routing; the actual SFT repeats preflight
on its selected full-corpus rows. The focused native/regression suite passed
29 tests, and Ruff / `git diff --check` passed. These checks establish pipeline
readiness, not learned CAD accuracy.

Broader action tools and actual multi-turn policy-gradient RL remain subsequent
implementation stages. They have not been silently claimed as implemented by
these data/SFT improvements.

V6 includes 12.7 mm / .5 inches in TRAIN. Old benchmarks holding out that
unit value are consequently no longer unseen-value tests for V6-trained
models. Use V6 fresh value holdout and disclose the curriculum version.
