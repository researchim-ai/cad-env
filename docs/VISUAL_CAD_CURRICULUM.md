# Visual CAD curriculum and evidence contract

This experiment trains Qwen3.5-4B for native FreeCAD tools on physical GPU 1
(RTX 3090, 24 GB). The starting checkpoint is `artifacts/qwen35-sft200-merged`,
which already has 200 steps of tool SFT. This is a continued-training experiment,
not a comparison of a new CAD model against an untouched foundation model.
The original foundation revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## What has actually been established

The previous 69-task validation set was solved 69/69 both with pictures and
with numeric state alone. That establishes bounded tool competence; it does
not establish visual grounding. Token accuracy is also insufficient: on the
new image-essential baseline, every first command executed, but only 35% of
first commands addressed a correct target and only 1/100 complete tasks passed.

Frozen, complete baselines for the current experiment:

- `vision-leap-baseline-grounding`: 1/100 native successes, strict mean reward
  0.01, on 25 groups of four counterfactual tasks.
- `vision-leap-baseline-complex`: 17/46 successes; 5/23 image-conditioned and
  12/23 text-conditioned. Mean dense reward 0.70569 includes partial credit.
- `vision-leap-baseline-retention`: 69/69 successes on the previous validation.

Stage A completed with 12/100 successes after 200 steps, versus 1/100 before.
The paired improvement is 11 percentage points (parent-group bootstrap 95% CI
5–17 points). This is below the predeclared 80% pilot gate.

Stage B completed 300 mixed SFT steps with vision adapters. Complete native
validation improved from 17/46 to 40/46 complex tasks: image 5/23 → 19/23,
text 12/23 → 21/23. The overall paired gain is 50 points (95% parent-bootstrap
CI 32.6–67.4); the prespecified image slice gains 60.9 points (CI 34.8–82.6).
Retention is 68/69 versus 69/69. Unlabelled marked-hole editing is still weak:
17/100 versus 1/100 (gain 16 points, CI 9–23). These are Stage B validation
results, not final test or image-ablation results.

Current run status lives in `artifacts/vision-leap-experiment/phase-a.json` and
`phase-b.json`, `phase-c.json`, `phase-d.json` and `phase-e.json`. A completed training log alone is never promoted to evidence
of CAD improvement. Only complete native evaluations and matched comparisons
qualify. Intermediate validation results must be labeled incomplete.

## Published recipes and the relevant distinctions

[IterCAD](https://arxiv.org/html/2606.13368v1) uses Qwen3.5-4B with dimensioned
multiview drawings and execution/visual feedback. Its recipe progresses from
expert trajectory SFT to on-policy repair SFT and geometry-aware GSPO with
Geometry-Viable Prefix Masking. Its CAD execution interface is CadQuery.
The transferable lesson is to teach valid trajectories first, then recovery,
and include failed generations in geometric evaluation. Its reported results
are not directly comparable to our native FreeCAD tool harness.

[CADBench](https://arxiv.org/html/2605.10873v1) broadens reconstruction evaluation
across CAD sources, geometry complexity, mesh noise and rendering styles.
It evaluates Qwen3.5-9B/27B, rather than providing direct evidence for our 4B
checkpoint. The useful lesson is to stratify complexity and observation shifts,
and report geometry quality alongside execution success.

[Parametric CAD Bench v3](https://arxiv.org/html/2611.02659v1) evaluates native
FreeCAD creation, creation/editing and drawing reconstruction with a separate
programmatic verifier. Geometry, dimensions and feature history all matter;
its leaderboard measures complete agent systems. Its
[validator](https://github.com/gNucleus-AI/freecad-validator) is a useful
external validation target. We have not run this external benchmark.

[CADWorld](https://cad-world.github.io/) evaluates GUI agents. Our restricted
JSON tool agent does not have the same action space or interaction budget.
[The official model card](https://huggingface.co/Qwen/Qwen3.5-4B) reports general
spatial and GUI capabilities; these do not establish FreeCAD competence.

## Data that require images

`visual-grounding-v1` has 832 accepted native tasks: 600 train, 100 validation,
132 test, grouped by parent geometry. A group contains four different marked
hole selections on the same initial part. Text, initial state, editable object
names and target aggregate volume/bounds are identical within the group.
Only image pixels identify the selected holes. Circle order and object aliases
are randomized. Agents receive current circle coordinates, not target labels.

For a deterministic policy without images, at most one of the four different
final geometries can be correct. Thus 25% is an information ceiling on this
complete grouped benchmark, not a universal random-guess baseline. The audit
verifies identical public inputs, distinct answers and images, correct marker
projection, split integrity and no cross-split physical geometry aliases.

`visual-grounding-v2` keeps parent splits and varies requested diameters
(10/12/14 mm in training). Dedicated test groups ask for 13 mm and 0.5 inches.
This prevents learning “always write 12” and tests numerical/unit transfer.
Both versions reserve tasks with three selected holes and changed rotations
and marker colors for testing. These are synthetic stress tests, not evidence
of robustness to arbitrary engineering scans.

The model sees a static goal image and numeric current-state feedback.
These tasks establish use of image-conditioned goals if controls succeed;
they do not establish reliance on images of the changing CAD viewport.

## Complex constructions

`complex-cad-v3` has 424 accepted tasks, paired text/image conditions on 212
parent geometries: 292 train, 46 validation, 86 test. Training contains eleven
construction graph signatures. Teachers use 8–14 actions, with supported
multi-level pads, boss patterns, coaxial holes and independent blind/through
pocket groups. Test includes longer shafts and mixed boss/pocket graphs absent
from training. Both text and image conditions are included for those graphs,
so a composition failure can be distinguished from a drawing-reading failure.

Image instructions omit the target dimensions; these occur in a 512×640
synthetic engineering drawing. Text conditions specify dimensions explicitly.
This supports separate measurement of instruction following, drawing reading
and sequential construction. Native oracles all passed execution, analytic
volume/bounds, geometric equivalence and parametric validity checks.

These datasets extend composition within the existing seven-operation API.
They do not add fillets, chamfers, shells, revolutions, arbitrary sketch curves,
assemblies, CAM or FEM. “More complex” must not be presented as unrestricted
industrial CAD coverage.

## Reward and evaluation protections

For marked-hole editing, the training/evaluation reward is strict native
success. An unchanged part, a wrong hole and an incomplete edit all score zero;
correct edits in an alternative order score one. Native calibration examples
are saved in `artifacts/visual-native-calibration/calibration.json`.

Global overlap and volume alone can reward an almost unchanged plate despite
missing a small functional feature. Dense geometric reward remains a diagnostic;
it is not the strict success criterion. Complex construction reports both
binary task success and dense reward, with failures retained in denominators.

The environment only accepts validated JSON operations. The grader checks the
native result against private reference geometry and analytic specifications,
including feature protection/editability. Target code, private task IDs, goal
specifications and image filenames are excluded from public model messages.
The policy's actual text/image hashes are logged per step.

All render variants, edit variants and scalar augmentations of one parent stay
in one split. Cross-corpus lineage is checked before mixed training. Validation
is for checkpoint selection; test and OOD are for the selected checkpoint only.
No held-out expert actions may be used to construct training repairs.

The frozen controller hashes are in
`artifacts/vision-leap-experiment/controller-freeze.json`. Before/after runs
use the same prompts, runtime, decoding, pixel/context and action budgets.
Changing tools or adding visual labels requires a new benchmark baseline.
The comparison resamples parent geometry groups, not correlated task rows.

## Training stages and what each tests

Stage A: 200 language-adapter SFT steps on image-essential v1 trajectories,
with a frozen vision encoder. It tests whether the existing visual features
can support the new coordinate-to-constraint binding. Strong token accuracy
is insufficient; native grounding and complete success decide the result.

Stage B: mixed continued SFT on variable-size visual tasks, complex drawing
and text tasks, and replay of previous train tasks. LoRA additionally adapts the final eight vision attention blocks and visual
merger. All other vision weights remain frozen. A guard verifies nonzero,
finite visual-adapter gradients, and a native checkpoint/resume smoke verifies
exact adapter restoration before the sustained run. This is a curriculum
comparison, not a controlled attribution of gains solely to vision adapters:
both data and trainable parameter sets change.

Both stages use BF16, SDPA, rank-16 LoRA, checkpointing and one visible GPU.
The modern pinned environment is `.venv-qwen35`, with dependencies recorded in
`requirements-qwen35.lock`. Every run stores source snapshots, data/export
hashes, package versions, gradients and peak GPU memory.

Predeclared pilot gates are in `contract.json`: 80% marked-hole success,
a 40-point matched-versus-no-image gap, 60% complex image success, and 95%
retention. A missed gate is a measured limitation and triggers diagnosis;
it cannot be renamed a successful qualitative jump.

## Required proof before a visual-learning claim

1. Complete matched before/after native evaluations, with group-bootstrap
   uncertainty and breakdowns by modality, family and construction horizon.
2. Matched-image versus no-goal, blank-goal and wrong-peer-goal controls.
   Wrong-goal grading still uses the original target. A separate peer-target
   check can establish coherent following of the substituted picture.
3. First-action target accuracy, complete-group success and full-task success,
   to distinguish perception from stopping, formatting or teacher order.
4. Retention on previous validation and final held-out test/OOD measurements.
5. Inspectable native artifacts and trajectories, including failures. No
   filtering for successful execution or cherry-picked examples.

Passing this establishes a bounded native CAD skill improvement. Wider claims
require an external native benchmark and new operations, plus tests on drawings
and viewport feedback beyond our own renderer.

## Next RL stage

SFT and a calibrated reward are prerequisites, not evidence that RL ran.
For GRPO/GSPO, sample several policy rollouts per TRAIN prompt and record strict
native reward, valid execution and within-group reward variance. Skip or replace
zero-variance groups; do not optimize JSON validity alone. Use full native
terminal validation and retain the SFT reference/KL control. Judge the update on
the same frozen validation contract and retention slices. On-policy repair SFT
should precede costly multi-turn RL when the student repeatedly makes the same
recoverable geometric mistake. This experiment currently implements and runs
SFT; it does not report an unperformed RL improvement.

## Reproduction commands

Run these from the repository root. Use new output directories; exporters,
training and evaluations refuse to overwrite populated outputs. The corpus
versions above are frozen local artifacts, not automatically downloaded data.

```bash
.venv-learning/bin/python -m cad_env.learning.visual_tasks \
  --output artifacts/visual-grounding-v2 --groups 192 --ood-groups 8 \
  --workers 4 --vary-diameter
.venv-learning/bin/python -m cad_env.learning.visual_audit \
  --dataset artifacts/visual-grounding-v2 \
  --output artifacts/visual-grounding-v2/information-audit.json
.venv-learning/bin/python -m cad_env.learning.complex_tasks \
  --output artifacts/complex-cad-v3 --seeds 64 --workers 4
```

The primary v1 baseline uses the same visual generator without
`--vary-diameter`. The originally frozen v1 generator predates scalar
augmentation; use the stored task/image/export hashes to identify that corpus.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-qwen35/bin/python \
  -m cad_env.learning.train_modern \
  --model artifacts/vision-leap-sft-a-merged \
  --dataset artifacts/visual-grounding-v2 \
  --extra-dataset artifacts/complex-cad-v3 \
  --extra-dataset artifacts/gpu1-corpus-v2 \
  --vision-lora --output artifacts/vision-leap-sft-b \
  --max-steps 300 --save-steps 100 --learning-rate 0.0001 \
  --batch-size 2 --gradient-accumulation 4 \
  --max-pixels 327680 --max-length 8192

CUDA_VISIBLE_DEVICES=1 .venv-qwen35/bin/python -m cad_env.learning.merge \
  --adapter artifacts/vision-leap-sft-b \
  --output artifacts/vision-leap-sft-b-merged

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-qwen35/bin/python \
  -m cad_env.learning.workflow evaluate \
  --model artifacts/vision-leap-sft-b-merged \
  --dataset artifacts/visual-grounding-v1 \
  --output artifacts/vision-leap-b-grounding \
  --disable-thinking --max-steps 8 --max-new-tokens 256 \
  --max-pixels 327680 --max-context 8192

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-qwen35/bin/python \
  -m cad_env.learning.visual_controls \
  --model artifacts/vision-leap-sft-b-merged \
  --dataset artifacts/visual-grounding-v1 \
  --output artifacts/vision-leap-b-controls

.venv-learning/bin/python -m cad_env.learning.peer_control \
  --dataset artifacts/visual-grounding-v1 \
  --evaluation artifacts/vision-leap-b-controls/wrong_goal/evaluation.json \
  --output artifacts/vision-leap-b-peer-control --workers 4
```

`visual_controls` memoizes only greedy public text plus image-content inputs;
every native episode is still executed and graded. Its latency must not be
compared against uncached runs. The cache has no access to target labels.
For final testing, explicitly select `--split test --final-test`; do this only
after freezing checkpoint selection and the reported claims.

## Additional diagnoses and follow-up experiments

Stage A selected `diameter_0` first on 87/100 validation tasks (the parent did
so on 43/100). First-action target accuracy fell from 35% to 27%, despite the
increase in full success. It nevertheless made exactly the required number
of distinct edits in every task. This separates marker counting/stopping from
binding a marked position to the correct CAD constraint. Image ablations are
needed before attributing even the counting behavior to vision.

The original multi-hole teacher used ascending circle indices. That introduces
an avoidable low-index prior. `--shuffle-edit-order` changes the order of valid
teacher actions without changing target geometry, parent splits or grading.
The expanded `visual-grounding-v3` contains 4,224 accepted tasks, zero rejected:
3,156 train tasks / 7,884 feedback rows, 520 validation and 548 test tasks.
Joint audits across the corpora found 1,539 parent groups with consistent splits.
The primary comparison still uses the original frozen 100 validation cases.

Stage C is conditional on Stage B missing the native pilot gates. It continues
from B with the expanded, shuffled corpus, complex tasks and old-task replay.
`--numeric-loss-weight 8` gives ASCII numeric tokens in assistant completions
more weight. It keeps causal next-token alignment and excludes prompt/image
positions; this is SFT weighting, not a custom RL objective. The optional
`--max-train-rows 128` selects a deterministic TRAIN subset for GPU smoke runs.
A full run omits that limit. Numerical and visual gradient guards and exact
checkpoint restoration must pass before sustained weighted training.

Numeric accuracy reported by this loss is teacher-forced training accuracy.
It is not a substitute for native validation or evidence of visual reasoning.
The pinned TRL custom-loss path may require an additional forward pass, so
measure its time/memory separately from the ordinary fused SFT profile.

An additional, separate `--entity-labels` condition overlays boxed circle IDs
on ALL visible holes. Crosshairs alone specify which holes should change.
This follows the interface idea of [Set-of-Mark](https://arxiv.org/abs/2310.11441):
make visible entities addressable in the action vocabulary. No target IDs are
inserted in text, and four-way public-input/aggregate-geometry controls remain
intact. This tests an easier visual-to-tool binding interface. It is a new
benchmark, with its own parent-model baseline; success there cannot replace a
failure on the unlabelled coordinate-binding benchmark. Its images are not
used in Stages A–C.

Future factory runs leave unconditioned text SFT/GRPO/step exports empty for
visual-goal tasks. Historical frozen corpora retain their original files and
hashes; their `supported_training_exports` gate permits only image-aware
feedback training. Use the separate Swift or bandit prompt exporters for RL.

`visual_bandit` calibrates strict native rewards on actual policy samples for
single-mark TRAIN tasks. It appends only a fixed public `finish` after the
sampled action, never an expert edit. Correct/wrong/no-op calibration is tested
against FreeCAD. Its report records mixed-reward groups and zero-variance rates.
This is a contextual-bandit preparation step; no multi-turn RL optimizer run is
implied. `control_metrics` refuses image-effect claims if model weights, cases,
budgets, runtime or decoding changed alongside the intervention.

## HTTP tool transport with goal images

The API can serve corpus tasks with public, session-scoped PNG URLs. Goal image
paths and private target IDs are not included in observations. It audits the
configured package, registers only accepted tasks, and can enforce TRAIN-only
sessions for learning services. Legacy procedural sessions remain available.

```bash
FREECAD_DATASET=artifacts/visual-grounding-v4-entities \
FREECAD_VISUAL=1 FREECAD_TRAIN_ONLY=1 \
  .venv-learning/bin/uvicorn cad_env.agent.api:create_app \
  --factory --host 127.0.0.1 --port 8000
```

Create a corpus episode with `POST /dataset/sessions` and JSON
`{"task_id":"<accepted TRAIN task ID from manifest.jsonl>"}`. The response
contains an opaque session ID, current state and `goal_images`/`images` URLs.
Fetch those PNG URLs and supply their pixels to the VLM alongside the shared
public prompt. Send each generated JSON action to
`POST /sessions/<session_id>/step` as `{"action":{...}}`.
`DELETE /sessions/<session_id>` closes the episode and image access.

The transport is intentionally sequential in this research process. Images
are small PNG byte responses; native validation and file serving are covered
by an ASGI integration test with correct edits, forbidden held-out sessions,
unknown images and session closure. Offline performance experiments above use
the same environment directly, rather than claiming an HTTP benchmark run.

The complex image-only validation subset has one shared nonvisual initial
prompt across all 23 cases. The native volume/bounds acceptance intervals
separate every target. Thus any deterministic blind policy can solve at most
1/23 cases; this is recorded in `complexity-audit.json`. Actual image ablations
remain part of the experiment rather than being replaced by that bound.

For future complex RL corpora, `complex_tasks --strict-reward` enables strict
native rewards (`complex-cad-v4-strict`). The frozen v3 experiment keeps its
original dense rewards for matched comparison. A native calibration in
`complex-reward-calibration/calibration.json` shows a blind hole incorrectly
made through receiving dense reward 0.7987 while failing the task; strict mode
returns zero. Stage B's remaining complex failures are in the perforated-plate
family, while its boss and shaft cases all passed. This motivates measuring
individual feature fidelity rather than optimizing global overlap alone.

`onpolicy_repairs` consumes a completed bandit calibration on TRAIN tasks,
independently revalidates a sampled wrong action, and constructs four repair
tasks sharing exactly that corrupted starting state. It explicitly supplies
original unmarked diameters in public instructions. Desired marked positions
still come only from image pixels. Every corrective sequence is checked against
an independently rebuilt native reference and the counterfactual information
audit. A prefix already correct for a peer target produces a legitimate
`finish`-only example. This prevents a shortcut such as always undoing the
previous edit. No validation/test errors are used to construct this corpus.

```bash
.venv-learning/bin/python -m cad_env.learning.onpolicy_repairs \
  --dataset artifacts/visual-grounding-v2 \
  --calibration artifacts/vision-leap-selected-bandit-calibration \
  --output artifacts/vision-leap-onpolicy-repairs --workers 4
```

Generation of repair examples does not imply a subsequent optimizer update.
Keep that status separate from completed SFT and future GRPO training.
