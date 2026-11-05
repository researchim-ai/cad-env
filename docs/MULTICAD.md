# Native multi-engine environments

`cad_env.multicad` is an optional extension alongside the existing FreeCAD
agent. Supported engines are `freecad`, `cadquery`, and `blender`. The original
FreeCAD worker, action schema and frozen evaluator remain unchanged.

## Execution contract

The same seven typed operations are interpreted by each engine: create a body,
rectangle/circle sketches, pad, pocket, parameter edit and finish. The portable
subset is one solid body, XY sketches, disjoint circles per profile and pad-linked
support planes. Nested circles in a single sketch are rejected; create bores
with pockets. Arbitrary Python, paths and unsupported operations are rejected.
Parameter changes rebuild dependent support heights, including after editing an
existing part. An explicit `finish` is required for strict task success.

FreeCAD uses its existing native constrained sketches and feature history.
CadQuery uses exact OCCT solids and exports STEP plus its replayable JSON source.
Blender uses actual polygonal meshes and exact-solver boolean modifiers, exports
`.blend`, and retains the action source as an object property. Source replay is
the editing contract for these latter engines: neither a STEP file nor an applied
Blender boolean claims to preserve a FreeCAD-style constrained feature tree.
Blender currently supports mechanical geometry through this adapter, not general
materials, animation, Geometry Nodes, sculpting or scene-composition tasks.

Each process receives validated JSON and runs trusted adapter code, with timeout
and isolated config directories. Process isolation is not an OS security sandbox.
Geometry workers hide CUDA devices and do not render on the training GPU.
Outputs include native runtime versions, source hashes, meshes and structured
results. Missing runtimes/crashes produce failures, never synthetic positives.

## Setup

Tested separately from the model-training environments with Python 3.12,
CadQuery 2.8.0 and official Blender 4.5.9 Linux x64. The downloaded Blender archive
SHA256 was `dcdc3eca6c9825bb35a8033b689c053f3cb5a9b0cd2a61b2eac2a49436b4ad3d`,
verified against the official `blender-4.5.9.sha256` release file.

```bash
python3 -m venv .venv-multicad
.venv-multicad/bin/python -m pip install -r requirements-multicad.lock
export BLENDER_BIN="$PWD/artifacts/runtimes/blender-4.5.9-linux-x64/blender"
# If FreeCAD is not on PATH: export FREECAD_CMD=/path/to/FreeCADCmd
```

Download a pinned Blender binary and its checksum from the
[official release directory](https://download.blender.org/release/Blender4.5/),
verify the checksum, and extract under the ignored `artifacts/runtimes/`.
`BLENDER_BIN` can point to another installation; its actual version is recorded.
Alternatively a Python interpreter with compatible `bpy` can be passed using
`--python`. Python 3.12 in this workspace had no matching bpy wheel, so native
Blender is used instead. Never install this geometry stack into the GPU training
venv: it has its own NumPy and other dependencies.

## Generate and inspect data

```bash
.venv-multicad/bin/python -m cad_env.multicad generate \
  --engines freecad cadquery blender \
  --seeds 3 --modes create edit --modalities image \
  --view-mode multiview --output artifacts/my-multicad-corpus

.venv-multicad/bin/python -m cad_env.multicad audit \
  --dataset artifacts/my-multicad-corpus
```

Original procedural families: bolt flanges with a central bore, counterbored
plates, coaxial stepped bores and grids of blind holes. Modes include construction,
dimension editing and correction of an erroneous initial dimension. The latter
repair tasks are synthetic; they are not on-policy repair trajectories.

Targets include a dimension specification image plus either one view or a
four-view collage. The text condition also duplicates dimensions in the prompt;
it is text plus images, not a pure-text ablation. A current-state image accompanies
every next-action SFT row. Before a solid exists, feedback describes the validated
feature graph and an empty view; the new environment does not claim native sketch
solver verification at that point. After solid features exist, geometry is executed
natively. Recorded teacher geometry is independently compared with a reference and
analytic volume/bounds before any positive rows are accepted.

The corpus stores private task definitions, reference and teacher results, native
projects, all goal views, current observations and split-specific feedback JSONL.
Only `exports/train.feedback.jsonl` is accepted by the training gate. Public model
messages exclude oracle actions, private analytic targets, task IDs and scores.
The image condition reads dimensions from goal pixels; the other geometry views
are not claimed independently essential because the specification is sufficient.

Normalized base/cutter geometry defines `split_group` for these one-base,
commuting-cut families; this is not a general BRep equivalence solver: all engines, text/image variants and
create/edit/repair versions stay together. This also allows the existing mixed-SFT
cross-root split guard to detect conflicting assignments. A small smoke corpus may
have no validation or test cases; it is not a publishable benchmark. Examine actual
split and lineage counts before measuring generalization.

## Agent evaluation and training integration

```bash
CUDA_VISIBLE_DEVICES=1 .venv-qwen35/bin/python -m cad_env.multicad evaluate \
  --dataset artifacts/my-multicad-corpus \
  --model artifacts/vision-leap-sft-c-merged \
  --engine blender --python "$PWD/.venv-multicad/bin/python" \
  --output artifacts/my-multicad-evaluation
```

This command competes for the training GPU: run only once its current queue is
free. Evaluation defaults to the full validation split; final test requires
`--split test --final-test`. Partial/crashed runs remain incomplete. Native strict
success is primary; sampled fixed-coordinate Chamfer/coverage diagnostics are
separate and are not official CADBench metrics.

`MultiCADEnv` in `cad_env/multicad/environment.py` exposes `reset`, `messages`, and
`step`. Invalid actions consume budget and return errors without accepting the
failed program. The same message builder is used for the exported supervision.
The audited feedback export can be used with `cad_env.learning.train_modern`
through `--dataset` or `--extra-dataset`; keep replay and retention evaluation.
A compatible column format alone is not proof that multi-engine SFT has been run:
no model-learning result on this new corpus is claimed yet.

`MultiCADReward` in `cad_env/multicad/rewards.py` is an offline/native reward hook
for **whole-program JSON-array** completions on registered TRAIN tasks. Its
contract differs from the one-action contextual-bandit GRPO pilot. Do not attach
it to that pilot or label multi-turn policy-gradient training as tested. The hook
is preparation for a separate training recipe with matched evaluation.

## Verified implementation evidence

Implementation checks completed with real installed runtimes:

- The native multi-engine checks passed, including correct/wrong/unfinished geometry in
  all three engines, parameter editing, invalid-action rejection, dataset audit
  and training-source tamper rejection. A separate prompt-parity test confirms
  the first exported observations match `MultiCADEnv.messages()`.
- All 91 pre-existing tests passed in the original learning environment.
- Three existing complex families (stepped shafts, boss plates and perforated
  plates) also passed native reference/analytic checks in CadQuery and Blender.
- `artifacts/multicad-sft-v2-smoke` contains six accepted native episodes and 30
  audited next-action rows, across three engines and create/edit modes. Its one
  physical design is only an integration smoke, not generalization evidence.
- Qwen3.5's actual processor and TRL vision collator passed all 30 rows, including
  completion-only masks, masked image tokens and template boundaries; the maximum
  sequence was 2,039 tokens under an 8,192-token budget. This was a CPU preflight,
  not an optimizer run. Evidence: `artifacts/multicad-training-preflight.json`.

A larger 72-episode generation job writes to
`artifacts/multicad-mechanical-v2-verified`; use its final summary and audit to
check completeness. The earlier v1 development corpora are superseded: their lineage hash used
construction order rather than normalized cutter geometry. They must not be
used for training or benchmark evidence. The v2 directory uses canonical
geometry groups, ignoring object names and hole construction order. The executor now rejects source changes within a run.

After the lineage correction, the full native suite passed **19/19 tests**.
`artifacts/multicad-test-v2-smoke` has six accepted TEST episodes (one separate
physical design, three engines and create/edit modes), with 54 held-out feedback
rows. The Blender sequential evaluator completed its two episodes using an
explicit oracle replay policy. This is a harness check, not model competence.
The native Blender reward hook returned 1 for the correct edit, 0 for a wrong
dimension and 0 for malformed output, and rejected a held-out task identity.
Evidence lives in `artifacts/multicad-blender-v2-oracle-evaluation/evaluation.json`
and `artifacts/multicad-reward-v2-smoke/verification.json`.

## Expanded mixed-SFT experiment

The approved follow-up is defined by `recipes/multicad-mixed-qwen35.json` and
`cad_env.learning.multicad_experiment`. Three CPU generation jobs produce separate
engine roots `artifacts/multicad-expanded-{freecad,cadquery,blender}` from 40 seeds
per family. All use the same canonical lineage split. After duplicate removal,
the FreeCAD root has 158 independent designs / 316 create+edit episodes, including
33 validation and 21 test geometries. Other engines are verified independently.
These new tasks randomize the initial thickness error instead of always using
+3 mm; `--varied-edits` records its curriculum version. Old v1 data remain readable.

The GPU driver waits for audited corpora and the earlier GPU queue. It evaluates
the same checkpoint C on each engine, trains 400 mixed SFT steps, merges, repeats
native validation, removes goal images as a causal control, and rechecks four
previous FreeCAD suites. Six sources receive an explicit budget of 1,024 rows
each: three new engines, simple replay, complex construction and visual grounding.
Small sources cycle deterministically; duplicate dataset paths remain forbidden.
Only TRAIN rows are sampled. Test remains unopened until prespecified validation,
image-use and retention gates pass. Source drift or a partial stage stops the run.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-qwen35/bin/python -m cad_env.learning.multicad_experiment \
  --recipe recipes/multicad-mixed-qwen35.json
```

The current run is already queued; do not launch a competing copy. Its state is
`artifacts/multicad-mixed-experiment-v3/status.json`. The previous GRPO pilot is a
separate intervention; mixed SFT starts from C to keep that comparison interpretable.
Preparation and queued commands are not completed optimizer or learning results.

Blender expansion runs as eight five-seed CPU shards, four at a time, using
`cad_env.learning.multicad_shards`. Each shard is natively generated and audited;
merging preserves task IDs, lineage splits, images and source hashes and removes
duplicate physical targets. The GPU recipe reads
`artifacts/multicad-expanded-blender-fixed`. The original sequential Blender
job and the first waiting GPU driver were superseded before any mixed-SFT update.

Current implementation verification: 95 main/experiment tests and 47 native
portable/advanced tests passed, plus the dedicated native shard-merge test.
The latter verifies duplicate removal, unchanged splits, asset integrity and a
successful audit after merging overlapping real FreeCAD shards. Advanced feature
training remains separate; its native capabilities and data are documented in
[SOLIDCAD.md](SOLIDCAD.md).


Blender grading repair: fractional-depth cylinder caps are baked onto the exact
requested Z planes, preventing float32 gaps from creating unintended enclosed
cavities. Ground-truth comparison now runs in a separate host-Python Manifold
process on the actual closed meshes. It checks symmetric difference, bounds,
analytic dimensions and body count without loosening tolerances; Blender's own
coplanar A-A boolean is not trusted for rewards. `requirements-multicad.lock`
includes manifold3d. Use the geometry Python argument when the policy runs in the
separate GPU environment. Older rejected Blender shards remain quarantined.

Native recording now reuses unchanged geometry and extends an unchanged feature
prefix incrementally; parameter changes still rebuild dependent geometry.
Regression tests cover previously failing fractional depths, complete recording,
wrong hole locations with the same overall bounds, and equivalent reordered cuts.
All 51 portable/advanced native tests passed after these repairs.

The current driver is detached from its terminal, with PID in
`artifacts/multicad-mixed-driver.pid` and log in
`artifacts/multicad-mixed-daemon.log`. Always inspect both process liveness and
status JSON; a stale waiting file does not prove a running queue.
