# Industrial analogue generator

The generator now has **nine procedural part families**: the original three
curriculum families plus six new families with practical mechanical layouts.
It preserves the running 90-part baseline and does not extend its frozen CAD API.

## New part families

* `adapter_flange`: circular flange, stepped hub/pilot, central through bore and
  angular bolt-circle patterns with optional counterbores.
* `bearing_cartridge`: rectangular mounting base, cylindrical housing, nominal
  bearing seat, shaft clearance, mounting counterbores and locating holes.
* `motor_mount_plate`: central pilot/bore, circular mounting pattern, external
  mounting holes and locating holes.
* `spacer_bushing`: concentric cylindrical steps, axial bore/seat and optional
  flange mounting pattern.
* `blind_cover`: flange with a **blind** central seat retaining bottom material,
  mounting pattern and counterbores.
* `alignment_fixture`: independently raised locating bosses with through/blind
  bores, mixed depths, mounting holes and locating holes.

Each has easy/medium/hard variants. Complexity increases **within each family**;
action-count ranges overlap between families. The initial industrial pilots use
10–26 / 24–64 / 38–84 teacher actions, respectively. Maximum rollout budget stays
128. These are not new tool capabilities: fillets, threads, horizontal drilling,
rotations, general profiles and assemblies remain outside the current API.

## Real references versus original geometry

These are **original procedural nominal parts, not imported industrial drawings**.
For a real reference, the bearing-cartridge family uses the published 6200 bearing
boundary dimensions: bore 10 mm, outside diameter 30 mm, width 9 mm.
[NSK's official 6200 page](https://www.nsk.com/my-en/engineering/products/bearings/ball-bearings/deep-groove-ball-bearings/single-row-deep-groove-ball-bearings/6200-apn.html)
is recorded in each task's provenance. The generated housing has a nominal
diameter-30/depth-9 seat and our own diameter-11 shaft clearance. Its surrounding
dimensions are original generated geometry. This is not an NSK housing replica,
validated bearing installation, or a prescribed engineering fit.

The [MISUMI bearing-housing category](https://uk.misumi-ec.com/vona2/mech/M0800000000/M0806000000/)
and [THK support-unit page](https://www.thk.com/us/en/products/ball_screw/support_unit/ek_bk_fk/)
were consulted for mechanical part typologies. Their PDF graphics, product CAD
files and complete drawings are not included in generated training data.

Task provenance separates sourced boundary dimensions, original housing geometry,
inspiration URLs and manufacturing approval (false). Native success verifies
nominal geometry and parametric history, not tolerances, surface finish, material,
retention/seals, loads or suitability for manufacture. ISO/GOST outputs remain
**draft profiles, not certified full conformance**.

## Quality and splitting

* Deterministic seed, frozen request and dependency hashes; all failed selected
  tasks are reported instead of being replaced by easier examples.
* Check positive depth, counterbore depth/radius, edge clearance, nonintersecting
  sibling pads, and at least 2 mm ligament between separate drill features.
  These constraints justify simple independent analytic volume calculations.
* Execute the teacher in real FreeCAD and check analytic volume/bounds. Export
  both profiles from the same native source and audit their labels/images.
* Drawing export independently checks every specified pad, hole, counterbore,
  support elevation and dependency against actual native features.
* Industrial lineage hashes ignore feature names/order and difficulty labels,
  using canonical primitive coordinates, elevations and dimensions. This is not
  a universal test for arbitrary equivalent B-reps/decompositions. Natural hash
  assignment separates TRAIN and development-validation; TEST is not exported.
* Public instructions explicitly explain that CB/ЦЕКОВКА is an **additional**
  coaxial pocket and that sketches and pockets need distinct names. Those rules
  apply to all tasks and contain no hidden target values. They address a missing
  convention in the earlier crop-only input, but improvement still needs testing.

Generation exports all teacher steps. This does not by itself fix the previous
SFT sampling imbalance or provide new model-rollout recovery supervision. Keep
counterbore/recovery coverage and visual-grounding ablations in the training plan.

## One entry point

Run from the repository root:

```bash
# All nine families, three levels, 100 parts per family/level:
# 2700 independent geometries, each with ISO and GOST => 5400 documents.
python scripts/generate_cad_data.py \
  --output artifacts/mixed-train-2700-v1 \
  --suite all --per-family-level 100 --split train --seed 41000000 \
  --exclude-pairs artifacts/curriculum-benchmark-90-v1/paired/pairs.jsonl
```

This is a **command example**, not a launched large run. `--suite industrial`
selects the six new families; `--suite curriculum` selects the original three.
`--levels easy medium`, `--split validation`, `--seed`, `--jobs`, repeated
`--exclude-pairs` and `--plan-only` control planning. Parameter spaces and sampling
windows are finite; disk and FreeCAD execution remain the scaling costs.

The combined `paired/pairs.jsonl` points to both representations of every part.
`dataset-sources.json` lists actual profile dataset roots for `--dataset` and
`--extra-dataset` in the existing SFT loader. Top-level orchestration directories
are not directly training datasets. Native tasks/references can also be used by
rollout/reward pipelines; this launcher does not run SFT/GRPO itself.

For selected new families, use the lower-level entry point:

```bash
python scripts/generate_industrial_drawings.py \
  --output artifacts/bearing-train-v1 \
  --families bearing_cartridge adapter_flange \
  --levels easy medium hard --per-family-level 20 --split train
```

This example requests 120 parts / 240 profile bundles. Training can use original
and new-family corpora together. A new industrial evaluation is a separate track:
do not silently add these families to an already-running benchmark.

## Native pilot artifacts

* `artifacts/industrial-analogues-train-18-v1`: one TRAIN part of every new
  family/level combination, both ISO and GOST.
* `artifacts/industrial-analogues-validation-18-v1`: separate validation parents
  covering the same combinations and profiles.

Inspect each `paired/summary.json` for actual completion and failures.
`task-statistics.json` describes the planned set, not model performance.
Both sets include PDF sheets, traceable VLM crops, native FCStd, teacher actions,
causal feedback exports and source provenance. No new model training or new-family
GPU benchmark was launched by the generator upgrade.

Both pilots completed: **36 independent parts / 72 profile documents**, with zero
TRAIN/validation lineage overlap. Together they contain 160 PDF pages, 672 VLM
crops, 7228 native feature checks and 2848 causal feedback rows. Each source CAD
and teacher program matches between ISO/GOST. All formatting checks passed while
the full-conformance claim remains false. See each `final-audit.json` and the
combined `artifacts/industrial-analogues-pilot-audit.json`.

Validation also passed 23 unit/integration tests and a separate 1800-case
dimensional-constraint smoke sweep (100 seeds × 6 families × 3 levels). The smoke
sweep is a procedural/schema test, not 1800 additional native-verified CAD parts.

Technical fonts are now downloaded at installation into an external cache. New
drawing bundles preserve font provenance without vendoring font binaries; see
[license scope](LICENSE_AUDIT.md). Previously generated pilot artifacts retain
their original source snapshots and notices.
