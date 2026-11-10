# Expanded FreeCAD tools (opt-in v2)

Composed mechanical families and paired drawing inputs are described in
[Composed v2 parts](FREECAD_V2_COMPOSITIONS.md).

`cad_env.freecad_v2` provides an expanded native FreeCAD action graph and a
single-action agent environment. It does not modify the seven-action v1
schema, native worker, prompts or frozen ISO/GOST benchmark. V1 and v2 scores
must be reported separately. Existing CAD-tuned weights have **not** been
trained or benchmarked on these new tools.

## Implemented operations

The contract has 20 action types, including inspection, parameter edits and finish:

- `sketch`: circle, rectangle, arbitrary closed polygon, or closed boundary
  combining straight segments and three-point circular arcs.
- Each sketch can specify a world `origin`, plane `normal`, and local `x_axis`.
  The local X direction is projected onto the plane; a parallel direction is
  rejected. Defaults are origin zero, XY plane and world X.
- `pad`, `pocket`: extrusion along the profile normal; a pocket cuts in the
  opposite direction. Pocket source and profile are explicit named objects.
- `revolve`: a profile rotated about an explicit world axis and origin.
- `loft`: a solid through 2–16 profiles, optionally ruled.
- `sweep`: a solid profile swept along a 3D polyline. The profile center must
  coincide with the path start and its normal must align with the first segment.
  Corners use FreeCAD's right-corner/miter transition; Frenet mode is optional.
- `fillet`, `chamfer`: selected source-local edges.
- `shell`: inward thickness with selected faces removed.
- `fuse`, `cut`, `common`: union, subtraction and intersection of named solids.
- `translate`, `rotate`, `mirror`: copies with explicit world transforms.
- `linear_pattern`, `polar_pattern`: 2–32 copies, including the original.
  Polar angle is total span; 360 degrees excludes a duplicated endpoint.
  Pattern copies remain separate solids. Overlapping copies are rejected;
  use individual translations and explicit boolean unions for overlapping parts.
- `set_parameter`: edit supported scalar dimensions and rebuild dependencies.
  Circle radius and rectangle length/width can be edited, as can extrusion
  depth, revolution/rotation angle, fillet radius, chamfer size and wall thickness.
- `finish`: explicitly select 1–32 final solid objects, including multiple bodies.
- `inspect`: request source-local edge/face descriptors for a named object.
  `offset=0`, `limit=32` by default, at most 64 entries per page.

Distances are mm, angles degrees. Directions are vectors. Negative rotation
directions can be expressed by reversing the axis. Names and numeric inputs
are bounded; arbitrary Python, filesystem paths and expressions are not actions.
Programs and episodes allow at most 128 actions/attempts. Native features have
additional limits of 128 solid instances and 4096 edges.

Example: a cylindrical through-hole cut from an extruded rectangle:

```json
[
  {"op":"sketch","name":"PlateProfile","profile":{"type":"rectangle","length":40,"width":30}},
  {"op":"pad","name":"Plate","profile":"PlateProfile","length":8},
  {"op":"sketch","name":"HoleProfile","profile":{"type":"circle","radius":3},"origin":[20,15,8]},
  {"op":"pocket","name":"DrilledPlate","source":"Plate","profile":"HoleProfile","length":8},
  {"op":"finish","targets":["DrilledPlate"]}
]
```

## Agent observations and reward

`Environment.step(action, reference=...)` replays committed actions in an
isolated CPU FreeCAD process. An invalid action leaves the previous committed
geometry intact and consumes one attempt. The environment records every attempt
in `trajectory.jsonl`, including errors. Process timeouts and crashes remain
infrastructure failures rather than ordinary CAD mistakes in the benchmark runner.

Public state contains named profiles/solids, their current feature definitions,
and topology counts. Detailed edge/face indices, lengths/areas and centers are
returned only for the object requested by the latest `inspect` action. This
avoids repeating the topology of every intermediate cut. Indices are
1-based. They are **not stable identifiers across topology-changing edits**:
inspect the updated state before choosing edges or faces again.

Reference programs remain outside the policy messages. Exact B-Rep symmetric
difference, bounds and solid-count checks determine success. Success also
requires an actual `finish` action; correct intermediate geometry is not enough.
Native scoring precedes tessellation, because triangulation can affect FreeCAD's
cached bounding boxes on curved shapes. The public observation omits reference
geometry and validation details.

## Generation and SFT

```bash
python -m cad_env.freecad_v2 \
  --output artifacts/my-freecad-v2-data \
  --per-family 10 --seed 1000
```

This attempts 190 parts: ten samples from each of 19 seeded feature families.
Each case is retained in the manifest even on native failure; failures are not
silently replaced by easier parts. Duplicate lineage aborts generation.
Sizes are sampled on a 0.01 mm grid; seed count is configurable, but this is a
finite family grammar rather than unlimited diversity in mechanical designs.

Every accepted part has:

- The original task specification and exact reference program.
- Native correct and deliberately translated negative controls. All 19 feature
  families also have analytic volume expectations, including selected-edge
  fillets/chamfers, tapered lofts, shells, patterns and profile edits.
- A saved `.FCStd` file, mesh, and orthographic goal renders (isometric, top, front).
- A native replay of every teacher prefix and one causal next-action example per
  step, including finish. The prompt contains dimensioned task text, goal images,
  current public state and earlier execution outcomes.
- Hashes, lineage-based 80/10/10 TRAIN/validation/TEST splits, and an audit report.

Exports are `exports/{train,validation,test}.feedback.jsonl`. The existing SFT
loader and training-source audit accept this package. Use only the TRAIN export
for optimization. These images are rendered views, **not ISO/GOST drawing sheets**;
dimensions are supplied as text. The existing industrial drawing exporter is
not automatically extended to every new feature by this implementation.

The teacher samples cover clean prefixes. They are not student-recovery data,
evidence of visual necessity, or evidence that a model has learned the tools.

## Separate model benchmark

After generating a package with validation cases, run with the pinned Qwen
environment, preserving its interpreter symlink:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv-qwen35/bin/python \
  scripts/benchmark_freecad_v2.py \
  --dataset artifacts/my-freecad-v2-data \
  --model artifacts/iso-gost-sft-pilot-v1/merged \
  --output artifacts/my-freecad-v2-benchmark --limit 20
```

This runner uses greedy single-action generation, at most 128 attempts,
goal images and current JSON, with no current-candidate image feedback. It is
serial and separate from the frozen batched v1 benchmark. It accepts only
development-validation cases and records the selected task IDs before loading
the model. GPU 1 is required. No v2 model benchmark or training run is implied
by the native operation tests. Existing v1 GRPO reward hooks must not be used
to execute v2 actions; the v2 environment supplies the native reward interface.

## Saved models and remaining scope

FreeCAD executes the actual OCCT operations, not a simulator. The saved document
contains individual `Part::Feature` nodes, feature definitions, explicit final
target flags and the full replay program. It is **source-replay parametric**:
`set_parameter` rebuilds the graph inside this environment. It does not install
Python proxies for live automatic recomputation when a user edits a property
in the FreeCAD GUI, and it does not produce a native PartDesign/Sketcher tree.

This covers the operation families listed above, not every FreeCAD command.
Remaining work includes constrained Sketcher editing, splines/NURBS, multiple
wire regions in one profile, face attachments with stable topology references,
variable-radius dress-ups, threads/helices, assembly joints, sheet metal,
CAM/FEM and general surface modeling. Loft/sweep/shell may reject infeasible
inputs through native kernel errors. No full standards-conformance or
manufacturing-readiness claim follows from these feature probes.

Native API references: [FreeCAD TopoShape](https://freecad.github.io/SourceDoc/d8/ded/classPart_1_1TopoShape.html)
and [Part API](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Part_API.md).
