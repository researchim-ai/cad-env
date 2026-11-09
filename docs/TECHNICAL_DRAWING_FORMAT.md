# Model-linked technical drawing format

Default profile: `cad-env.a3-first-angle.nominal.v1`.

Separate draft ISO and Russian ESKD profiles are now selected explicitly with
`--standard iso` / `--standard gost`. See [edition policy, checks and remaining
conformance review](DRAWING_STANDARDS.md). They do not upgrade the existing corpus
or establish full conformance automatically.

This is a nominal-geometry drawing/data format, not a claim of full ISO, ASME or
ESKD conformance. It does not invent material, surface finish, fit, thread or
manufacturing-tolerance requirements that were absent from the input. A standards
review remains distinct from the software checks described here.

## Outputs

Each bundle contains an unchanged source FCStd/STEP, an editable FCStd document
with native TechDraw views, section and extent dimensions, A3 landscape SVG sheets,
a multipage vector PDF, 2400-pixel page PNGs, readable VLM crops, crop-coordinate
metadata, native measurement/feature checks, source-code snapshots and a hashed
manifest. Re-export after editing the model. The PDFs/schedules are snapshots;
the native view and extent-dimension objects are editable/model-linked. Editing a
file invalidates the bundle audit until a new export is produced.

The overview uses first-angle placement (top below front; right view to the left),
explicit convention and units, one shared scale, native hidden-line removal,
model-derived center marks, an A-A cutting-plane indicator, and a native section.
Section hatching is clipped against exact B-rep section material, including voids;
its spacing is fixed in paper units. Arrowheads and hatch segments use plain
vector paths to survive the headless Qt SVG/PDF renderer.

Continuation sheets list every supported base, pad and drilling feature: global
XY position, named support, diameter, height/depth and counterbore size/depth. The
native worker checks every listed circular feature against its actual sketch and
pad/pocket length, its support elevation, and checks that no pad/pocket is omitted.
Native extent dimensions are checked against the reference bounding box. The
source body's volume/bounds must stay unchanged after adding drawing objects.

## Commands

Run from the repository root with the installed FreeCAD command-line executable
(`FREECAD_CMD` can override discovery). System `/usr/bin/python3` must provide
PySide2 QtGui/QtSvg; the project Python needs Pillow and, for the new profiles, fontTools
(`pip install -r requirements-drawings.txt`). The exporter isolates FreeCAD
settings/caches and uses Qt's offscreen renderer. Linux with the installed
FreeCAD 1.1 / Qt 5.15 was tested; other platforms are not qualified here.

```bash
python -m cad_env.drawings.export --model part.FCStd --task task.json --output artifacts/part-drawing
python -m cad_env.drawings.dataset --source artifacts/large-composition-v1 --output artifacts/technical-corpus-new
.venv-qwen35/bin/python -m cad_env.drawings.benchmark --dataset artifacts/technical-corpus-v2 --model artifacts/release-trial-v1/candidate/merged --output artifacts/technical-input-preflight.json
```

The task adapter currently supports the three large-composition families with
`base` / `pads` / `cuts` specifications. An unsupported or incomplete feature
schedule is rejected. Export without `--task` explicitly produces a geometry-view
preview with overall dimensions only; its manifest marks it as not fully
specified. Generic STEP alone cannot reveal the designer's missing dimensional
requirements or parametric intent.

## VLM input policy

Keep the full PDF/SVG as the authoritative drawing. A3 at a small pixel budget is
not readable enough for training. The exporter derives separately labeled crops
of each view, then six-row schedule crops: position columns and size/depth columns
share explicit feature IDs. `tiles.json` records source sheet and crop rectangles.
The original sheet pixels are retained without inferred/hidden target labels.
The layout enforces an estimated minimum text height under a 327680-pixel image
budget; this is a layout check, not a measured OCR accuracy guarantee.

Use the manifest's complete `vlm_tiles` list. Larger parts require more than the
old single-image budget. Do not silently drop sheets/crops or truncate targets to
fit. The processor preflight measures long causal validation prompts with the
actual Qwen processor. GPU-memory and completion-mask preflight still precede any
future training run; CPU input acceptance is not proof that a 3090 can train every
long sequence at the requested batch size.

## Corpus conversion and evidence

The converter creates a new dataset, leaving its source untouched. It processes
TRAIN and development-validation only. Geometry parents, reference actions,
labels, causal pre-action state/history and native success evidence are preserved.
Only public drawing instructions and goal images change. Modality pairs share one
drawing bundle and retain their source split. Original rows/tasks are carried as
private audit evidence; they are never placed in model prompts.

The audit checks hashes and FCStd evidence, unchanged task/teacher semantics,
deterministic prompt/image retargeting, complete drawing bundles, and SVGs
regenerated from native proof plus the checked specification. Training's source
guard invokes this audit for `technical-drawing-v1`. An observation-format change
needs a fresh model baseline and must not be reported as a model-weight gain.

## Current limits

- Manufactured-source drawings have not yet been imported or independently scored.
- GD&T, tolerances, material/finish requirements and manufacturing process planning
  are not implemented by this profile.
- Unsupported geometry/tool operations remain unsupported; 128 actions only
  expands composition length.
- Data generated from known parametric families remains synthetic, even when
  native TechDraw produces the drawing views.


## Verified pilot — 2026-11-09

`artifacts/technical-corpus-v2` contains 48 tasks / 24 independent native drawing
bundles. There are 1624 TRAIN feedback rows and 1624 development-validation rows.
The corpus and training-source semantic guards passed for all bundles and rows.
Output directories must be new: v1 was a preserved, rejected development attempt;
v2 is the audited corpus. Original tasks and their target geometry were retained.

A pinned Qwen processor accepted the long validation inputs; the largest checked
prompt had 18480 tokens and 22 image crops. A real GPU1 inference on that input
returned a valid `finish` action in 13.29 seconds, with 11.70 GiB peak allocated
and 13.04 GiB peak reserved memory. This was a teacher-state input execution smoke,
not a complete CAD rollout or training-memory qualification. No new SFT was started.

56 related tests passed, including native PDF/drawing generation, incorrect-depth
and missing-feature rejection, modified-image rejection, rehashed wrong-label and
future-state rejection, sealed TEST access prevention, and existing agent tests.
The three extreme examples' PDFs were rendered with Poppler for visual QA.

A separate six-question / three-design crop-reading diagnostic matched all requested
numeric fields on the full images after explicitly labeled literal-format recovery
(6/6), versus 0/6 on blank controls. The raw outputs used single-quoted Python-style
dictionaries, so strict JSON exactness was 0/6. Both scores and raw responses are
preserved in `artifacts/technical-drawing-readability-v1.json`. This small component
probe is not a CAD rollout result, and recovery does not affect CAD action parsing
or rewards. Reproduce with `scripts/probe_drawing_readability.py --dataset ...
--model ... --output <new-file>` on GPU1. The response grader never executes model
code and rejects duplicate keys, non-finite values and callable expressions.
