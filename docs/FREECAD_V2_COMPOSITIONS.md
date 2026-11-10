# Composed v2 parts and paired drawing inputs

The expanded generator combines several operations in one part rather than
testing an isolated feature. It preserves the frozen seven-action ISO/GOST
benchmark. New programs use the opt-in `freecad.v2` contract.

Three original families are available at easy, medium and hard levels:

- **Flanged bushing:** a revolved stepped annular section with flange mounting
  holes and counterbores.
- **Pedestal mount:** an extruded base, tapered ruled rectangular loft, boolean
  union and counterbored holes around the pedestal.
- **Gusset bracket:** an L-shaped XZ section extruded along Y, with primary holes
  and counterbores on both horizontal and vertical legs.

Easy parts have four primary holes and four counterbores, medium parts twelve
of each, hard parts twenty-four of each. Teacher programs contain 19–23,
51–55 and 99–103 actions respectively, including an explicit finish. Each
geometry has an independent analytic volume expectation. Intermediate teacher
actions execute in native FreeCAD; final programs must match the reference
and volume. Deliberately translated candidates are checked as negative controls.
Tests additionally remove one counterbore and require native failure.

## Paired drawing package

The native worker exports **only the selected final shape** to STEP. The drawing
exporter reads that STEP, rather than combining all intermediate Part features.
Both profiles share the same STEP hash, geometric specification and split group.

Each ISO/GOST package contains native hidden-line projections and a section,
followed by dimensional specification sheets. The appendix describes geometric
forms, profile vertices, plane orientation, hole entry coordinates, outward
normals, primary diameters/depths and counterbore diameters/depths. It contains
no JSON action program or teacher object names. It is a dimensioned appendix,
not a complete replacement for manufacturing annotations on each view.

Coordinates use semicolons between components, so GOST decimal commas are
unambiguous. Values retain six decimal places where needed. Goal input crops
include the existing readable view crops and every appendix line; appendix
crops are checked against the 327680-pixel-per-image budget and a minimum
estimated text size of 10.5 pixels. This estimate is not an OCR accuracy score.

ISO/GOST framing uses existing **experimental profiles**. Full conformance is
not established, including manufacturing tolerances, material, fits, finish and
complete normative annotation. Every wrapper declares
`can_claim_full_conformance: false`. Fonts remain in the external installation
cache; no font binaries are copied into the new bundles.

For SFT, both profiles supervise the **same causal native teacher step**. Only
goal drawing images and public pre-action state/history enter the prompt. The
dimension specification is visible in the appendix images; it is not separately
transcribed into the prompt. Reference actions and future state remain private.

## Generate a small mixed-split pilot

```bash
python -m cad_env.freecad_v2.generate_compositions \
  --output artifacts/my-v2-compositions \
  --per-family-level 1 --seed 10000
```

This attempts nine geometries and eighteen paired drawing packages. Native
failures remain in the manifest and do not enter training. A failure is not
replaced with an easier geometry. Existing outputs are never overwritten.

Exports remain compatible with the current single-action SFT loader:
`exports/train.feedback.jsonl`, `validation.feedback.jsonl`,
`test.feedback.jsonl`. Every row records its profile. Only TRAIN is optimized.
The generator stores validation/TEST selections, source snapshots, per-file
hashes, native traces and a dataset audit. This pilot is an integration check,
not a statistically sufficient model benchmark.

## Freeze balanced TRAIN and control plans

```bash
python scripts/generate_v2_curriculum.py \
  --output artifacts/my-v2-validation-plan \
  --split validation --per-family-level 3 --seed 50000 --plan-only

python scripts/generate_v2_curriculum.py \
  --output artifacts/my-v2-train \
  --split train --per-family-level 10 --seed 60000 \
  --exclude-plan artifacts/my-v2-validation-plan/task-plan.json
```

With `--plan-only`, no FreeCAD process, rendering or optimization is launched.
Without it, the script first freezes `task-plan.json`, then generates its exact
tasks under `dataset/`. Quotas apply to each of three families and three levels.
Selection searches sequential seeds for the requested **natural hash split**;
there is no relabeling or selection on model/native execution success. Native
failures after planning stay failures. Duplicate/excluded lineages are skipped
before native execution. Split hashes are canonical specification identities,
not a universal detector of equivalent B-Rep geometry across arbitrary programs.

Development plans prepared in this workspace:

- `artifacts/freecad-v2-control-validation-v1/task-plan.json`: 27 validation parts.
- `artifacts/freecad-v2-control-test-v1/task-plan.json`: 27 TEST parts.
- `artifacts/freecad-v2-train-plan-v1/task-plan.json`: 90 TRAIN parts.

These three are **plans**, not completed rendered datasets. They have no shared
parent groups. They were frozen before any v2 model optimization or evaluation.
TEST should remain unused during method and checkpoint selection.

## Evaluate a model on drawing inputs

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv-qwen35/bin/python \
  scripts/benchmark_freecad_v2.py \
  --dataset artifacts/my-v2-compositions \
  --profile iso \
  --model artifacts/iso-gost-sft-pilot-v1/merged \
  --output artifacts/my-v2-iso-evaluation --limit 20
```

Repeat with `--profile gost` and a fresh output directory. Drawing packages
require an explicit profile; the runner does not silently substitute the old
goal renders. It audits both profiles before using the crop paths.

No model improvement is implied by successful generation. Before scaling SFT,
complete a fixed validation baseline, test image/token-budget handling and
compare normal inputs against blank/wrong-image diagnostics separately.

## Completed integration pilot

`artifacts/freecad-v2-compositions-paired-v3` contains nine accepted geometries,
18 paired drawing packages, 42 final PDF sheets and 148 VLM crops. All nine
correct native controls passed; all nine translated controls failed as intended.
There are 1,038 next-action rows: 786 TRAIN, 46 validation and 206 TEST, with
both profiles sharing each parent and step. The dataset audit checked 4,140 file
hashes; all paired drawing audits passed. Earlier v1/v2 output directories are
interrupted development artifacts and must not be used as completed datasets.

Long-trajectory state was initially too large: repeating full topology for every
intermediate body produced over 400,000 characters. Public state now retains all
feature definitions and topology counts, with paged details available through
`inspect`. The hard flange's state shrank to about 22,000 characters. This changes
only the opt-in v2 contract; frozen v1 evaluation is unaffected.

CPU preflight with the pinned Qwen processor checked all 1,038 samples, including
image-token expansion. The largest full training example has **21,215 tokens**
and inputs use at most **10 images**, below the 32,768-token/22-image limits.
No examples were truncated. Per-image pixel cap is 327,680. This is a context
budget check, not a model quality or visual-grounding result.

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' .venv-qwen35/bin/python \
  scripts/check_v2_context.py \
  --dataset artifacts/freecad-v2-compositions-paired-v3 \
  --model artifacts/iso-gost-sft-pilot-v1/merged \
  --output /tmp/v2-context-check.json
```
