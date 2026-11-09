# ISO and Russian ESKD drawing profiles

Status as of **2026-11-09**: two explicitly selected, experimentally supported
profiles. **Full ISO or GOST conformance has not been confirmed.** This document
records the implemented subset and remaining review. Successful CAD execution,
PDF rendering, passing tests and a conformance declaration are different results.

## Selection and compatibility

- `--standard nominal` retains the original `cad-env.a3-first-angle.nominal.v1`
  profile, including its layout. Existing `technical-corpus-v2` is in this profile;
  it must not be relabeled ISO or GOST.
- `--standard iso`: `cad-env.iso-a3.2026.v1`, English, A3 landscape, left margin
  20 mm and other margins 10 mm, 180 mm title-block width. Its 50 mm title-block
  height and cell layout are project choices, not an ISO-prescribed form.
- `--standard gost`: `cad-env.gost-r-a3.2026.v1`, Russian, A3 landscape, left
  margin 20 mm and other margins 5 mm, 185 x 55 mm first-sheet title-block envelope
  and 185 x 15 mm continuation envelope, duplicate designation rotated 180 degrees
  and inventory fields. Detailed form-cell verification is still pending.

Both new profiles use a pinned, Cyrillic-capable technical font rendered as vector
outlines. Heights are physical cap heights, chosen from 2.5, 3.5, 5, 7 and 10 mm;
they are not CSS font sizes. Using a font intended for ISO lettering does **not**
prove every glyph, stroke ratio and spacing meets ISO 3098 or GOST 2.304.
Install the pinned font, editable source and license texts outside the repository
with `python -m cad_env.drawings.font_cache --install`. They live in the user cache;
new bundles contain vector outlines and hashed `font-provenance.json`, without
font binaries or editable font sources. Legacy bundles retain their original
snapshots and notices. Downloading the font does not change its license terms.

Both profiles preserve first-angle placement, the native solid, feature checks,
section material/voids, dimension values and feature schedule. New profiles use
`⌀` for diameter; Russian numeric dimensions use decimal commas. First extent
lines are placed 15 mm from the contour, to separate cutting-plane labels from
dimension text. The minimum first offset in GOST 2.307-2011 clause 5.11 is 10 mm;
that single check does not verify all dimensioning requirements.

## Normative basis and edition policy

Official sources:

- [ISO 5457:1999](https://www.iso.org/standard/29017.html), with Amd 1:2010:
  drawing-sheet sizes and layout.
- [ISO 7200:2004](https://www.iso.org/standard/35446.html), confirmed in 2025:
  title-block data fields, sections 5.1–5.3, and width in section 6. Mandatory
  document data include owner, identification, issue date, sheet number, title,
  approval person, creator and document type. Identification/title/sheet numbering
  are generated; externally supplied administrative identities are never invented.
- [ISO 5456-2:1996](https://www.iso.org/standard/11502.html), section 5.1:
  first-angle orthographic placement.
- [ISO 128-2:2022](https://www.iso.org/standard/83355.html): line conventions.
- [ISO 129-1:2018](https://www.iso.org/standard/64007.html): dimension presentation.
- [ISO 3098-1:2015](https://www.iso.org/standard/65679.html),
  [ISO 3098-6:2000](https://www.iso.org/standard/33201.html): lettering, including
  Cyrillic. Complete normative glyph comparison has not been performed.
- [ГОСТ Р 2.104-2023](https://protect.gost.ru/gost/details/69d422a2-3c0e-4561-a521-0c99f9cf4417),
  effective 2024-03-01, including 2024 corrections: title blocks, table 1,
  annexes А/Б, forms 1 and 2а. This is the Russian national profile, not a claim
  about every country's application of ESKD.
- [ГОСТ 2.301-68](https://protect.gost.ru/gost/details/aa681f87-ff49-4115-ad25-169e15d2af06):
  formats in the selected edition policy.
- ГОСТ 2.302-68, 2.303-68, 2.304-81, 2.305-2008 and 2.306-68:
  scales, lines, lettering, views/sections and hatching. The profile lists these
  targets; a list of references is not evidence of full clause-by-clause review.
- [ГОСТ 2.307-2011](https://protect.gost.ru/gost/details/62de9da6-e1a7-4023-955e-7dd9777eaab6):
  dimensions in the selected edition policy.

The [ГОСТ Р 2.307-2026 registry entry](https://protect.gost.ru/gost/details/eddb3107-58be-4811-b237-5ab1d0d4c4d5)
sets its effective date to **2027-10-01**, with early application allowed.
The selected profile does not silently opt into early application. A standard
being adopted and being in force are separate dates. Recheck editions before a
production release; the profile identifier freezes this project's current target.

The review used official registry cards and accessible original normative text
copies for ISO 7200, ISO 5456-2 and GOST R 2.104. Not all full ISO texts or normative
figures were available for complete inspection. Unverified figures and clauses
remain pending rather than being reported as passed.

## Reproduction

Install the Python-side dependencies with `pip install -r requirements-drawings.txt`.
System FreeCAD and PySide2 QtSvg/QtGui remain required as described in
`TECHNICAL_DRAWING_FORMAT.md`. Output directories must be new.

```bash
python -m cad_env.drawings.export --model part.FCStd --task task.json --standard iso --metadata document.json --output artifacts/part-iso
python -m cad_env.drawings.export --model part.FCStd --task task.json --standard gost --metadata document.json --output artifacts/part-gost
python -m cad_env.drawings.dataset --source artifacts/large-composition-v1 --standard iso --metadata document.json --output artifacts/corpus-iso-new
python -m cad_env.drawings.dataset --source artifacts/large-composition-v1 --standard gost --metadata document.json --output artifacts/corpus-gost-new
```

`document.json` accepts explicit owner, designation, title, issue_date (ISO date),
creator/approver, document_type, status/revision/version, material/mass,
developer/checker/technologist/norm_checker/approval, liter, inventory_number,
registration and boolean synthetic_example. Unknown fields and invalid types are
rejected. Null administrative values stay blank. Material, mass, signatures and
approval are not guessed from geometry. Mark generated demonstration documents
with `"synthetic_example": true`. Some profile fields have width checks that reject
oversized metadata instead of silently clipping it; arbitrary long text wrapping
is not implemented.

`--require-conformance` is a strict release gate. It currently refuses full
conformance for either profile, retaining the generated draft and report for
review. Supplying invented names or signatures cannot complete a normative review.

## Evidence and limits

Every new-profile bundle has `conformance.json` with named checks, references,
evidence and pending review. `format_checks_passed` covers only implemented
formatting checks. `status` additionally includes required administrative data and
synthetic-document release eligibility. `can_claim_full_conformance` stays false
while review remains incomplete. The bundle audit independently recomputes the
report and detects a manually changed declaration even if its hash is updated.

Automatic checks cover A3 size, frame margins and line-group width, title-block
outer dimensions, physical declared lettering heights and pinned font identity,
GOST duplicate-designation presence/orientation, first-angle view arrangement,
selected scale subset, first dimension offset and diameter notation. Tests also
measure a real glyph outline independently of its declared SVG attributes.

Remaining work before a full declaration:

1. Verify every title-block internal cell/additional field against the selected
   normative figures, including corrections and applicability exceptions.
2. Independently review full glyph forms/stroke ratios/spacing, projection symbol,
   arrows, sections, hidden lines and complete dimensioning conventions.
3. Define real product requirements: tolerances, fits, datum system, surface
   requirements, threads, materials and technical notes where applicable. Current
   supported tasks reconstruct nominal geometry and cannot infer these details.
4. Validate actual organizational metadata, approval/signature/registration
   evidence and document lifecycle under the selected paper/electronic workflow.
5. Review additional formats, orientations, geometry families and external
   industrial source drawings before expanding the supported scope.

The small built-in conformance checker is not an external normative certification.
Training results on these drawings must be labeled synthetic nominal-geometry
results until these gaps are resolved.

## Reproducible validation completed on 2026-11-09

`artifacts/standards-profiles-v3` contains the same three extreme source models
(plate, multi-support fixture and stepped flange) exported in both profiles:
6 bundles, 20 A3 pages. All six native/artifact audits and implemented formatting
checks passed. Their document status remains `nonconforming`: synthetic examples
have no genuine release approval; developer/norm-controller/material/registration
or creator/approver fields are intentionally missing. Pending normative review
also prevents any complete conformance claim, even if all metadata were supplied.

`verification.json` records each bundle's audit, failures, page/crop counts and
estimated minimum font size. Each bundle has 14–22 crops, with a minimum estimated
10.562 pixels at the 327680-pixel image budget. New-profile caption height was
reduced to preserve readability with real 2.5 mm schedule lettering; the legacy
profile's crop layout remains unchanged. This remains an estimate, not an OCR or
VLM-task accuracy measurement.

All 20 PDF pages were rendered with Poppler and visually reviewed. The regression
command `python -m pytest -q tests/test_extreme_tasks.py tests/test_source_mixing.py
tests/test_technical_drawings.py tests/test_drawing_readability.py` passed **27 tests**. It includes both native profile exports, continuation fields, independent
glyph-height measurement, overflow rejection, a strict conformance-gate failure,
rehashed false-report rejection and small paired corpus conversion in both
profiles with unchanged labels and task/split identities. Ruff and `git diff
--check` passed.

The unchanged nominal `technical-corpus-v2` also passed its semantic audit:
48 episodes, 24 drawing bundles and 3248 feedback rows. No new training was started
and no ISO/GOST-profile model-performance result is claimed.

## Bulk generation: 500 independent tasks

The reproducible command is recorded in `recipes/iso-gost-500.json`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m cad_env.drawings.batch --output artifacts/iso-gost-500-v1 --count 500 --shard-size 25 --seed-start 9000000 --jobs 4 --native-workers 2 --exclude-dataset artifacts/large-composition-v1
```

This selects **500 distinct CAD geometries**, with 250 ISO-profile tasks and 250
GOST-profile tasks. Each profile has 200 TRAIN and 50 development-validation tasks,
balanced across challenge/extreme tiers and approximately balanced across the
three supported families. There are no repeated modality variants counted as new
geometries. Geometry hashes determine splits; they are never relabeled. The old
pilot's parents are excluded. TEST specifications are skipped before native work.

This is a new IID synthetic training/development corpus, not an external industrial
benchmark, and not a paired ISO-versus-GOST benchmark: the two groups contain
different independent geometries. It retains the nominal-geometry limitations
and incomplete conformance status described above.

The frozen `plan.json` records the actual specifications before execution.
Generation runs in 20 independently audited shards, each with a native `source`
corpus, a converted `dataset` and a `result.json`. `batch-summary.json` reports only
finished, audited shards as completed. A failed task is never silently replaced by
an easier random seed. Re-running the same command audits and reuses completed
work; incomplete output is retained under `failed-attempts` before rebuilding.
Changes to the drawing implementation are rejected on resume.

The batch root is an index, **not a single training dataset**. Its
`training_datasets` list contains shard paths, each with the existing
`technical-drawing-v1` contract, full native/semantic audits, TRAIN/validation JSONL
and image assets. The trainer's existing `--extra-dataset` option can combine these
shards after memory/context qualification. It must not treat the index itself as
an ordinary dataset or mix validation feedback into training. No training is
launched by this generator.

### Matched ISO/GOST comparison on the same 500 geometries

The requested final comparison is **500 geometries x 2 profiles = 1000 drawing
bundles**. The initial batch supplies one profile per geometry; the paired runner
adds the opposite profile from exactly the same native source. It does not sample
another 500 parts. See `recipes/iso-gost-paired-500.json`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/generate_paired_drawings.py --primary artifacts/iso-gost-500-v1 --output artifacts/iso-gost-paired-500-v1 --jobs 4
```

The primary run may still be active: the paired runner waits for each audited
primary shard before exporting its opposite representation. It writes a matched
`pairs.jsonl` index with ISO and GOST dataset/task/PDF/model/image paths for every
physical parent. The index intentionally refers to both the original and added
datasets; retain both directories. For each pair, source-FCStd hashes, oracle
actions and split identities must agree. Each opposite corpus passes the existing
native, formatting and causal-data audits. Status is complete only after every
planned pair is available. Both profiles retain the incomplete full-conformance
status and nominal-geometry scope above.

The natural split is preserved in both representations: 400 TRAIN and 100
validation geometries. This is a matched comparison corpus, not a newly sealed
external TEST benchmark. No model is trained by either generation command.

Verified paired generation completed on 2026-11-09:

- 500 independent geometries, each exported once in ISO and once in GOST:
  **1000 documents, 2828 PDF pages, 13984 VLM crops**.
- All 40 corpus shards passed native/artifact/causal semantic audits. Paired source
  FCStd hashes, target actions and split identities agree for all 500 pairs.
- 170456 native feature checks across both representations; minimum estimated
  text height at the fixed image budget is 10.562 px (not measured OCR accuracy).
- Geometry distribution: 172 drilled plates, 164 fixtures, 164 stepped flanges;
  250 challenge and 250 extreme cases. Oracle programs have 32–108 actions.
- 400 TRAIN / 100 development-validation parents retain the same split in both
  representations. Paired comparison indexes cover all 500; no additional parts
  were sampled when completing the second profile. No model training or model
  benchmark run was launched.
- `artifacts/iso-gost-paired-500-v1/final-audit.json` records final evidence;
  `pairs.jsonl` indexes 500 matched cases and `cases.jsonl` indexes 1000 separate
  profile cases. The pair index uses absolute local dataset paths.

Recheck the entire matched set with the final-audit command in
`recipes/iso-gost-paired-500.json`. Full normative conformance remains unconfirmed;
the production-document gaps described above still apply.
