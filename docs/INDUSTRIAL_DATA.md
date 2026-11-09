# Industrial CAD data: scope and acceptance criteria

The release goal is useful reconstruction and editing of real mechanical parts.
Current generated drawings are synthetic parameter schedules, not audited
ISO/ASME/ESKD drawings. Native FreeCAD success proves nominal shape and editable
history checks for the implemented tools; it does not prove manufacturability,
correct GD&T interpretation, material choice, load capacity or regulatory compliance.

## Three separate evaluation tracks

1. **Synthetic composition stress-test (implemented).** Large, deterministic
   geometry with exact native teachers. `large-composition-v1` has 48 accepted
   tasks from 24 physical parents: 12 TRAIN, 12 development-validation. Each
   parent has drawing-only and text-plus-the-same-drawing conditions. Challenge
   teachers use 32–48 actions; extreme teachers use 72–108. The action language
   now accepts up to 128 actions. All drawings/variants of one physical parent
   stay together. TEST is not exported. Difficulty is fixed before model scoring.
2. **Model-derived nominal drawings (implemented; standards review pending).**
   `cad-env.a3-first-angle.nominal.v1` uses actual B-reps and native FreeCAD
   TechDraw views, sections and extent dimensions, with center/hidden lines,
   scale, units, title/revision metadata and verified complete feature schedules.
   See TECHNICAL_DRAWING_FORMAT.md. Formal ISO/ASME/ESKD conformance remains
   unassessed. Choose a named standards profile and edition before a conformance
   review; a page border/title block alone is insufficient.
3. **Real industrial documents (not yet imported).** Original PDFs/vector drawings
   paired with licensed reference STEP/native CAD, reviewed by someone able to
   interpret the drawing. Keep real-source evaluation outside the procedural
   training family. Report source and supported-feature coverage, not just score
   on whichever convenient parts happen to work.

## Standards references checked on 2026-11-08

- [ISO 128-1](https://www.iso.org/standard/65296.html): general representation.
- [ISO 129-1](https://www.iso.org/standard/64007.html): presentation of dimensions
  and associated tolerances. Its scope explicitly distinguishes presentation
  from application/meaning of dimensional tolerances; nominal reconstruction
  alone is not tolerance interpretation.
- [ASME Y14](https://www.asme.org/codes-standards/y14-standards): distinct drawing,
  projection and dimensioning/GD&T standards.
- [FreeCAD TechDraw examples](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/TechDraw_General_Examples.md):
  model-derived views, sections and dimensions. The installed headless FreeCAD
  exposes TechDraw projection APIs; this availability probe is not a completed
  industrial drawing exporter.

These public references establish scope. Complete standards conformance has not
been checked against normative texts or reviewed by a drafting expert.

## Data acceptance and splitting

Preserve original document bytes, page numbers, source URL, owner/author, license
and allowed use, part/revision/assembly lineage, unit system, projection convention,
reference geometry provenance, verified nominal dimensions, tolerance annotations
and unsupported/ambiguous requirements. Hash raw inputs and derived assets.
Split by design lineage/source family, including revisions and assembly relatives,
not by PDF page, raster image, action row or arbitrary filename. Match coordinates
and units explicitly between drawing and reference geometry. Keep the source
PDF/vector plus a rendered VLM view; controlled scan/noise variants share lineage.

Proposed external evaluation candidate:
[CADGenBench](https://github.com/huggingface/cadgenbench) has drawing-to-STEP and
editing tasks with public inputs and private ground truth. Its
[dataset card](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data)
reports 81 mechanical fixtures sourced from Mecado and specifies evaluation use.
Do not mix its benchmark inputs into SFT. Local exact scoring is unavailable
without the private reference geometry; no submission has been made.

A download button is not evidence of training/redistribution permission. For
example, [TraceParts API terms](https://developers.traceparts.com/docs/api-terms-of-use)
restrict use and redistribution. Such catalog assets must not be silently bulk
copied into a public training corpus.

## Tool and scoring coverage

The current action set covers bodies, rectangular/circular sketches, pads,
pockets and scalar parameter edits. Missing industrial operations include
arbitrary profile/arc constraints, revolutions, fillets/chamfers, threads,
patterns and assemblies. Adding 128-action support increases composition length;
it does not add these capabilities. New tools require separately versioned
observations, native execution tests and a fresh comparable baseline.

Measure exact native nominal-shape success, feature coverage and dimensions,
through/blind depth correctness, support/dependency correctness, parametric
editability, protected-feature preservation, invalid calls, unnecessary actions
and successful stopping separately. CAD solver tolerance is not a manufacturing
fit/tolerance specification. Global shape overlap can hide a critical incorrect
hole, so retain strict success and add feature-level diagnostics.

Evaluate text-plus-drawing, drawing-only and image ablations independently.
Calibrate model difficulty on development data without choosing individual cases
because a particular checkpoint failed. Keep an attainable hard tier and an
extreme frontier tier; zero success on a small pilot is a floor, not proof that
more training will solve it. Include ambiguity and missing-information tasks as
explicit clarification/abstention tasks once the agent protocol supports them.
