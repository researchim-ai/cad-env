# Experimental advanced solid graph

`cad_env.solidcad` is a separate versioned tool contract. It does not extend or
change the frozen seven-action FreeCAD benchmark or the ongoing portable mixed
SFT experiment. Inputs are bounded JSON, not arbitrary Python.

Implemented native operations in FreeCAD and CadQuery:

- Boxes and cylinders at explicit origins; boolean fuse and cut.
- Polygon extrusion, including concave profiles.
- Closed profiles with three-point circular arcs, followed by extrusion.
- Revolution of a radial/Z polygon through up to 360 degrees.
- Fillets and chamfers on all edges of the selected object.
- Inward shells with a top opening.
- Multiple separate final solids selected by an explicit finish action.

FreeCAD uses its native Part kernel; CadQuery uses OCCT.
Each engine saves a native artifact and replayable source. These source-replay
models do not claim fully constrained sketches, editable native feature trees,
assembly joints, CAM or FEM. The FreeCAD artifact currently stores the resulting
shape and source program in one Part feature; separate final solids form a compound.

## Native validation

The retained engines are checked against correct references and deliberately
wrong dimensions. Polygon/arc areas, revolved-bushing volume, hollow-box volume,
multiple solids and booleans have analytic checks. Fillet/chamfer probes require
nonempty volume smaller than the base. Native exact-BRep tolerance is 0.00001.
Older three-engine test counts are historical and do not describe this release.

```bash
.venv-multicad/bin/python -m cad_env.solidcad \
  --seeds 3 --output artifacts/my-solid-feature-probes
```

Nine original families: polygon bracket, arc-ended plate, revolved bushing,
rounded block, chamfered block, open box, separate bodies, bored block and fused
blocks. A native feature corpus is being generated at
`artifacts/solidcad-feature-corpus-v2`; inspect `summary.json` before using it.
Failed native cases never enter positive exports.

Exports are draft `train.program.jsonl`, validation and test rows with a
**whole-program** JSON-array completion. They are deliberately separate from
portable `train.feedback.jsonl` single-action SFT. No model has been trained or
benchmarked on this advanced contract yet. A dedicated audited training profile,
larger independent lineages, stronger editability checks and matched evaluation
are required before claiming advanced CAD agent competence. This corpus's graph
hash is only a construction-lineage grouping; it is not a general geometric
alias detector across arbitrary boolean/revolve programs.

The three-seed feature corpus completed with **78 accepted native episodes and
zero failed cases**, after removing repeated construction targets. Its 45 TRAIN
program rows passed the actual Qwen3.5 processor/TRL collator: completion masks,
masked image tokens and template boundaries all checked; maximum sequence 756
tokens, maximum target 151 tokens. This is CPU data preparation, not training.
Evidence: `artifacts/solidcad-processor-preflight.json`.
