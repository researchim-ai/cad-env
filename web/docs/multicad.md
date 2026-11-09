# Native CAD executors

Supported engines are `freecad` and `cadquery`. The optional runtime with GPL
Python adapters was removed on 2026-11-09; it is not an installation option.
Archived multi-engine measurements remain historical artifacts, not new results.

Both supported engines execute validated JSON tools. FreeCAD retains native
parametric feature history; CadQuery saves STEP plus replayable action source.
CadQuery source replay does not claim native constrained sketches. Independent
native checks compare candidate/reference geometry, analytic dimensions and body
count. The seven-action contract remains separate from the advanced solid graph.

## Setup

Install FreeCAD separately and use an isolated geometry environment:

```bash
python -m venv .venv-multicad
.venv-multicad/bin/pip install -r requirements-multicad.lock
python -m cad_env.multicad generate \
  --output artifacts/my-multicad --engines freecad cadquery \
  --python .venv-multicad/bin/python --seeds 3
```

Generation supports create/edit/repair tasks, text/image conditions, single/multiple
views and varied edits. Native execution, lineage, artifacts and exports are
audited. Failed teachers never become accepted examples. For evaluation/training
commands use `python -m cad_env.multicad --help` and the corresponding subcommand's
`--help`. Rewards use registered TRAIN tasks only. A passing teacher is not a
trained-model result, and a geometry score is not exact task success.

Use frozen run manifests when reproducing historical measurements. Removing a
runtime does not turn older three-engine figures into current two-engine figures.
