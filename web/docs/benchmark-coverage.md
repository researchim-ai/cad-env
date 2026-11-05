# External benchmark task axes and our generator coverage

Reviewed against official sources on 2026-11-05. Our generated corpora are original
synthetic development data, not official benchmark reproductions or leaderboard
submissions. External evaluation cases must remain out of training.

## CADBench: reconstruct geometry from observations

[Paper](https://arxiv.org/html/2605.10873v1) and
[official runner](https://github.com/anniedoris/CADBench).

CADBench covers six families: CAD-Base (DeepCAD), CAD-Fusion, CAD-Extrude,
CAD-All-Ops, CAD-Mechanical and CAD-Organic. STEP-based families are stratified
by BRep face count; mesh-based objects add broader geometric diversity. Inputs
include single-view grayscale, multi-view grayscale, photorealistic images,
clean meshes and noisy meshes. It measures volumetric and surface similarity,
executability and program compactness. The released runner currently consumes
CadQuery Python, not our restricted action JSON.

Implemented development coverage:

- Exact native mechanical construction in FreeCAD/CadQuery and polygonal
  mechanical construction in Blender, with independent reference/analytic checks.
- Single-view or multi-view goal images, plus explicit visual dimensions.
- Parameter/feature and oracle-horizon variation; new flanges, bores, counterbores
  and blind-hole patterns, in addition to our existing shaft/boss/plate curricula.
- Native execution validity, strict geometric correctness, native projects and
  current-state feedback. CadQuery/Blender workers also report solid-volume IoU;
  sampled Chamfer/coverage diagnostics use fixed coordinates and a declared seed.

Not implemented: official CADBench import/adapter, its principal-axis alignment
and metric aggregation, full operation vocabulary, measured face-count bins,
diversity sampling, organic meshes, mesh-conditioned agents, noisy-mesh goals,
PBR goal generation, or benchmark-equivalent reconstruction from renders alone.
Our specification images reveal dimensions not present in CADBench renders;
these are different observation contracts and their scores cannot be pooled.

An experimental separate `solidcad.v1` contract now executes polygon and arc
extrusions, revolves, fillets, chamfers, top-open shells, boolean operations and
multiple final solids in all three engines. It passed native positive/negative
feature tests; see [advanced solid-graph scope](SOLIDCAD.md). These operations
are not part of the frozen portable benchmark or its queued mixed-SFT model.
Sweep/loft, general sketch constraints, assemblies and richer manufacturing
workflows remain absent. Native operation support is not evidence of learned
model competence or full CADBench operation coverage.

## CADWorld: operate CAD through a long workflow

[Official project and task examples](https://cad-world.github.io/).

CADWorld has 200 FreeCAD tasks across sketch, part design, assembly, manufacturing,
FEM, appearance, point clouds, macros, measurement, mesh and TechDraw. Examples
include fully constrained tangent arc/circle sketches. Its main setting uses GUI
screenshots and actions with task-specific checks on saved engineering state.

Our current overlap is part construction, numeric edits, preservation of existing
features and native artifact inspection. Synthetic repairs provide recovery
practice. We do not implement the GUI action benchmark, tangent/arc constraints,
assemblies, CAM, FEM, point clouds or drawing-document generation. New Blender
support does not establish those FreeCAD workflows. These need separate action
sets and independent result checkers, not only a new render.

## Native FreeCAD validation

[freecad-validator](https://github.com/gNucleus-AI/freecad-validator) provides an
external reference for artifact-oriented validation. Our checks likewise separate
successful execution from correct geometry and inspect saved native data. We have
not run its external suite or claimed equivalence with its grading rules.

## Data and evidence rules

Keep physical-design lineages together across engines, modalities and edits.
Reserve complete geometry families and operation combinations in a future
benchmark version; the initial multi-engine split is a physical-lineage split,
not an entire-family or unseen-operation holdout. Do not tune on the existing
final test. Report each engine and task type separately, including failures and
retention of the existing FreeCAD skill. Rendering style, units and multi-edit
combinations need independent development and fresh final holdouts.

The first question for mixed training is whether it helps transfer without
forgetting FreeCAD. Merely executing the same common JSON tool program in three
engines does not prove the model knows three native APIs. Native Python API
code-generation would be a separate action contract and benchmark.
