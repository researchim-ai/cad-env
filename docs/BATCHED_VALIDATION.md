# Batched validation and checkpoint-500 measurements

The sequential run is preserved in `artifacts/checkpoint500-evaluation`.
Completed native cohorts before the restart:

- FreeCAD: **41/66 → 65/66** strict success (62.1% → 98.5%), 33 physical
  parent designs. Paired gain +36.36 pp, cluster-bootstrap 95% CI
  [+27.27, +45.45] pp.
- CadQuery: **36/66 → 64/66** (54.5% → 97.0%). These are the same underlying
  designs as FreeCAD; do not treat their counts as independent evidence.
- The sequential old-task retention cohort was interrupted at **57/69**;
  it is partial and cannot be presented as a completed benchmark.

No training or queued SFT was restarted. Checkpoint-500 is merged once into
`artifacts/checkpoint500-evaluation/merged` and evaluated on physical GPU 1.

## Implementation

`batch_policy.py` adds greedy batched generation to the existing model loader,
without changing the historical `policy.py`. A single GPU-owner thread collects
requests from independent episode workers, pads on the left, keeps image order,
and returns each completion to its caller. Exceptions reach all waiting callers.
Each CAD episode still waits for its own tool feedback before its next action.

`parallel_protocol.py` preserves the native scoring, public observations, task
budgets and per-task seeds of the original protocol. CAD work is process-isolated;
reports retain task identity/physical lineage and record the concurrent execution
profile separately. Timings are not directly comparable with serial per-episode
latencies, which exclude reset and do not include waiting for other requests.
The CLI is `python -m cad_env.learning.batch_evaluate`.

The restart uses **batch limit 4 / 8 episode workers**, one model copy. Completed
serial FreeCAD/CadQuery outputs are retained; the four remaining cohorts restart
in fresh directories in `artifacts/checkpoint500-batch-evaluation`.

## Measurements and parity

On eight real multimodal prompts, exact response text matched **8/8** for both
batch 2 and 4 against scalar greedy inference. This small initial timing test
was slower for the batched generation itself: 5.66 s (scalar), 8.23 s (batch 2),
7.18 s (batch 4), with peak allocated memory 8.85 / 9.16 / 9.80 GiB. These are
initial microbenchmarks with heterogeneous prompts, not a warmed repeated
throughput study. Spare VRAM does not imply that a larger batch will be faster.

An end-to-end eight-episode native pilot reproduced **8/8 complete action
trajectories**. Its native episode wall span after runtime probing was about
27.7 seconds. The corresponding scalar episodes spent 33.1 seconds in policy
calls and steps alone, excluding their reset overhead. This supports a modest
end-to-end benefit from overlapping CPU work despite the inference microbenchmark.

The full batch retention cohort completed **69/69** successfully. All **57/57**
overlapping action trajectories from the interrupted scalar run matched exactly,
and all success outcomes matched. That is empirical overlap evidence, not a
guarantee of universal floating-point determinism for every future batch shape.

Batch retention took **143.1 s** from runtime-probe completion to final report,
excluding model loading/audit. Its 62 generation calls had batch sizes:
54×4, 1×3, 4×2, 3×1. The earlier interrupted scalar run completed 57 tasks over
about 196.6 s in the corresponding runtime-probe/report interval. These runs
have different task counts, so do not advertise their ratio as a controlled
same-cohort speedup.

Final cohort results and paired comparisons are recorded in the batch driver's
`status.json`; do not combine their timing distributions with the serial run.
The historical frozen worker/schema/prompt files remain unchanged.

## Website snapshot

The local website is in `web`, with actual audited TRAIN reference meshes and
recorded completed measurements. It labels serial mechanical cohorts separately
from batch retention. Refresh it with `python web/scripts/snapshot.py`; it is a
static snapshot rather than live GPU telemetry. No benchmark TEST examples are
used as public mesh demos.

## Completed checkpoint-500 validation

All six cohorts are complete (447 tasks across separate curricula). Preserve
execution profiles when reporting them:

- FreeCAD, serial: 41/66 → **65/66**.
- CadQuery, serial: 36/66 → **64/66**.
- Simple retention, batch: 69/69 → **69/69**.
- Complex builds, batch: 39/46 → **43/46**; paired +8.70 pp,
  parent-cluster bootstrap 95% CI [+2.17, +17.39] pp.
- Unlabelled visual selection, batch: 30/100 → **30/100**; no aggregate gain.
- Entity-labelled selection, batch: 86/100 → **88/100**; paired +2 pp,
  95% CI [-3, +7] pp, so the improvement is not supported.

The native construction gains are substantial on these generated tasks, while
visual selection remains a limitation. This is not an official external CADBench
score or a broad CAD competence claim. TEST was not opened. No further optimizer
updates were run. The batch comparison uses the same public task contract, but
its execution profile differs from the historical sequential parent evaluation;
parity evidence is the recorded 8-prompt test and 57 shared complete trajectories.

The website was checked in the in-app browser at desktop (1440 px) and mobile
(390 px): no horizontal overflow, no JavaScript errors, functional part selection,
wireframe, zoom/reset, copy controls, navigation and field-guide page. The
selected Python regression suite passed 42 tests before the final shutdown-drain
regression was added; all three batcher tests then passed. Syntax checks, Ruff,
local-link/native-mesh validation and `git diff --check` passed.
