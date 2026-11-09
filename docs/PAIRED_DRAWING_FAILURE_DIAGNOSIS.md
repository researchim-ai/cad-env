# ISO/GOST SFT pilot: failure diagnosis

This report describes a **22 completed episode snapshot** of the ongoing
`artifacts/iso-gost-sft-pilot-v1` evaluation, not its final result. The main
benchmark and policy prompts were not changed for these diagnostics.

## Confirmed behavior

There are two exact successes in this snapshot. Mean native geometry score on
these **same completed cases** is 0.986235 before and 0.992762 after the new SFT.
The earlier whole-benchmark average must not be compared against this partial
subset: cases finish at different times and this is not a random interim sample.

Of 11 agent-finished episodes, a program-level cut signature diagnostic matches
222 of 224 primary cuts, but only 16 of 108 counterbores. Across all 22 episodes,
including early stagnation, it matches 255/400 primary cuts and 24/188
counterbores. These signatures compare circle position, radius and pocket depth,
ignore object names/order, and are **not native feature-completeness scores**:
supports, overlap and geometric equivalence still require native verification.

Eight cases have every primary cut signature but lack at least one counterbore.
High global geometry scores hide these small, required details. Strict failure
is justified; lowering the tolerance would conceal the actual problem.

### Counterbore omission: native intervention

`x-e574d9a3146ceb08921916cf`, ISO: the agent creates the base and all 12 primary
holes with correct dimensions, skips all six counterbores, then finishes.
Native geometry score is 0.9977586224. The corresponding input SIZE/DEPTH crop
contains the counterbore diameter/depth column; it is not missing from the input.
Visual inspection confirms the values are present, but does not prove the model
recognizes them after processing.

Replay the student's committed actions, removing its final finish; append only
the six teacher counterbores (12 actions), then finish. Native validation gives
geometry score 1, symmetric difference 0 and exact task success. This is a
**teacher-assisted causal diagnostic**, not a model success, and is stored outside
the benchmark in `diagnostics/teacher-cb-intervention`.

### Object name collision: native intervention

`x-6fe436081bf9ee4d852c0b25`, GOST: the agent creates sketch `Hole01Ream`, then
requests a pocket also named `Hole01Ream`, with that sketch as its profile.
The backend rejects the duplicate name. Repeated attempts exhaust the stagnation
guard. Replay of the same prefix/action fails; changing only the new pocket's
name to `Hole01ReamCut` makes execution succeed. This does not establish success
on the rest of the task. Evidence is in `diagnostics/name-collision-intervention`.

## Training coverage and limits

The one-epoch run contains **1000 next-action examples, not 1000 complete CAD
trajectories**, from 267 distinct TRAIN parents. ISO/GOST examples are paired:
they intentionally share their underlying teacher decisions.

* 56 examples supervise starting a counterbore sketch: 28 paired decisions.
* 66 examples supervise its pocket; much of that decision is already constrained
  by the preceding correct sketch and object state.
* Only eight examples are marked recovery (0.8%).
* Only 16 supervise finish. These are positive teacher endpoints, not explicit
  counterexamples to finishing with an omitted feature.

The completion mask, image mask and template boundaries were checked across all
1000 samples. Resume round-trip verified all 340 adapter tensors exactly; language
and vision gradients were nonzero on all 100 resumed updates. These checks rule
out some plumbing failures, but do **not** prove image grounding or agent success.

The evidence is consistent with an imbalance between clean-prefix next-action
training and autonomous execution: after skipping a counterbore or changing an
object name, the agent encounters states scarcely represented in training.
This remains a mechanism hypothesis; it needs controlled prefix/image probes.
More epochs on the same easy rows are not an established fix.

The document crop pipeline also excludes the schedule's explanatory notes above
and below the table. The core instruction supplies some conventions, but a
future input version should preserve those notes explicitly. The current frozen
benchmark must remain unchanged for the before/after comparison. These drawings
use experimental ISO/GOST profiles; full standards conformance is not established.

## Next controlled experiments

1. Complete the frozen 40 episodes and run the diagnostic on the final snapshot.
   Keep validation failures for diagnosis, never insert their teachers into TRAIN.
2. On fresh TRAIN parents, collect rollouts with omitted counterbores, renamed
   sketches and duplicate-name attempts. Validate recovery actions in native
   FreeCAD; include those states as supervised examples. Reject incompatible
   repairs instead of blindly replaying the teacher's original object names.
3. Replace uniform row selection with coverage quotas for first counterbore
   decisions, later-page rows, mixed depths, recovery and completion checks.
   Count distinct parents and decisions as well as profile duplicates.
4. Isolate perception from execution: compare original-image input, blank-image
   input and a teacher-transcribed schedule **in a separate diagnostic**. Compare
   correct teacher prefixes with the student's divergent prefixes. Transcribed
   teacher fields are an upper-bound intervention, never a deployable baseline.
5. Trial an image-to-feature checklist followed by typed CAD execution. Track
   primary holes and counterbores separately; finish only after checking every
   extracted feature. The checklist must be predicted from public inputs, not
   supplied by the hidden evaluator. Evaluate this changed protocol separately.

No additional training has been launched by this diagnosis. The original
post-training evaluation continues on GPU 1.

## Reproduce the CPU snapshot analysis

```bash
python scripts/diagnose_paired_failures.py \
  --run artifacts/iso-gost-sft-pilot-v1 \
  --baseline artifacts/paired-20-model-benchmark-v2 \
  --output artifacts/iso-gost-sft-pilot-v1/failure-diagnosis.json
```

`tests/test_paired_diagnosis.py` checks renamed objects, wrong depths, unpocketed
sketches, grouped circles and duplicate multiplicity. Native interventions and
all benchmark scores remain separate artifacts.
