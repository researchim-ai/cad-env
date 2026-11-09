# Matched ISO/GOST drawing-agent pilot — 2026-11-09

Completed run: `artifacts/paired-20-model-benchmark-v2`.
This is a bounded development benchmark of the current agent protocol, not an
industrial drawing-conformance test or a full-corpus model result.

## Inputs and protocol

Twenty independent validation parents were selected before inference using a
fixed hash order (seed 20261109), approximately balanced across three families and
two difficulty tiers: 8 plates, 6 multi-support fixtures and 6 stepped flanges;
10 challenge / 10 extreme. Each model received both ISO and GOST representations
of each parent: **40 episodes per checkpoint, 80 episodes total**.

The two checkpoints are the original local Qwen3.5-4B and the previously tuned
`release-trial-v1/candidate/merged`. Neither was trained on the new 500-parent
paired corpus in this experiment. Full checkpoint identities/hashes are recorded
in each model's `evaluation.json`.

Every episode starts with an empty CAD document. The policy sees raster drawing
crops and its own causal public JSON state/history. It receives no teacher prefix,
numeric target specification, reference program, or target 3D model. The current
profile uses goal images and JSON state; it does not render the current candidate
back to the model (`goal_only_observation`). General page footnotes/title blocks
outside the selected crops are not supplied as additional full-page images.

Inference uses physical GPU1 RTX3090, BF16, greedy decoding, batch size 2, four
native episode workers, 32768 context tokens, 327680 pixels per image and 1024 new
tokens per action. The original per-task action budget is retained, up to 128.
Six consecutive actions leaving the public state unchanged terminate an episode
as stagnation. This guard is part of the measured protocol, so these results do
not represent unrestricted retrying up to the full action budget.

Success requires an agent-issued `finish` and native shape/parametric/specification
success at the task's 1e-5 tolerance. Non-finished episodes also receive a separate
evaluator-only final geometry probe; that probe is not an agent action. Native
geometry score is volumetric B-rep IoU from OCCT boolean missing/extra volumes,
with no pose/scale normalization. Infrastructure failures are kept separate from
model failures. **All 80 episodes completed without infrastructure errors.**

The initial v1 was an interrupted token-budget qualification using 256 tokens;
it is marked non-reportable. The final v2 uses the same selected parents and a
1024-token limit sufficient for supported multi-circle actions. No v1 outcomes
are pooled into v2.

## Measured results

- Original Qwen3.5-4B: ISO **0/20** and GOST **0/20** exact successes.
  Mean solid IoU: ISO **0.06218**, GOST **0.04709**. Valid final solids: 5/20 and
  4/20. There were 271 invalid actions among 374 total actions (72.46%). All forty
  episodes ended at the stagnation guard.
- Previously tuned checkpoint: ISO **0/20** and GOST **0/20** exact successes.
  Mean solid IoU: ISO **0.79887**, GOST **0.82869**. Valid final solids: **40/40**.
  There were 168 invalid actions among 906 actions (18.54%). Twenty-one episodes
  ended with the agent's finish; nineteen ended at the stagnation guard.
- The six native oracle controls, one per family/tier, all achieved success with
  reward 1.0 under the same native grader.

A higher IoU can reflect the stock or main body while missing small but critical
holes, counterbores, depths or individual steps. It does not replace task success.
The observed GOST/ISO difference is descriptive on 20 paired parents, not evidence
that one representation is generally better.

## Concrete failure evidence

The original model mostly fails object references/types or duplicates names.
The tuned model constructs valid solids, but still has reference/name failures,
misses feature details and groups holes of incompatible depths.

For `x-e574d9a3146ceb08921916cf`, the tuned ISO rollout correctly proposed the
204 x 174 x 16 mm stock. It then put the holes at (163.2, 130.5) and (163.2, 43.5)
into the same `HoleSketch02` and applied a 4.8 mm Pocket. Their specified depths
are respectively 4.8 and 8 mm. The native result consequently fails despite an
IoU of about 0.995. Counterbores also need separate matching of diameter/depth.

A current environment constraint contributed failures: circle sketches accept
1–16 circles. The tuned checkpoint sometimes proposes roughly 30 circles and
then tries to use the rejected sketch. This is recorded as an API failure, not
an image-reading failure. The supported reference programs remain executable
within the same constraints. Making the constraint explicit in the public guide
and testing better grouping are separate follow-up changes, not part of this run.

Exploratory numeric diagnostics count only the first proposed base rectangle:
the original model proposed the correct length/width in all 14 applicable cases
for each profile. The tuned model did so in 14/14 ISO cases and 12/14 GOST cases
(the other two did not propose a rectangle). This narrow diagnostic does not
establish correct reading of every feature or causal image grounding; there were
no blank-image ablations in this pilot.

## Evidence and reproduction

- `selection.json`: exact parents and matched input paths.
- `status.json`: all eighty episode results, checkpoint identities and protocol.
- `<checkpoint>/evaluation.json`: native outcomes, batch statistics and memory.
- `<checkpoint>/<standard>/<task>/responses.jsonl`: raw model outputs and input hashes.
- Native episode trajectories, FCStd outputs and final-score directories: actual
  student actions, state/error feedback and OCCT validation.
- `oracle-controls.json`: separate positive controls, never counted as model successes.
- `analysis.json`: aggregate metrics and concrete error examples.

The command and settings are in `recipes/paired-drawing-benchmark-20.json`.
Use a fresh output directory to rerun. Aggregate a completed run with:

```bash
python scripts/analyze_paired_benchmark.py --run artifacts/paired-20-model-benchmark-v2
```

No training was started, no TEST split was opened, and no checkpoint was promoted.
