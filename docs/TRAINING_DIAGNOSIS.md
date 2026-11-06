# Training diagnosis — 2026-11-06

## Evidence and scope

No new SFT was launched during this investigation. GPU1 ran a bounded component
probe of the raw Qwen3.5-4B and the completed hard-pilot checkpoint. Both use
greedy generation, native thinking disabled, 1024 output tokens, identical
prompts and pixel budgets. Eighty questions per model cover 20 validation
parents per task family. These prompts differ from training's tool-action
contract: this is a diagnostic, not a replacement benchmark. TEST stays sealed.

Reproducible artifacts:
- `artifacts/training-review-v1/local-findings.json`: native final-state counts.
- `artifacts/training-review-v1/mixture-census.json`: actual sampled mixture.
- `artifacts/training-review-v1/component-probe-v2/status.json`: completed probe.
- `artifacts/training-review-v1/component-probe-v2/*-responses.json`: raw answers,
  parseability, exactness and possible output-cap flags.
- `cad_env/learning/perception_probe.py`: public inputs and private offline grading.

Raw → trained component exactness:
- Marker count: 18/20 → 13/20; both 20/20 parseable.
- Full marker-to-public-circle-index set: 0/20 → 0/20. Raw only 5/20
  parseable versus trained 20/20: raw/trained conditional precision is not
  a fair population comparison. Trained mean precision .623, recall .661.
- Full hole table: 20/20 → 20/20; raw answers need JSON salvage whereas
  trained answers are strict JSON. Thus table OCR is not the main explanation
  for the counterbore construction failures.
- Fixed table crop input: 6/20 → 20/20; raw only 6/20 parseable. This does not
  establish that cropping harms raw visual perception: format/cap failures
  prevent that interpretation.

Marker-count decline is a possible regression on a small, new-prompt probe;
repeat with paired variants before claiming catastrophic forgetting. Selection
failure is consistent with weak spatial binding, not proof of a capacity limit.
Construction failure despite good table extraction points toward maintaining
complete feature sets, translating rows to operations, and action sequencing.

## What published recipes actually do

[IterCAD](https://arxiv.org/html/2606.13368v1) uses the same Qwen3.5-4B:
28K trajectories, including 8K student trajectories corrected by a teacher,
then multi-turn geometry-feedback RL. Its released
[SFT script](https://github.com/KnowledgeXLab/IterCAD/blob/main/train/IterCAD_SFT.sh)
uses full tuning, three epochs and eight GPUs. This is evidence of a different
training scale and interaction loop, not a requirement to copy that budget.
Its released RL/evaluation also disable native thinking; our flag alone cannot
explain failure. Explicit reasoning and output budgets differ.

[DeciCAD](https://arxiv.org/html/2610.21755) supports semantic CAD tools,
causal next-action SFT and LoRA: the interface choice is defensible. It teaches
recovery from observed failures and online decisions, including termination.
Its data quality and harness results caution against treating more JSON or
more examples as sufficient. It uses a different model and benchmark.

[CADFusion](https://github.com/microsoft/CADFusion) alternates sequential
learning and visual-feedback preference optimization. Its instructions stress
collecting feedback on the current model's own generations. Our four-parent
single-edit repair set does not cover current multi-edit failures.

## Changes justified before another training run

1. Separate spatial binding, full feature coverage, stopping and native geometry
   metrics. Preserve strict whole-task success and image-removal controls.
2. Add current rendered CAD feedback to a controlled dataset variant; current
   hard editing supplies changing JSON state but only a static goal image.
3. Collect genuine current-student multi-edit failure histories on TRAIN,
   teacher-correct them with native verification, and supervise recovery and
   finish decisions. Keep causal history before each action.
4. Increase independent designs, feature graphs and image layouts rather than
   repeat a small set of teacher steps. Audit equivalent-valid next-action
   ordering; canonical ordering is an ablation, not an established cure.
5. Test explicit all-feature/grounding supervision against the existing
   next-action objective with the same budget and parent splits. Change one
   factor per pilot, within one epoch and the user's short-run budget.
6. Only then evaluate trajectory RL. Single-action bandit feedback does not
   reproduce multi-turn GRPO/GSPO. Check credit assignment, early stopping,
   unchanged-feature protection and exact geometry, not only global IoU.

The perception-plan prototype is untrained and unqualified. A deterministic
compiler changes the controller protocol; any score gain must be reported
separately. Neither LoRA rank, numeric-token weighting, number of steps nor
Liger kernels has been causally established as the visual failure's cause.

## Implemented next experiment

`grounding_data.py` creates and semantically audits separate TRAIN-only VQA
packages. Actual visual package: 512 rows (256 counts + 256 complete selections),
64 parent designs, 256 distinct images after exact-pixel deduplication. Actual
construction package: 48 complete hole-table targets from 48 parents. These
are auxiliary questions, not a deterministic CAD executor. Dataset hashes,
public-only prompt reconstruction, exact targets and split lineage are checked.

`train_modern --action-contract single_action_auxiliary` explicitly allows
mixing action and auxiliary-grounding exports. The default action contract
remains unchanged. The standard processor, completion-only masking, vision
adapter checks and memory preflight apply to the mixed data.

`multistep_recovery.py` collects real TRAIN student prefixes and native teacher
continuations. It intercepts a proposed finish before execution, supervises the
alternative action at that exact causal state, restores unwanted edits and
completes missing targets. It never reopens a terminated environment. With
`--current-render`, native current candidate renders are included as well.
The four-parent smoke test produced two native-successful recoveries and two
rejections because the remaining eight-step episode budget could not fit the
repairs. Rejections are retained as evidence, never as successful supervision.
The output is explicitly candidate feedback, not an approved training package;
it needs packaging/semantic audit before mixing into SFT. No broad recovery
improvement is claimed from this smoke test.

`grounding_experiment.py` runs two arms from the same hard-pilot checkpoint:
- Control: 1024 existing hard-edit action rows.
- Auxiliary: the identical first 800 sampled action rows, plus 192 visual VQA
  and 32 construction-table rows (all sampled without replacement).
- Each arm: 128 updates, effective batch 8, one prepared-mixture epoch;
  LR 5e-6, numeric weight 8, same vision LoRA and qualified fused/checkpoint
  profile. Checkpoints every 32 updates; 20 optimizer minutes per arm;
  60-minute total subprocess deadline. A stopped arm invalidates the matched
  128-update comparison; it is never silently counted as complete.
- This matches optimizer updates and examples, not supervised tokens/FLOPs or
  wall time. There are no recovery examples or observation changes in either
  arm, isolating the replacement with explicit perception supervision.
- Full 160-task native visual validation and the 80-case-per-model component
  probe compare both arms. TEST stays sealed; no automatic promotion or next
  training is permitted. Broader retention and image controls are still required
  before adopting a candidate even if this local ablation succeeds.

The experiment saves command logs, source hashes, data audits, optimizer/vision
gradient checks and durable status. Entry point (from the pinned environment):

```bash
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
  .venv-qwen35/bin/python -m cad_env.learning.grounding_experiment \
  --output artifacts/grounding-ablation-v1
```

Launch status: `artifacts/grounding-ablation-v1` started on physical GPU1.
The initial control preflight checked all 1024 rows: image tokens masked,
completion-only masks and template boundaries valid; longest row 1904 tokens.
Focused regression checks passed 22 tests, including deliberate prompt/label/
split corruption, missing auxiliary targets, and causal native-recovery logic.
Training benefit is pending the completed two-arm evaluation.
