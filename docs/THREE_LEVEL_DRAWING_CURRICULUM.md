# Three-level paired drawing curriculum

Previously, the paired 500-part drawing corpus included only `challenge` and
`extreme` tasks (400 TRAIN, 100 development-validation). It was not a balanced
easy-to-hard benchmark. The existing SFT pilot used 1000 individual next-action
rows, not 1000 complete tasks.

The new generator exposes three fixed structural levels across the same three
families: heterogeneous plates, multi-support fixtures and terraced flanges.

* **easy:** 10–16 teacher actions; two-hole plates, a one-boss fixture, or a
  two-level flange. Some counterbores and varying depths are present.
* **medium:** 32–48 teacher actions; maps to the existing `challenge` geometry.
* **hard:** 72–108 teacher actions; maps to existing `extreme` geometry, with an
  execution budget of at most 128 actions.

Levels are selected before running a policy. Action count is a structural proxy,
not a guarantee of model difficulty. No tasks are removed because a model fails.
Within each level, the 30-part benchmark has 10 parts per family.

Each physical task is generated once and exported into **both ISO and GOST**
representations with identical native source, teacher program and split. The
benchmark therefore has 90 independent parts, 180 drawing bundles and 180 agent
episodes. Both representations of one part are correlated observations, not two
independent geometries. These are synthetic **draft profiles**, not certified
ISO/GOST conformance or a claim of arbitrary industrial CAD coverage.

## Current runs

The following jobs were launched, not yet reported complete:

* `artifacts/curriculum-benchmark-90-v1`: 30 validation parts per level, starting
  at seed 21000000. After generation, the original local Qwen3.5-4B model at
  `artifacts/models/qwen35-4b` is evaluated without our adapters. It waits for
  `artifacts/iso-gost-sft-pilot-v1/experiment.json` to reach a terminal state and
  refuses to start if GPU1 still has a compute process. GPU0 is not used.
* `artifacts/curriculum-train-90-v1`: a separate starter TRAIN corpus with 30
  parts per level, starting at seed 22000000; generation only, no model training.

Both plans exclude all 500 previous paired lineages. TRAIN/validation assignment
uses the original natural lineage hash, sampled to quotas without relabeling
splits. The new TRAIN and validation plans have zero lineage overlap. The new
benchmark is **development-validation**, not an untouched final TEST set.

The per-part factory verifies teacher execution and analytic geometry in native
FreeCAD, exports causal step feedback, then converts both representations and
audits their image inputs/teachers. Native failures remain failures of the frozen
plan: they are not silently replaced. Native rejection of any selected task makes
the job fail. Unchanged commands can resume audited completed shards.

## Generate a benchmark

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/generate_curriculum_drawings.py \
  --output artifacts/curriculum-benchmark-90-v1 \
  --per-level 30 --split validation --seed 21000000 \
  --exclude-pairs artifacts/iso-gost-paired-500-v1/pairs.jsonl \
  --jobs 2 --benchmark-base \
  --wait-experiment artifacts/iso-gost-sft-pilot-v1/experiment.json
```

## Generate training data at a chosen scale

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/generate_curriculum_drawings.py \
  --output artifacts/curriculum-train-large-v1 \
  --per-level 1000 --split train --seed 23000000 \
  --exclude-pairs artifacts/curriculum-benchmark-90-v1/paired/pairs.jsonl \
  --exclude-pairs artifacts/iso-gost-paired-500-v1/pairs.jsonl \
  --jobs 2
```

This example requests **3000 geometries / 6000 drawing bundles**, not 1000 SFT
rows. It is a command example, **not a launched large run**. `--levels easy`,
`--levels medium hard`, `--per-level N`, `--seed` and `--shard-size` control
selection. `--plan-only` freezes a deterministic plan without native generation.
Counts can be any positive integer for generation; the baseline launcher requires
a per-level multiple of three for equal family quotas. Finite parameter spaces,
sampling-window exhaustion, storage and native runtime remain real limits;
this is not unbounded generation of any possible CAD operation.

## Outputs and learning use

* `primary/plan.json`: frozen task definitions, seeds, lineages, source tiers.
* `task-statistics.json`: requested counts and teacher action ranges.
* `primary/shards/*/source`: native verified teachers and causal trajectories.
* `primary/shards/*/dataset` and `paired/shards/*/dataset`: paired profile corpora,
  including PDF sheets, PNG crops, editable FCStd and feedback exports.
* `paired/pairs.jsonl`: physical tasks with pointers to both representations.
* `paired/summary.json`: actual completed counts and failures.
* For TRAIN shards: `exports/train.feedback.jsonl`, ready for the existing visual
  SFT loader; pass each profile's **dataset directory** using `--dataset` and
  `--extra-dataset`. Do not pass the top-level orchestration directory as a dataset.
* `baseline/status.json` and episode artifacts: baseline progress and native scores.
* `baseline/levels.json`: after all episodes, counts, successes, geometry scores
  and invalid actions separately for each level and profile.

Data generation supplies SFT examples and native reference/reward tasks usable
by rollout pipelines. It does **not** itself run GRPO or create rollout advantages.
More data alone also does not resolve the diagnosed counterbore omissions and
error recovery gap; coverage-aware sampling and rollouts remain necessary.
