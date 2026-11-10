# Coverage-controlled drawing SFT

The first ISO/GOST pilot used uniform next-action sampling. Its 1,000 rows
included only 56 counterbore sketches, 66 counterbore pockets and 16 finishes.
These are row counts across both profiles, not independent geometries.

The completed 40-episode pilot has nine exact successes. The program signature
diagnostic matches 535/796 primary cuts but only 26/224 counterbores. These
counts include both drawing representations and incomplete rollouts; they are
not independent feature trials or a native geometric completeness metric.
The native intervention in `PAIRED_DRAWING_FAILURE_DIAGNOSIS.md` supplies
stronger evidence for one concrete omitted-counterbore case.

`scripts/prepare_drawing_coverage.py` prepares a deterministic TRAIN-only index
plan. Every selected decision has both ISO and GOST source row indices. It
checks paired teacher equality, source task lineage and explicit held-out parent
groups; source export and selection hashes are recorded. It never loads a model.

The exclusive category targets are 20% counterbore sketches, 20% counterbore
pockets, 10% finish, 20% other operations in the last 35% of a teacher program,
and 5% existing recovery-labelled rows. Remaining slots sample unused decisions.
Selection rotates between parents within each category. Unavailable quotas are
reported and filled from unused rows; no cycling or repeated decisions is used.
The `CB` classification relies on this generator's teacher naming convention.
Late program steps are a proxy for long-horizon coverage, not proof that a row
requires looking at a later drawing page.

```bash
python scripts/prepare_drawing_coverage.py \
  --paired artifacts/iso-gost-paired-500-v1 \
  --heldout artifacts/paired-20-model-benchmark-v2/selection.json \
    artifacts/curriculum-benchmark-90-v1/checkpoint250/selection.json \
  --decisions 1000 --output artifacts/drawing-coverage-v2/plan.json
```

This selects 1,000 paired decisions, hence 2,000 image-conditioned examples.
The prepared v2 plan covers 378 distinct TRAIN parents: 212 counterbore sketch
decisions, 216 counterbore pockets, 269 late decisions, 105 finishes, 52 existing
recovery-labelled decisions and 146 other decisions. Each occurs in both
profiles. All requested category quotas are available.
It writes an **index plan**, not a standalone training dataset. The current
trainer's source budgets/seeds cannot reproduce quota sampling: do not pass
this plan to the uniform pilot runner and assume it trained on these rows.
The next integration must materialize or explicitly load the selected indices,
verify source hashes and image containment, and record the consumed row IDs.

Existing recovery labels describe generator-provided error states. They are
not evidence of actual student mistakes or verified autonomous recovery.
Separate TRAIN rollouts and native-validated continuations are still needed.
No additional optimization or benchmark improvement is claimed by this plan.
