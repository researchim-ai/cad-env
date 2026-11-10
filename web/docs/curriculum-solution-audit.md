# Full audit of the completed v1 curriculum benchmark

The audit passed on the same **90 validation tasks**, with ISO and GOST
representations checked separately for the original and CAD-tuned models.
The original benchmark results were not modified. CPU FreeCAD replay and
artifact inspection were used; no new model generation was performed.

Confirmed results by representation:

- Original Qwen: ISO 0/90; GOST 0/90.
- CAD-tuned checkpoint: ISO 47/90; GOST 47/90.
- Easy: ISO 30/30; GOST 25/30.
- Medium: ISO 14/30; GOST 16/30.
- Hard: ISO 3/30; GOST 6/30.
- Every successful output has zero B-Rep symmetric-difference volume against
  its reference. These are not borderline tolerance passes.
- Across the 90 paired tasks: 40 solved in both representations, 7 only in ISO,
  7 only in GOST, and 36 in neither.
- No missing, duplicate or incorrectly aggregated benchmark records were found.

Results are not combined into a score over the two representations: the unit
of comparison is a task, represented by either ISO or GOST drawings.

## Checks performed

1. Recomputed per-level/profile aggregates and compared them with `levels.json`
   and the published before/after JSON. Checked baseline/candidate protocol
   equality and checkpoint file identities.
2. Reconstructed **8,544 policy requests/actions** from response logs and native
   traces. Verified prompt hashes, goal-image hashes, committed actions, invalid
   action counts and actual agent finish. The scoring-only appended finish was
   not treated as an agent finish.
3. Checked original saved FCStd hashes, schema/worker hashes and scoring requests.
4. Re-executed the committed programs for every task, model and representation in fresh FreeCAD processes, with the
   frozen task reference and analytic volume/bounds checks. Verified runtimes
   and compared replayed success with recorded success.
5. Used a separate native verifier, which does not import the benchmark evaluator,
   to open the original saved candidate, replayed candidate, dataset oracle and
   drawing's native model. Compared exact shapes, analytic volume/bounds, valid
   solid count and constrained-sketch editability. A real parameter perturbation
   had to change the solid for a successful editable model.
6. Compared circular cut signatures for the successful programs in both representations as an
   additional feature diagnostic. All expected cuts match. Signature matching
   ignores names/order/support details and supplements the B-Rep check.

One GOST easy solution (`x-0e89c9cb915807072f0826e0`) repeats the same counterbore
cut twice. All required features are present and the final geometry is exact.
The benchmark grades completed geometry/editability, not minimum action count,
so its success remains correct. Efficiency can be reported as a separate metric.

## Audit artifacts and reproduction

Final report: `artifacts/curriculum-solutions-audit-v1/status.json` (`passed`,
all tasks and representations checked, 8,544 logged actions verified, no global errors or case discrepancies). Each case has
fresh native replay artifacts and `independent.json`. Executed verifier source
snapshots and their hashes are retained in the audit directory.

```bash
python scripts/audit_curriculum_solutions.py \
  --root artifacts/curriculum-benchmark-90-v1 \
  --output artifacts/my-fresh-solution-audit --workers 4
```

The first auditing pass incorrectly rebuilt policy history using normalized
execution values instead of raw model JSON (integer/float serialization and key
order matter for string hashes). Its diagnostic report is preserved in
`first-pass.json`. Trace reconstruction was corrected and all prompts were
rechecked against their original hashes; completed native checks were retained.
This was an auditing implementation issue, not a change to inference inputs,
student solutions or benchmark scores.

This audit confirms the saved benchmark's accounting and native solution
correctness. It does not establish full ISO/GOST conformance, visual necessity,
independence from all historical model pretraining, or final TEST performance.
The candidate had multiple stages of CAD tuning before evaluation; the total
gain must not be attributed solely to the last ISO/GOST SFT pilot.
