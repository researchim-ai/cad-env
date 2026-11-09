"""Audit matched ISO/GOST observations, cross-shard identities and native sources."""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.dataset import audit_technical_dataset
from cad_env.drawings.export import sha


def audit_pairs(primary, paired, workers=4):
    primary, paired = Path(primary).resolve(), Path(paired).resolve()
    base = json.loads((primary / "batch-summary.json").read_text())
    summary = json.loads((paired / "summary.json").read_text())
    request = json.loads((paired / "request.json").read_text())
    if base["status"] != "complete" or summary["status"] != "complete":
        raise ValueError("Both generation stages must finish before the final audit")
    if request["primary_plan_sha256"] != sha(primary / "plan.json"):
        raise ValueError("Frozen source plan changed")
    plan = json.loads((primary / "plan.json").read_text())
    expected = {t["task_id"]: t for s in plan["shards"] for t in s["tasks"]}
    pairs = [json.loads(s) for s in (paired / "pairs.jsonl").read_text().splitlines()]
    if len(pairs) != plan["count"] or {p["task_id"] for p in pairs} != set(expected):
        raise ValueError("Paired index has missing or duplicate tasks")
    datasets = {str((primary / r["dataset"]).resolve()) for r in base["results"]}
    datasets.update(r["added_dataset"] for r in summary["shards"])
    datasets = sorted(datasets)
    for path in datasets:
        if not any(Path(path).is_relative_to(root) for root in (primary, paired)):
            raise ValueError("Unsafe dataset path")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        audits = list(pool.map(audit_technical_dataset, datasets))
    errors = [
        str(path) + ": " + str(audit["errors"][:3])
        for path, audit in zip(datasets, audits)
        if not audit["ok"]
    ]
    if errors:
        raise ValueError(str(errors))
    counts, splits, families, tiers, feedback = (Counter() for _ in range(5))
    cases = []
    pages = images = native_checks = 0
    smallest_font = float("inf")
    groups = set()
    for pair in pairs:
        task = expected[pair["task_id"]]
        group = task["split_group"]
        if (
            group in groups
            or pair["split_group"] != group
            or set(pair["representations"]) != {"iso", "gost"}
        ):
            raise ValueError("Missing profile or repeated/changed physical parent")
        groups.add(group)
        splits[task["split"]] += 1
        families[task["family"]] += 1
        tiers[task["difficulty"]] += 1
        model_hashes = set()
        for standard, representation in pair["representations"].items():
            dataset = Path(representation["dataset"]).resolve()
            if str(dataset) not in datasets:
                raise ValueError("Pair refers to an unaudited dataset")
            actual = json.loads(Path(representation["task"]).read_text())
            if any(
                actual[key] != task[key]
                for key in (
                    "task_id",
                    "split_group",
                    "split",
                    "parameters",
                    "oracle_actions",
                    "reference_actions",
                )
            ):
                raise ValueError("Paired task changed geometry, teacher or split")
            if actual["drawing_standard"] != standard:
                raise ValueError("Wrong profile label")
            bundle = dataset / actual["drawing_bundle"]
            manifest = json.loads((bundle / "manifest.json").read_text())
            model_hashes.add(sha(bundle / "source.fcstd"))
            if representation["source_sha256"] != manifest["source_sha256"]:
                raise ValueError("Index source hash disagrees with bundle")
            tiles = json.loads((bundle / "tiles.json").read_text())["tiles"]
            smallest_font = min(
                smallest_font, *(t["estimated_min_font_px_at_327680"] for t in tiles)
            )
            native_checks += len(
                json.loads((bundle / "native.json").read_text())["native_spec_checks"]
            )
            pages += manifest["page_count"]
            images += len(manifest["vlm_tiles"])
            counts[standard] += 1
            cases.append(
                {
                    "case_id": task["task_id"] + ":" + standard,
                    "pair_id": task["task_id"],
                    "split_group": group,
                    "split": task["split"],
                    "standard": standard,
                    "dataset": str(dataset),
                    "task": representation["task"],
                    "pdf": representation["pdf"],
                    "native_model": representation["native_model"],
                    "images": representation["vlm_images"],
                }
            )
        if len(model_hashes) != 1:
            raise ValueError("ISO and GOST use different source FCStd files")
    for dataset in datasets:
        for kind, count in json.loads((Path(dataset) / "summary.json").read_text())[
            "counts"
        ].items():
            feedback[kind] += count
    if counts != {"iso": plan["count"], "gost": plan["count"]} or smallest_font < 10.5:
        raise ValueError("Unexpected profile count or unreadable crop budget")
    result = {
        "ok": True,
        "status": "complete",
        "independent_geometries": len(groups),
        "matched_pairs": len(pairs),
        "profile_counts": dict(counts),
        "drawing_documents": len(cases),
        "pdf_pages": pages,
        "vlm_images": images,
        "native_feature_checks": native_checks,
        "minimum_estimated_font_px": smallest_font,
        "split_counts_by_geometry": dict(splits),
        "families": dict(families),
        "difficulties": dict(tiers),
        "feedback_rows_across_both_profiles": dict(feedback),
        "audited_datasets": len(datasets),
        "same_native_sources": True,
        "same_teacher_actions": True,
        "same_splits": True,
        "training_started": False,
        "model_evaluation_run": False,
        "test_opened": False,
        "can_claim_full_conformance": False,
        "scope": "Matched synthetic nominal-geometry inputs for representation comparison. Full ISO/GOST conformance and manufacturing requirements remain incomplete.",
    }
    (paired / "cases.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in sorted(cases, key=lambda c: c["case_id"]))
    )
    (paired / "final-audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Positive worker count required")
    print(json.dumps(audit_pairs(args.primary, args.paired, args.workers), indent=2))


if __name__ == "__main__":
    main()
