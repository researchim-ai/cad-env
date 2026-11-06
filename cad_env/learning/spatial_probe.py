"""Paired no-training diagnosis of coordinate-frame and entity-ID burden.

Four views of each fixed validation task keep geometry, goal selection and
public numeric circle state constant. All-entity labels are generated for EVERY
circle and checked against native public circle ordering; they never identify
which entity is selected. Metrics belong to a changed observation protocol,
not to the old native CAD benchmark. Private expected sets stay in grading.
"""

import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops

from .grounding_data import SELECT
from .perception_data import public_initial, sha
from .perception_plan import public_payload
from .perception_probe import grade, messages
from .visual_tasks import render_goal


def variants(task):
    result = {}
    for canonical in (False, True):
        for labels in (False, True):
            t = copy.deepcopy(task)
            t["goal_spec"]["entity_labels"] = labels
            if canonical:
                t["goal_spec"]["angle"] = 0
            key = ("canonical" if canonical else "original") + (
                "_ids" if labels else "_plain"
            )
            result[key] = t
    return result


def build_cases(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    from .audit import audit_dataset

    audit = audit_dataset(root)
    if not audit["ok"]:
        raise ValueError(audit["errors"])
    seen, dataset = set(), []
    for entry in map(json.loads, (root / "manifest.jsonl").read_text().splitlines()):
        if (
            not entry["accepted"]
            or entry["split"] != "validation"
            or entry["split_group"] in seen
        ):
            continue
        folder = root / "episodes" / entry["task_id"]
        task = json.loads((folder / "task.json").read_text())
        if " inches" in task["instruction"]:
            continue
        seen.add(entry["split_group"])
        public = public_payload(public_initial(root, task))["state"]
        sketch = next(o for o in public["objects"] if o.get("circles"))
        circles = [
            {k: c[k] for k in ("index", "x", "y", "radius")} for c in sketch["circles"]
        ]
        # Verify raster label i really refers to current native circle index i.
        drawn = task["goal_spec"]["circles"]
        if len(drawn) != len(circles) or any(
            c["index"] != i
            or any(abs(c[k] - drawn[i][k]) > 1e-6 for k in ("x", "y", "radius"))
            for i, c in enumerate(circles)
        ):
            raise ValueError("Image/native index mismatch")
        for condition, t in variants(task).items():
            image = out / "images" / f"{entry['task_id']}-{condition}.png"
            render_goal(t, image)
            if condition == "original_plain":
                with Image.open(image) as a, Image.open(folder / "goal.png") as b:
                    if (
                        a.size != b.size
                        or ImageChops.difference(
                            a.convert("RGB"), b.convert("RGB")
                        ).getbbox()
                    ):
                        raise ValueError(
                            "Regenerated baseline pixels differ from frozen goal"
                        )
            question = SELECT + json.dumps(circles)
            if condition.endswith("_ids"):
                question += " Every black boxed integer labels the CURRENT circle index; ALL circles are labelled. Only colored crosshairs indicate selection. Read the labels of ALL marked circles."
            dataset.append(
                {
                    "task_id": entry["task_id"],
                    "parent": entry["split_group"],
                    "condition": condition,
                    "kind": "selection",
                    "messages": messages(question, [str(image)]),
                    "images": [str(image)],
                    "expected": sorted(task["goal_spec"]["selected"]),
                    "source_image_sha256": sha(folder / "goal.png"),
                    "image_sha256": sha(image),
                }
            )
    if not dataset:
        raise ValueError("No eligible validation tasks")
    return dataset


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", default="artifacts/hard-unlabelled-v1")
    p.add_argument(
        "--model", default="artifacts/grounding-ablation-v1/auxiliary/merged"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if not 0 < a.minutes <= 20:
        p.error("At most twenty diagnostic minutes")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = build_cases(a.dataset, out)
    (out / "private-grading.json").write_text(json.dumps(dataset, indent=2))
    state = {
        "status": "prepared",
        "optimizer_updates": 0,
        "test_opened": False,
        "cases": len(dataset),
        "parents": len({c["parent"] for c in dataset}),
        "summaries": {},
        "scope": "Paired component diagnostic; all-entity labels and canonical camera change the observation protocol; not native CAD success",
    }

    def persist():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / "status.json")

    persist()
    if a.prepare_only:
        return
    from .batch_policy import BatchedHFPolicy

    deadline = time.monotonic() + a.minutes * 60
    try:
        policy = BatchedHFPolicy(
            a.model,
            temperature=0,
            max_new_tokens=256,
            max_context=8192,
            max_pixels=327680,
            chat_template_kwargs={"enable_thinking": False},
        )
        state.update(
            status="running",
            model_identity=policy.identity,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        persist()
        rows = []
        for start in range(0, len(dataset), 4):
            if time.monotonic() > deadline:
                raise TimeoutError("Diagnostic budget exhausted")
            batch = dataset[start : start + 4]
            responses = policy.generate_batch(
                [(c["messages"], c["images"]) for c in batch]
            )
            for c, response in zip(batch, responses):
                rows.append(
                    {
                        "task_id": c["task_id"],
                        "parent": c["parent"],
                        "condition": c["condition"],
                        "response": response,
                        "grade": grade(response, c),
                        "possible_token_cap": len(
                            policy.tokenizer.encode(response, add_special_tokens=False)
                        )
                        >= 254,
                    }
                )
            (out / "responses.json").write_text(json.dumps(rows, indent=2))
            state["completed_cases"] = len(rows)
            persist()
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["condition"]].append(row)
        for key, values in buckets.items():
            parsed = [v["grade"] for v in values if v["grade"]["parseable"]]
            state["summaries"][key] = {
                "cases": len(values),
                "exact": sum(v["grade"]["exact"] for v in values),
                "parseable": len(parsed),
                "token_caps": sum(v["possible_token_cap"] for v in values),
                "mean_precision_parseable": sum(v["precision"] for v in parsed)
                / len(parsed)
                if parsed
                else None,
                "mean_recall_parseable": sum(v["recall"] for v in parsed) / len(parsed)
                if parsed
                else None,
            }
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
