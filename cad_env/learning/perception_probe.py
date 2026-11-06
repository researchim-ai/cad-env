"""Bounded no-training probes isolating counting, grounding and drawing OCR.

These are component diagnostics on validation, not native CAD success scores.
Only messages/images are passed to the model; expected values stay in grading.
"""

import argparse
import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path


def messages(question, images):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Answer the visual question. Return only the requested JSON object. Do not issue CAD actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}]
            + [{"type": "image"} for _ in images],
        },
    ]


def cases(output, one_per_parent=False):
    from .perception_data import public_initial
    from .perception_plan import PublicCropper, public_payload

    out = Path(output)
    cropper = PublicCropper(out / "public-crops")
    result = []
    seen_parents = set()
    roots = {
        "visual": Path("artifacts/hard-unlabelled-v1"),
        "construction": Path("artifacts/hard-construction-v2"),
    }
    for line in (roots["visual"] / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"] or entry["split"] != "validation":
            continue
        root = roots["visual"]
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        if " inches" in task["instruction"]:
            continue  # one public unit variant per identical pixel image
        if one_per_parent and entry["split_group"] in seen_parents:
            continue
        seen_parents.add(entry["split_group"])
        images = [str((root / "episodes" / entry["task_id"] / "goal.png").resolve())]
        common = {"task_id": entry["task_id"], "parent": entry["split_group"]}
        question = 'Count every COLORED crosshair marker visible in the image. Count markers, not black hole outlines. Return {"count": integer}.'
        result.append(
            {
                **common,
                "kind": "count",
                "messages": messages(question, images),
                "images": images,
                "expected": len(task["goal_spec"]["selected"]),
            }
        )
        state = public_payload(public_initial(root, task))["state"]
        sketch = next(o for o in state["objects"] if o.get("circles"))
        public = [
            {k: c[k] for k in ["index", "x", "y", "radius"]} for c in sketch["circles"]
        ]
        question = (
            'Match ALL colored crosshairs in the image to the CURRENT sketch circle indices below. Use the drawn X/Y arrows for rotation. Include no unmarked circle and omit no marked circle. Return {"indices": [integers sorted ascending]}. Current public circles: '
            + json.dumps(public)
        )
        result.append(
            {
                **common,
                "kind": "selection",
                "messages": messages(question, images),
                "images": images,
                "expected": sorted(task["goal_spec"]["selected"]),
            }
        )
    for line in (roots["construction"] / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if (
            not entry["accepted"]
            or entry["split"] != "validation"
            or entry["family"] != "counterbored_plate"
        ):
            continue
        root = roots["construction"]
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        if task["input_modality"] != "image":
            continue
        paths = [str((root / "episodes" / entry["task_id"] / "goal.png").resolve())]
        expected = [
            [h["x"], h["y"], 2 * h["radius"], h["depth"]]
            for h in task["parameters"]["holes"]
        ]
        question = 'Read EVERY row of the visible HOLES table as [X,Y,DIAMETER,DEPTH]. Preserve separate rows that share XY but have a different diameter or depth. Include all rows, without generating CAD code. Return {"holes": [[x,y,diameter,depth],...]}. All values are mm.'
        for kind, images in [
            ("hole_table", paths),
            ("hole_table_zoom", cropper.images(public_initial(root, task), paths)),
        ]:
            result.append(
                {
                    "task_id": entry["task_id"],
                    "parent": entry["split_group"],
                    "kind": kind,
                    "messages": messages(question, images),
                    "images": images,
                    "expected": expected,
                }
            )
    return result


def grade(raw, case):
    strict = True
    try:
        value = json.loads(raw)
    except ValueError:
        strict = False
        candidates = []
        key = (
            "count"
            if case["kind"] == "count"
            else "indices"
            if case["kind"] == "selection"
            else "holes"
        )
        for position, char in enumerate(raw):
            if char != "{":
                continue
            try:
                candidate, _ = json.JSONDecoder().raw_decode(raw[position:])
            except ValueError:
                continue
            if isinstance(candidate, dict) and key in candidate:
                candidates.append(candidate)
        if not candidates:
            return {"parseable": False, "strict_json": False, "exact": False}
        value = candidates[-1]
    kind = case["kind"]
    expected = case["expected"]
    result = {"parseable": True, "strict_json": strict, "exact": False}
    if not isinstance(value, dict):
        return {**result, "parseable": False}
    if kind == "count":
        count = value.get("count")
        if type(count) is not int:
            return {**result, "parseable": False}
        return {
            **result,
            "exact": count == expected,
            "absolute_error": abs(count - expected),
        }
    if kind == "selection":
        indices = value.get("indices")
        if (
            not isinstance(indices, list)
            or any(type(i) is not int for i in indices)
            or len(set(indices)) != len(indices)
        ):
            return {**result, "parseable": False}
        a, b = set(expected), set(indices)
        tp = len(a & b)
        return {
            **result,
            "exact": a == b,
            "precision": tp / len(b) if b else 0.0,
            "recall": tp / len(a),
            "predicted_count": len(b),
            "target_count": len(a),
        }
    holes = value.get("holes")
    if not isinstance(holes, list) or any(
        not isinstance(h, list)
        or len(h) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in h
        )
        for h in holes
    ):
        return {**result, "parseable": False}
    a, b = (
        Counter(tuple(round(v, 2) for v in h) for h in expected),
        Counter(tuple(round(v, 2) for v in h) for h in holes),
    )
    tp = sum((a & b).values())
    return {
        **result,
        "exact": a == b,
        "row_precision": tp / len(holes) if holes else 0.0,
        "row_recall": tp / len(expected),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--one-per-parent", action="store_true")
    p.add_argument("--raw-model", default="artifacts/models/qwen35-4b")
    p.add_argument("--trained-model", default="artifacts/hard-experiment-v1/merged")
    a = p.parse_args()
    if not 0 < a.minutes <= 20:
        p.error("At most twenty diagnostic minutes")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    dataset = cases(out, a.one_per_parent)
    (out / "cases-private-grading.json").write_text(json.dumps(dataset, indent=2))
    models = {
        "raw": a.raw_model,
        "trained": a.trained_model,
    }
    state = {
        "status": "starting",
        "optimizer_updates": 0,
        "scope": "Diagnostic component VQA on validation, not native CAD success; greedy non-thinking fixed across both models",
        "results": {},
        "case_counts": dict(Counter(c["kind"] for c in dataset)),
    }

    def persist():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    persist()
    deadline = time.monotonic() + a.minutes * 60
    import torch

    from .batch_policy import BatchedHFPolicy

    try:
        for name, model in models.items():
            state["status"] = name
            persist()
            policy = BatchedHFPolicy(
                model,
                temperature=0,
                max_new_tokens=1024,
                max_context=8192,
                max_pixels=327680,
                chat_template_kwargs={"enable_thinking": False},
            )
            records = []
            for i in range(0, len(dataset), 4):
                if time.monotonic() > deadline:
                    raise TimeoutError("Diagnostic budget exhausted")
                batch = dataset[i : i + 4]
                answers = policy.generate_batch(
                    [(c["messages"], c["images"]) for c in batch]
                )
                for c, raw in zip(batch, answers):
                    records.append(
                        {
                            "task_id": c["task_id"],
                            "parent": c["parent"],
                            "kind": c["kind"],
                            "response": raw,
                            "possible_token_cap": len(
                                policy.tokenizer.encode(raw, add_special_tokens=False)
                            )
                            >= 1022,
                            "grade": grade(raw, c),
                        }
                    )
                (out / (name + "-responses.json")).write_text(
                    json.dumps(records, indent=2)
                )
                state["completed_cases"] = len(records)
                persist()
            buckets = defaultdict(list)
            for row in records:
                buckets[row["kind"]].append(row["grade"])
            summary = {}
            for kind, rows in buckets.items():
                summary[kind] = {
                    "cases": len(rows),
                    "exact": sum(r["exact"] for r in rows),
                    "parseable": sum(r["parseable"] for r in rows),
                    "strict_json": sum(r["strict_json"] for r in rows),
                }
                for field in [
                    "precision",
                    "recall",
                    "row_precision",
                    "row_recall",
                    "absolute_error",
                ]:
                    values = [r[field] for r in rows if field in r]
                    if values:
                        summary[kind][field + "_mean_parseable"] = sum(values) / len(
                            values
                        )
            state["results"][name] = {"model": policy.identity, "summary": summary}
            persist()
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        state["status"] = "complete"
        persist()
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        persist()
        raise


if __name__ == "__main__":
    main()
