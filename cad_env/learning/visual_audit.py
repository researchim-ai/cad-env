"""Information-control audit: pixels must carry the counterfactual edit choice."""

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from cad_env.agent.observations import agent_messages

from .audit import audit_dataset
from .visual_tasks import canonical_digest


def audit_visual(root, *, base_report=None):
    from PIL import Image

    root = Path(root).resolve()
    base = audit_dataset(root) if base_report is None else base_report
    errors = list(base["errors"])
    manifest = [
        json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()
    ]
    groups = defaultdict(list)
    geometry_splits = defaultdict(set)
    tiers = Counter()
    counts = Counter()
    for entry in manifest:
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        if not entry["accepted"]:
            errors.append("Rejected visual oracle: " + entry["task_id"])
            continue
        spec = task["goal_spec"]
        requested = re.search(r"to diameter ([0-9.]+) (mm|inches)", task["instruction"])
        edits = [a for a in task["oracle_actions"] if a["op"] == "set_parameter"]
        initial = json.loads(
            (root / "episodes" / entry["task_id"] / "oracle/result.json").read_text()
        )["initial_observation"]
        expected_reference = copy.deepcopy(
            task.get("goal_base_actions", task["initial_actions"])
        )
        try:
            diameter = float(requested[1]) * (25.4 if requested[2] == "inches" else 1)
            sketch = expected_reference[3]
            current = next(
                o["circles"] for o in initial["objects"] if o["name"] == sketch["name"]
            )
            current = {h["index"]: h["radius"] for h in current}
            desired = {
                i: diameter if i in spec["selected"] else 2 * h["radius"]
                for i, h in enumerate(spec["circles"])
            }
            needed = {
                i
                for i, d in desired.items()
                if not math.isclose(2 * current[i], d, abs_tol=1e-9)
            }
            indices = [int(a["parameter"].removeprefix("diameter_")) for a in edits]
            if (
                set(indices) != needed
                or len(indices) != len(needed)
                or any(a["target"] != sketch["name"] for a in edits)
                or any(
                    not math.isclose(a["value"], desired[i], abs_tol=1e-9)
                    for a, i in zip(edits, indices)
                )
                or diameter <= max(2 * h["radius"] for h in spec["circles"])
            ):
                raise ValueError("Edit selection or requested diameter differs")
            for index, value in desired.items():
                sketch["circles"][index]["radius"] = value / 2
            if task["reference_actions"] != expected_reference + [{"op": "finish"}]:
                raise ValueError("Native reference differs from the visible request")
        except (ValueError, TypeError, KeyError, IndexError, StopIteration):
            errors.append("Teacher/request mismatch: " + task["task_id"])
        image_path = root / task["goal_images"][0]
        obs = {
            "state": initial,
            "remaining_steps": task["max_steps"],
            "goal_images": [str(image_path)],
            "images": [],
        }
        messages = agent_messages(task, obs, visual=True)
        serialized = json.dumps(messages, sort_keys=True)
        if any(
            k in serialized
            for k in [
                '"goal_spec"',
                '"oracle_actions"',
                '"reference_actions"',
                '"goal_base_actions"',
                '"repair_provenance"',
                task["task_id"],
            ]
        ):
            errors.append("Private target/ID leaked: " + task["task_id"])
        groups[task["condition_group"]].append(
            {
                "task": task,
                "text": canonical_digest(messages),
                "initial": canonical_digest(initial),
                "goal": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "oracle": canonical_digest(task["oracle_actions"]),
                "reference": canonical_digest(task["reference_actions"]),
            }
        )
        geometry_splits[
            canonical_digest(
                [
                    spec["length"],
                    spec["width"],
                    spec["height"],
                    sorted((c["x"], c["y"], c["radius"]) for c in spec["circles"]),
                ]
            )
        ].add(task["split"])
        tiers[task["evaluation_tier"]] += 1
        counts[task["split"]] += 1
        # Verify marker pixels against CAD coordinates independently of task labels.
        im = Image.open(image_path).convert("RGB")
        angle = math.radians(spec["angle"])
        c, s = math.cos(angle), math.sin(angle)
        factor = 300 / max(
            abs(c) * spec["length"] + abs(s) * spec["width"],
            abs(s) * spec["length"] + abs(c) * spec["width"],
        )
        color = {"red": (200, 35, 40), "blue": (25, 85, 190), "purple": (140, 35, 160)}[
            spec["style"]
        ]
        for i, hole in enumerate(spec["circles"]):
            x, y = hole["x"] - spec["length"] / 2, hole["y"] - spec["width"] / 2
            px, py = (
                round(256 + factor * (c * x - s * y)),
                round(264 - factor * (s * x + c * y)),
            )
            marked = any(
                sum(
                    (im.getpixel((px + dx, py + dy))[k] - color[k]) ** 2
                    for k in range(3)
                )
                < 50**2
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )
            if marked != (i in spec["selected"]):
                errors.append("Marker/teacher mismatch: " + task["task_id"])
    for group, rows in groups.items():
        if len(rows) != 4:
            errors.append("Incomplete counterfactual group: " + group)
        if (
            len({r["text"] for r in rows}) != 1
            or len({r["initial"] for r in rows}) != 1
        ):
            errors.append("Nonvisual counterfactual inputs differ: " + group)
        if (
            len({r["goal"] for r in rows}) != len(rows)
            or len({r["oracle"] for r in rows}) != len(rows)
            or len({r["reference"] for r in rows}) != len(rows)
        ):
            errors.append("Counterfactual goals/answers are not distinct: " + group)
        if len({r["task"]["split"] for r in rows}) != 1:
            errors.append("Counterfactual split leakage: " + group)
        if len({canonical_digest(r["task"]["analytic_metrics"]) for r in rows}) != 1:
            errors.append("Aggregate dimensions expose counterfactual choice: " + group)
    if any(len(s) != 1 for s in geometry_splits.values()):
        errors.append("Renamed/permuted geometry crosses splits")
    validation_groups = sum(
        rows[0]["task"]["split"] == "validation" for rows in groups.values()
    )
    identical_nonvisual = bool(groups) and all(
        len(rows) == 4
        and len({r["text"] for r in rows}) == 1
        and len({r["initial"] for r in rows}) == 1
        for rows in groups.values()
    )
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "condition_groups": len(groups),
        "counts": dict(counts),
        "tiers": dict(tiers),
        "validation_groups": validation_groups,
        "all_counterfactual_nonvisual_inputs_identical": identical_nonvisual,
        "greedy_no_image_success_ceiling": len(groups) / len(manifest)
        if manifest
        and identical_nonvisual
        and all(len({r["reference"] for r in rows}) == 4 for rows in groups.values())
        else None,
        "note": "Ceiling applies to deterministic policies on complete groups with identical nonvisual observations; it is not a generic random-guess baseline.",
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit_visual(a.dataset)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["ok"])
