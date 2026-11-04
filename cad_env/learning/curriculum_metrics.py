"""Diagnostic slices and counterfactual consistency beyond overall CAD success."""

import json
from collections import Counter, defaultdict
from pathlib import Path


def compare_modalities(dataset, baseline, candidate):
    """Prespecified modality slices derived only from two complete matched runs."""
    import hashlib

    from .metrics import paired_comparison
    from .protocol import digest

    root = Path(dataset).resolve()
    paths = [Path(baseline), Path(candidate)]
    reports = [json.loads(p.read_text()) for p in paths]
    if (
        digest(json.loads((root / "summary.json").read_text()))
        != reports[0]["dataset_digest"]
    ):
        raise ValueError("Modality metadata comes from another corpus")
    overall = paired_comparison(*reports)
    groups = defaultdict(set)
    for task_id in reports[0]["task_ids"]:
        task = json.loads((root / "episodes" / task_id / "task.json").read_text())
        groups[task.get("input_modality", "visual")].add(task_id)
    slices = {}
    for modality, identifiers in groups.items():
        sliced = [
            {
                **r,
                "task_ids": [i for i in r["task_ids"] if i in identifiers],
                "episodes": [e for e in r["episodes"] if e["task_id"] in identifiers],
            }
            for r in reports
        ]
        slices[modality] = paired_comparison(*sliced)
    return {
        "overall": overall,
        "modalities": slices,
        "source_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        "note": "Modality analyses of complete runs; no failed cases removed and no separate debug run promoted",
    }


def summarize_curriculum(dataset, evaluation):
    root = Path(dataset).resolve()
    path = Path(evaluation).resolve()
    report = json.loads(path.read_text())
    if report["status"] != "complete":
        raise ValueError("Only complete evaluations may be summarized")
    slices = defaultdict(list)
    groups = defaultdict(list)
    first_valid = []
    first_grounded = []
    first_fields = defaultdict(list)
    first_parameters = defaultdict(int)
    edit_count_correct = []
    distinct_count_correct = []
    primitive_flags = {}
    step_counts = []
    for episode in report["episodes"]:
        task = json.loads(
            (root / "episodes" / episode["task_id"] / "task.json").read_text()
        )
        for label in [
            f"tier:{task.get('evaluation_tier', 'iid')}",
            f"input:{task.get('input_modality', 'visual')}",
            f"family:{task['family']}",
            f"oracle_steps:{len(task['oracle_actions'])}",
            f"input_tier:{task.get('input_modality', 'visual')}:{task.get('evaluation_tier', 'iid')}",
            f"family_input:{task['family']}:{task.get('input_modality', 'visual')}",
        ]:
            slices[label].append(episode)
        groups[task.get("condition_group", task["split_group"])].append(episode)
        rows = [
            json.loads(l)
            for l in (path.parent / episode["trajectory"]).read_text().splitlines()
        ]
        first = rows[0]
        first_action = first["action"] if isinstance(first["action"], dict) else {}
        first_valid.append(bool(first["info"]["execution_ok"]))
        if task["mode"] == "create":

            def primitive(action):
                if not isinstance(action, dict):
                    return None
                if action.get("op") == "sketch_rectangle":
                    return ("rectangle", action.get("length"), action.get("width"))
                if action.get("op") == "sketch_circles":
                    return (
                        "circles",
                        json.dumps(action.get("circles"), sort_keys=True),
                    )
                return None

            expected = next(
                primitive(a)
                for a in task["reference_actions"]
                if primitive(a) is not None
            )
            actual = next(
                (
                    primitive(r["action"])
                    for r in rows
                    if primitive(r["action"]) is not None
                ),
                None,
            )
            # Compare numeric circle values independently of JSON int/float spelling.
            if expected and actual and expected[0] == actual[0] == "circles":
                correct = json.loads(expected[1]) == json.loads(actual[1])
            else:
                correct = expected == actual
            primitive_flags[episode["trajectory"]] = correct
        if task["family"] == "visual_grounding":
            acceptable = [
                a for a in task["oracle_actions"] if a["op"] == "set_parameter"
            ]
            first_grounded.append(first_action in acceptable)
            for field in ("op", "target", "parameter", "value"):
                first_fields[field].append(
                    any(first_action.get(field) == a[field] for a in acceptable)
                )
            first_parameters[str(first_action.get("parameter"))] += 1
            edits = [
                r["action"]
                for r in rows
                if r["info"].get("execution_ok")
                and isinstance(r["action"], dict)
                and r["action"].get("op") == "set_parameter"
            ]
            edit_count_correct.append(len(edits) == len(acceptable))
            distinct_count_correct.append(
                len({(a.get("target"), a.get("parameter")) for a in edits})
                == len(acceptable)
            )
        step_counts.append(len(task["oracle_actions"]))

    def metrics(rows):
        primitive = [
            primitive_flags[r["trajectory"]]
            for r in rows
            if r["trajectory"] in primitive_flags
        ]
        return {
            "tasks": len(rows),
            "success": sum(r["task_success"] for r in rows) / len(rows),
            "reward": sum(r["reward"] for r in rows) / len(rows),
            "initial_primitive_parameter_accuracy": sum(primitive) / len(primitive)
            if primitive
            else None,
        }

    return {
        "status": "complete",
        "slices": {k: metrics(v) for k, v in sorted(slices.items())},
        "condition_groups": len(groups),
        "all_variants_success_rate": sum(
            all(e["task_success"] for e in g) for g in groups.values()
        )
        / len(groups),
        "first_action_execution_validity": sum(first_valid) / len(first_valid),
        "initial_primitive_parameter_accuracy": sum(primitive_flags.values())
        / len(primitive_flags)
        if primitive_flags
        else None,
        "first_action_grounding_accuracy": sum(first_grounded) / len(first_grounded)
        if first_grounded
        else None,
        "first_action_field_accuracy": {
            k: sum(values) / len(values) for k, values in first_fields.items()
        },
        "first_action_parameter_counts": dict(first_parameters),
        "edit_action_count_accuracy": sum(edit_count_correct) / len(edit_count_correct)
        if edit_count_correct
        else None,
        "distinct_edit_count_accuracy": sum(distinct_count_correct)
        / len(distinct_count_correct)
        if distinct_count_correct
        else None,
        "oracle_step_range": [min(step_counts), max(step_counts)],
        "note": "First-action grounding is a diagnostic, not end-to-end CAD success. Counterfactual all-variant success uses every member, including failures.",
    }


def audit_complex(root, *, base_report=None):
    import hashlib

    from .audit import audit_dataset

    root = Path(root).resolve()
    audit = audit_dataset(root) if base_report is None else base_report
    errors = list(audit["errors"])
    geometry_splits = defaultdict(set)
    pairs = defaultdict(list)
    train_signatures = set()
    ood = []
    split_groups = defaultdict(set)
    image_inputs = defaultdict(lambda: defaultdict(list))
    steps = []
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"]:
            errors.append("Rejected complex oracle: " + entry["task_id"])
            continue
        t = json.loads((root / "episodes" / entry["task_id"] / "task.json").read_text())
        p = t["parameters"]
        for key in ["holes", "bosses"]:
            if key in p:
                p[key] = sorted(p[key], key=lambda x: json.dumps(x, sort_keys=True))
        geometry = hashlib.sha256(
            json.dumps([t["family"], p], sort_keys=True).encode()
        ).hexdigest()
        geometry_splits[geometry].add(t["split"])
        pairs[t["split_group"]].append(t)
        signature = json.dumps(t["construction_signature"])
        if t["split"] == "train":
            train_signatures.add(signature)
        if t["evaluation_tier"] != "iid":
            ood.append((t["task_id"], signature))
        split_groups[t["split"]].add(t["split_group"])
        steps.append(len(t["oracle_actions"]))
        if t["input_modality"] == "image" and any(
            c.isdigit() for c in t["public_task"]["instruction"]
        ):
            errors.append("Numeric target leaked into image-only instruction")
        if t["input_modality"] == "image":
            from cad_env.agent.observations import agent_messages

            native = json.loads(
                (root / "episodes" / t["task_id"] / "oracle/result.json").read_text()
            )
            public = agent_messages(
                t,
                {
                    "state": native["initial_observation"],
                    "remaining_steps": t["max_steps"],
                    "goal_images": ["image-placeholder"],
                    "images": [],
                },
                visual=True,
            )
            key = hashlib.sha256(
                json.dumps(public, sort_keys=True).encode()
            ).hexdigest()
            image_inputs[t["split"]][key].append(t)
    if any(len(s) > 1 for s in geometry_splits.values()):
        errors.append("Geometry reordered across splits")
    for task_id, signature in ood:
        if signature in train_signatures:
            errors.append("OOD construction graph seen in train: " + task_id)
    for group, rows in pairs.items():
        if len({r["split"] for r in rows}) != 1:
            errors.append("Modality pair crosses split: " + group)
    blind_ceilings = {}
    for split, aliases in image_inputs.items():
        upper, total = 0, 0
        for tasks in aliases.values():
            # A candidate cannot satisfy two targets with disjoint permitted
            # volume or bounds intervals. Connected components give a safe
            # upper bound even when aggregate metrics cannot distinguish parts.
            parent = list(range(len(tasks)))

            def find(i, parent=parent):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i, a in enumerate(tasks):
                for j, b in enumerate(tasks[:i]):
                    x, y = a["analytic_metrics"], b["analytic_metrics"]
                    ta, tb = a["tolerance"], b["tolerance"]
                    separated = any(
                        abs(u - v) > ta + tb for u, v in zip(x["bounds"], y["bounds"])
                    )
                    separated |= max(
                        x["volume"] * (1 - ta), y["volume"] * (1 - tb)
                    ) > min(x["volume"] * (1 + ta), y["volume"] * (1 + tb))
                    if not separated:
                        parent[find(i)] = find(j)

            upper += max(Counter(find(i) for i in range(len(tasks))).values())
            total += len(tasks)
        blind_ceilings[split] = {
            "image_tasks": total,
            "distinct_nonvisual_inputs": len(aliases),
            "max_blind_successes": upper,
            "greedy_no_image_success_ceiling": upper / total,
        }
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": audit,
        "groups_by_split": {k: len(v) for k, v in split_groups.items()},
        "training_construction_graphs": len(train_signatures),
        "unseen_ood_graph_cases": len(ood),
        "oracle_step_range": [min(steps), max(steps)],
        "image_and_text_geometry_pairs_grouped": bool(not errors),
        "image_information_control": blind_ceilings,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--evaluation")
    p.add_argument("--baseline")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.baseline:
        if not a.evaluation:
            p.error("--baseline requires --evaluation")
        result = compare_modalities(a.dataset, a.baseline, a.evaluation)
    else:
        result = (
            summarize_curriculum(a.dataset, a.evaluation)
            if a.evaluation
            else audit_complex(a.dataset)
        )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
