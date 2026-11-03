"""Adversarial verifier calibration, independent of model quality or policy samples."""

import copy
import json
from pathlib import Path

from cad_env.agent import FreeCADBackend, make_task

from .rewards import score_completion


def verifier_calibration(output_dir, backend=None):
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new calibration directory")
    root.mkdir(parents=True, exist_ok=True)
    backend = backend or FreeCADBackend()
    task = make_task("plate", 11, "create")
    cases = []

    def add(name, actions, expected, case_task=task):
        cases.append((name, case_task, actions, expected))

    add("correct_plate", copy.deepcopy(task["oracle_actions"]), True)
    shifted = copy.deepcopy(task["oracle_actions"])
    shifted[3]["circles"][0]["x"] += 2
    add("same_volume_wrong_hole_position", shifted, False)
    blind = copy.deepcopy(task["oracle_actions"])
    blind[4]["length"] /= 2
    add("blind_instead_of_through_hole", blind, False)
    radius = copy.deepcopy(task["oracle_actions"])
    radius[3]["circles"][0]["radius"] *= 1.3
    add("wrong_hole_radius", radius, False)
    extra = copy.deepcopy(task["oracle_actions"])
    extra.insert(-1, {"op": "create_body", "name": "ExtraBody"})
    add("extra_body", extra, False)
    add("empty_finish", [{"op": "finish"}], False)
    add("invalid_json", None, False)
    edit = make_task("plate", 11, "edit")
    overshoot = [
        edit["oracle_actions"][0],
        {
            "op": "set_parameter",
            "target": "HolePocket",
            "parameter": "length",
            "value": 100,
        },
        {"op": "finish"},
    ]
    add("equal_shape_but_protected_parameter_changed", overshoot, False, edit)
    ring = make_task("bushing", 11, "create")
    p = ring["parameters"]
    alternative = [
        {"op": "create_body", "name": "RingBody"},
        {
            "op": "sketch_circles",
            "name": "Annulus",
            "body": "RingBody",
            "circles": [
                {"x": 0, "y": 0, "radius": p["radius"]},
                {"x": 0, "y": 0, "radius": p["hole_radius"]},
            ],
        },
        {
            "op": "pad",
            "name": "RingPad",
            "body": "RingBody",
            "profile": "Annulus",
            "length": p["height"],
        },
        {"op": "finish"},
    ]
    add("different_correct_construction", alternative, True, ring)
    results = []
    for name, t, actions, expected in cases:
        score = score_completion(
            t,
            json.dumps(actions) if actions is not None else "invalid JSON",
            root / name,
            backend,
        )
        results.append(
            {
                "case": name,
                "expected_success": expected,
                "observed_success": score["task_success"],
                "passed": score["task_success"] == expected,
                "reward": score["reward"],
                "validation": score.get("validation"),
            }
        )
    report = {
        "passed": all(r["passed"] for r in results),
        "cases": results,
        "scope": "Verifier calibration, not a learned-policy benchmark or GRPO readiness evidence.",
    }
    (root / "calibration.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = verifier_calibration(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
