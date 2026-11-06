"""Offline TRAIN teacher for complete multi-hole recovery from public state.

This module consumes private TRAIN labels to generate supervision. It must never
be called by a deployed policy or used as part of benchmark action selection.
"""

import math


def repair_actions(task, initial_state, current_state):
    if task.get("split") != "train" or task.get("mode") != "edit":
        raise ValueError("Recovery supervision requires TRAIN edit tasks")
    initial = {o["name"]: o for o in initial_state["objects"] if o.get("circles")}
    current = {o["name"]: o for o in current_state["objects"] if o.get("circles")}
    desired = {
        (name, f"diameter_{c['index']}"): 2 * c["radius"]
        for name, sketch in initial.items()
        for c in sketch["circles"]
    }
    selected = set()
    for action in task["oracle_actions"]:
        if action["op"] != "set_parameter":
            continue
        key = (action["target"], action["parameter"])
        if key not in desired or key in selected:
            raise ValueError("Ambiguous or unknown target dimension")
        selected.add(key)
        desired[key] = action["value"]
    if not selected:
        raise ValueError("No editable teacher targets")
    actual = {
        (name, f"diameter_{c['index']}"): 2 * c["radius"]
        for name, sketch in current.items()
        for c in sketch["circles"]
    }
    if set(actual) != set(desired):
        raise ValueError(
            "Feature topology changed; parameter-only teacher cannot repair it"
        )
    for value in [*desired.values(), *actual.values()]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid diameter")
    # Restore unrequested edits first, then complete requested edits. This order
    # is deterministic, but is not claimed to be the only valid action order.
    keys = sorted(
        desired, key=lambda k: (k in selected, k[0], int(k[1].split("_")[-1]))
    )
    actions = [
        {"op": "set_parameter", "target": k[0], "parameter": k[1], "value": desired[k]}
        for k in keys
        if not math.isclose(actual[k], desired[k], rel_tol=0, abs_tol=1e-6)
    ]
    return actions + [{"op": "finish"}]
