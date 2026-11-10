"""Process entry point; public state never includes hidden reference geometry."""

import json
import sys
import traceback
from pathlib import Path

from .schema import validate


def main():
    path = Path(sys.argv[1])
    request = json.loads(path.read_text())
    result = {"execution_ok": False, "reward": 0.0, "contract": "freecad.v2"}
    try:
        from . import native

        actions = validate(request["actions"], terminal=False)
        shape, objects, state, targets = native.build(actions)
        result.update(
            state=state,
            execution_ok=True,
            runtime={"freecad": ".".join(native.App.Version()[:3])},
        )
        result["artifact"] = native.save(path.parent, actions, objects, state, targets)
        if shape is not None:
            result["geometry"] = native.metrics(shape)
        validation = {"task_success": False}
        if request["reference"] is not None and shape is not None:
            target, _, _, _ = native.build(validate(request["reference"]))
            tm = native.metrics(target)
            delta = native.difference(shape, target) / max(tm["volume"], 1.0)
            bounds = max(
                abs(a - b)
                for a, b in zip(native.metrics(shape)["bounds"], tm["bounds"])
            )
            validation.update(
                relative_symmetric_difference=delta,
                bounds_error=bounds,
                task_success=actions[-1]["op"] == "finish"
                and delta <= 1e-5
                and bounds <= 1e-5
                and len(shape.Solids) == len(target.Solids),
            )
        expected = request.get("expected_volume")
        if expected is not None:
            validation["analytic_matches"] = (
                shape is not None
                and abs(shape.Volume - expected) / max(expected, 1) <= 1e-5
            )
            validation["task_success"] &= validation["analytic_matches"]
        result["validation"] = validation
        result["reward"] = float(validation["task_success"])
        if shape is not None:
            shape.exportStep(str(path.parent / "final.step"))
        # Tessellation changes OCCT's cached bounds on curved shapes. Score
        # exact geometry before generating display triangulation.
        if shape is not None:
            (path.parent / "mesh.json").write_text(json.dumps(native.mesh(shape)))
    except Exception as exc:  # noqa: BLE001 - native failures must be structured
        result.update(
            execution_ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    (path.parent / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
