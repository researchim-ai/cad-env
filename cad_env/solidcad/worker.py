import json
import sys
import traceback
from pathlib import Path

from .schema import validate


def main():
    path = Path(sys.argv[1])
    out = path.parent
    request = json.loads(path.read_text())
    result = {"execution_ok": False, "reward": 0.0}
    try:
        engine = request["engine"]
        if engine == "freecad":
            from . import freecad_engine as native
        elif engine == "cadquery":
            from . import cq_engine as native
        else:
            raise ValueError("Unknown solid engine")
        program = validate(request["program"])
        shape = native.build(program)
        m = native.metrics(shape)
        if not m["valid"] or m["volume"] <= 0 or m["solids"] < 1:
            raise ValueError("Invalid final solid graph")
        validation = {"task_success": False}
        if request["reference"] is not None:
            target = native.build(validate(request["reference"]))
            target_metrics = native.metrics(target)
            delta = native.difference(shape, target)
            relative = delta / max(target_metrics["volume"], 1.0)
            bounds = max(
                (abs(a - b) for a, b in zip(m["bounds"], target_metrics["bounds"]))
            )
            validation.update(
                relative_symmetric_difference=relative,
                bounds_error=bounds,
                task_success=relative <= request["tolerance"]
                and bounds <= request["tolerance"]
                and (m["solids"] == target_metrics["solids"]),
            )
        expected = request.get("expected_volume")
        if expected is not None:
            tolerance = request["tolerance"]
            validation["analytic_matches"] = (
                abs(m["volume"] - expected) / max(expected, 1.0) <= tolerance
            )
            validation["task_success"] &= validation["analytic_matches"]
        (out / "mesh.json").write_text(json.dumps(native.mesh(shape)))
        artifact = native.save(shape, out, program)
        result.update(
            execution_ok=True,
            geometry=m,
            validation=validation,
            reward=float(validation["task_success"]),
            artifact=artifact,
            runtime={"engine": engine, "version": native.VERSION},
            contract="solidcad.v1: source replay; no constrained-sketch or assembly-joint guarantee",
        )
    except Exception as exc:  # noqa: BLE001 - native failures become structured results
        result.update(
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
