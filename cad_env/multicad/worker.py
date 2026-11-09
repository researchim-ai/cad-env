"""Trusted worker entrypoint; input is validated JSON, never arbitrary Python."""

import json
import sys
import traceback
from pathlib import Path

from .plan import resolve


def main():
    path = Path(sys.argv[1])
    out = path.parent
    result = {"execution_ok": False, "reward": 0.0}
    try:
        request = json.loads(path.read_text())
        if request["engine"] == "cadquery":
            from . import cq_engine as engine
        else:
            raise ValueError("Invalid worker engine")
        candidate = engine.build(resolve(request["actions"]))
        m = engine.metrics(candidate)
        if not m["valid"] or m["volume"] <= 0 or m["solids"] != 1:
            raise ValueError("Candidate is not one valid nonempty solid")
        validation = {
            "task_success": False,
            "geometry_matches": False,
            "finished": bool(
                request["actions"] and request["actions"][-1]["op"] == "finish"
            ),
        }
        tolerance = request["tolerance"]
        reference = request["reference_actions"]
        if reference is not None:
            target = engine.build(resolve(reference))
            target_metrics = engine.metrics(target)
            delta = engine.difference(candidate, target)
            scale = max(target_metrics["volume"], 1.0)
            bounds_error = max(
                (abs(a - b) for a, b in zip(m["bounds"], target_metrics["bounds"]))
            )
            validation.update(
                symmetric_difference_volume=delta,
                relative_symmetric_difference=delta / scale,
                bounds_error=bounds_error,
                geometry_matches=delta / scale <= tolerance
                and bounds_error <= tolerance,
            )
            union = (m["volume"] + target_metrics["volume"] + delta) / 2
            validation["native_volume_iou"] = (
                max(0.0, (m["volume"] + target_metrics["volume"] - delta) / 2) / union
            )
        analytic = request.get("expected_metrics")
        if analytic:
            analytic_tolerance = tolerance
            analytic_ok = (
                abs(m["volume"] - analytic["volume"]) / max(analytic["volume"], 1.0)
                <= analytic_tolerance
            )
            if "bounds" in analytic:
                analytic_ok &= (
                    max((abs(a - b) for a, b in zip(m["bounds"], analytic["bounds"])))
                    <= tolerance
                )
            validation.update(
                analytic_matches=bool(analytic_ok),
                analytic_tolerance=analytic_tolerance,
            )
        validation["task_success"] = bool(
            reference is not None
            and validation["geometry_matches"]
            and validation["finished"]
            and validation.get("analytic_matches", True)
        )
        if request["mesh"]:
            (out / "mesh.json").write_text(json.dumps(engine.mesh(candidate)))
        events = []
        before = {"objects": [], "geometry": None}
        if request.get("record"):
            directory = out / "checkpoints"
            directory.mkdir()
            last_features, last_shape = ([], None)
            for i, action in enumerate(request["actions"]):
                features, objects = resolve(
                    request["actions"][: i + 1], allow_empty=True, return_objects=True
                )
                if features == last_features:
                    shape = last_shape
                elif (
                    hasattr(engine, "extend")
                    and len(features) == len(last_features) + 1
                    and (features[:-1] == last_features)
                ):
                    shape = engine.extend(last_shape, features[-1])
                else:
                    shape = engine.build(features) if features else None
                last_features, last_shape = (features, shape)
                after = {
                    "objects": objects,
                    "geometry": engine.metrics(shape) if shape is not None else None,
                }
                mesh_path = f"checkpoints/{i:03d}-mesh.json"
                (out / mesh_path).write_text(
                    json.dumps(
                        engine.mesh(shape)
                        if shape is not None
                        else {"vertices": [], "faces": []}
                    )
                )
                events.append(
                    {
                        "step_id": i,
                        "action": action,
                        "observation_before": before,
                        "observation_after": after,
                        "checkpoint_mesh": mesh_path,
                    }
                )
                before = after
        artifact = engine.save(candidate, out, request["actions"])
        result.update(
            execution_ok=True,
            geometry=m,
            validation=validation,
            reward=float(validation["task_success"]),
            artifact=artifact,
            events=events,
            observation=before if events else {"geometry": m},
            runtime={"engine": request["engine"], "version": engine.VERSION},
            parametric_contract="action-source replay; no native constrained-sketch guarantee",
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
