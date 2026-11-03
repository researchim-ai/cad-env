"""Standalone trusted worker, run only by FreeCADCmd (no ML dependencies)."""

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import validate_program


def main(request_path, result_path):
    import FreeCAD as App
    import Part
    import Sketcher

    request = json.loads(Path(request_path).read_text())
    out = Path(result_path).parent
    result = {"execution_ok": False, "backend": "freecad", "events": []}
    stage = "execution"

    def dump(path, data):
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
        tmp.replace(path)

    def get(doc, name, kind=None):
        obj = doc.getObject(name)
        if obj is None or (kind and not obj.isDerivedFrom(kind)):
            raise ValueError("Missing object or incorrect type: " + name)
        return obj

    def snapshot(doc):
        objects = []
        for obj in doc.Objects:
            row = {
                "name": obj.Name,
                "type": obj.TypeId,
                "state": list(obj.State),
                "dependencies": sorted(o.Name for o in obj.OutList),
                "expressions": [
                    [str(k), str(v)] for k, v in getattr(obj, "ExpressionEngine", [])
                ],
            }
            if obj.isDerivedFrom("Sketcher::SketchObject"):
                row["fully_constrained"] = bool(obj.FullyConstrained)
                row["constraints"] = {
                    obj.Constraints[i].Name: obj.getDatum(i).Value
                    for i in range(obj.ConstraintCount)
                    if obj.Constraints[i].Name
                }
            elif obj.TypeId in ("PartDesign::Pad", "PartDesign::Pocket"):
                row["length"] = obj.Length.Value
            objects.append(row)
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
        geometry = None
        if len(bodies) == 1 and not bodies[0].Shape.isNull():
            s = bodies[0].Shape
            bb = s.BoundBox
            geometry = {
                "volume": s.Volume,
                "area": s.Area,
                "solids": len(s.Solids),
                "valid": bool(s.isValid()),
                "bounds": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
                "center": [s.CenterOfMass.x, s.CenterOfMass.y, s.CenterOfMass.z],
            }
        return {"objects": objects, "geometry": geometry, "body_count": len(bodies)}

    def apply(doc, action):
        op = action["op"]
        if op == "finish":
            return
        if "name" in action and doc.getObject(action["name"]):
            raise ValueError("Duplicate object name: " + action["name"])
        if op == "create_body":
            doc.addObject("PartDesign::Body", action["name"])
        elif op in ("sketch_rectangle", "sketch_circles"):
            body = get(doc, action["body"], "PartDesign::Body")
            sk = body.newObject("Sketcher::SketchObject", action["name"])
            if op == "sketch_rectangle":
                l, w = action["length"], action["width"]
                pts = [(0, 0), (l, 0), (l, w), (0, w)]
                for i in range(4):
                    p, q = pts[i], pts[(i + 1) % 4]
                    sk.addGeometry(
                        Part.LineSegment(App.Vector(*p, 0), App.Vector(*q, 0)), False
                    )
                for i in range(4):
                    sk.addConstraint(
                        Sketcher.Constraint("Coincident", i, 2, (i + 1) % 4, 1)
                    )
                for i in (0, 2):
                    sk.addConstraint(Sketcher.Constraint("Horizontal", i))
                for i in (1, 3):
                    sk.addConstraint(Sketcher.Constraint("Vertical", i))
                sk.addConstraint(Sketcher.Constraint("Coincident", 0, 1, -1, 1))
                for edge, value, name in ((0, l, "length"), (1, w, "width")):
                    idx = sk.addConstraint(Sketcher.Constraint("Distance", edge, value))
                    sk.renameConstraint(idx, name)
            else:
                if "support" in action:
                    support = get(doc, action["support"], "PartDesign::Pad")
                    if support.getParentGeoFeatureGroup() != body:
                        raise ValueError("Support must belong to the same body")
                    # Link the support plane to its pad and sketch, avoiding face indices.
                    sk.setExpression(
                        "Placement.Base.z",
                        support.Profile[0].Name
                        + ".Placement.Base.z + "
                        + support.Name
                        + ".Length",
                    )
                for circle in action["circles"]:
                    x, y, r = circle["x"], circle["y"], circle["radius"]
                    i = sk.addGeometry(
                        Part.Circle(App.Vector(x, y, 0), App.Vector(0, 0, 1), r), False
                    )
                    if x == 0 and y == 0:
                        sk.addConstraint(Sketcher.Constraint("Coincident", i, 3, -1, 1))
                    else:
                        if x == 0:
                            sk.addConstraint(
                                Sketcher.Constraint("DistanceX", i, 3, 0.0)
                            )
                        else:
                            sk.addConstraint(Sketcher.Constraint("DistanceX", i, 3, x))
                        if y == 0:
                            sk.addConstraint(
                                Sketcher.Constraint("DistanceY", i, 3, 0.0)
                            )
                        else:
                            sk.addConstraint(Sketcher.Constraint("DistanceY", i, 3, y))
                    idx = sk.addConstraint(Sketcher.Constraint("Diameter", i, r * 2))
                    sk.renameConstraint(idx, "diameter_" + str(i))
        elif op in ("pad", "pocket"):
            body = get(doc, action["body"], "PartDesign::Body")
            sk = get(doc, action["profile"], "Sketcher::SketchObject")
            if sk.getParentGeoFeatureGroup() != body:
                raise ValueError("Profile must belong to the same body")
            feature = body.newObject(
                "PartDesign::Pad" if op == "pad" else "PartDesign::Pocket",
                action["name"],
            )
            feature.Profile = sk
            feature.Length = action["length"]
        elif op == "set_parameter":
            obj = get(doc, action["target"])
            param = action["parameter"]
            if obj.isDerivedFrom("Sketcher::SketchObject"):
                names = [obj.Constraints[i].Name for i in range(obj.ConstraintCount)]
                if param not in names:
                    raise ValueError("No such named sketch dimension: " + param)
                obj.setDatum(
                    names.index(param), App.Units.Quantity(str(action["value"]) + " mm")
                )
            elif (
                obj.TypeId in ("PartDesign::Pad", "PartDesign::Pocket")
                and param == "length"
            ):
                obj.Length = action["value"]
            else:
                raise ValueError("Unsupported parameter for target")
        doc.recompute()
        errors = [o.Name for o in doc.Objects if "Invalid" in o.State]
        if errors:
            raise ValueError("Recompute failed for: " + ", ".join(errors))
        for obj in doc.Objects:
            if hasattr(obj, "Visibility"):
                obj.Visibility = obj.TypeId == "PartDesign::Body"
        for obj in doc.Objects:
            if obj.TypeId == "PartDesign::Body" and obj.Tip:
                obj.Tip.Visibility = True

    def build(name, actions):
        doc = App.newDocument(name)
        for action in validate_program(actions):
            apply(doc, action)
        return doc

    def shape(doc):
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
        if len(bodies) != 1 or bodies[0].Shape.isNull():
            raise ValueError("Expected exactly one non-empty PartDesign body")
        s = bodies[0].Shape
        if not s.isValid() or len(s.Solids) != 1 or s.Volume <= 0:
            raise ValueError("Expected one valid positive-volume solid")
        return s

    def capture_mesh(doc):
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
        solid = None
        if len(bodies) == 1 and not bodies[0].Shape.isNull() and bodies[0].Shape.Solids:
            solid = bodies[0].Shape
        if solid is None:
            for obj in reversed(doc.Objects):
                if (
                    obj.TypeId in ("PartDesign::Pad", "PartDesign::Pocket")
                    and not obj.Shape.isNull()
                    and obj.Shape.Solids
                ):
                    solid = obj.Shape
                    break
        vertices, faces = solid.tessellate(0.15) if solid is not None else ([], [])
        lines = []
        if doc.Objects and doc.Objects[-1].isDerivedFrom("Sketcher::SketchObject"):
            for edge in doc.Objects[-1].Shape.Edges:
                points = edge.discretize(Deflection=0.1)
                lines.append([[v.x, v.y, v.z] for v in points[:2048]])
        return {
            "vertices": [[v.x, v.y, v.z] for v in vertices],
            "faces": faces,
            "lines": lines,
        }

    def compare(a, b):
        # Both shapes use the original mm coordinates: no scale/pose normalization.
        missing, extra = b.cut(a).Volume, a.cut(b).Volume
        union = a.Volume + missing
        relative_error = (abs(missing) + abs(extra)) / max(union, 1e-12)
        return {
            "geometry_score": max(0.0, min(1.0, 1 - relative_error)),
            "relative_symmetric_difference": relative_error,
            "missing_volume": missing,
            "extra_volume": extra,
        }

    try:
        result["runtime"] = {
            "freecad": App.Version(),
            "occt": Part.OCC_VERSION,
            "python": sys.version,
            "worker_version": "1",
        }
        initial = validate_program(request.get("initial_actions", []))
        actions = validate_program(request.get("actions", []))
        doc = build("Candidate", initial)
        result["initial_observation"] = snapshot(doc)
        if initial:
            doc.saveAs(str(out / "input.FCStd"))
        if request.get("mesh"):
            dump(out / "initial_mesh.json", capture_mesh(doc))
        for i, action in enumerate(actions):
            before = snapshot(doc)
            step_started = time.monotonic()
            apply(doc, action)
            checkpoint = None
            if request.get("record"):
                checkpoint = f"checkpoints/{i:03d}.FCStd"
                (out / "checkpoints").mkdir(exist_ok=True)
                doc.saveAs(str(out / checkpoint))
            checkpoint_mesh = None
            if checkpoint and request.get("mesh"):
                checkpoint_mesh = f"checkpoints/{i:03d}-mesh.json"
                dump(out / checkpoint_mesh, capture_mesh(doc))
            result["events"].append(
                {
                    "step_id": i,
                    "action": action,
                    "observation_before": before,
                    "observation_after": snapshot(doc),
                    "checkpoint": checkpoint,
                    "checkpoint_mesh": checkpoint_mesh,
                    "duration_seconds": time.monotonic() - step_started,
                    "checkpoint_sha256": hashlib.sha256(
                        (out / checkpoint).read_bytes()
                    ).hexdigest()
                    if checkpoint
                    else None,
                }
            )
        result["observation"] = snapshot(doc)
        candidate_path = out / "output.FCStd"
        doc.saveAs(str(candidate_path))
        App.closeDocument(doc.Name)
        doc = App.openDocument(str(candidate_path))
        doc.recompute()
        result["execution_ok"] = True
        result["observation"] = snapshot(doc)
        result["artifact"] = "output.FCStd"
        if request.get("mesh"):
            dump(out / "mesh.json", capture_mesh(doc))
            result["mesh"] = "mesh.json"
        if request.get("reference_actions") is not None:
            # This is trusted evaluator code. Agent actions cannot execute Python or write scores.
            stage = "evaluation"
            validation = {
                "shape_valid": False,
                "parametric": False,
                "history_preserved": False,
                "task_success": False,
                "reward": 0.0,
                "geometry_score": 0.0,
                "evaluator_error": None,
            }
            result["validation"] = validation
            try:
                candidate = shape(doc)
            except ValueError as exc:
                validation["candidate_error"] = str(exc)
            else:
                validation["shape_valid"] = True
                reference = build("Reference", request["reference_actions"])
                reference_shape = shape(reference)
                metrics = compare(candidate, reference_shape)
                validation.update(metrics)
                sketches = [
                    o for o in doc.Objects if o.isDerivedFrom("Sketcher::SketchObject")
                ]
                # Exercise a named dimension and check that the final body actually changes.
                parametric = bool(sketches) and all(
                    o.FullyConstrained for o in sketches
                )
                mutation_ok = False
                for sk in sketches:
                    named = [
                        i for i in range(sk.ConstraintCount) if sk.Constraints[i].Name
                    ]
                    if not named:
                        continue
                    idx = named[0]
                    original = sk.getDatum(idx)
                    old_shape = candidate.copy()
                    try:
                        sk.setDatum(idx, App.Units.Quantity(original.Value * 1.01))
                        doc.recompute()
                        changed = shape(doc)
                        mutation_ok = (
                            abs(changed.Volume - old_shape.Volume)
                            > old_shape.Volume * 1e-8
                        )
                        mutation_ok = mutation_ok and not any(
                            "Invalid" in o.State for o in doc.Objects
                        )
                    except Exception:
                        mutation_ok = False
                    finally:
                        sk.setDatum(idx, original)
                        doc.recompute()
                    if mutation_ok:
                        break
                validation["parametric"] = parametric and mutation_ok
                before_objects = {
                    (o["name"], o["type"])
                    for o in result["initial_observation"]["objects"]
                }
                after_objects = {(o.Name, o.TypeId) for o in doc.Objects}
                # Edits preserve all object identities and feature types; create has no initial history.
                validation["history_preserved"] = (
                    not initial or before_objects == after_objects
                )
                protected_ok = True
                allowed = request.get("allowed_edits")
                if initial and allowed is not None:
                    permitted = {tuple(pair) for pair in allowed}
                    original_rows = {
                        o["name"]: o for o in result["initial_observation"]["objects"]
                    }
                    current_rows = {o["name"]: o for o in snapshot(doc)["objects"]}
                    for name, old in original_rows.items():
                        new = current_rows.get(name, {})
                        if old.get("dependencies") != new.get(
                            "dependencies"
                        ) or old.get("expressions") != new.get("expressions"):
                            protected_ok = False
                        for parameter, value in old.get("constraints", {}).items():
                            if (name, parameter) not in permitted and abs(
                                new.get("constraints", {}).get(parameter, float("inf"))
                                - value
                            ) > 1e-7:
                                protected_ok = False
                        if (
                            "length" in old
                            and (name, "length") not in permitted
                            and abs(new.get("length", float("inf")) - old["length"])
                            > 1e-7
                        ):
                            protected_ok = False
                validation["protected_parameters_preserved"] = protected_ok
                tolerance = request.get("tolerance", 1e-5)
                exact = metrics["relative_symmetric_difference"] <= tolerance
                gates = (
                    validation["parametric"]
                    and validation["history_preserved"]
                    and protected_ok
                )
                expected = request.get("expected_metrics")
                spec_pass = True
                if expected:
                    measured = snapshot(doc)["geometry"]
                    volume_error = abs(measured["volume"] - expected["volume"]) / max(
                        expected["volume"], 1e-12
                    )
                    bounds_error = max(
                        abs(a - b)
                        for a, b in zip(measured["bounds"], expected["bounds"])
                    )
                    spec_pass = volume_error <= tolerance and bounds_error <= tolerance
                    validation["spec_errors"] = {
                        "relative_volume": volume_error,
                        "bounds_mm": bounds_error,
                    }
                validation["spec_pass"] = spec_pass
                validation["task_success"] = bool(exact and gates and spec_pass)
                validation["reward"] = (
                    0.0
                    if not gates
                    else (
                        0.8 * metrics["geometry_score"]
                        + 0.2 * float(exact and spec_pass)
                    )
                )
                # Restore original geometry in saved output after the editability probe.
                doc.saveAs(str(candidate_path))
    except Exception as exc:
        result["error"] = str(exc)
        result["error_origin"] = stage
        if stage == "evaluation" and "validation" in result:
            result["validation"]["evaluator_error"] = str(exc)
        result["error_type"] = type(exc).__name__
        result["traceback"] = traceback.format_exc()
    finally:
        for name in list(App.listDocuments()):
            App.closeDocument(name)
        dump(result_path, result)
