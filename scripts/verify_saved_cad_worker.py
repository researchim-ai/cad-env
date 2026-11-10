"""Independent read-only FCStd geometry checks; run only by FreeCADCmd.

Deliberately does not import the benchmark evaluator or its success functions.
"""

import json
import sys
from pathlib import Path


def main():
    import FreeCAD as App

    request = json.loads(Path(sys.argv[1]).read_text())

    def load(path):
        doc = App.openDocument(path)
        doc.recompute()
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
        s = (
            bodies[0].Shape.copy()
            if len(bodies) == 1 and not bodies[0].Shape.isNull()
            else None
        )
        return doc, s

    def valid(shape):
        return (
            shape is not None
            and shape.isValid()
            and len(shape.Solids) == 1
            and shape.Volume > 0
        )

    def compare(a, b):
        if not valid(a):
            return {"valid": False, "exact": False}
        if not valid(b):
            raise ValueError("Invalid independent reference shape")
        missing = b.cut(a).Volume
        extra = a.cut(b).Volume
        relative = (abs(missing) + abs(extra)) / max(a.Volume + missing, 1e-12)
        bb = a.BoundBox
        bounds = [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]
        volume_error = (
            abs(a.Volume - request["analytic"]["volume"])
            / request["analytic"]["volume"]
        )
        bounds_error = max(
            abs(x - y) for x, y in zip(bounds, request["analytic"]["bounds"])
        )
        return {
            "valid": True,
            "exact": relative <= request["tolerance"],
            "missing_volume": missing,
            "extra_volume": extra,
            "relative_error": relative,
            "analytic_volume_error": volume_error,
            "analytic_bounds_error": bounds_error,
            "spec_pass": volume_error <= request["tolerance"]
            and bounds_error <= request["tolerance"],
        }

    def editable(doc, shape):
        sketches = [o for o in doc.Objects if o.isDerivedFrom("Sketcher::SketchObject")]
        if (
            not valid(shape)
            or not sketches
            or not all(s.FullyConstrained for s in sketches)
        ):
            return False
        for sk in sketches:
            for index in range(sk.ConstraintCount):
                if not sk.Constraints[index].Name:
                    continue
                old = sk.getDatum(index)
                try:
                    sk.setDatum(index, App.Units.Quantity(old.Value * 1.01))
                    doc.recompute()
                    body = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"][0]  # noqa: RUF015
                    changed = body.Shape
                    ok = (
                        valid(changed)
                        and abs(changed.Volume - shape.Volume) > shape.Volume * 1e-8
                        and not any("Invalid" in o.State for o in doc.Objects)
                    )
                except Exception:  # noqa: BLE001 - native mutation may fail with OCCT exception types
                    ok = False
                finally:
                    sk.setDatum(index, old)
                    doc.recompute()
                if ok:
                    return True
        return False

    result = {}
    try:
        reference_doc, reference = load(request["reference"])
        if not valid(reference):
            raise ValueError("Invalid oracle artifact")
        result["reference"] = compare(reference, reference)
        for key in ("saved", "replayed", "drawing"):
            doc, shape = load(request[key])
            item = compare(shape, reference)
            if key != "drawing":
                item["parametric"] = editable(doc, shape)
                item["independent_success"] = bool(
                    request["actual_finish"]
                    and item["valid"]
                    and item["exact"]
                    and item.get("spec_pass")
                    and item["parametric"]
                )
            result[key] = item
            App.closeDocument(doc.Name)
        App.closeDocument(reference_doc.Name)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - persist native audit failures explicitly
        result.update(ok=False, error=repr(exc))
    finally:
        for name in list(App.listDocuments()):
            App.closeDocument(name)
    Path(request["output"]).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
