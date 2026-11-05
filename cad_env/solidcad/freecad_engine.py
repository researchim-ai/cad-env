import FreeCAD as App
import Part

VERSION = ".".join(App.Version()[:3])


def build(program):
    objects = {}
    for a in program:
        op = a["op"]
        origin = App.Vector(*a.get("origin", [0, 0, 0]))
        if op == "finish":
            return Part.makeCompound([objects[n] for n in a["targets"]])
        if op == "box":
            s = Part.makeBox(a["length"], a["width"], a["height"], origin)
        elif op == "cylinder":
            s = Part.makeCylinder(a["radius"], a["height"], origin)
        elif op in {"polygon_extrude", "revolve_profile"}:
            pts = [
                App.Vector(x, 0, y) if op == "revolve_profile" else App.Vector(x, y, 0)
                for x, y in a["points"]
            ]
            wire = Part.makePolygon(pts + [pts[0]])
            face = Part.Face(wire)
            s = (
                face.revolve(App.Vector(0, 0, 0), App.Vector(0, 0, 1), a["angle"])
                if op == "revolve_profile"
                else face.extrude(App.Vector(0, 0, a["height"]))
            )
        elif op == "arc_extrude":
            start = App.Vector(*a["start"], 0)
            edges = []
            for segment in a["segments"]:
                end = App.Vector(*segment["to"], 0)
                edges.append(
                    Part.Arc(start, App.Vector(*segment["mid"], 0), end).toShape()
                    if segment["type"] == "arc"
                    else Part.makeLine(start, end)
                )
                start = end
            s = Part.Face(Part.Wire(edges)).extrude(App.Vector(0, 0, a["height"]))
        elif op == "fillet":
            base = objects[a["source"]]
            s = base.makeFillet(a["radius"], base.Edges)
        elif op == "chamfer":
            base = objects[a["source"]]
            s = base.makeChamfer(a["size"], base.Edges)
        elif op == "shell":
            base = objects[a["source"]]
            top = max(f.CenterOfMass.z for f in base.Faces)
            faces = [f for f in base.Faces if abs(f.CenterOfMass.z - top) < 1e-6]
            s = base.makeThickness(faces, -a["thickness"], 0.0001)
        elif op == "fuse":
            s = objects[a["left"]].fuse(objects[a["right"]]).removeSplitter()
        elif op == "cut":
            s = objects[a["left"]].cut(objects[a["right"]]).removeSplitter()
        if s.isNull() or not s.isValid() or s.Volume <= 0:
            raise ValueError("Invalid/empty solid feature")
        objects[a["name"]] = s


def metrics(s):
    b = s.BoundBox
    return {
        "volume": s.Volume,
        "area": s.Area,
        "bounds": [b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax],
        "valid": s.isValid(),
        "solids": len(s.Solids),
    }


def difference(a, b):
    return a.cut(b).Volume + b.cut(a).Volume


def mesh(s):
    v, f = s.tessellate(0.1)
    return {"vertices": [[p.x, p.y, p.z] for p in v], "faces": [list(t) for t in f]}


def save(s, out, program):
    import json

    doc = App.newDocument("SolidGraph")
    obj = doc.addObject("Part::Feature", "Result")
    obj.Shape = s
    obj.addProperty("App::PropertyString", "ActionSource")
    obj.ActionSource = json.dumps(program)
    doc.recompute()
    doc.saveAs(str(out / "output.FCStd"))
    App.closeDocument(doc.Name)
    restored = App.openDocument(str(out / "output.FCStd"))
    if (
        abs(restored.getObject("Result").Shape.Volume - s.Volume) / max(s.Volume, 1)
        > 1e-6
    ):
        raise ValueError("FCStd round-trip mismatch")
    App.closeDocument(restored.Name)
    return "output.FCStd"
