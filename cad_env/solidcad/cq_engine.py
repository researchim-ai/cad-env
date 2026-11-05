import cadquery as cq

from cad_env.multicad import cq_engine as base

metrics = base.metrics
mesh = base.mesh
difference = base.difference

VERSION = cq.__version__


def build(program):
    objects = {}
    for a in program:
        op = a["op"]
        o = a.get("origin", [0, 0, 0])
        if op == "finish":
            return cq.Compound.makeCompound([objects[n] for n in a["targets"]])
        if op == "box":
            shape = cq.Solid.makeBox(
                a["length"], a["width"], a["height"], cq.Vector(*o)
            )
        elif op == "cylinder":
            shape = cq.Solid.makeCylinder(a["radius"], a["height"], cq.Vector(*o))
        elif op in {"polygon_extrude", "revolve_profile"}:
            pts = [
                cq.Vector(x, 0, y) if op == "revolve_profile" else cq.Vector(x, y, 0)
                for x, y in a["points"]
            ]
            wire = cq.Wire.makePolygon(pts, close=True)
            shape = (
                cq.Solid.revolve(wire, [], a["angle"], (0, 0, 0), (0, 0, 1))
                if op == "revolve_profile"
                else cq.Solid.extrudeLinear(wire, [], (0, 0, a["height"]))
            )
        elif op == "arc_extrude":
            start = cq.Vector(*a["start"], 0)
            edges = []
            for segment in a["segments"]:
                end = cq.Vector(*segment["to"], 0)
                edges.append(
                    cq.Edge.makeThreePointArc(start, cq.Vector(*segment["mid"], 0), end)
                    if segment["type"] == "arc"
                    else cq.Edge.makeLine(start, end)
                )
                start = end
            shape = cq.Solid.extrudeLinear(
                cq.Wire.assembleEdges(edges), [], (0, 0, a["height"])
            )
        elif op == "fillet":
            s = objects[a["source"]]
            shape = s.fillet(a["radius"], s.Edges())
        elif op == "chamfer":
            s = objects[a["source"]]
            shape = s.chamfer(a["size"], None, s.Edges())
        elif op == "shell":
            shape = (
                cq.Workplane(obj=objects[a["source"]])
                .faces(">Z")
                .shell(-a["thickness"])
                .val()
            )
        elif op == "fuse":
            shape = objects[a["left"]].fuse(objects[a["right"]]).clean()
        elif op == "cut":
            shape = objects[a["left"]].cut(objects[a["right"]]).clean()
        if not shape.isValid() or shape.Volume() <= 0:
            raise ValueError("Invalid/empty solid feature")
        objects[a["name"]] = shape


def save(shape, out, program):
    import json

    cq.exporters.export(shape, str(out / "output.step"))
    (out / "source.actions.json").write_text(json.dumps(program, indent=2))
    restored = cq.importers.importStep(str(out / "output.step")).val()
    if abs(restored.Volume() - shape.Volume()) / max(shape.Volume(), 1) > 1e-6:
        raise ValueError("STEP round-trip mismatch")
    return "output.step"
