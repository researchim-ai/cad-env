"""Native OCCT feature graph. Imported only inside the FreeCAD process."""

import copy
import json

import FreeCAD as App
import Part

from cad_env.solidcad import freecad_engine as geometry

difference = geometry.difference
mesh = geometry.mesh
metrics = geometry.metrics


def vec(v):
    return App.Vector(*v)


def placement(a):
    z = vec(a.get("normal", [0, 0, 1]))
    z.normalize()
    x = vec(a.get("x_axis", [1, 0, 0]))
    if "x_axis" not in a and abs(x.dot(z)) > 0.99:
        x = App.Vector(0, 1, 0)
    x = x - z * x.dot(z)
    if x.Length < 1e-8:
        raise ValueError("Plane x_axis is parallel to normal")
    x.normalize()
    y = z.cross(x)
    return App.Placement(vec(a.get("origin", [0, 0, 0])), App.Rotation(x, y, z, "ZXY"))


def sketch(a):
    p = a["profile"]
    if p["type"] == "circle":
        wire = Part.Wire([Part.makeCircle(p["radius"])])
    elif p["type"] in ("rectangle", "polygon"):
        pts = (
            p["points"]
            if p["type"] == "polygon"
            else [[0, 0], [p["length"], 0], [p["length"], p["width"]], [0, p["width"]]]
        )
        points = [App.Vector(x, y, 0) for x, y in pts]
        wire = Part.makePolygon(points + [points[0]])
    else:
        start = App.Vector(*p["start"], 0)
        edges = []
        for segment in p["segments"]:
            end = App.Vector(*segment["to"], 0)
            edges.append(
                Part.Arc(start, App.Vector(*segment["mid"], 0), end).toShape()
                if segment["type"] == "arc"
                else Part.makeLine(start, end)
            )
            start = end
        wire = Part.Wire(edges)
    # Face construction rejects invalid/self-intersecting or nonclosed wires.
    face = Part.Face(wire)
    if not wire.isClosed() or not face.isValid() or face.Area < 1e-8:
        raise ValueError("Invalid closed sketch profile")
    wire.Placement = placement(a)
    return wire


def topology(shape, offset=0, limit=32):
    return {
        "edges": [
            {
                "index": i + 1,
                "length": e.Length,
                "center": [e.CenterOfMass.x, e.CenterOfMass.y, e.CenterOfMass.z],
            }
            for i, e in enumerate(shape.Edges)
            if offset <= i < offset + limit
        ],
        "faces": [
            {
                "index": i + 1,
                "area": f.Area,
                "center": [f.CenterOfMass.x, f.CenterOfMass.y, f.CenterOfMass.z],
            }
            for i, f in enumerate(shape.Faces)
            if offset <= i < offset + limit
        ],
    }


def selected(items, indices):
    if any(i > len(items) for i in indices):
        raise ValueError("Topology index exceeds source edge/face count")
    return [items[i - 1] for i in indices]


def compound(shapes):
    solids = [solid for shape in shapes for solid in shape.Solids]
    if len(solids) > 128:
        raise ValueError("Solid instance budget exceeded")
    for i, left in enumerate(solids):
        for right in solids[i + 1 :]:
            if (
                left.BoundBox.intersect(right.BoundBox)
                and left.common(right).Volume > 1e-7
            ):
                raise ValueError("Overlapping compound solids; explicitly fuse instead")
    return Part.makeCompound(shapes)


def build(actions):
    # Apply edits in event order, rebuilding dependencies before continuing.
    definitions, order, objects, normals = {}, [], {}, {}

    def feature(a):
        op = a["op"]
        source = objects.get(a.get("source"))
        origin = vec(a.get("origin", [0, 0, 0]))
        if op == "sketch":
            normals[a["name"]] = placement(a).Rotation.multVec(App.Vector(0, 0, 1))
            return sketch(a)
        if op in ("pad", "pocket"):
            direction = normals[a["profile"]] * a["length"]
            if op == "pocket":
                direction = -direction
            tool = Part.Face(objects[a["profile"]]).extrude(direction)
            return tool if op == "pad" else source.cut(tool).removeSplitter()
        if op == "revolve":
            return Part.Face(objects[a["profile"]]).revolve(
                origin, vec(a["axis"]), a["angle"]
            )
        if op == "loft":
            return Part.makeLoft(
                [objects[n] for n in a["profiles"]], True, a.get("ruled", False)
            )
        if op == "sweep":
            points = [vec(p) for p in a["path"]]
            if (points[0] - objects[a["profile"]].CenterOfMass).Length > 1e-5:
                raise ValueError("Sweep profile center must equal path start")
            tangent = points[1] - points[0]
            tangent.normalize()
            if abs(tangent.dot(normals[a["profile"]])) < 1 - 1e-6:
                raise ValueError("Sweep profile must be normal to initial path segment")
            path = Part.Wire(Part.makePolygon(points).Edges)
            return path.makePipeShell(
                [objects[a["profile"]]], True, a.get("frenet", False), 1
            )
        if op in ("fillet", "chamfer"):
            edges = selected(source.Edges, a["edges"])
            return (
                source.makeFillet(a["radius"], edges)
                if op == "fillet"
                else source.makeChamfer(a["size"], edges)
            )
        if op == "shell":
            return source.makeThickness(
                selected(source.Faces, a["faces"]), -a["thickness"], 1e-5
            )
        if op in ("fuse", "cut", "common"):
            return getattr(objects[a["left"]], op)(objects[a["right"]]).removeSplitter()
        if op in ("translate", "rotate", "mirror"):
            s = source.copy()
            if op == "translate":
                s.translate(vec(a["vector"]))
            elif op == "rotate":
                s.rotate(origin, vec(a["axis"]), a["angle"])
            else:
                s = s.mirror(origin, vec(a["normal"]))
            return s
        if op in ("linear_pattern", "polar_pattern"):
            copies = []
            for i in range(a["count"]):
                s = source.copy()
                if op == "linear_pattern":
                    s.translate(vec(a["vector"]) * i)
                else:
                    # Angle is total span; 360 excludes a duplicate last copy.
                    divisor = a["count"] if a["angle"] == 360 else a["count"] - 1
                    s.rotate(origin, vec(a["axis"]), a["angle"] * i / divisor)
                copies.append(s)
            return compound(copies)
        raise ValueError("Unsupported native feature")

    def compute(a):
        shape = feature(a)
        if shape.isNull() or not shape.isValid():
            raise ValueError("Invalid feature: " + a["name"])
        if len(shape.Solids) > 128 or len(shape.Edges) > 4096:
            raise ValueError("Feature complexity budget exceeded")
        if a["op"] != "sketch" and (not shape.Solids or shape.Volume <= 1e-8):
            raise ValueError("Feature must contain a nonempty solid: " + a["name"])
        objects[a["name"]] = shape

    targets = []
    for a in actions:
        if a["op"] == "finish":
            targets = a["targets"]
        elif a["op"] == "inspect":
            pass
        elif a["op"] == "set_parameter":
            target = definitions[a["target"]]
            if target["op"] == "sketch":
                target["profile"][a["parameter"]] = a["value"]
            else:
                target[a["parameter"]] = a["value"]
            for name in order:
                compute(definitions[name])
        else:
            definitions[a["name"]] = copy.deepcopy(a)
            order.append(a["name"])
            compute(a)
    solids = [n for n in order if definitions[n]["op"] != "sketch"]
    targets = targets or solids[-1:]
    shape = compound([objects[n] for n in targets]) if targets else None
    state = [
        {
            "name": n,
            "kind": "profile" if definitions[n]["op"] == "sketch" else "solid",
            "operation": definitions[n],
            "edge_count": len(objects[n].Edges),
            "face_count": len(objects[n].Faces),
            **(
                topology(
                    objects[n],
                    actions[-1].get("offset", 0),
                    actions[-1].get("limit", 32),
                )
                if actions[-1]["op"] == "inspect" and actions[-1]["target"] == n
                else {}
            ),
        }
        for n in order
    ]
    return shape, objects, state, targets


def save(out, actions, objects, state, targets):
    doc = App.newDocument("AdvancedCAD")
    try:
        nodes = {}
        for item in state:
            n = item["name"]
            obj = doc.addObject("Part::Feature", n)
            obj.Shape = objects[n]
            obj.addProperty("App::PropertyString", "ActionSource")
            obj.ActionSource = json.dumps(item["operation"])
            obj.addProperty("App::PropertyBool", "FinalTarget")
            obj.FinalTarget = n in targets
            nodes[n] = obj
        source = doc.addObject("App::FeaturePython", "ProgramSource")
        source.addProperty("App::PropertyString", "ProgramJSON")
        source.ProgramJSON = json.dumps(actions)
        source.addProperty("App::PropertyString", "Contract")
        source.Contract = (
            "freecad.v2: explicit source replay, not PartDesign feature proxies"
        )
        doc.recompute()
        doc.saveAs(str(out / "output.FCStd"))
    finally:
        App.closeDocument(doc.Name)
    restored = App.openDocument(str(out / "output.FCStd"))
    try:
        for name, shape in objects.items():
            saved = restored.getObject(name).Shape
            if (
                saved.isNull()
                or not saved.isValid()
                or abs(saved.Volume - shape.Volume) > 1e-5
            ):
                raise ValueError("FCStd round-trip failed: " + name)
    finally:
        App.closeDocument(restored.Name)
    return "output.FCStd"
