"""Exact OCCT solid execution through CadQuery."""

import cadquery as cq

VERSION = cq.__version__


def build(features):
    solid = None
    for feature in features:
        p = feature["profile_spec"]
        z = feature["z"]
        depth = feature["length"]
        bottom = z if feature["op"] == "pad" else z - depth
        if p["op"] == "sketch_rectangle":
            tool = cq.Solid.makeBox(
                p["length"], p["width"], depth, cq.Vector(0, 0, bottom)
            )
        else:
            tools = [
                cq.Solid.makeCylinder(
                    c["radius"], depth, cq.Vector(c["x"], c["y"], bottom)
                )
                for c in p["circles"]
            ]
            tool = tools[0]
            for other in tools[1:]:
                tool = tool.fuse(other)
        solid = (
            tool
            if solid is None
            else (solid.fuse(tool) if feature["op"] == "pad" else solid.cut(tool))
        )
        solid = solid.clean()
        if not solid.isValid() or len(solid.Solids()) != 1:
            raise ValueError("Each feature must produce one valid solid")
    return solid


def metrics(solid):
    b = solid.BoundingBox()
    return {
        "volume": solid.Volume(),
        "area": solid.Area(),
        "bounds": [b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax],
        "valid": solid.isValid(),
        "solids": len(solid.Solids()),
    }


def difference(a, b):
    return a.cut(b).Volume() + b.cut(a).Volume()


def mesh(solid):
    v, f = solid.tessellate(0.1)
    return {"vertices": [[p.x, p.y, p.z] for p in v], "faces": [list(t) for t in f]}


def save(solid, out, actions):
    cq.exporters.export(solid, str(out / "output.step"))
    # The restricted source is retained; STEP alone carries no editable program.
    (out / "source.actions.json").write_text(
        __import__("json").dumps(actions, indent=2)
    )
    restored = cq.importers.importStep(str(out / "output.step")).val()
    if abs(restored.Volume() - solid.Volume()) > max(1.0, solid.Volume()) * 1e-6:
        raise ValueError("STEP round-trip changed volume")
    return "output.step"
