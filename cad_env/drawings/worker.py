"""Executed only by native FreeCADCmd; no simulated projections or labels."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def main(request_path, result_path):
    import FreeCAD as App
    import Part
    import TechDraw

    request = json.loads(Path(request_path).read_text())
    out = Path(result_path).parent
    source = Path(request["model"])
    if source.suffix.lower() == ".fcstd":
        doc = App.openDocument(str(source))
        bodies = [obj for obj in doc.Objects if obj.TypeId == "PartDesign::Body"]
        if len(bodies) != 1:
            raise ValueError(
                "Exactly one PartDesign body required for FCStd drawing export"
            )
        model = bodies[0]
    else:
        doc = App.newDocument("TechnicalDrawing")
        model = doc.addObject("Part::Feature", "SourcePart")
        model.Shape = Part.read(str(source))
    shape = model.Shape.copy()
    if not shape.isValid() or len(shape.Solids) != 1:
        raise ValueError("Reference must be one valid solid")
    box = shape.BoundBox
    bounds = [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax]
    span = [box.XLength, box.YLength, box.ZLength]
    volume = shape.Volume
    spec = request.get("parameters", {})
    checks = []
    axes = []

    def close(actual, expected, label):
        if not math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-9):
            raise ValueError("Drawing specification disagrees with CAD: " + label)
        checks.append({"field": label, "native": actual, "specified": expected})

    def feature(name, kind, x, y, radius, length):
        obj = doc.getObject(name)
        if obj is None or obj.TypeId != kind:
            raise ValueError("Missing native feature: " + name)
        sketch = obj.Profile[0]
        if len(sketch.Geometry) != 1:
            raise ValueError("Schedule expects a single circular profile: " + name)
        circle = sketch.Geometry[0]
        for field, actual, expected in [
            ("x", circle.Center.x, x),
            ("y", circle.Center.y, y),
            ("radius", circle.Radius, radius),
            ("length", obj.Length.Value, length),
        ]:
            close(actual, expected, name + "." + field)
        top = sketch.Placement.Base.z
        z0, z1 = (
            (top, top + length) if kind == "PartDesign::Pad" else (top - length, top)
        )
        axes.append({"x": x, "y": y, "z0": z0, "z1": z1, "radius": radius})
        return sketch

    if spec:
        expected_pads = {p["name"] for p in spec.get("pads", [])}
        if "base" in spec:
            expected_pads.add("BasePad")
        expected_cuts = {h["name"] for h in spec.get("cuts", [])}
        expected_cuts.update(
            h["name"] + "CB" for h in spec.get("cuts", []) if h["cb_radius"] is not None
        )
        if expected_pads != {
            o.Name for o in doc.Objects if o.TypeId == "PartDesign::Pad"
        } or expected_cuts != {
            o.Name for o in doc.Objects if o.TypeId == "PartDesign::Pocket"
        }:
            raise ValueError(
                "Feature schedule does not cover every native pad and pocket"
            )
        if "base" in spec:
            b = spec["base"]
            close(doc.getObject("BasePad").Length.Value, b["height"], "base.height")
            sketch = doc.getObject("BaseSketch")
            close(sketch.getDatum("length").Value, b["length"], "base.length")
            close(sketch.getDatum("width").Value, b["width"], "base.width")
        for pad in spec.get("pads", []):
            profile = feature(
                pad["name"],
                "PartDesign::Pad",
                pad["x"],
                pad["y"],
                pad["radius"],
                pad["height"],
            )
            support = doc.getObject(pad["support"]) if pad["support"] else None
            expected_z = (
                (support.Profile[0].Placement.Base.z + support.Length.Value)
                if support
                else 0
            )
            close(profile.Placement.Base.z, expected_z, pad["name"] + ".support_z")
            if support is not None and support not in profile.OutList:
                raise ValueError(
                    "Native pad support dependency mismatch: " + pad["name"]
                )

        for cut in spec.get("cuts", []):
            profile = feature(
                cut["name"],
                "PartDesign::Pocket",
                cut["x"],
                cut["y"],
                cut["radius"],
                cut["depth"],
            )
            support = doc.getObject(cut["support"])
            expected_z = support.Profile[0].Placement.Base.z + support.Length.Value
            close(profile.Placement.Base.z, expected_z, cut["name"] + ".support_z")
            if support not in profile.OutList:
                raise ValueError(
                    "Native drilling support dependency mismatch: " + cut["name"]
                )

            if cut["cb_radius"] is not None:
                cb = feature(
                    cut["name"] + "CB",
                    "PartDesign::Pocket",
                    cut["x"],
                    cut["y"],
                    cut["cb_radius"],
                    cut["cb_depth"],
                )
                if support not in cb.OutList:
                    raise ValueError(
                        "Native counterbore support dependency mismatch: " + cut["name"]
                    )
                close(
                    cb.Placement.Base.z,
                    expected_z,
                    cut["name"] + ".counterbore_support_z",
                )
    doc.saveAs(str(out / "model-with-drawings.FCStd"))
    page = doc.addObject("TechDraw::DrawPage", "TechnicalPage")
    template = doc.addObject("TechDraw::DrawSVGTemplate", "DrawingTemplate")
    template.Template = str(out / "template.svg")
    page.Template = template
    page.ProjectionType = "First Angle"
    scale = max(
        s
        for s in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
        if s <= min(130 / span[0], 95 / span[1], 68 / span[2])
    )
    page.Scale = scale
    views = {}
    definitions = [
        ("front", (0, -1, 0), (1, 0, 0), (190, 70)),
        ("top", (0, 0, 1), (1, 0, 0), (190, 178)),
        ("right", (1, 0, 0), (0, 1, 0), (70, 70)),
    ]

    def svg_layers(view):
        wrapper = ET.fromstring("<svg>" + TechDraw.viewPartAsSvg(view) + "</svg>")
        layers = list(wrapper)
        for i, group in enumerate(layers):
            group.set("stroke-width", str((0.35 if i == 0 else 0.18) / scale))
            if i:
                group.set("stroke-dasharray", f"{2 / scale} {1 / scale}")
            for element in group.iter():
                element.attrib.pop("id", None)
        # Visible lines take precedence over coincident hidden lines.
        return "".join(
            ET.tostring(group, encoding="unicode") for group in layers[1:] + layers[:1]
        )

    for name, direction, xdirection, center in definitions:
        view = doc.addObject("TechDraw::DrawViewPart", "Drawing" + name.title())
        view.Source = [model]
        view.Direction = direction
        view.XDirection = xdirection
        view.HardHidden = True
        view.ScaleType = "Custom"
        view.Scale = 1
        view.X, view.Y = center[0], 297 - center[1]
        page.addView(view)
        doc.recompute()
        if view.getStatusString() != "Valid":
            raise ValueError("Invalid native projection: " + name)
        views[name] = {
            "svg": svg_layers(view),
            "center_mm": center,
            "object": view.Name,
            "direction": direction,
            "geometric_center": list(view.getGeometricCenter()),
        }
        view.Scale = scale
    doc.recompute()
    dimension_offset = (
        9 if request.get("drawing_standard", "nominal") == "nominal" else 15
    )
    dimensions = []
    for view_name, axis, expected in [
        ("top", 0, span[0]),
        ("top", 1, span[1]),
        ("front", 1, span[2]),
    ]:
        view = doc.getObject(views[view_name]["object"])
        dimension = TechDraw.makeExtentDim(view, [], axis)
        doc.recompute()
        view_center = views[view_name]["center_mm"]
        dimension.X = view_center[0] + (
            span[0] * scale / 2 + dimension_offset if axis else 0
        )
        dimension.Y = 297 - (
            view_center[1] + (span[1] * scale / 2 + dimension_offset if not axis else 0)
        )
        value = dimension.getRawValue()
        close(value, expected, "dimension." + view_name + "." + str(axis))
        dimensions.append(
            {
                "object": dimension.Name,
                "view": view_name,
                "axis": axis,
                "value_mm": value,
            }
        )
    # Native section object plus exact B-rep intersection faces for hatch regions.
    section_y = (
        spec.get("cuts", [{}])[0].get("y", (box.YMin + box.YMax) / 2)
        if spec.get("cuts")
        else (box.YMin + box.YMax) / 2
    )
    section = doc.addObject("TechDraw::DrawViewSection", "DrawingSectionA")
    section.BaseView = doc.getObject(views["top"]["object"])
    section.Source = [model]
    section.SectionOrigin = (
        (box.XMin + box.XMax) / 2,
        section_y,
        (box.ZMin + box.ZMax) / 2,
    )
    section.SectionNormal = (0, -1, 0)
    section.Direction = (0, -1, 0)
    section.XDirection = (1, 0, 0)
    section.SectionSymbol = "A"
    section.HardHidden = False
    section.ScaleType = "Custom"
    section.Scale = 1
    section.X, section.Y = 340, 297 - 178
    page.addView(section)
    doc.recompute()
    if section.getStatusString() != "Valid":
        raise ValueError("Native section could not be constructed")
    center = section.getGeometricCenter()
    reach = max(abs(x) for x in bounds) + max(span) + 10
    plane = Part.makePlane(
        4 * reach,
        4 * reach,
        App.Vector(-2 * reach, section_y, -2 * reach),
        App.Vector(0, 1, 0),
    )
    cap = shape.common(plane)
    if cap.Area <= 0 or not cap.Faces:
        raise ValueError("Section plane has no solid material")
    contours = []
    for face in cap.Faces:
        rings = []
        for wire in face.Wires:
            points = wire.discretize(Deflection=0.025)
            rings.append(
                [[point.x - center.x, -(point.z - center.z)] for point in points]
            )
        contours.append(rings)
    hatch_segments = []
    step = 2 * math.sqrt(2) / scale
    offset = box.ZMin - box.XMax - step
    limit = box.ZMax - box.XMin + step
    while offset <= limit:
        hatch_line = Part.makeLine(
            App.Vector(-2 * reach, section_y, -2 * reach + offset),
            App.Vector(2 * reach, section_y, 2 * reach + offset),
        )
        clipped = cap.common(hatch_line)
        for edge in clipped.Edges:
            points = [v.Point for v in edge.Vertexes]
            if len(points) == 2:
                hatch_segments.append(
                    [[p.x - center.x, -(p.z - center.z)] for p in points]
                )
        offset += step
    views["section"] = {
        "svg": svg_layers(section),
        "center_mm": [340, 178],
        "object": section.Name,
        "direction": [0, -1, 0],
        "geometric_center": list(center),
        "hatch_contours": contours,
        "hatch_segments": hatch_segments,
    }
    section.Scale = scale
    doc.recompute()
    close(model.Shape.Volume, volume, "unchanged.volume")
    for actual, expected in zip(
        [
            model.Shape.BoundBox.XMin,
            model.Shape.BoundBox.YMin,
            model.Shape.BoundBox.ZMin,
            model.Shape.BoundBox.XMax,
            model.Shape.BoundBox.YMax,
            model.Shape.BoundBox.ZMax,
        ],
        bounds,
    ):
        close(actual, expected, "unchanged.bounds")
    doc.save()
    result = {
        "status": "complete",
        "freecad": App.Version(),
        "bounds_mm": bounds,
        "volume_mm3": volume,
        "scale": scale,
        "projection": "first_angle",
        "units": "mm",
        "views": views,
        "dimensions": dimensions,
        "dimension_offset_mm": dimension_offset,
        "section": {
            "plane_y_mm": section_y,
            "cap_area_mm2": cap.Area,
            "contours_chord_deflection_mm": 0.025,
        },
        "native_spec_checks": checks,
        "feature_axes": axes,
        "specification_verified": bool(spec),
        "support_dependencies_verified": bool(spec),
        "editable_source": source.suffix.lower() == ".fcstd",
        "native_document": "model-with-drawings.FCStd",
    }
    Path(result_path).write_text(json.dumps(result, indent=2))
