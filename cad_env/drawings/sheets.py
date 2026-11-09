"""A3, first-angle nominal geometry sheets; all coordinates are millimeters."""

import math
import re
from html import escape

from .profiles import ACTIVE, METADATA, get_profile, validate_metadata

WIDTH, HEIGHT = 420, 297
PROFILE = "cad-env.a3-first-angle.nominal.v1"


def text(x, y, value, size=3.2, anchor="start", **attrs):
    profile = ACTIVE.get()
    if profile is not None:
        from .lettering import outline_text
        from .standard_frames import TRANSLATIONS

        value = str(value)
        if profile.key == "gost":
            if value in (
                "ORTHOGONAL VIEWS AND SECTION",
                "FEATURE AND DRILLING SCHEDULE",
            ):
                x = 110
            value = TRANSLATIONS.get(value, value)
            value = re.sub(r"(?<![A-Za-z0-9_])D(?=[0-9])", "⌀", value)
            value = re.sub(r"(?<![A-Za-z])([0-9]+)\.([0-9]+)", r"\1,\2", value)
        else:
            value = re.sub(r"(?<![A-Za-z0-9_])D(?=[0-9])", "⌀", value)
        return outline_text(x, y, value, size, anchor, **attrs)
    attrs.setdefault("stroke", "none")
    extra = " ".join(
        f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"'
        for k, v in attrs.items()
    )
    return f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" text-anchor="{anchor}" {extra}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, style=""):
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" {style}/>'


def frame(identity, title, sheet, total, scale="1:1"):
    if ACTIVE.get() is not None:
        from .standard_frames import render_frame

        return render_frame(identity, title, sheet, total, scale)
    values = [
        '<rect x="20" y="10" width="390" height="277" fill="none" stroke-width="0.5"/>',
        '<path d="M230 251H410 M230 263H410 M230 275H410 M230 251V287 M350 263V287 M380 263V287" fill="none"/>',
        text(234, 256, "CAD-ENV / NOMINAL GEOMETRY", 2.8),
        text(234, 261, title.replace("_", " ").upper()[:62], 3.2),
        text(234, 268, "DRAWING ID", 2.2),
        text(234, 273, identity, 3.1),
        text(354, 268, "REV", 2.2),
        text(354, 273, "A", 3.1),
        text(384, 268, "SHEET", 2.2),
        text(384, 273, f"{sheet}/{total}", 3.1),
        text(234, 280, "mm  |  FIRST-ANGLE  |  A3", 2.8),
        text(234, 285, f"SCALE {scale}  |  MATERIAL: NOT SPECIFIED", 2.6),
        text(24, 268, "NOMINAL MODEL RECONSTRUCTION", 3.1),
        text(
            24,
            274,
            "No manufacturing tolerances, finish or material requirements inferred.",
            2.7,
        ),
        text(
            24,
            280,
            "Drafting profile: A3 / first-angle. Standards conformance not assessed.",
            2.7,
        ),
        text(24, 286, "Dimensions govern; do not measure the raster image.", 2.7),
    ]
    return "".join(values)


def document(body):
    profile = ACTIVE.get()
    if profile is not None:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297" data-drafting-standard="'
            + profile.key
            + '">'
            '<rect width="420" height="297" fill="white"/><g stroke="black" stroke-width="'
            + str(profile.thin)
            + '" fill="black">'
            + body
            + "</g></svg>"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">'
        '<defs><marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="3" markerHeight="3" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0L6 3L0 6Z" fill="black"/></marker>'
        '<pattern id="hatch" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><path d="M0 0V2" stroke="black" stroke-width="0.16"/></pattern></defs>'
        '<rect width="420" height="297" fill="white"/>'
        '<g stroke="black" stroke-width="0.18" fill="black" font-family="DejaVu Sans, sans-serif">'
        + body
        + "</g></svg>"
    )


def scale_label(scale):
    if scale <= 1:
        return f"1:{1 / scale:g}"
    return f"{scale:g}:1"


def overview(native, identity, title, total):
    s = native["scale"]
    b = native["bounds_mm"]
    dx, dy, dz = (b[i + 3] - b[i] for i in range(3))
    body = frame(identity, title, 1, total, scale_label(s))
    body += text(24, 20, "ORTHOGONAL VIEWS AND SECTION", 4.2)
    body += text(
        24, 26, "Model-derived hidden-line removal; identical scale for all views.", 2.8
    )
    for name, view in native["views"].items():
        x, y = view["center_mm"]
        local = ""
        if name == "section":
            for first, last in view["hatch_segments"]:
                local += line(
                    first[0],
                    first[1],
                    last[0],
                    last[1],
                    f'stroke-width="{(ACTIVE.get().thin if ACTIVE.get() else 0.16) / s}"',
                )
        local += view["svg"]
        body += (
            f'<g id="view-{name}" transform="translate({x} {y}) scale({s})">{local}</g>'
        )
        view_height = dy if name == "top" else dz
        caption = {
            "front": "FRONT",
            "top": "TOP",
            "right": "RIGHT (first-angle)",
            "section": f"SECTION A-A / Y={native['section']['plane_y_mm']:g}",
        }[name]
        if ACTIVE.get() is None:
            body += text(x, y + view_height * s / 2 + 17, caption, 3.2, "middle")
        elif name == "section":
            body += text(
                x,
                y - view_height * s / 2 - 7,
                "А-А" if ACTIVE.get().key == "gost" else "A-A",
                5,
                "middle",
            )
    # Center marks belong to actual cylindrical features, not arbitrary box centers.
    marked = set()
    for axis in native.get("feature_axes", []):
        for name in ("top", "front", "right"):
            x, y = native["views"][name]["center_mm"]
            center = native["views"][name]["geometric_center"]
            style = f'stroke-width="{ACTIVE.get().thin if ACTIVE.get() else 0.18}" stroke-dasharray="6 1 1 1"'
            if name == "top":
                px, py = (
                    x + (axis["x"] - center[0]) * s,
                    y - (axis["y"] - center[1]) * s,
                )
                key = (name, round(px, 6), round(py, 6))
                if key in marked:
                    continue
                marked.add(key)
                extent = min(4, axis["radius"] * s + 1.5)
                body += line(px - extent, py, px + extent, py, style) + line(
                    px, py - extent, px, py + extent, style
                )
            else:
                horizontal = (
                    axis["x"] - center[0] if name == "front" else axis["y"] - center[1]
                )
                px = x + horizontal * s
                y0, y1 = (
                    y - (axis["z1"] - center[2]) * s - 2,
                    y - (axis["z0"] - center[2]) * s + 2,
                )
                key = (name, round(px, 6), round(y0, 6), round(y1, 6))
                if key not in marked:
                    body += line(px, y0, px, y1, style)
                marked.add(key)
    # A-A cutting line and direction arrows are tied to the native plane.
    x, y = native["views"]["top"]["center_mm"]
    cy = native["views"]["top"]["geometric_center"][1]
    cut_y = y - (native["section"]["plane_y_mm"] - cy) * s
    left, right = x - dx * s / 2 - 5, x + dx * s / 2 + 5
    if ACTIVE.get() is None:
        body += line(
            left, cut_y, right, cut_y, 'stroke-width="0.35" stroke-dasharray="8 1 1 1"'
        )
    else:
        weight = 1 if ACTIVE.get().key == "gost" else 0.7
        body += line(left - 3, cut_y, left + 5, cut_y, f'stroke-width="{weight}"')
        body += line(right - 5, cut_y, right + 3, cut_y, f'stroke-width="{weight}"')
    for end in (left, right):
        body += line(
            end,
            cut_y + 5,
            end,
            cut_y - 4,
            'stroke-width="0.35"',
        )
        body += f'<path d="M{end},{cut_y - 4} l-0.65,2.5 h1.3 Z" stroke="none"/>'
        body += text(
            end,
            cut_y + 10,
            "А" if ACTIVE.get() and ACTIVE.get().key == "gost" else "A",
            5 if ACTIVE.get() else 3.5,
            "middle",
        )
    dimensions = {(d["view"], d["axis"]): d["value_mm"] for d in native["dimensions"]}

    gap = 15 if ACTIVE.get() else 9

    def horizontal(x, y, width, value):
        body_line = line(
            x - width / 2,
            y,
            x + width / 2,
            y,
            "",
        )
        body_line += line(
            x - width / 2, y - (gap if ACTIVE.get() else 8), x - width / 2, y + 2
        ) + line(x + width / 2, y - (gap if ACTIVE.get() else 8), x + width / 2, y + 2)
        for sign in [-1, 1]:
            end = x + sign * width / 2
            body_line += (
                f'<path d="M{end},{y} l{-sign * 2.5},-0.65 v1.3 Z" stroke="none"/>'
            )
        return body_line + text(x, y - 1.4, f"{value:g}", 3.2, "middle", stroke="none")

    def vertical(x, y, height, value):
        result = line(
            x,
            y - height / 2,
            x,
            y + height / 2,
            "",
        )
        result += line(
            x - (gap if ACTIVE.get() else 8), y - height / 2, x + 2, y - height / 2
        ) + line(
            x - (gap if ACTIVE.get() else 8), y + height / 2, x + 2, y + height / 2
        )
        for sign in [-1, 1]:
            end = y + sign * height / 2
            result += (
                f'<path d="M{x},{end} l-0.65,{-sign * 2.5} h1.3 Z" stroke="none"/>'
            )
        result += (
            f'<g transform="translate({x - 1.4} {y}) rotate(-90)">'
            + text(0, 0, f"{value:g}", 3.2, "middle", stroke="none")
            + "</g>"
        )
        return result

    body += horizontal(190, 178 + dy * s / 2 + gap, dx * s, dimensions["top", 0])
    body += vertical(190 + dx * s / 2 + gap, 178, dy * s, dimensions["top", 1])
    body += vertical(190 + dx * s / 2 + gap, 70, dz * s, dimensions["front", 1])
    return document(body)


def schedules(parameters, identity, title, total):
    rows = []
    if "base" in parameters:
        base = parameters["base"]
        rows.append(
            (
                "BASE",
                [
                    "BasePad",
                    "XY / Z=0",
                    "0",
                    "0",
                    f"{base['length']:g} x {base['width']:g}",
                    f"{base['height']:g}",
                    "-",
                ],
            )
        )
    for pad in parameters.get("pads", []):
        rows.append(
            (
                "PAD",
                [
                    pad["name"],
                    pad["support"] or "XY / Z=0",
                    f"{pad['x']:g}",
                    f"{pad['y']:g}",
                    f"D{2 * pad['radius']:g}",
                    f"{pad['height']:g}",
                    "-",
                ],
            )
        )
    for hole in parameters.get("cuts", []):
        cb = (
            "-"
            if hole["cb_radius"] is None
            else f"D{2 * hole['cb_radius']:g} / {hole['cb_depth']:g}"
        )
        rows.append(
            (
                "HOLE",
                [
                    hole["name"],
                    hole["support"],
                    f"{hole['x']:g}",
                    f"{hole['y']:g}",
                    f"D{2 * hole['radius']:g}",
                    f"{hole['depth']:g}",
                    cb,
                ],
            )
        )
    pages = []
    columns = [23, 46, 87, 126, 154, 179, 237, 281, 407]
    for start in range(0, len(rows), 18):
        page = 2 + start // 18
        body = frame(identity, title, page, total, "NTS (schedule)")
        body += text(24, 22, "FEATURE AND DRILLING SCHEDULE", 4.2)
        body += text(
            24,
            30,
            "Coordinates in the global model XY system. Diameters and depths in mm.",
            3,
        )
        body += text(
            24,
            37,
            "PAD: height upward from support. HOLE: depth downward from support top.",
            3,
        )
        body += text(
            24, 44, "CB: counterbore diameter / depth from the same support top.", 3
        )
        headers = [
            "TYPE",
            "FEATURE",
            "SUPPORT",
            "X",
            "Y",
            "SIZE",
            "H / DEPTH",
            "CB (D / DEPTH)",
        ]
        for i, label in enumerate(headers):
            body += text(columns[i] + 1, 57, label, 2.7)
        count = min(18, len(rows) - start)
        for i in range(count + 2):
            body += line(columns[0], 50 + i * 9, columns[-1], 50 + i * 9)
        for x in columns:
            body += line(x, 50, x, 50 + (count + 1) * 9)
        for index, (kind, values) in enumerate(rows[start : start + 18]):
            for col, value in enumerate([kind] + values):
                body += text(columns[col] + 1, 66 + index * 9, value, 2.9)
        body += text(
            24,
            231,
            "Do not merge holes with different depths into a single through cut.",
            3.1,
        )
        body += text(
            24,
            238,
            "All listed rows are required. Names identify feature supports, not operation order.",
            3.1,
        )
        body += text(
            24,
            245,
            "No fit, surface finish, thread or material requirements are inferred from nominal geometry.",
            2.8,
        )
        pages.append(document(body))
    return pages


def _make_sheets(native, parameters, identity, title):
    rows = (
        int("base" in parameters)
        + len(parameters.get("pads", []))
        + len(parameters.get("cuts", []))
    )
    total = 1 + math.ceil(rows / 18)
    return [overview(native, identity, title, total)] + schedules(
        parameters, identity, title, total
    )


def make_sheets(
    native, parameters, identity, title, *, standard="nominal", metadata=None
):
    import copy
    import xml.etree.ElementTree as ET

    profile = get_profile(standard)
    values = validate_metadata(metadata or {})
    token = ACTIVE.set(profile)
    meta_token = METADATA.set(values)
    try:
        if profile:
            native = copy.deepcopy(native)
            for view in native["views"].values():
                wrapper = ET.fromstring("<svg>" + view["svg"] + "</svg>")
                for group in wrapper:
                    weight = (
                        profile.thin if group.get("stroke-dasharray") else profile.wide
                    )
                    group.set("stroke-width", str(weight / native["scale"]))
                view["svg"] = "".join(
                    ET.tostring(group, encoding="unicode") for group in wrapper
                )
        return _make_sheets(native, parameters, identity, title)
    finally:
        METADATA.reset(meta_token)
        ACTIVE.reset(token)
