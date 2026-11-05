"""Bounded solid graph: profiles, revolves, dress-up, shells and multiple bodies."""

import copy
import math

from cad_env.agent.schema import identifier, number

VERSION = "solidcad.v1"
TOOLS = {
    "box": ({"name", "length", "width", "height"}, {"origin"}),
    "cylinder": ({"name", "radius", "height"}, {"origin"}),
    "polygon_extrude": ({"name", "points", "height"}, set()),
    "arc_extrude": ({"name", "start", "segments", "height"}, set()),
    "revolve_profile": ({"name", "points", "angle"}, set()),
    "fillet": ({"name", "source", "radius"}, set()),
    "chamfer": ({"name", "source", "size"}, set()),
    "shell": ({"name", "source", "thickness"}, set()),
    "fuse": ({"name", "left", "right"}, set()),
    "cut": ({"name", "left", "right"}, set()),
    "finish": ({"targets"}, set()),
}


def point(value, n=2):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("Invalid coordinate tuple")
    return [number(x, False) for x in value]


def validate(program):
    if not isinstance(program, list) or not 1 <= len(program) <= 64:
        raise ValueError("Expected 1..64 solid actions")
    objects = set()
    result = []
    for i, raw in enumerate(program):
        if not isinstance(raw, dict) or raw.get("op") not in TOOLS:
            raise ValueError("Unknown solid tool")
        a = copy.deepcopy(raw)
        required, optional = TOOLS[a["op"]]
        if set(a) != {"op", *required, *optional.intersection(a)}:
            raise ValueError("Missing/unexpected solid arguments")
        if a["op"] == "finish":
            if (
                i != len(program) - 1
                or not isinstance(a["targets"], list)
                or not 1 <= len(a["targets"]) <= 8
                or len(set(a["targets"])) != len(a["targets"])
            ):
                raise ValueError("Finish requires 1..8 unique final objects")
            if any(t not in objects for t in a["targets"]):
                raise ValueError("Unknown final object")
        else:
            identifier(a["name"])
            if a["name"] in objects:
                raise ValueError("Duplicate solid name")
            for k in ["source", "left", "right"]:
                if k in a and a[k] not in objects:
                    raise ValueError("Unknown solid dependency")
            for k in [
                "length",
                "width",
                "height",
                "radius",
                "size",
                "thickness",
                "angle",
            ]:
                if k in a:
                    a[k] = number(a[k])
            if "angle" in a and a["angle"] > 360:
                raise ValueError("Revolve angle exceeds 360")
            if "origin" in a:
                a["origin"] = point(a["origin"], 3)
            if "points" in a:
                if not isinstance(a["points"], list) or not 3 <= len(a["points"]) <= 32:
                    raise ValueError("Profile requires 3..32 vertices")
                a["points"] = [point(p) for p in a["points"]]
                if a["op"] == "revolve_profile" and any(p[0] < 0 for p in a["points"]):
                    raise ValueError("Radial coordinates must be nonnegative")
            if a["op"] == "arc_extrude":
                a["start"] = point(a["start"])
                if (
                    not isinstance(a["segments"], list)
                    or not 2 <= len(a["segments"]) <= 16
                ):
                    raise ValueError("Profile requires 2..16 segments")
                for segment in a["segments"]:
                    if not isinstance(segment, dict) or segment.get("type") not in {
                        "line",
                        "arc",
                    }:
                        raise ValueError("Unknown profile segment")
                    keys = (
                        {"type", "to", "mid"}
                        if segment["type"] == "arc"
                        else {"type", "to"}
                    )
                    if set(segment) != keys:
                        raise ValueError("Incorrect arc/line arguments")
                    segment["to"] = point(segment["to"])
                    if "mid" in segment:
                        segment["mid"] = point(segment["mid"])
                if a["segments"][-1]["to"] != a["start"]:
                    raise ValueError("Profile must close explicitly")
            objects.add(a["name"])
        result.append(a)
    if result[-1]["op"] != "finish":
        raise ValueError("Solid programs must finish explicitly")
    return result


def arc_points(start, mid, end, count=128):
    ax, ay = start
    bx, by = mid
    cx, cy = end
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        raise ValueError("Arc points must be non-collinear")
    u = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    v = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    radius = math.hypot(ax - u, ay - v)
    if radius > 10000:
        raise ValueError("Arc radius out of range")
    a = math.atan2(ay - v, ax - u)
    b = math.atan2(by - v, bx - u)
    c = math.atan2(cy - v, cx - u)
    sweep = (c - a) % (2 * math.pi)
    if (b - a) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    return [
        [
            u + radius * math.cos(a + sweep * i / count),
            v + radius * math.sin(a + sweep * i / count),
        ]
        for i in range(1, count + 1)
    ]
