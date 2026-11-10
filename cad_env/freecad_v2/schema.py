"""Bounded JSON features, explicit planes and references. No arbitrary Python."""

import copy
import math

from cad_env.agent.schema import identifier, number

MAX_ACTIONS = 128
TOOLS = {
    "sketch": ({"name", "profile"}, {"origin", "normal", "x_axis"}),
    "pad": ({"name", "profile", "length"}, set()),
    "pocket": ({"name", "source", "profile", "length"}, set()),
    "revolve": ({"name", "profile", "axis", "angle"}, {"origin"}),
    "loft": ({"name", "profiles"}, {"ruled"}),
    "sweep": ({"name", "profile", "path"}, {"frenet"}),
    "fillet": ({"name", "source", "radius", "edges"}, set()),
    "chamfer": ({"name", "source", "size", "edges"}, set()),
    "shell": ({"name", "source", "thickness", "faces"}, set()),
    "fuse": ({"name", "left", "right"}, set()),
    "cut": ({"name", "left", "right"}, set()),
    "common": ({"name", "left", "right"}, set()),
    "translate": ({"name", "source", "vector"}, set()),
    "rotate": ({"name", "source", "axis", "angle"}, {"origin"}),
    "mirror": ({"name", "source", "normal"}, {"origin"}),
    "linear_pattern": ({"name", "source", "vector", "count"}, set()),
    "polar_pattern": ({"name", "source", "axis", "count", "angle"}, {"origin"}),
    "set_parameter": ({"target", "parameter", "value"}, set()),
    "inspect": ({"target"}, {"offset", "limit"}),
    "finish": ({"targets"}, set()),
}


def vector(v, dimension=3, nonzero=False):
    if not isinstance(v, list) or len(v) != dimension:
        raise ValueError("Invalid coordinate vector")
    result = [number(x, False) for x in v]
    if nonzero and math.sqrt(sum(x * x for x in result)) < 1e-8:
        raise ValueError("Zero direction")
    return result


def profile(p):
    if not isinstance(p, dict):
        raise ValueError("Invalid profile object")  # noqa: TRY004 - JSON validation errors share one type
    kind = p.get("type")
    keys = {
        "circle": {"type", "radius"},
        "rectangle": {"type", "length", "width"},
        "polygon": {"type", "points"},
        "segments": {"type", "start", "segments"},
    }
    if not isinstance(kind, str) or kind not in keys or set(p) != keys[kind]:
        raise ValueError("Unknown profile or unexpected fields")
    if kind in ("circle", "rectangle"):
        for key in set(p) - {"type"}:
            p[key] = number(p[key])
    elif kind == "polygon":
        if not isinstance(p["points"], list) or not 3 <= len(p["points"]) <= 64:
            raise ValueError("Polygon requires 3..64 points")
        p["points"] = [vector(v, 2) for v in p["points"]]
        if len({tuple(v) for v in p["points"]}) != len(p["points"]):
            raise ValueError("Repeated polygon vertex; closure is implicit")
    else:
        p["start"] = vector(p["start"], 2)
        if not isinstance(p["segments"], list) or not 2 <= len(p["segments"]) <= 64:
            raise ValueError("Expected 2..64 line/arc segments")
        for s in p["segments"]:
            if not isinstance(s, dict) or s.get("type") not in ("line", "arc"):
                raise ValueError("Unknown segment")
            if set(s) != (
                {"type", "to", "mid"} if s["type"] == "arc" else {"type", "to"}
            ):
                raise ValueError("Unexpected segment fields")
            s["to"] = vector(s["to"], 2)
            if "mid" in s:
                s["mid"] = vector(s["mid"], 2)
        if p["segments"][-1]["to"] != p["start"]:
            raise ValueError("Segment profile must close explicitly")
    return p


EDITABLE = {
    "pad": {"length"},
    "pocket": {"length"},
    "revolve": {"angle"},
    "fillet": {"radius"},
    "chamfer": {"size"},
    "shell": {"thickness"},
    "rotate": {"angle"},
    "polar_pattern": {"angle"},
}


def validate(actions, *, terminal=True):
    if not isinstance(actions, list) or not 0 < len(actions) <= MAX_ACTIONS:
        raise ValueError("Expected 1..128 actions")
    result, types, definitions = [], {}, {}
    for index, raw in enumerate(actions):
        a = copy.deepcopy(raw)
        if (
            not isinstance(a, dict)
            or not isinstance(a.get("op"), str)
            or a["op"] not in TOOLS
        ):
            raise ValueError("Unknown v2 action")
        op = a["op"]
        required, optional = TOOLS[op]
        if not required <= a.keys() or set(a) - {"op", *required, *optional}:
            raise ValueError("Missing or unexpected arguments")
        if op == "finish":
            targets = a["targets"]
            if (
                index != len(actions) - 1
                or not isinstance(targets, list)
                or not 1 <= len(targets) <= 32
            ):
                raise ValueError("finish must be last with 1..32 solid targets")
            if any(
                not isinstance(n, str) or types.get(n) != "solid" for n in targets
            ) or len(set(targets)) != len(targets):
                raise ValueError("Unknown, repeated or nonsolid finish target")
        elif op in ("set_parameter", "inspect"):
            target = a["target"]
            if not isinstance(target, str) or target not in definitions:
                raise ValueError("Unknown parameter target")
            if op == "inspect":
                for key, default, lo, hi in (
                    ("offset", 0, 0, 4096),
                    ("limit", 32, 1, 64),
                ):
                    value = a.get(key, default)
                    if type(value) is not int or not lo <= value <= hi:
                        raise ValueError("Invalid topology page")
                result.append(a)
                continue
            definition = definitions[target]
            allowed = EDITABLE.get(definition["op"], set())
            if definition["op"] == "sketch":
                allowed = {"circle": {"radius"}, "rectangle": {"length", "width"}}.get(
                    definition["profile"]["type"], set()
                )
            if not isinstance(a["parameter"], str) or a["parameter"] not in allowed:
                raise ValueError("Unsupported parameter for target")
            a["value"] = number(a["value"])
            if a["parameter"] == "angle" and a["value"] > 360:
                raise ValueError("Angle exceeds 360 degrees")
        else:
            identifier(a["name"])
            if a["name"] in types:
                raise ValueError("Duplicate object name")
            for key in ("source", "left", "right"):
                if key in a and (
                    not isinstance(a[key], str) or types.get(a[key]) != "solid"
                ):
                    raise ValueError("Unknown solid dependency")
            if op == "sketch":
                a["profile"] = profile(a["profile"])
            elif "profile" in a and (
                not isinstance(a["profile"], str)
                or types.get(a["profile"]) != "profile"
            ):
                raise ValueError("Unknown sketch profile")
            if op == "loft":
                if (
                    not isinstance(a["profiles"], list)
                    or not 2 <= len(a["profiles"]) <= 16
                ):
                    raise ValueError("Loft requires 2..16 profiles")
                if any(
                    not isinstance(n, str) or types.get(n) != "profile"
                    for n in a["profiles"]
                ):
                    raise ValueError("Unknown loft profile")
                if len(set(a["profiles"])) != len(a["profiles"]):
                    raise ValueError("Repeated loft profile")
            for key in ("origin", "normal", "axis", "x_axis", "vector"):
                if key in a:
                    a[key] = vector(a[key], nonzero=key != "origin")
            if "path" in a:
                if not isinstance(a["path"], list) or not 2 <= len(a["path"]) <= 64:
                    raise ValueError("Sweep needs 2..64 path points")
                a["path"] = [vector(v) for v in a["path"]]
                if any(p == q for p, q in zip(a["path"], a["path"][1:])):
                    raise ValueError("Zero length path segment")
            for key in ("length", "radius", "size", "thickness", "angle"):
                if key in a:
                    a[key] = number(a[key])
            if "angle" in a and a["angle"] > 360:
                raise ValueError("Angle exceeds 360 degrees")
            if "count" in a and (
                type(a["count"]) is not int or not 2 <= a["count"] <= 32
            ):
                raise ValueError("Pattern count must be 2..32")
            for key in ("edges", "faces"):
                if key in a:
                    v = a[key]
                    if (
                        not isinstance(v, list)
                        or not 1 <= len(v) <= 128
                        or any(type(i) is not int or i < 1 for i in v)
                        or len(set(v)) != len(v)
                    ):
                        raise ValueError(
                            "Expected unique positive 1-based topology indices"
                        )
            for key in ("ruled", "frenet"):
                if key in a and type(a[key]) is not bool:
                    raise ValueError("Expected boolean")
            types[a["name"]] = "profile" if op == "sketch" else "solid"
            definitions[a["name"]] = a
        result.append(a)
    if terminal and result[-1]["op"] != "finish":
        raise ValueError("Explicit finish required")
    return result
