"""Versioned, deliberately small action language for real FreeCAD rollouts."""

import math
import re
from typing import Any

SCHEMA_VERSION = "cad-env.v1"
MAX_ACTIONS = 64
TOOLS = {
    "create_body": {"required": ["name"], "optional": []},
    "sketch_rectangle": {
        "required": ["name", "body", "length", "width"],
        "optional": [],
    },
    "sketch_circles": {
        "required": ["name", "body", "circles"],
        "optional": ["support"],
    },
    "pad": {"required": ["name", "body", "profile", "length"], "optional": []},
    "pocket": {"required": ["name", "body", "profile", "length"], "optional": []},
    "set_parameter": {"required": ["target", "parameter", "value"], "optional": []},
    "finish": {"required": [], "optional": []},
}
ACTION_GUIDE = """Return ONLY a JSON array of actions (or {"actions": [...]}).
All lengths are mm. Names are stable identifiers, not labels. Supported actions:
{"op":"create_body","name":"Body"}
{"op":"sketch_rectangle","name":"BaseSketch","body":"Body","length":40,"width":30}
{"op":"sketch_circles","name":"HoleSketch","body":"Body","support":"Pad","circles":[{"x":20,"y":15,"radius":3}]}
{"op":"pad","name":"Pad","body":"Body","profile":"BaseSketch","length":6}
{"op":"pocket","name":"Hole","body":"Body","profile":"HoleSketch","length":6}
{"op":"set_parameter","target":"HoleSketch","parameter":"diameter_0","value":8}
{"op":"finish"}
Rectangles start at (0,0) on XY. Circle sketches without support are on XY.
A supported circle sketch is on the top of the named pad, linked by an expression.
Circle sketch named dimensions: diameter_0, diameter_1, ... .
Rectangle named dimensions: length, width. Pad/pocket editable parameter: length.
Use explicit body/profile names. Finish only after the requested solid is complete."""


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]{0,47}", value
    ):
        raise ValueError("Object names must match [A-Za-z][A-Za-z0-9_]{0,47}")
    return value


def number(value: Any, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric value")
    value = float(value)
    if not math.isfinite(value) or abs(value) > 10000 or (positive and value <= 0):
        raise ValueError("Dimension out of range (positive, finite, at most 10000 mm)")
    return value


def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(action, dict)
        or not isinstance(action.get("op"), str)
        or action["op"] not in TOOLS
    ):
        raise ValueError("Unknown CAD action")
    spec = TOOLS[action["op"]]
    allowed = {"op", *spec["required"], *spec["optional"]}
    if set(action) - allowed or any(k not in action for k in spec["required"]):
        raise ValueError("Missing or unexpected action arguments")
    result = dict(action)
    for key in ("name", "body", "profile", "support", "target"):
        if key in result:
            identifier(result[key])
    for key in ("length", "width", "value"):
        if key in result:
            result[key] = number(result[key])
    if "parameter" in result:
        if not isinstance(result["parameter"], str) or not re.fullmatch(
            r"length|width|diameter_[0-9]+", result["parameter"]
        ):
            raise ValueError("Unsupported editable parameter")
    if "circles" in result:
        circles = result["circles"]
        if not isinstance(circles, list) or not 1 <= len(circles) <= 16:
            raise ValueError("A circle sketch needs 1..16 circles")
        result["circles"] = []
        for circle in circles:
            if not isinstance(circle, dict) or set(circle) != {"x", "y", "radius"}:
                raise ValueError("Circle requires x, y, radius")
            result["circles"].append(
                {
                    "x": number(circle["x"], False),
                    "y": number(circle["y"], False),
                    "radius": number(circle["radius"]),
                }
            )
    return result


def validate_program(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
        raise ValueError("Program must be a list with at most 64 actions")
    result = [validate_action(a) for a in actions]
    if any(a["op"] == "finish" for a in result[:-1]):
        raise ValueError("finish must be the final action")
    return result
