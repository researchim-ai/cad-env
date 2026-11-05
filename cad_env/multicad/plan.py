"""Resolve the shared action language without executing user Python.

Parameter edits rebuild the feature graph, including dependent support planes.
The portable subset is one body, XY rectangles and non-overlapping circles.
"""

import copy
import math

from cad_env.agent.schema import validate_program


def resolve(actions, *, allow_empty=False, return_objects=False):
    definitions, order, body = {}, [], None
    for a in validate_program(actions):
        op = a["op"]
        if op == "finish":
            continue
        if op == "set_parameter":
            if a["target"] not in definitions:
                raise ValueError("Parameter target does not exist")
            target = definitions[a["target"]]
            parameter = a["parameter"]
            if parameter.startswith("diameter_") and target["op"] == "sketch_circles":
                index = int(parameter[9:])
                if index >= len(target["circles"]):
                    raise ValueError("Circle index out of range")
                target["circles"][index]["radius"] = a["value"] / 2
            elif parameter in {"length", "width"} and parameter in target:
                target[parameter] = a["value"]
            else:
                raise ValueError("Parameter is not editable on this object")
            continue
        name = a["name"]
        if name in definitions:
            raise ValueError("Duplicate object name")
        if op == "create_body":
            if body is not None:
                raise ValueError("Portable contract supports exactly one body")
            body = name
        elif body is None or a["body"] != body:
            raise ValueError("Unknown body")
        if "support" in a:
            s = definitions.get(a["support"], {})
            if s.get("op") != "pad":
                raise ValueError("Support must be a preceding pad")
        if op in {"pad", "pocket"}:
            p = definitions.get(a["profile"], {})
            if p.get("op") not in {"sketch_rectangle", "sketch_circles"}:
                raise ValueError("Profile must be a preceding sketch")
        definitions[name] = copy.deepcopy(a)
        order.append(name)
    features, tops = [], {}
    for name in order:
        a = definitions[name]
        if a["op"] not in {"pad", "pocket"}:
            continue
        profile = copy.deepcopy(definitions[a["profile"]])
        z = tops[profile["support"]] if "support" in profile else 0
        if profile["op"] == "sketch_circles":
            circles = profile["circles"]
            for i, c in enumerate(circles):
                for other in circles[:i]:
                    if (
                        math.hypot(c["x"] - other["x"], c["y"] - other["y"])
                        <= c["radius"] + other["radius"]
                    ):
                        raise ValueError(
                            "Portable circle profiles must be disjoint; use a separate pocket for bores"
                        )
        features.append({**a, "profile_spec": profile, "z": z})
        tops[name] = z + a["length"] if a["op"] == "pad" else z
    if (not features and not allow_empty) or (features and features[0]["op"] != "pad"):
        raise ValueError("A solid must start with a pad")
    return (features, list(definitions.values())) if return_objects else features
