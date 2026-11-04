"""Deterministic procedural teachers. Public instructions and private references."""

import copy
import hashlib
import json
import random
from typing import Any

from .schema import ACTION_GUIDE, SCHEMA_VERSION, validate_program

FAMILIES = ("plate", "bushing", "flange", "stepped_shaft")
MODES = ("create", "edit", "repair")


def program(family: str, p: dict[str, Any]):
    a = [{"op": "create_body", "name": "Body"}]
    if family == "plate":
        a += [
            {
                "op": "sketch_rectangle",
                "name": "BaseSketch",
                "body": "Body",
                "length": p["length"],
                "width": p["width"],
            }
        ]
    else:
        a += [
            {
                "op": "sketch_circles",
                "name": "BaseSketch",
                "body": "Body",
                "circles": [{"x": 0, "y": 0, "radius": p["radius"]}],
            }
        ]
    a += [
        {
            "op": "pad",
            "name": "BasePad",
            "body": "Body",
            "profile": "BaseSketch",
            "length": p["height"],
        }
    ]
    if family == "stepped_shaft":
        a += [
            {
                "op": "sketch_circles",
                "name": "StepSketch",
                "body": "Body",
                "support": "BasePad",
                "circles": [{"x": 0, "y": 0, "radius": p["step_radius"]}],
            },
            {
                "op": "pad",
                "name": "StepPad",
                "body": "Body",
                "profile": "StepSketch",
                "length": p["step_height"],
            },
        ]
    else:
        x, y = (p["length"] / 2, p["width"] / 2) if family == "plate" else (0, 0)
        circles = [{"x": x, "y": y, "radius": p["hole_radius"]}]
        if family == "flange":
            # Cardinal positions keep exact dimensions and reproducible references.
            r = p["bolt_circle_radius"]
            circles += [
                {"x": r, "y": 0, "radius": p["bolt_radius"]},
                {"x": 0, "y": r, "radius": p["bolt_radius"]},
                {"x": -r, "y": 0, "radius": p["bolt_radius"]},
                {"x": 0, "y": -r, "radius": p["bolt_radius"]},
            ]
        a += [
            {
                "op": "sketch_circles",
                "name": "HoleSketch",
                "body": "Body",
                "support": "BasePad",
                "circles": circles,
            },
            {
                "op": "pocket",
                "name": "HolePocket",
                "body": "Body",
                "profile": "HoleSketch",
                "length": p["height"],
            },
        ]
    return validate_program(a + [{"op": "finish"}])


def make_task(
    family: str = "plate", seed: int = 0, mode: str = "create"
) -> dict[str, Any]:
    if family not in FAMILIES or mode not in MODES:
        raise ValueError("Unsupported family or task mode")
    rng = random.Random(f"{family}:{seed}")
    p = {
        "height": rng.randint(4, 12),
        "radius": rng.randint(20, 32),
        "length": rng.randint(40, 70),
        "width": rng.randint(30, 50),
        "hole_radius": rng.randint(3, 6),
        "bolt_radius": 2,
        "step_radius": rng.randint(8, 14),
        "step_height": rng.randint(8, 20),
    }
    p["bolt_circle_radius"] = round(p["radius"] * 0.7, 2)
    reference = program(family, p)
    initial = []
    if family == "plate":
        description = f"Create a rectangular plate {p['length']} x {p['width']} x {p['height']} mm, with one centered through hole of diameter {2 * p['hole_radius']} mm."
    elif family == "bushing":
        description = f"Create a cylindrical bushing: outer diameter {2 * p['radius']} mm, height {p['height']} mm, coaxial through bore diameter {2 * p['hole_radius']} mm."
    elif family == "flange":
        description = f"Create a circular flange: outer diameter {2 * p['radius']} mm, thickness {p['height']} mm, central bore diameter {2 * p['hole_radius']} mm. Add four through holes diameter {2 * p['bolt_radius']} mm at the cardinal points of a bolt circle diameter {2 * p['bolt_circle_radius']} mm."
    else:
        description = f"Create a coaxial stepped shaft: base diameter {2 * p['radius']} mm and height {p['height']} mm; upper step diameter {2 * p['step_radius']} mm and height {p['step_height']} mm."
    if mode != "create":
        initial_p = copy.deepcopy(p)
        parameter = "step_height" if family == "stepped_shaft" else "hole_radius"
        initial_p[parameter] += 2
        initial = program(family, initial_p)[:-1]
        target = "StepPad" if family == "stepped_shaft" else "HoleSketch"
        field = "length" if family == "stepped_shaft" else "diameter_0"
        value = p[parameter] if family == "stepped_shaft" else p[parameter] * 2
        oracle = [
            {
                "op": "set_parameter",
                "target": target,
                "parameter": field,
                "value": value,
            },
            {"op": "finish"},
        ]
        prefix = "Edit" if mode == "edit" else "Repair the incorrect dimension in"
        description = f"{prefix} the existing {family}. Set {target}.{field} to {value} mm. Preserve every other dimension and the existing feature history."
    else:
        oracle = reference
    # Group all hole/step-height variations of a base design together. This also
    # keeps coincident geometries from different seeds in the same split.
    base_keys = {
        "plate": ("length", "width", "height"),
        "bushing": ("radius", "height"),
        "flange": ("radius", "height", "bolt_circle_radius", "bolt_radius"),
        "stepped_shaft": ("radius", "height", "step_radius"),
    }[family]
    identity = json.dumps([family, {k: p[k] for k in base_keys}], sort_keys=True)
    group = hashlib.sha256(identity.encode()).hexdigest()[:16]
    bucket = int(group[:8], 16) % 100
    split = "train" if bucket < 80 else ("validation" if bucket < 90 else "test")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"{family}-{seed:06d}-{mode}",
        "family": family,
        "seed": seed,
        "mode": mode,
        "instruction": description,
        "parameters": p,
        "initial_actions": initial,
        "reference_actions": reference,
        "oracle_actions": oracle,
        "split_group": group,
        "split": split,
        "max_steps": 32,
        "tolerance": 1e-5,
        "require_parametric": True,
    }


def public_task(task):
    if "public_task" in task:
        allowed = ("task_id", "family", "mode", "instruction", "max_steps")
        result = {k: task["public_task"][k] for k in allowed}
        result["max_steps"] = task["max_steps"]
        return result
    return {
        k: task[k] for k in ("task_id", "family", "mode", "instruction", "max_steps")
    }


def prompt_messages(task, observation=None):
    # Initial recipe is legitimate context for edit/repair; the target recipe stays private.
    context = {"task": public_task(task), "initial_actions": task["initial_actions"]}
    if observation is not None:
        context["observation"] = observation
    return [
        {
            "role": "system",
            "content": "You are a FreeCAD agent. Build one editable PartDesign solid.\n"
            + ACTION_GUIDE,
        },
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def expected_metrics(task):
    """Independent closed-form checks, not extracted from the oracle CAD shape."""
    import math

    if "analytic_metrics" in task:
        return copy.deepcopy(task["analytic_metrics"])
    p, family = task["parameters"], task["family"]
    h = p["height"]
    if family == "plate":
        volume = (p["length"] * p["width"] - math.pi * p["hole_radius"] ** 2) * h
        bounds = [0, 0, 0, p["length"], p["width"], h]
    else:
        r = p["radius"]
        volume = math.pi * r**2 * h
        zmax = h
        if family in ("bushing", "flange"):
            volume -= math.pi * p["hole_radius"] ** 2 * h
        if family == "flange":
            volume -= 4 * math.pi * p["bolt_radius"] ** 2 * h
        if family == "stepped_shaft":
            volume += math.pi * p["step_radius"] ** 2 * p["step_height"]
            zmax += p["step_height"]
        bounds = [-r, -r, 0, r, r, zmax]
    return {"volume": volume, "bounds": bounds}


def allowed_edits(task):
    """Private edit contract; all other named dimensions/dependencies are protected."""
    if task["mode"] == "create":
        return None
    return [
        [a["target"], a["parameter"]]
        for a in task["oracle_actions"]
        if a["op"] == "set_parameter"
    ]
