"""Procedural mechanical families inspired by external evaluation axes.

These are original synthetic tasks, not copies or claimed CADBench coverage.
"""

import copy
import hashlib
import json
import math
import random

from cad_env.agent.schema import validate_program

from .plan import resolve

FAMILIES = ("bolt_flange", "counterbored_plate", "stepped_bore", "blind_pocket_grid")


def geometry_group(program):
    """Canonical grouping for one-base, commuting-cut mechanical programs."""
    geometry = []
    for feature in resolve(program):
        profile = feature["profile_spec"]
        if profile["op"] == "sketch_rectangle":
            shape = ["rectangle", profile["length"], profile["width"]]
        else:
            shape = [
                "circles",
                sorted(
                    [round(c["x"], 6), round(c["y"], 6), round(c["radius"], 6)]
                    for c in profile["circles"]
                ),
            ]
        geometry.append([feature["op"], shape, feature["z"], feature["length"]])
    return hashlib.sha256(
        json.dumps(
            sorted(geometry, key=lambda x: json.dumps(x)), sort_keys=True
        ).encode()
    ).hexdigest()


def make_task(family, seed, mode="create", modality="image", *, varied_edits=False):
    if (
        family not in FAMILIES
        or mode not in {"create", "edit", "repair"}
        or modality not in {"text", "image"}
    ):
        raise ValueError("Unknown family, mode or modality")
    rng = random.Random(f"multicad-v1:{family}:{seed}")
    height = rng.randint(8, 14)
    length = rng.randrange(72, 97, 4)
    width = rng.randrange(56, 77, 4)
    program = [{"op": "create_body", "name": "Body"}]
    lines = []
    if family == "bolt_flange":
        radius = rng.randint(32, 42)
        program.append(
            {
                "op": "sketch_circles",
                "name": "BaseSketch",
                "body": "Body",
                "circles": [{"x": 0, "y": 0, "radius": radius}],
            }
        )
        volume = math.pi * radius**2 * height
        bounds = [-radius, -radius, 0, radius, radius, height]
        lines.append(
            f"Circular flange: outer diameter {2 * radius}, thickness {height}. Center X=Y=0."
        )
    else:
        program.append(
            {
                "op": "sketch_rectangle",
                "name": "BaseSketch",
                "body": "Body",
                "length": length,
                "width": width,
            }
        )
        volume = length * width * height
        bounds = [0, 0, 0, length, width, height]
        lines.append(
            f"Rectangular plate: length {length}, width {width}, thickness {height}. Origin at lower-left."
        )
    program.append(
        {
            "op": "pad",
            "name": "BasePad",
            "body": "Body",
            "profile": "BaseSketch",
            "length": height,
        }
    )
    holes = []
    if family == "bolt_flange":
        n = rng.choice([4, 6, 8])
        pitch = round(radius * 0.68, 2)
        hole_radius = rng.choice([2, 3])
        holes = [
            (
                round(pitch * math.cos(2 * math.pi * i / n), 6),
                round(pitch * math.sin(2 * math.pi * i / n), 6),
                hole_radius,
                height,
            )
            for i in range(n)
        ]
        holes.insert(0, (0, 0, rng.randint(6, 10), height))
    elif family == "stepped_bore":
        r = rng.randint(4, 6)
        outer = r + 4
        holes = [
            (length / 2, width / 2, r, height),
            (length / 2, width / 2, outer, height / 3),
        ]
    else:
        positions = [(length * x / 4, width * y / 3) for x in (1, 2, 3) for y in (1, 2)]
        rng.shuffle(positions)
        for x, y in positions[: rng.randint(3, 6)]:
            r = rng.choice([2, 3])
            depth = (
                height
                if family == "counterbored_plate"
                else rng.choice([height / 3, height / 2])
            )
            holes.append((round(x, 6), round(y, 6), r, depth))
    removed = 0.0
    for i, (x, y, r, depth) in enumerate(holes):
        program.extend(
            [
                {
                    "op": "sketch_circles",
                    "name": f"HoleSketch{i}",
                    "body": "Body",
                    "support": "BasePad",
                    "circles": [{"x": x, "y": y, "radius": r}],
                },
                {
                    "op": "pocket",
                    "name": f"Hole{i}",
                    "body": "Body",
                    "profile": f"HoleSketch{i}",
                    "length": depth,
                },
            ]
        )
        lines.append(
            f"Hole {i + 1}: X={x:g}, Y={y:g}, diameter={2 * r:g}, depth from top={depth:g}."
        )
        # The second coaxial bore overlaps the first; subtract only its annulus.
        removed += (
            math.pi
            * (r * r - (holes[0][2] ** 2 if family == "stepped_bore" and i == 1 else 0))
            * depth
        )
        if family == "counterbored_plate":
            large = r + 2
            recess = height / 3
            program.extend(
                [
                    {
                        "op": "sketch_circles",
                        "name": f"RecessSketch{i}",
                        "body": "Body",
                        "support": "BasePad",
                        "circles": [{"x": x, "y": y, "radius": large}],
                    },
                    {
                        "op": "pocket",
                        "name": f"Recess{i}",
                        "body": "Body",
                        "profile": f"RecessSketch{i}",
                        "length": recess,
                    },
                ]
            )
            lines.append(
                f"Hole {i + 1} counterbore: diameter={2 * large:g}, depth={recess:g}."
            )
            removed += math.pi * (large * large - r * r) * recess
    program.append({"op": "finish"})
    program = validate_program(program)
    # Hash physical target, never engine or image modality, to keep variants together.
    key = geometry_group(program)
    bucket = int(key[:8], 16) % 10
    split = "train" if bucket < 7 else ("validation" if bucket < 9 else "test")
    initial = []
    teacher = program
    if mode != "create":
        initial = copy.deepcopy(program[:-1])
        delta = rng.choice([-4, -2, 2, 3, 5]) if varied_edits else 3
        initial[2]["length"] = height + delta
        teacher = [
            {
                "op": "set_parameter",
                "target": "BasePad",
                "parameter": "length",
                "value": height,
            },
            {"op": "finish"},
        ]
    instruction = (
        "Build the specified editable solid."
        if mode == "create"
        else (
            "Correct the erroneous base thickness."
            if mode == "repair"
            else "Change the base thickness to the requested value."
        )
    )
    if mode != "create":
        # Depths stay as specified when the support follows the modified base.
        instruction += " Preserve all sketch dimensions, hole depths and feature names."
    return {
        **({"curriculum_version": "varied-edits-v2"} if varied_edits else {}),
        "schema_version": "multicad.v1",
        "family": family,
        "seed": seed,
        "mode": mode,
        "input_modality": modality,
        "split_group": key,
        "split": split,
        "instruction": instruction,
        "specification": lines,
        "reference_actions": program,
        "initial_actions": initial,
        "oracle_actions": teacher,
        "analytic_metrics": {"volume": volume - removed, "bounds": bounds},
        "units": "mm",
        "complexity": {"holes": len(holes), "oracle_actions": len(teacher)},
        "max_steps": min(64, len(teacher) + 4),
    }
