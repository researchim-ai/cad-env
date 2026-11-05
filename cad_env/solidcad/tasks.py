"""Original native feature probes; not a replacement for a held-out benchmark."""

import hashlib
import json
import math
import random

from .schema import validate

FAMILIES = (
    "polygon_bracket",
    "arc_plate",
    "revolved_bushing",
    "rounded_block",
    "chamfered_block",
    "open_box",
    "two_bodies",
    "boolean_bracket",
    "fused_blocks",
)


def make_task(family, seed):
    if family not in FAMILIES:
        raise ValueError("Unknown solid family")
    rng = random.Random(f"solidcad-v1:{family}:{seed}")
    l = rng.randint(28, 45)
    w = rng.randint(20, 30)
    h = rng.randint(8, 14)
    program = [{"op": "box", "name": "Base", "length": l, "width": w, "height": h}]
    expected = None
    description = ""
    targets = ["Result"]
    if family == "polygon_bracket":
        t = rng.randint(4, 7)
        points = [[0, 0], [l, 0], [l, t], [t, t], [t, w], [0, w]]
        program = [
            {"op": "polygon_extrude", "name": "Result", "points": points, "height": h}
        ]
        expected = t * (l + w - t) * h
        description = f"Extrude the closed L profile {points} by {h} mm."
    elif family == "arc_plate":
        r = rng.randint(5, 9)
        segments = [
            {"type": "line", "to": [l, 0]},
            {"type": "arc", "mid": [l + r, r], "to": [l, 2 * r]},
            {"type": "line", "to": [0, 2 * r]},
            {"type": "line", "to": [0, 0]},
        ]
        program = [
            {
                "op": "arc_extrude",
                "name": "Result",
                "start": [0, 0],
                "segments": segments,
                "height": h,
            }
        ]
        expected = (l * 2 * r + math.pi * r * r / 2) * h
        description = f"Create a {l} by {2 * r} mm rectangular plate with a semicircular right end of radius {r}, extruded {h} mm. Left edge X=0, bottom Y=0."
    elif family == "revolved_bushing":
        inner = rng.randint(3, 5)
        outer = inner + rng.randint(4, 8)
        points = [[inner, 0], [outer, 0], [outer, h], [inner, h]]
        program = [
            {"op": "revolve_profile", "name": "Result", "points": points, "angle": 360}
        ]
        expected = math.pi * (outer * outer - inner * inner) * h
        description = (
            f"Revolve the radial/Z profile {points} through 360 degrees about Z."
        )
    elif family in {"rounded_block", "chamfered_block"}:
        size = rng.choice([1.0, 1.5, 2.0])
        op = "fillet" if family == "rounded_block" else "chamfer"
        program.append(
            {
                "op": op,
                "name": "Result",
                "source": "Base",
                ("radius" if op == "fillet" else "size"): size,
            }
        )
        description = f"Create a {l} x {w} x {h} mm block at origin, then {op} every edge by {size} mm."
    elif family == "open_box":
        t = rng.choice([1.0, 2.0])
        program.append(
            {"op": "shell", "name": "Result", "source": "Base", "thickness": t}
        )
        expected = l * w * h - (l - 2 * t) * (w - 2 * t) * (h - t)
        description = f"Create a {l} x {w} x {h} mm open-top box with inward wall/bottom thickness {t} mm; preserve outside dimensions."
    elif family == "two_bodies":
        r = rng.randint(3, 6)
        gap = 3 * r + 4
        program = [
            {"op": "cylinder", "name": "First", "radius": r, "height": h},
            {
                "op": "cylinder",
                "name": "Second",
                "radius": r,
                "height": h,
                "origin": [gap, 0, 0],
            },
        ]
        targets = ["First", "Second"]
        expected = 2 * math.pi * r * r * h
        description = f"Keep two separate cylinders, radius {r} mm, height {h} mm, origins [0,0,0] and [{gap},0,0]. Do not fuse them."
    elif family == "boolean_bracket":
        r = 3
        program.extend(
            [
                {
                    "op": "cylinder",
                    "name": "Tool",
                    "radius": r,
                    "height": h,
                    "origin": [l / 2, w / 2, 0],
                },
                {"op": "cut", "name": "Result", "left": "Base", "right": "Tool"},
            ]
        )
        expected = l * w * h - math.pi * r * r * h
        description = f"Create a {l} x {w} x {h} mm block and cut a centered vertical through bore of diameter {2 * r} mm."
    elif family == "fused_blocks":
        program.extend(
            [
                {
                    "op": "box",
                    "name": "Other",
                    "length": l,
                    "width": w,
                    "height": h,
                    "origin": [l / 2, 0, 0],
                },
                {"op": "fuse", "name": "Result", "left": "Base", "right": "Other"},
            ]
        )
        expected = 1.5 * l * w * h
        description = f"Fuse equal {l} x {w} x {h} mm blocks placed at origins [0,0,0] and [{l / 2},0,0]. Produce one solid."
    program.append({"op": "finish", "targets": targets})
    program = validate(program)
    key = hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
    bucket = int(key[:8], 16) % 10
    return {
        "schema_version": "solidcad.v1",
        "family": family,
        "seed": seed,
        "instruction": description,
        "units": "mm",
        "program": program,
        "expected_volume": expected,
        "split_group": key,
        "split": "train" if bucket < 7 else ("validation" if bucket < 9 else "test"),
        "base_volume": l * w * h
        if family in {"rounded_block", "chamfered_block", "open_box"}
        else None,
    }
