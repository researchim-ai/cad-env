"""Seeded original operation probes with independent analytic volume checks."""

import hashlib
import json
import math
import random

from .schema import validate

FAMILIES = (
    "tilted_pad",
    "polygon_pad",
    "arc_pad",
    "pocket",
    "revolve",
    "loft",
    "sweep",
    "fillet",
    "chamfer",
    "shell",
    "fuse",
    "cut",
    "common",
    "translate",
    "rotate",
    "mirror",
    "linear_pattern",
    "polar_pattern",
    "parameter_edit",
)


def make_task(family, seed):
    if family not in FAMILIES:
        raise ValueError("Unknown advanced family")
    rng = random.Random(f"freecad.v2:{family}:{seed}")
    l, w, h = [
        round(rng.uniform(lo, hi), 2) for lo, hi in [(25, 40), (15, 24), (8, 14)]
    ]
    sk = {
        "op": "sketch",
        "name": "Profile",
        "profile": {"type": "rectangle", "length": l, "width": w},
    }
    pad = {"op": "pad", "name": "Base", "profile": "Profile", "length": h}
    a, volume, target = [sk, pad], l * w * h, "Base"
    if family == "tilted_pad":
        sk.update(origin=[3, -4, 2], normal=[1, 1, 1], x_axis=[1, -1, 0])
    elif family == "polygon_pad":
        sk["profile"] = {
            "type": "polygon",
            "points": [[0, 0], [l, 0], [l, 4], [4, 4], [4, w], [0, w]],
        }
        volume = 4 * (l + w - 4) * h
    elif family == "arc_pad":
        sk["profile"] = {
            "type": "segments",
            "start": [0, 0],
            "segments": [
                {"type": "line", "to": [l, 0]},
                {"type": "arc", "mid": [l + 5, 5], "to": [l, 10]},
                {"type": "line", "to": [0, 10]},
                {"type": "line", "to": [0, 0]},
            ],
        }
        volume = (l * 10 + math.pi * 25 / 2) * h
    elif family == "pocket":
        a.extend(
            [
                {
                    "op": "sketch",
                    "name": "Hole",
                    "profile": {"type": "circle", "radius": 3},
                    "origin": [l / 2, w / 2, h],
                },
                {
                    "op": "pocket",
                    "name": "Result",
                    "source": "Base",
                    "profile": "Hole",
                    "length": h,
                },
            ]
        )
        volume -= math.pi * 9 * h
        target = "Result"
    elif family == "revolve":
        sk.update(
            profile={"type": "rectangle", "length": l, "width": h}, normal=[0, -1, 0]
        )
        a = [
            sk,
            {
                "op": "revolve",
                "name": "Result",
                "profile": "Profile",
                "axis": [0, 0, 1],
                "angle": 360,
            },
        ]
        volume, target = math.pi * l * l * h, "Result"
    elif family == "loft":
        a = [
            sk,
            {
                "op": "sketch",
                "name": "Top",
                "profile": {"type": "rectangle", "length": l / 2, "width": w / 2},
                "origin": [l / 4, w / 4, h],
            },
            {
                "op": "loft",
                "name": "Result",
                "profiles": ["Profile", "Top"],
                "ruled": True,
            },
        ]
        volume, target = l * w * h * 7 / 12, "Result"
    elif family == "sweep":
        sk["profile"] = {"type": "circle", "radius": 3}
        a = [
            sk,
            {
                "op": "sweep",
                "name": "Result",
                "profile": "Profile",
                "path": [[0, 0, 0], [0, 0, h]],
            },
        ]
        volume, target = math.pi * 9 * h, "Result"
    elif family in ("fillet", "chamfer", "shell"):
        a.append({"op": "inspect", "target": "Base"})
        item = {"op": family, "name": "Result", "source": "Base"}
        if family == "fillet":
            item.update(radius=1, edges=[1])
            volume = l * w * h - (1 - math.pi / 4) * h
        elif family == "chamfer":
            item.update(size=1, edges=[1])
            volume = l * w * h - h / 2
        else:
            item.update(thickness=1, faces=[6])
            volume = l * w * h - (l - 2) * (w - 2) * (h - 1)
        a.append(item)
        target = "Result"
    elif family in ("fuse", "cut", "common"):
        a.extend(
            [
                {
                    "op": "translate",
                    "name": "Shift",
                    "source": "Base",
                    "vector": [l / 2, 0, 0],
                },
                {"op": family, "name": "Result", "left": "Base", "right": "Shift"},
            ]
        )
        volume *= 1.5 if family == "fuse" else 0.5
        target = "Result"
    elif family in ("translate", "rotate", "mirror", "linear_pattern", "polar_pattern"):
        item = {"op": family, "name": "Result", "source": "Base"}
        if family == "translate":
            item["vector"] = [3, 4, 5]
        elif family == "rotate":
            item.update(axis=[0, 0, 1], angle=90)
        elif family == "mirror":
            item.update(normal=[1, 0, 0])
        elif family == "linear_pattern":
            item.update(vector=[l + 5, 0, 0], count=3)
            volume *= 3
        else:
            a.append(
                {
                    "op": "translate",
                    "name": "Offset",
                    "source": "Base",
                    "vector": [l + 5, 0, 0],
                }
            )
            item.update(source="Offset", axis=[0, 0, 1], angle=360, count=3)
            volume *= 3
        a.append(item)
        target = "Result"
    elif family == "parameter_edit":
        a.append(
            {
                "op": "set_parameter",
                "target": "Profile",
                "parameter": "width",
                "value": w + 2,
            }
        )
        volume = l * (w + 2) * h
    a.append({"op": "finish", "targets": [target]})
    a = validate(a)
    group = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:24]
    # 80/10/10 lineage split shared across all future representations.
    bucket = int(group[:8], 16) % 10
    split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
    base = f"Start with a rectangle from (0,0) to ({l},{w}) on XY and extrude by {h} along +Z. "
    descriptions = {
        "tilted_pad": f"Extrude a {l} by {w} rectangle by {h}. Its plane origin is (3,-4,2), normal (1,1,1), local X (1,-1,0).",
        "polygon_pad": f"Extrude the XY polygon [(0,0),({l},0),({l},4),(4,4),(4,{w}),(0,{w})] by {h}.",
        "arc_pad": f"Extrude by {h} an XY plate with boundary (0,0) to ({l},0), then an arc through ({l + 5},5) to ({l},10), then (0,10) and (0,0).",
        "pocket": base
        + f"Cut a radius-3 through hole centered at ({l / 2},{w / 2}) from Z={h} down to Z=0.",
        "revolve": f"Rotate the XZ rectangle (0,0),({l},0),({l},{h}),(0,{h}) through 360 degrees about world Z.",
        "loft": f"Create a solid ruled loft between an XY rectangle {l} by {w} at (0,0,0) and a rectangle {l / 2} by {w / 2} at ({l / 4},{w / 4},{h}).",
        "sweep": f"Sweep a radius-3 XY circle centered at the origin along the line from (0,0,0) to (0,0,{h}).",
        "fillet": base + "Round only the vertical edge at X=0,Y=0 with radius 1.",
        "chamfer": base + "Chamfer only the vertical edge at X=0,Y=0 by 1.",
        "shell": base
        + "Remove the top face and hollow inward with wall and bottom thickness 1.",
        "fuse": base + f"Unite it with a copy translated by ({l / 2},0,0).",
        "cut": base + f"Subtract a copy translated by ({l / 2},0,0).",
        "common": base
        + f"Keep only the intersection with a copy translated by ({l / 2},0,0).",
        "translate": base + "Keep only a copy moved by (3,4,5).",
        "rotate": base
        + "Keep only a copy rotated 90 degrees about Z through the origin.",
        "mirror": base + "Keep only the mirror across the world YZ plane.",
        "linear_pattern": base
        + f"Keep three separate copies with step vector ({l + 5},0,0), including the original.",
        "polar_pattern": base
        + f"First move by ({l + 5},0,0). Keep three separate copies at 0,120,240 degrees about world Z.",
        "parameter_edit": base
        + f"Then change the profile width to {w + 2}; keep the extrusion depth unchanged.",
    }
    return {
        "task_id": "v2-" + group,
        "split_group": group,
        "split": split,
        "family": family,
        "instruction": descriptions[family]
        + " All lengths in mm. Retain only the requested final solid(s).",
        "actions": a,
        "expected_volume": volume,
    }
