"""Original mechanical compositions with declarative specs independent of teachers."""

import hashlib
import json
import math
import random

from .schema import validate

FAMILIES = ("flanged_bushing", "pedestal_mount", "gusset_bracket")
LEVELS = ("easy", "medium", "hard")


def make_task(family, level, seed):
    if family not in FAMILIES or level not in LEVELS:
        raise ValueError("Unknown composition family/level")
    rng = random.Random(f"composition-v2:{family}:{level}:{seed}")
    n = {"easy": 2, "medium": 6, "hard": 12}[level]
    t, cb_depth = round(rng.uniform(6, 9), 2), 2.0
    holes, forms, actions = [], [], []
    if family == "flanged_bushing":
        r, shaft, bore, height = (
            round(rng.uniform(46, 50), 2),
            14.0,
            5.0,
            round(rng.uniform(23, 28), 2),
        )
        profile = [
            [bore, 0],
            [r, 0],
            [r, t],
            [shaft, t],
            [shaft, height],
            [bore, height],
        ]
        forms = [
            {
                "kind": "revolved radial/Z polygon",
                "points": profile,
                "axis": [0, 0, 1],
                "angle": 360,
            }
        ]
        actions = [
            {
                "op": "sketch",
                "name": "Section",
                "profile": {"type": "polygon", "points": profile},
                "normal": [0, -1, 0],
            },
            {
                "op": "revolve",
                "name": "Base",
                "profile": "Section",
                "axis": [0, 0, 1],
                "angle": 360,
            },
        ]
        volume = math.pi * (
            (r * r - bore * bore) * t + (shaft * shaft - bore * bore) * (height - t)
        )
        for i in range(n * 2):
            angle = 2 * math.pi * i / (n * 2)
            holes.append(
                {
                    "center": [
                        round(34 * math.cos(angle), 6),
                        round(34 * math.sin(angle), 6),
                        t,
                    ],
                    "normal": [0, 0, 1],
                    "radius": 1.5,
                    "depth": t,
                    "cb_radius": 2.5,
                    "cb_depth": cb_depth,
                }
            )
    elif family == "pedestal_mount":
        length, width, height = (
            round(rng.uniform(155, 170), 2),
            100.0,
            round(rng.uniform(18, 24), 2),
        )
        forms = [
            {
                "kind": "rectangular base",
                "length": length,
                "width": width,
                "height": t,
                "origin": [0, 0, 0],
            },
            {
                "kind": "ruled rectangular loft",
                "bottom_origin": [length / 2 - 12, 40, t],
                "bottom_size": [24, 20],
                "top_origin": [length / 2 - 7, 44, t + height],
                "top_size": [14, 12],
            },
        ]
        actions = [
            {
                "op": "sketch",
                "name": "PlateProfile",
                "profile": {"type": "rectangle", "length": length, "width": width},
            },
            {"op": "pad", "name": "Plate", "profile": "PlateProfile", "length": t},
            {
                "op": "sketch",
                "name": "Bottom",
                "profile": {"type": "rectangle", "length": 24, "width": 20},
                "origin": [length / 2 - 12, 40, t],
            },
            {
                "op": "sketch",
                "name": "Top",
                "profile": {"type": "rectangle", "length": 14, "width": 12},
                "origin": [length / 2 - 7, 44, t + height],
            },
            {
                "op": "loft",
                "name": "Pedestal",
                "profiles": ["Bottom", "Top"],
                "ruled": True,
            },
            {"op": "fuse", "name": "Base", "left": "Plate", "right": "Pedestal"},
        ]
        # The unequal aspect ratios make this a prismoid, not a similar-section
        # pyramid frustum. Integrate the product of linear side lengths exactly.
        volume = length * width * t + height * (
            24 * 20 + (24 * (-8) + 20 * (-10)) / 2 + (-10) * (-8) / 3
        )
        for i in range(n * 2):
            angle = 2 * math.pi * i / (n * 2)
            holes.append(
                {
                    "center": [
                        round(length / 2 + 36 * math.cos(angle), 6),
                        round(50 + 36 * math.sin(angle), 6),
                        t,
                    ],
                    "normal": [0, 0, 1],
                    "radius": 1.5,
                    "depth": t,
                    "cb_radius": 2.5,
                    "cb_depth": cb_depth,
                }
            )
    else:
        length, height, width = (
            round(rng.uniform(140, 150), 2),
            round(rng.uniform(120, 130), 2),
            40.0,
        )
        points = [[0, 0], [length, 0], [length, t], [t, t], [t, height], [0, height]]
        forms = [
            {
                "kind": "XZ L profile extruded toward -Y",
                "points": points,
                "origin": [0, width, 0],
                "normal": [0, -1, 0],
                "depth": width,
            }
        ]
        actions = [
            {
                "op": "sketch",
                "name": "BracketProfile",
                "profile": {"type": "polygon", "points": points},
                "origin": [0, width, 0],
                "normal": [0, -1, 0],
            },
            {"op": "pad", "name": "Base", "profile": "BracketProfile", "length": width},
        ]
        volume = (length * t + height * t - t * t) * width
        for i in range(n):
            x = t + 12 + (length - t - 24) * i / max(1, n - 1)
            z = t + 12 + (height - t - 24) * i / max(1, n - 1)
            holes.extend(
                [
                    {
                        "center": [round(x, 6), width / 2, t],
                        "normal": [0, 0, 1],
                        "radius": 1.5,
                        "depth": t,
                        "cb_radius": 2.5,
                        "cb_depth": cb_depth,
                    },
                    {
                        "center": [t, width / 2, round(z, 6)],
                        "normal": [1, 0, 0],
                        "radius": 1.5,
                        "depth": t,
                        "cb_radius": 2.5,
                        "cb_depth": cb_depth,
                    },
                ]
            )
    source = "Base"
    for i, hole in enumerate(holes, 1):
        for suffix, radius, depth in [
            ("Drill", hole["radius"], hole["depth"]),
            ("CB", hole["cb_radius"], hole["cb_depth"]),
        ]:
            sketch, cut = f"H{i:02d}{suffix}Sketch", f"H{i:02d}{suffix}Cut"
            actions.extend(
                [
                    {
                        "op": "sketch",
                        "name": sketch,
                        "profile": {"type": "circle", "radius": radius},
                        "origin": hole["center"],
                        "normal": hole["normal"],
                    },
                    {
                        "op": "pocket",
                        "name": cut,
                        "source": source,
                        "profile": sketch,
                        "length": depth,
                    },
                ]
            )
            source = cut
        volume -= math.pi * (
            hole["radius"] ** 2 * hole["depth"]
            + (hole["cb_radius"] ** 2 - hole["radius"] ** 2) * hole["cb_depth"]
        )
    actions.append({"op": "finish", "targets": [source]})
    spec = {
        "forms": forms,
        "holes": holes,
        "units": "mm",
        "convention": "Hole depths opposite outward normal; counterbore depth is measured from the same entry plane.",
    }
    group = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    bucket = int(group[:8], 16) % 10
    return {
        "task_id": "c2-" + group,
        "split_group": group,
        "split": "train" if bucket < 8 else "validation" if bucket == 8 else "test",
        "family": family,
        "difficulty": level,
        "design_spec": spec,
        "actions": validate(actions),
        "expected_volume": volume,
        "instruction": "Build the single solid defined by the supplied drawing sheets and dimensional specification. Include every primary hole and counterbore. Dimensions in mm.",
    }
