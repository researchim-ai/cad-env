"""Original procedural analogues of mechanical parts, not vendor CAD replicas.

All features fit the existing frozen CAD API. Dimensional clearance constraints
are conservative geometric checks, not load/tolerance/manufacturability approval.
"""

import hashlib
import json
import math
import random

from cad_env.agent.schema import MAX_ACTIONS, validate_program
from cad_env.learning.extreme_tasks import make_extreme_task

FAMILIES = (
    "adapter_flange",
    "bearing_cartridge",
    "motor_mount_plate",
    "spacer_bushing",
    "blind_cover",
    "alignment_fixture",
)
LEVELS = ("easy", "medium", "hard")
VERSION = "industrial-analogues-v1"
REFERENCES = {
    "bearing_cartridge": "https://uk.misumi-ec.com/vona2/mech/M0800000000/M0806000000/",
    "adapter_flange": "https://www.thk.com/us/en/products/ball_screw/support_unit/ek_bk_fk/",
}


class PartBuilder:
    def __init__(self):
        self.p = {"pads": [], "cuts": []}
        self.actions = [{"op": "create_body", "name": "Body"}]
        self.supports = {}
        self.volume = 0.0
        self.bounds = None
        self.constraints = []

    def rectangle(self, length, width, height):
        self.p["base"] = {"length": length, "width": width, "height": height}
        self.actions.extend(
            [
                {
                    "op": "sketch_rectangle",
                    "name": "BaseSketch",
                    "body": "Body",
                    "length": length,
                    "width": width,
                },
                {
                    "op": "pad",
                    "name": "BasePad",
                    "body": "Body",
                    "profile": "BaseSketch",
                    "length": height,
                },
            ]
        )
        self.supports["BasePad"] = {"top": height, "rectangle": (length, width)}
        self.volume = length * width * height
        self.bounds = [0, 0, 0, length, width, height]

    def circle(self, name, support, x, y, radius):
        a = {
            "op": "sketch_circles",
            "name": name,
            "body": "Body",
            "circles": [{"x": x, "y": y, "radius": radius}],
        }
        if support:
            a["support"] = support
        self.actions.append(a)

    def inside(self, support, x, y, radius, margin=2):
        s = self.supports[support]
        if "rectangle" in s:
            length, width = s["rectangle"]
            gap = min(x - radius, y - radius, length - x - radius, width - y - radius)
        else:
            gap = s["radius"] - math.hypot(x - s["x"], y - s["y"]) - radius
        if gap < margin - 1e-6:
            raise ValueError("Insufficient edge clearance on " + support)
        self.constraints.append(
            {
                "kind": "edge_clearance",
                "support": support,
                "actual_mm": gap,
                "minimum_mm": margin,
            }
        )

    def pad(self, name, support, x, y, radius, height):
        if support:
            self.inside(support, x, y, radius)
        z = self.supports[support]["top"] if support else 0
        # Sibling pads are disjoint; stacked concentric pads are permitted.
        for s in self.p["pads"]:
            if (
                s["support"] == support
                and math.hypot(x - s["x"], y - s["y"]) < radius + s["radius"] + 2
            ):
                raise ValueError("Sibling pad overlap or inadequate spacing")
        self.circle(name + "Sketch", support, x, y, radius)
        self.actions.append(
            {
                "op": "pad",
                "name": name,
                "body": "Body",
                "profile": name + "Sketch",
                "length": height,
            }
        )
        self.p["pads"].append(
            {
                "name": name,
                "support": support,
                "x": x,
                "y": y,
                "radius": radius,
                "height": height,
            }
        )
        self.supports[name] = {"top": z + height, "x": x, "y": y, "radius": radius}
        self.volume += math.pi * radius**2 * height
        if self.bounds is None:
            self.bounds = [x - radius, y - radius, 0, x + radius, y + radius, height]
        else:
            self.bounds[-1] = max(self.bounds[-1], z + height)

    def hole(self, name, support, x, y, radius, depth, cb_radius=None, cb_depth=None):
        outer = cb_radius or radius
        self.inside(support, x, y, outer)
        if not 0 < depth <= self.supports[support]["top"]:
            raise ValueError("Invalid drilling depth")
        if cb_radius is not None and not (cb_radius > radius and 0 < cb_depth < depth):
            raise ValueError("Invalid counterbore")
        for h in self.p["cuts"]:
            gap = (
                math.hypot(x - h["x"], y - h["y"])
                - outer
                - (h["cb_radius"] or h["radius"])
            )
            if gap < 2 - 1e-6:
                raise ValueError("Drill features intersect or lack ligament")
            self.constraints.append(
                {"kind": "drill_ligament", "actual_mm": gap, "minimum_mm": 2}
            )
        self.circle(name + "Sketch", support, x, y, radius)
        self.actions.append(
            {
                "op": "pocket",
                "name": name,
                "body": "Body",
                "profile": name + "Sketch",
                "length": depth,
            }
        )
        if cb_radius is not None:
            self.circle(name + "CBProfile", support, x, y, cb_radius)
            self.actions.append(
                {
                    "op": "pocket",
                    "name": name + "CB",
                    "body": "Body",
                    "profile": name + "CBProfile",
                    "length": cb_depth,
                }
            )
        self.p["cuts"].append(
            {
                "name": name,
                "support": support,
                "x": x,
                "y": y,
                "radius": radius,
                "depth": depth,
                "cb_radius": cb_radius,
                "cb_depth": cb_depth,
            }
        )
        self.volume -= math.pi * radius**2 * depth
        if cb_radius is not None:
            self.volume -= math.pi * (cb_radius**2 - radius**2) * cb_depth


def make_industrial_task(family, seed, level="medium"):
    if family not in FAMILIES or level not in LEVELS:
        raise ValueError("Unknown industrial family/level")
    rng = random.Random(f"{VERSION}:{family}:{level}:{seed}")
    rank = LEVELS.index(level)
    b = PartBuilder()
    bearing = None
    if family in ("adapter_flange", "blind_cover", "spacer_bushing"):
        radius = rng.randrange(55, 81, 5)
        height = rng.randint(8, 14)
        b.pad("Flange", None, 0, 0, radius, height)
        top = "Flange"
        if family == "spacer_bushing":
            for i in range(rank + 1):
                name = f"Step{i + 1}"
                b.pad(name, top, 0, 0, radius * (0.65 - i * 0.1), rng.randint(7, 15))
                top = name
            bore = round(radius * 0.16, 2)
            b.hole("Shaft", top, 0, 0, bore, b.supports[top]["top"], bore + 3, 3)
            count = (0, 4, 8)[rank]
        else:
            b.pad("Hub", top, 0, 0, radius * 0.42, rng.randint(8, 16))
            top = "Hub"
            if rank:
                b.pad("Pilot", top, 0, 0, radius * 0.32, rng.randint(3, 7))
                top = "Pilot"
            depth = (
                b.supports[top]["top"]
                if family == "adapter_flange"
                else b.supports[top]["top"] - 3
            )
            b.hole(
                "Shaft" if family == "adapter_flange" else "BlindSeat",
                top,
                0,
                0,
                round(radius * 0.15, 2),
                depth,
                round(radius * 0.24, 2),
                3,
            )
            count = (4, 8, 16)[rank]
        angle = rng.choice([0, 15, 22.5])
        for i in range(count):
            a = math.radians(angle + 360 * i / count)
            x, y = (
                round(radius * 0.77 * math.cos(a), 5),
                round(radius * 0.77 * math.sin(a), 5),
            )
            r = rng.choice([2, 2.5, 3])
            cb = rank > 0 and i % 2 == 0
            b.hole(
                f"Mount{i + 1:02d}",
                "Flange",
                x,
                y,
                r,
                height,
                r + 1.5 if cb else None,
                height / 2 if cb else None,
            )
    else:
        length, width, height = (
            rng.randrange(150, 191, 10),
            rng.randrange(100, 131, 10),
            rng.randint(8, 14),
        )
        b.rectangle(length, width, height)
        if family == "alignment_fixture":
            count = (2, 4, 8)[rank]
            cols, rows = ((2, 1), (2, 2), (4, 2))[rank]
            for i in range(count):
                x, y = (
                    round(length * (i % cols + 1) / (cols + 1), 5),
                    round(width * (i // cols + 1) / (rows + 1), 5),
                )
                name = f"Locator{i + 1:02d}"
                h = rng.randint(8, 20)
                b.pad(name, "BasePad", x, y, 8, h)
                depth = height + h if i % 2 == 0 else round(h * 0.6, 3)
                cb = rank > 0 and i % 2 == 1
                b.hole(
                    f"Pin{i + 1:02d}",
                    name,
                    x,
                    y,
                    2,
                    depth,
                    4 if cb else None,
                    2 if cb else None,
                )
        else:
            bearing = (
                {
                    "designation": "6200",
                    "bore_mm": 10,
                    "outside_mm": 30,
                    "width_mm": 9,
                    "source_url": "https://www.nsk.com/my-en/engineering/products/bearings/ball-bearings/deep-groove-ball-bearings/single-row-deep-groove-ball-bearings/6200-apn.html",
                }
                if family == "bearing_cartridge"
                else None
            )
            hub = 23 if bearing else round(min(length, width) * 0.2, 3)
            b.pad("Hub", "BasePad", length / 2, width / 2, hub, rng.randint(10, 24))
            top = "Hub"
            if rank:
                b.pad(
                    "Pilot",
                    "Hub",
                    length / 2,
                    width / 2,
                    hub - 2 if bearing else hub * 0.8,
                    4,
                )
                top = "Pilot"
            if rank == 2 and bearing:
                b.pad("Shoulder", top, length / 2, width / 2, hub - 4, 4)
                top = "Shoulder"
            b.hole(
                "Shaft",
                top,
                length / 2,
                width / 2,
                5.5 if bearing else hub * 0.35,
                b.supports[top]["top"],
                15 if bearing else hub * 0.58,
                9 if bearing else 4,
            )
            if family == "motor_mount_plate":
                count = (4, 8, 12)[rank]
                for i in range(count):
                    a = math.radians(45 + 360 * i / count)
                    x, y = (
                        round(length / 2 + width * 0.36 * math.cos(a), 5),
                        round(width / 2 + width * 0.36 * math.sin(a), 5),
                    )
                    b.hole(
                        f"Motor{i + 1:02d}",
                        "BasePad",
                        x,
                        y,
                        2.5,
                        height,
                        4 if rank else None,
                        height / 2 if rank else None,
                    )
        for i, (x, y) in enumerate(
            [(12, 12), (length - 12, 12), (12, width - 12), (length - 12, width - 12)]
        ):
            b.hole(
                f"Bolt{i + 1:02d}",
                "BasePad",
                x,
                y,
                3,
                height,
                5 if rank else None,
                height / 2 if rank else None,
            )
        if rank:
            for i, y in enumerate([width / 3, 2 * width / 3]):
                b.hole(f"Dowel{i + 1:02d}", "BasePad", 12, round(y, 5), 2, height * 0.6)
        if rank == 2:
            for i, y in enumerate([width / 3, 2 * width / 3]):
                b.hole(
                    f"DowelR{i + 1:02d}",
                    "BasePad",
                    length - 12,
                    round(y, 5),
                    2,
                    height * 0.6,
                )
    program = validate_program(b.actions + [{"op": "finish"}])
    # Geometry lineage excludes observation format and difficulty labels.
    canonical = {
        "base": b.p.get("base"),
        "pads": sorted(
            (
                s["x"],
                s["y"],
                b.supports[s["name"]]["top"] - s["height"],
                s["radius"],
                s["height"],
            )
            for s in b.p["pads"]
        ),
        "cuts": sorted(
            (
                h["x"],
                h["y"],
                b.supports[h["support"]]["top"],
                h["radius"],
                h["depth"],
                h["cb_radius"] or 0,
                h["cb_depth"] or 0,
            )
            for h in b.p["cuts"]
        ),
    }
    group = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    bucket = int(group[:8], 16) % 100
    task = make_extreme_task("heterogeneous_plate", seed, "challenge", "image")
    budget = min(MAX_ACTIONS, len(program) + 20)
    task.update(
        task_id="i-" + hashlib.sha256(group.encode()).hexdigest()[:24],
        family=family,
        parameters=b.p,
        reference_actions=program,
        oracle_actions=program,
        analytic_metrics={"volume": b.volume, "bounds": b.bounds},
        split_group=group,
        condition_group=group,
        split="train" if bucket < 75 else "validation" if bucket < 90 else "test",
        max_steps=budget,
        difficulty=level,
        benchmark_version=VERSION,
        goal_spec={"kind": "large_composition", "parameters": b.p},
        construction_signature=[(a["op"], len(a.get("circles", []))) for a in program],
        design_provenance={
            "kind": "original_procedural_industrial_analogue",
            "vendor_replica": False,
            "dimensions_from_catalogue": bool(bearing),
            "inspiration_url": REFERENCES.get(family),
            "manufacturing_approval": False,
        },
        design_checks=b.constraints,
    )
    if bearing:
        task["design_provenance"]["bearing_boundary_reference"] = bearing
        task["design_provenance"]["scope"] = (
            "Only bearing boundary dimensions are sourced; housing is original and nominal, fits/seals/retention are not validated."
        )
    task["instruction"] += (
        " CB (or ЦЕКОВКА) is an ADDITIONAL coaxial larger-diameter pocket at the stated depth; "
        "it does not replace the primary bore. All schedule rows and their CB entries are required. "
        "Sketches and pockets must have distinct unique object names. "
        "Check feature completeness before finish."
    )
    task["public_task"]["instruction"] = task["instruction"]
    task["public_task"]["max_steps"] = budget
    return task
