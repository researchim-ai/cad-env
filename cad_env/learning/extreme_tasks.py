"""Native-verified large compositions using the existing typed FreeCAD tools.

Difficulty is fixed before policy evaluation, never selected from model failures.
The pilot exports TRAIN and development-validation only; sealed TEST is untouched.
"""

import argparse
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from cad_env.agent.schema import MAX_ACTIONS, SCHEMA_VERSION, validate_program

FAMILIES = ("heterogeneous_plate", "multi_support_fixture", "terraced_flange_stack")
TIERS = ("challenge", "extreme")
VERSION = "large-composition-v1"
IMAGE_INSTRUCTION = (
    "Build one editable parametric solid matching the engineering drawing. "
    "All dimensions are millimeters. Hole depths are downward from their named "
    "support top. Preserve every specified feature. Finish only when complete."
)


def make_extreme_task(family, seed, tier="extreme", modality="image"):
    if (
        family not in FAMILIES
        or tier not in (*TIERS, "easy")
        or modality not in ("text", "image")
    ):
        raise ValueError("Unknown family, difficulty or modality")
    rng = random.Random(f"{VERSION}:{family}:{tier}:{seed}")
    extreme = tier == "extreme"
    easy = tier == "easy"
    program = [{"op": "create_body", "name": "Body"}]
    p = {"pads": [], "cuts": []}

    def circles(name, support, values):
        action = {
            "op": "sketch_circles",
            "name": name,
            "body": "Body",
            "circles": values,
        }
        if support:
            action["support"] = support
        program.append(action)

    def pad(name, support, x, y, radius, height):
        sketch = name + "Sketch"
        circles(sketch, support, [{"x": x, "y": y, "radius": radius}])
        program.append(
            {
                "op": "pad",
                "name": name,
                "body": "Body",
                "profile": sketch,
                "length": height,
            }
        )
        p["pads"].append(
            {
                "name": name,
                "support": support,
                "x": x,
                "y": y,
                "radius": radius,
                "height": height,
            }
        )

    def cut(support, x, y, radius, depth, cb_radius=None, cb_depth=None):
        name = f"Hole{len(p['cuts']) + 1:02d}"
        circles(name + "Sketch", support, [{"x": x, "y": y, "radius": radius}])
        program.append(
            {
                "op": "pocket",
                "name": name,
                "body": "Body",
                "profile": name + "Sketch",
                "length": depth,
            }
        )
        if cb_radius is not None:
            circles(
                name + "CBProfile", support, [{"x": x, "y": y, "radius": cb_radius}]
            )
            program.append(
                {
                    "op": "pocket",
                    "name": name + "CB",
                    "body": "Body",
                    "profile": name + "CBProfile",
                    "length": cb_depth,
                }
            )
        p["cuts"].append(
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

    if family in ("heterogeneous_plate", "multi_support_fixture"):
        p["base"] = {
            "length": rng.randrange(200, 241, 4),
            "width": rng.randrange(150, 181, 4),
            "height": rng.randint(16, 24),
        }
        b = p["base"]
        program += [
            {
                "op": "sketch_rectangle",
                "name": "BaseSketch",
                "body": "Body",
                "length": b["length"],
                "width": b["width"],
            },
            {
                "op": "pad",
                "name": "BasePad",
                "body": "Body",
                "profile": "BaseSketch",
                "length": b["height"],
            },
        ]
        volume = b["length"] * b["width"] * b["height"]
        bounds = [0, 0, 0, b["length"], b["width"], b["height"]]
        if family == "heterogeneous_plate":
            columns, rows = (2, 1) if easy else (8, 4) if extreme else (4, 3)
            positions = [
                (
                    round(b["length"] * (x + 1) / (columns + 1), 2),
                    round(b["width"] * (y + 1) / (rows + 1), 2),
                )
                for y in range(rows)
                for x in range(columns)
            ]
            rng.shuffle(positions)
            for i, (x, y) in enumerate(positions):
                radius = rng.choice([2, 2.5, 3, 3.5, 4])
                depth = (
                    b["height"]
                    if i % 3 == 0
                    else round(b["height"] * rng.choice([0.3, 0.5, 0.7]), 2)
                )
                cb_depth = min(4, depth / 2)
                cut(
                    "BasePad",
                    x,
                    y,
                    radius,
                    depth,
                    radius + rng.choice([2, 3]) if i % 2 == 0 else None,
                    cb_depth if i % 2 == 0 else None,
                )
        else:
            count = 1 if easy else 12 if extreme else 4
            for i in range(count):
                columns, rows = (1, 1) if easy else (4, 3) if extreme else (2, 2)
                x = round(b["length"] * (i % columns + 1) / (columns + 1), 2)
                y = round(b["width"] * (i // columns + 1) / (rows + 1), 2)
                radius, height = rng.randint(11, 14), rng.randint(10, 26)
                pad(f"Boss{i + 1}", "BasePad", x, y, radius, height)
                volume += math.pi * radius**2 * height
                bounds[-1] = max(bounds[-1], b["height"] + height)
            for i, boss in enumerate(p["pads"]):
                depth = (
                    boss["height"] + b["height"]
                    if i % 2 == 0
                    else round(boss["height"] * 0.6, 2)
                )
                radius = rng.choice([2, 3, 4])
                cut(
                    boss["name"],
                    boss["x"],
                    boss["y"],
                    radius,
                    depth,
                    radius + 3 if i % 2 else None,
                    3 if i % 2 else None,
                )
            for x, y in [
                (12, 12),
                (b["length"] - 12, 12),
                (12, b["width"] - 12),
                (b["length"] - 12, b["width"] - 12),
            ]:
                cut("BasePad", x, y, 2.5, b["height"])
    else:
        count = 2 if easy else 18 if extreme else 8
        radius = rng.randint(114, 120) if extreme else rng.randint(66, 72)
        total = 0
        for i in range(count):
            height = rng.randint(5, 13)
            pad(f"Level{i + 1}", f"Level{i}" if i else None, 0, 0, radius, height)
            total += height
            radius -= rng.randint(5, 6)
        volume = sum(math.pi * s["radius"] ** 2 * s["height"] for s in p["pads"])
        bounds = [
            -p["pads"][0]["radius"],
            -p["pads"][0]["radius"],
            0,
            p["pads"][0]["radius"],
            p["pads"][0]["radius"],
            total,
        ]
        cut(p["pads"][-1]["name"], 0, 0, rng.choice([2, 3]), total)
        z = 0
        for i, stage in enumerate(p["pads"][:-1]):
            z += stage["height"]
            offset = (stage["radius"] + p["pads"][i + 1]["radius"]) / 2
            depth = z if i % 2 == 0 else stage["height"] / 2
            # Different terrace radii keep all drilling cylinders disjoint.
            for sign in [-1, 1]:
                cut(stage["name"], sign * offset, 0, 1.1, depth)
    for hole in p["cuts"]:
        volume -= math.pi * hole["radius"] ** 2 * hole["depth"]
        if hole["cb_radius"] is not None:
            volume -= (
                math.pi
                * (hole["cb_radius"] ** 2 - hole["radius"] ** 2)
                * hole["cb_depth"]
            )
    program = validate_program(program + [{"op": "finish"}])
    group = hashlib.sha256(
        json.dumps([VERSION, family, p], sort_keys=True).encode()
    ).hexdigest()[:24]
    bucket = int(group[:8], 16) % 100
    split = "train" if bucket < 75 else "validation" if bucket < 90 else "test"
    text = IMAGE_INSTRUCTION + " Geometry schedule: " + json.dumps(p, sort_keys=True)
    instruction = text if modality == "text" else IMAGE_INSTRUCTION
    budget = min(MAX_ACTIONS, len(program) + 20)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "x-"
        + hashlib.sha256(f"{group}:{modality}".encode()).hexdigest()[:24],
        "family": family,
        "mode": "create",
        "instruction": instruction,
        "public_task": {
            "task_id": "cad-task",
            "family": "complex_part",
            "mode": "create",
            "instruction": instruction,
            "max_steps": budget,
        },
        "parameters": p,
        "reference_actions": program,
        "oracle_actions": program,
        "initial_actions": [],
        "analytic_metrics": {"volume": volume, "bounds": bounds},
        "split_group": group,
        "condition_group": group,
        "split": split,
        "max_steps": budget,
        "tolerance": 1e-5,
        "require_parametric": True,
        "reward_mode": "strict_success",
        "observation_profile": "compact_v1",
        "goal_only_observation": True,
        "evaluation_tier": "iid",
        "difficulty": tier,
        "generator_seed": seed,
        "input_modality": modality,
        "benchmark_version": VERSION,
        "goal_spec": {"kind": "large_composition", "parameters": p},
        "construction_signature": [
            (a["op"], len(a.get("circles", []))) for a in program
        ],
    }


def render_extreme_goal(task, path):
    """One readable specification sheet; all positions, supports and depths public."""
    from PIL import Image, ImageDraw, ImageFont

    p = task["parameters"]
    im = Image.new("RGB", (960, 1200), "white")
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=24):
        d.text(
            (x, y),
            value,
            fill="#102636",
            font=ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size
            ),
        )

    text(24, 14, "PARAMETRIC CONSTRUCTION / mm", 32)
    text(24, 57, task["family"].upper(), 25)
    if "base" in p:
        b = p["base"]
        text(
            24, 94, f"BASE at XY origin: L={b['length']} W={b['width']} H={b['height']}"
        )
        scale = min(720 / b["length"], 220 / b["width"])
        ox, oy = 90, 370
        d.rectangle(
            (ox, oy - b["width"] * scale, ox + b["length"] * scale, oy),
            outline="black",
            width=2,
        )
        for item in p["pads"] + p["cuts"]:
            x, y, r = (
                ox + item["x"] * scale,
                oy - item["y"] * scale,
                item["radius"] * scale,
            )
            d.ellipse((x - r, y - r, x + r, y + r), outline="black", width=2)
            text(x + 4, y - 26, item["name"].replace("Hole", "H"), 20)
        text(24, 387, "Top view: +X right; +Y up. Base starts at Z=0.", 21)
    else:
        total = sum(s["height"] for s in p["pads"])
        scale = min(760 / (2 * p["pads"][0]["radius"]), 250 / total)
        z = 0
        for item in p["pads"]:
            d.rectangle(
                (
                    480 - item["radius"] * scale,
                    375 - (z + item["height"]) * scale,
                    480 + item["radius"] * scale,
                    375 - z * scale,
                ),
                outline="black",
                width=2,
            )
            z += item["height"]
        text(24, 387, "Section outline: coaxial levels, bottom -> top; Z=0.", 21)
    y = 425
    if p["pads"]:
        text(24, y, "PADS: name/support   X,Y    diameter,height", 23)
        y += 29
        for item in p["pads"]:
            text(
                24,
                y,
                f"{item['name']}/{item['support'] or 'XY'}   {item['x']:g},{item['y']:g}   D{2 * item['radius']:g} H{item['height']:g}",
                22,
            )
            y += 26
    text(24, 1090, "Drilling schedules continue on the following goal sheets.", 23)
    text(24, 1130, "Depths: downward from each named support's top.", 23)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    pages = [path]
    # Fixed-size pages keep small print readable under the VLM pixel budget.
    # Every sheet is a task GOAL; no oracle, private state or answer action shown.
    for offset in range(0, len(p["cuts"]), 16):
        sheet = Image.new("RGB", (960, 1200), "white")
        d = ImageDraw.Draw(sheet)
        number = 2 + offset // 16
        text(24, 20, f"DRILLING SCHEDULE / mm / SHEET {number}", 30)
        text(24, 75, "D=diameter; CB=counterbore (diameter,depth)", 25)
        text(24, 115, "Depth is downward from SUPPORT TOP, including CB.", 25)
        text(24, 170, "HOLE   SUPPORT         X      Y     D  DEPTH    CB(D,DEP)", 24)
        for row, hole in enumerate(p["cuts"][offset : offset + 16]):
            cb = (
                "-"
                if hole["cb_radius"] is None
                else f"{2 * hole['cb_radius']:g},{hole['cb_depth']:g}"
            )
            line = (
                f"{hole['name']:<6} {hole['support']:<10} "
                f"{hole['x']:>6g} {hole['y']:>6g} {2 * hole['radius']:>5g} "
                f"{hole['depth']:>6g} {cb:>10}"
            )
            text(24, 225 + row * 48, line, 25)
        text(24, 1060, "Build EVERY row at its indicated position and depth.", 25)
        text(24, 1100, "Only holes reaching the support's bottom are through.", 25)
        text(24, 1140, "Keep the model parametric. Finish after all features.", 25)
        target = path.with_name(f"goal-sheet-{number}.png")
        sheet.save(target)
        pages.append(target)
    return pages


def select_tasks(per_split=2, seed_start=8000000):
    tasks = []
    for family in FAMILIES:
        for tier in TIERS:
            remaining = {"train": per_split, "validation": per_split}
            for seed in range(seed_start, seed_start + 100000):
                task = make_extreme_task(family, seed, tier)
                split = task["split"]
                if remaining.get(split):
                    tasks += [task, make_extreme_task(family, seed, tier, "text")]
                    remaining[split] -= 1
                if not any(remaining.values()):
                    break
            if any(remaining.values()):
                raise ValueError("Insufficient naturally split parents")
    return tasks


def audit_extreme(root):
    from .curriculum_metrics import audit_complex

    root = Path(root)
    report = audit_complex(root)
    errors = list(report["errors"])
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        rebuilt = make_extreme_task(
            task["family"],
            task["generator_seed"],
            task["difficulty"],
            task["input_modality"],
        )
        if {k: v for k, v in task.items() if k != "goal_images"} != json.loads(
            json.dumps(rebuilt)
        ):
            errors.append(
                "Task differs from deterministic specification: " + task["task_id"]
            )
        if task["split"] == "test":
            errors.append("Pilot must not export TEST")
        with tempfile.TemporaryDirectory(prefix="cad-extreme-audit-") as temporary:
            pages = render_extreme_goal(rebuilt, Path(temporary) / "goal.png")
            if len(pages) != len(task["goal_images"]):
                errors.append("Drawing sheet count differs: " + task["task_id"])
            else:
                for page, relative in zip(pages, task["goal_images"], strict=True):
                    path = (root / relative).resolve()
                    if (
                        not path.is_relative_to(root.resolve())
                        or not path.is_file()
                        or page.read_bytes() != path.read_bytes()
                    ):
                        errors.append(
                            "Drawing differs from numeric specification: "
                            + task["task_id"]
                        )
    return {**report, "ok": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parents-per-split", type=int, default=2)
    args = parser.parse_args()
    if args.parents_per_split < 1:
        parser.error("Positive parent count required")
    from .factory import generate_dataset

    tasks = select_tasks(args.parents_per_split)
    summary = generate_dataset(
        args.output,
        workers=4,
        timeout=120,
        preferences=False,
        task_specs=tasks,
        families=FAMILIES,
        modes=("create",),
    )
    root = Path(args.output)
    summary.update(
        benchmark=VERSION,
        drawing_profile="synthetic_parameter_schedule",
        standards_conformance="not_assessed",
        manufacturability_verified=False,
        strict_reward=True,
        supported_training_exports=["train.feedback.jsonl"],
        parent_counts=dict(
            Counter(t["split"] for t in {t["split_group"]: t for t in tasks}.values())
        ),
        scope="Prespecified challenge/extreme development pilot. No policy failure filtering or TEST exports; text condition includes the same drawing.",
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_extreme(root)
    (root / "extreme-audit.json").write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "accepted": summary["accepted"],
                "attempted": len(tasks),
                "audit_ok": audit["ok"],
                "parents": summary["parent_counts"],
            }
        )
    )
    if summary["accepted"] != len(tasks) or not audit["ok"]:
        raise RuntimeError("Extreme native/semantic audit failed")


if __name__ == "__main__":
    main()
