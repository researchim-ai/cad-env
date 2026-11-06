"""Fresh long-horizon construction and unlabelled multi-edit curricula.

All parents retain natural hash splits. Existing manifest parents, including
sealed TEST identities, are excluded without reading their target geometry.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from cad_env.agent.schema import SCHEMA_VERSION, validate_program


def make_hard_task(family, seed, modality="image"):
    rng = random.Random(f"hard-cad-v1:{family}:{seed}")
    body = "Body"
    program = [{"op": "create_body", "name": body}]
    if family == "hollow_stacked_shaft":
        radii = [rng.randint(35, 40)]
        for _ in range(rng.choice([4, 5])):
            radii.append(radii[-1] - rng.randint(4, 5))
        p = {
            "stages": [{"radius": r, "height": rng.randint(5, 12)} for r in radii],
            "bore_radius": rng.choice([2, 3]),
            "mount_radius": 1.1,
        }
        total = sum(s["height"] for s in p["stages"])
        for i, stage in enumerate(p["stages"]):
            sketch = {
                "op": "sketch_circles",
                "name": f"Profile{i}",
                "body": body,
                "circles": [{"x": 0, "y": 0, "radius": stage["radius"]}],
            }
            if i:
                sketch["support"] = f"Level{i - 1}"
            program += [
                sketch,
                {
                    "op": "pad",
                    "name": f"Level{i}",
                    "body": body,
                    "profile": f"Profile{i}",
                    "length": stage["height"],
                },
            ]
        program += [
            {
                "op": "sketch_circles",
                "name": "BoreSketch",
                "body": body,
                "support": f"Level{len(radii) - 1}",
                "circles": [{"x": 0, "y": 0, "radius": p["bore_radius"]}],
            },
            {
                "op": "pocket",
                "name": "AxialBore",
                "body": body,
                "profile": "BoreSketch",
                "length": total,
            },
            {
                "op": "sketch_circles",
                "name": "MountSketch",
                "body": body,
                "support": "Level0",
                "circles": [
                    {"x": sign * (radii[0] - 2.2), "y": 0, "radius": p["mount_radius"]}
                    for sign in [-1, 1]
                ],
            },
            {
                "op": "pocket",
                "name": "MountHoles",
                "body": body,
                "profile": "MountSketch",
                "length": p["stages"][0]["height"],
            },
        ]
        volume = sum(math.pi * s["radius"] ** 2 * s["height"] for s in p["stages"])
        volume -= (
            math.pi * p["bore_radius"] ** 2 * total
            + 2 * math.pi * p["mount_radius"] ** 2 * p["stages"][0]["height"]
        )
        bounds = [-radii[0], -radii[0], 0, radii[0], radii[0], total]
        text = (
            "Build a coaxial stepped shaft centered at X=Y=0, starting Z=0. Stages bottom-to-top (diameter,height), mm: "
            + json.dumps([[2 * s["radius"], s["height"]] for s in p["stages"]])
            + f". Add a central axial through bore diameter {2 * p['bore_radius']} mm through all stages."
            + f" Drill two diameter 2.2 mm mounting holes through only the bottom flange at X=+/-{radii[0] - 2.2:g}, Y=0. Keep all features parametric."
        )
    elif family == "counterbored_plate":
        p = {
            "length": rng.randrange(84, 113, 4),
            "width": rng.randrange(64, 89, 4),
            "height": rng.randint(10, 16),
            "bore_radius": rng.choice([2, 3]),
            "counter_radius": rng.choice([5, 6]),
            "counter_depth": rng.randint(2, 5),
        }
        centers = [
            (round(p["length"] * x, 2), round(p["width"] * y, 2))
            for x, y in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75)]
        ]
        p["holes"] = [
            {"x": x, "y": y, "radius": p["bore_radius"], "depth": p["height"]}
            for x, y in centers
        ]
        p["holes"] += [
            {"x": x, "y": y, "radius": p["counter_radius"], "depth": p["counter_depth"]}
            for x, y in centers
        ]
        p["holes"].append(
            {
                "x": 0.75 * p["length"],
                "y": 0.75 * p["width"],
                "radius": 2,
                "depth": round(p["height"] / 2, 2),
            }
        )
        program += [
            {
                "op": "sketch_rectangle",
                "name": "BaseSketch",
                "body": body,
                "length": p["length"],
                "width": p["width"],
            },
            {
                "op": "pad",
                "name": "BasePad",
                "body": body,
                "profile": "BaseSketch",
                "length": p["height"],
            },
        ]
        for i, holes in enumerate([p["holes"][:3], p["holes"][3:6], p["holes"][6:]]):
            program += [
                {
                    "op": "sketch_circles",
                    "name": f"DrillSketch{i}",
                    "body": body,
                    "support": "BasePad",
                    "circles": [{k: h[k] for k in ["x", "y", "radius"]} for h in holes],
                },
                {
                    "op": "pocket",
                    "name": f"Cut{i}",
                    "body": body,
                    "profile": f"DrillSketch{i}",
                    "length": holes[0]["depth"],
                },
            ]
        volume = (
            p["length"] * p["width"] * p["height"]
            - 3 * math.pi * p["bore_radius"] ** 2 * p["height"]
        )
        volume -= (
            3
            * math.pi
            * (p["counter_radius"] ** 2 - p["bore_radius"] ** 2)
            * p["counter_depth"]
            + math.pi * 4 * p["holes"][-1]["depth"]
        )
        bounds = [0, 0, 0, p["length"], p["width"], p["height"]]
        text = (
            f"Create a plate {p['length']} x {p['width']} x {p['height']} mm at XY origin. "
            f"At centers {json.dumps(centers)} drill diameter {2 * p['bore_radius']} mm through holes, "
            f"then enlarge only their top {p['counter_depth']} mm to diameter {2 * p['counter_radius']} mm (counterbores). "
            f"At ({p['holes'][-1]['x']},{p['holes'][-1]['y']}) drill a separate blind hole diameter 4 mm, depth {p['holes'][-1]['depth']} mm. Depths are measured downward from base top. Keep all features parametric."
        )
    else:
        raise ValueError("Unknown hard family")
    program = validate_program(program + [{"op": "finish"}])
    group = hashlib.sha256(
        json.dumps([family, p], sort_keys=True).encode()
    ).hexdigest()[:20]
    bucket = int(group[:8], 16) % 100
    split = "train" if bucket < 75 else "validation" if bucket < 90 else "test"
    instruction = (
        text
        if modality == "text"
        else "Create one editable parametric solid matching the engineering drawing. All dimensions are in millimeters. Follow the indicated feature supports and drilling depths; finish only when complete."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "h-"
        + hashlib.sha256(f"{group}:{modality}".encode()).hexdigest()[:24],
        "family": family,
        "mode": "create",
        "instruction": instruction,
        "public_task": {
            "task_id": "cad-task",
            "family": "complex_part",
            "mode": "create",
            "instruction": instruction,
            "max_steps": 24,
        },
        "parameters": p,
        "reference_actions": program,
        "oracle_actions": program,
        "initial_actions": [],
        "analytic_metrics": {"volume": volume, "bounds": bounds},
        "split_group": group,
        "condition_group": group,
        "split": split,
        "max_steps": 24,
        "tolerance": 1e-5,
        "require_parametric": True,
        "reward_mode": "strict_success",
        "observation_profile": "compact_v1",
        "goal_only_observation": True,
        "evaluation_tier": "iid",
        "input_modality": modality,
        "goal_spec": {"kind": family, "parameters": p},
        "construction_signature": [
            (a["op"], len(a.get("circles", []))) for a in program
        ],
    }


def render_hollow_goal(task, path):
    from PIL import Image, ImageDraw, ImageFont

    p = task["parameters"]
    im = Image.new("RGB", (768, 960), "white")
    d = ImageDraw.Draw(im)

    def text(x, y, s, size=24):
        d.text(
            (x, y),
            s,
            fill="black",
            font=ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
            ),
        )

    text(24, 16, "HOLLOW STEPPED SHAFT / mm", 30)
    total = sum(s["height"] for s in p["stages"])
    scale = min(480 / (2 * p["stages"][0]["radius"]), 310 / total)
    z = 0
    for stage in p["stages"]:
        d.rectangle(
            (
                384 - stage["radius"] * scale,
                400 - (z + stage["height"]) * scale,
                384 + stage["radius"] * scale,
                400 - z * scale,
            ),
            outline="black",
            width=3,
        )
        z += stage["height"]
    d.rectangle(
        (
            384 - p["bore_radius"] * scale,
            400 - total * scale,
            384 + p["bore_radius"] * scale,
            400,
        ),
        fill="white",
        outline="black",
        width=2,
    )
    text(24, 440, "STAGES BOTTOM -> TOP: DIAMETER, HEIGHT")
    for i, stage in enumerate(p["stages"]):
        text(32, 485 + 40 * i, f"{i + 1}: D={2 * stage['radius']}  H={stage['height']}")
    text(24, 755, f"CENTRAL BORE: D={2 * p['bore_radius']}, THROUGH ALL STAGES")
    text(24, 800, "TWO MOUNTING HOLES: D=2.2, THROUGH BOTTOM ONLY", 21)
    text(24, 840, f"XY CENTERS: (+/-{p['stages'][0]['radius'] - 2.2:g}, 0)")
    text(24, 895, "All stages coaxial; Z=0 at bottom. Keep feature history.", 20)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.resize((512, 640)).save(path)


def existing_parents():
    return {
        json.loads(line)["split_group"]
        for p in Path("artifacts").glob("*/manifest.jsonl")
        for line in p.read_text().splitlines()
        if line.strip()
    }


def construction_tasks(denied=()):
    tasks = []
    denied = set(denied)
    for family in ["counterbored_plate", "hollow_stacked_shaft"]:
        remaining = {"train": 48, "validation": 20, "test": 8}
        for seed in range(4000000, 4010000):
            task = make_hard_task(family, seed, "text")
            split = task["split"]
            group = task["split_group"]
            if remaining[split] and group not in denied:
                tasks += [task, make_hard_task(family, seed, "image")]
                denied.add(group)
                remaining[split] -= 1
            if not any(remaining.values()):
                break
        if any(remaining.values()):
            raise ValueError("Insufficient fresh construction parents")
    return tasks


def visual_tasks(denied=()):
    from .visual_tasks import canonical_digest, make_visual_group

    tasks = []
    denied = set(denied)
    remaining = {"train": 64, "validation": 20, "test": 8}
    for seed in range(5000000, 5010000):
        count = 2 + seed % 5
        inches = [0.375, 0.5, 0.625][seed % 3]
        variants = []
        for unit in ["mm", "inches"]:
            group = make_visual_group(
                seed,
                selected_count=count,
                target_diameter=round(inches * 25.4, 6),
                public_diameter=f"{inches:g} inches"
                if unit == "inches"
                else f"{round(inches * 25.4, 6):g} mm",
                shuffle_edit_order=True,
            )
            ids = {
                t["task_id"]: "u-"
                + canonical_digest([t["task_id"], "hard-unlabelled-v1", unit])[:24]
                for t in group
            }
            for t in group:
                t["task_id"] = ids[t["task_id"]]
                t["condition_group"] += "-hard-" + unit
                t["goal_spec"]["angle"] = (seed % 12) * 30
                t["counterfactual_goal_images"] = [
                    f"episodes/{ids[Path(p).parent.name]}/goal.png"
                    for p in t["counterfactual_goal_images"]
                ]
                t["curriculum_version"] = "hard-unlabelled-v1"
                variants.append(t)
        split = variants[0]["split"]
        parent = variants[0]["split_group"]
        if remaining[split] and parent not in denied:
            tasks += variants
            denied.add(parent)
            remaining[split] -= 1
        if not any(remaining.values()):
            break
    if any(remaining.values()):
        raise ValueError("Insufficient fresh unlabelled parents")
    return tasks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["construction", "visual"], required=True)
    args = p.parse_args()
    from .curriculum_metrics import audit_complex
    from .factory import generate_dataset
    from .visual_audit import audit_visual

    tasks = (
        construction_tasks(existing_parents())
        if args.kind == "construction"
        else visual_tasks(existing_parents())
    )
    summary = generate_dataset(
        args.output,
        workers=4,
        preferences=False,
        task_specs=tasks,
        families=tuple(sorted({t["family"] for t in tasks})),
        modes=("create",) if args.kind == "construction" else ("edit",),
    )
    summary.update(
        benchmark="complex-cad-hard-v1"
        if args.kind == "construction"
        else "visual-grounding-hard-unlabelled-v1",
        supported_training_exports=["train.feedback.jsonl"],
        strict_reward=True,
        parent_counts=dict(
            Counter(t["split"] for t in {t["split_group"]: t for t in tasks}.values())
        ),
        scope="Fresh parents only. New curriculum version includes longer compositions in TRAIN; old TEST identities excluded, no claim that previous composition categories remain unseen.",
    )
    root = Path(args.output)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_complex(root) if args.kind == "construction" else audit_visual(root)
    (root / "hard-audit.json").write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "accepted": summary["accepted"],
                "requested": len(tasks),
                "audit_ok": audit["ok"],
                "parents": summary["parent_counts"],
            }
        )
    )
    if summary["accepted"] != len(tasks) or not audit["ok"]:
        raise RuntimeError("Hard corpus failed native or semantic audit")


if __name__ == "__main__":
    main()
