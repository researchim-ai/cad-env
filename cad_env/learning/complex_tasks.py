"""Compositional native CAD teachers and dimensioned image-to-CAD drawings."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from cad_env.agent.schema import SCHEMA_VERSION, validate_program


def make_complex_task(
    family, seed, modality="image", tier="iid", *, strict_reward=False
):
    rng = random.Random(f"complex-v1:{family}:{seed}")
    p = {
        "length": rng.randrange(72, 97, 4),
        "width": rng.randrange(52, 73, 4),
        "height": rng.randint(7, 12),
    }
    body = "Body"
    program = [{"op": "create_body", "name": body}]
    if family == "stacked_shaft":
        count = rng.choice([3, 4]) if tier == "iid" else rng.choice([5, 6])
        radii = [rng.randint(34, 40)]
        for _ in range(count - 1):
            radii.append(radii[-1] - rng.randint(4, 5))
        p = {"stages": [{"radius": r, "height": rng.randint(5, 14)} for r in radii]}
        for i, stage in enumerate(p["stages"]):
            sketch = {
                "op": "sketch_circles",
                "name": f"Profile{i}",
                "body": body,
                "circles": [{"x": 0, "y": 0, "radius": stage["radius"]}],
            }
            if i:
                sketch["support"] = f"Level{i - 1}"
            program.extend(
                [
                    sketch,
                    {
                        "op": "pad",
                        "name": f"Level{i}",
                        "body": body,
                        "profile": f"Profile{i}",
                        "length": stage["height"],
                    },
                ]
            )
        metrics = {
            "volume": sum(
                math.pi * s["radius"] ** 2 * s["height"] for s in p["stages"]
            ),
            "bounds": [
                -radii[0],
                -radii[0],
                0,
                radii[0],
                radii[0],
                sum(s["height"] for s in p["stages"]),
            ],
        }
        text = (
            "Create a coaxial circular stepped shaft. Stages from bottom to top (diameter, height) in mm: "
            + json.dumps([[2 * s["radius"], s["height"]] for s in p["stages"]])
            + ". Start at Z=0, center every stage on X=Y=0, and place each stage on the preceding stage."
        )
    else:
        program.extend(
            [
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
        )
        p["holes"] = []
        if family == "perforated_plate":
            positions = [
                (round(p["length"] * x / 4, 2), round(p["width"] * y / 3, 2))
                for x in (1, 2, 3)
                for y in (1, 2)
            ]
            rng.shuffle(positions)
            positions = positions[: rng.randint(4, 6)]
            for i, (x, y) in enumerate(positions):
                p["holes"].append(
                    {
                        "x": x,
                        "y": y,
                        "radius": rng.randint(2, 3),
                        "depth": p["height"]
                        if i % 2 == 0
                        else rng.randint(2, p["height"] - 2),
                    }
                )
            groups = {}
            for h in p["holes"]:
                groups.setdefault(h["depth"], []).append(h)
            for i, (depth, holes) in enumerate(sorted(groups.items())):
                program.extend(
                    [
                        {
                            "op": "sketch_circles",
                            "name": f"DrillSketch{i}",
                            "body": body,
                            "support": "BasePad",
                            "circles": [
                                {k: h[k] for k in ("x", "y", "radius")} for h in holes
                            ],
                        },
                        {
                            "op": "pocket",
                            "name": f"Cut{i}",
                            "body": body,
                            "profile": f"DrillSketch{i}",
                            "length": depth,
                        },
                    ]
                )
            volume = p["length"] * p["width"] * p["height"] - sum(
                math.pi * h["radius"] ** 2 * h["depth"] for h in p["holes"]
            )
            bounds = [0, 0, 0, p["length"], p["width"], p["height"]]
            text = (
                f"Create a plate {p['length']} x {p['width']} x {p['height']} mm at XY origin. Drill downward from its top surface. Holes (X,Y,diameter,depth), in mm: "
                + json.dumps(
                    [[h["x"], h["y"], 2 * h["radius"], h["depth"]] for h in p["holes"]]
                )
                + ". Depth equal to plate thickness means through. Keep all features parametric."
            )
        elif family in ("boss_plate", "mixed_boss_plate"):
            p.update(
                boss_radius=rng.randint(6, 8),
                boss_height=rng.randint(5, 11),
                bore_radius=rng.randint(2, 3),
            )
            positions = [
                (round(p["length"] * x / 3, 2), round(p["width"] * y / 3, 2))
                for x, y in [(1, 1), (2, 1), (1, 2), (2, 2)]
            ]
            rng.shuffle(positions)
            positions = positions[: rng.randint(2, 4)]
            p["bosses"] = [
                {"x": x, "y": y, "radius": p["boss_radius"]} for x, y in positions
            ]
            program.extend(
                [
                    {
                        "op": "sketch_circles",
                        "name": "BossSketch",
                        "body": body,
                        "support": "BasePad",
                        "circles": p["bosses"],
                    },
                    {
                        "op": "pad",
                        "name": "BossPad",
                        "body": body,
                        "profile": "BossSketch",
                        "length": p["boss_height"],
                    },
                ]
            )
            p["holes"] = [
                {
                    "x": x,
                    "y": y,
                    "radius": p["bore_radius"],
                    "depth": p["height"] + p["boss_height"],
                }
                for x, y in positions
            ]
            program.extend(
                [
                    {
                        "op": "sketch_circles",
                        "name": "BoreSketch",
                        "body": body,
                        "support": "BossPad",
                        "circles": [
                            {k: h[k] for k in ("x", "y", "radius")} for h in p["holes"]
                        ],
                    },
                    {
                        "op": "pocket",
                        "name": "BoreCut",
                        "body": body,
                        "profile": "BoreSketch",
                        "length": p["height"] + p["boss_height"],
                    },
                ]
            )
            volume = (
                p["length"] * p["width"] * p["height"]
                + len(positions) * math.pi * p["boss_radius"] ** 2 * p["boss_height"]
                - sum(math.pi * h["radius"] ** 2 * h["depth"] for h in p["holes"])
            )
            if family == "mixed_boss_plate":
                extra = {
                    "x": p["length"] / 2,
                    "y": p["width"] / 2,
                    "radius": 2,
                    "depth": round(p["height"] / 2, 2),
                }
                p["extra_hole"] = extra
                program.extend(
                    [
                        {
                            "op": "sketch_circles",
                            "name": "RecessSketch",
                            "body": body,
                            "support": "BasePad",
                            "circles": [{k: extra[k] for k in ("x", "y", "radius")}],
                        },
                        {
                            "op": "pocket",
                            "name": "Recess",
                            "body": body,
                            "profile": "RecessSketch",
                            "length": extra["depth"],
                        },
                    ]
                )
                volume -= math.pi * extra["radius"] ** 2 * extra["depth"]
            bounds = [0, 0, 0, p["length"], p["width"], p["height"] + p["boss_height"]]
            text = (
                f"Create a plate {p['length']} x {p['width']} x {p['height']} mm at XY origin. On its top add cylindrical bosses OD {2 * p['boss_radius']} mm, height {p['boss_height']} mm, at XY positions "
                + json.dumps(positions)
                + f". Bore each boss coaxially, diameter {2 * p['bore_radius']} mm, through both boss and base."
            )
            if "extra_hole" in p:
                text += f" Add a central blind hole diameter 4 mm and depth {p['extra_hole']['depth']} mm measured from the base plate top, outside the bosses."
        else:
            raise ValueError("Unknown complex family")
        metrics = {"volume": volume, "bounds": bounds}
    program = validate_program(program + [{"op": "finish"}])
    group = hashlib.sha256(
        json.dumps([family, p], sort_keys=True).encode()
    ).hexdigest()[:20]
    bucket = int(group[:8], 16) % 100
    split = (
        "test"
        if tier != "iid"
        else ("train" if bucket < 75 else "validation" if bucket < 90 else "test")
    )
    instruction = (
        text
        if modality == "text"
        else "Create one editable parametric solid matching the engineering drawing. Dimensions are in millimeters. Top views use X right and Y up; shaft stages are listed bottom-to-top. Respect through and blind depths. Finish only when complete."
    )
    task_id = "c-" + hashlib.sha256(f"{group}:{modality}".encode()).hexdigest()[:24]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
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
        "analytic_metrics": metrics,
        "split_group": group,
        "condition_group": group,
        "split": split,
        "max_steps": 24,
        "tolerance": 1e-5,
        "require_parametric": True,
        "observation_profile": "compact_v1",
        "goal_only_observation": True,
        "evaluation_tier": tier,
        "input_modality": modality,
        **({"reward_mode": "strict_success"} if strict_reward else {}),
        "goal_spec": {"kind": family, "parameters": p},
        "construction_signature": [
            (a["op"], len(a.get("circles", []))) for a in program
        ],
    }


def render_complex_goal(task, path):
    if task["family"] == "hollow_stacked_shaft":
        from .hard_tasks import render_hollow_goal

        return render_hollow_goal(task, path)
    from PIL import Image, ImageDraw, ImageFont

    p = task["parameters"]
    family = task["family"]
    scale = 2
    im = Image.new("RGB", (512 * scale, 640 * scale), "white")
    d = ImageDraw.Draw(im)

    def font(n):
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n * scale
        )

    def text(x, y, s, n=18):
        d.text((x * scale, y * scale), s, font=font(n), fill="black")

    text(18, 12, "ENGINEERING DRAWING / mm", 22)
    if family == "stacked_shaft":
        text(18, 45, "COAXIAL CIRCULAR STAGES; CENTER X=Y=0", 15)
        stages = p["stages"]
        total = sum(s["height"] for s in stages)
        rmax = stages[0]["radius"]
        factor = min(270 / (2 * rmax), 240 / total)
        z = 0
        for s in stages:
            x0 = (256 - s["radius"] * factor) * scale
            x1 = (256 + s["radius"] * factor) * scale
            y0 = (325 - (z + s["height"]) * factor) * scale
            y1 = (325 - z * factor) * scale
            d.rectangle((x0, y0, x1, y1), outline="black", width=2 * scale)
            z += s["height"]
        text(18, 350, "BOTTOM -> TOP: diameter D, height H", 18)
        for i, s in enumerate(stages):
            text(
                24,
                383 + i * 30,
                f"{i + 1}:  D={2 * s['radius']}    H={s['height']}",
                21,
            )
    else:
        text(18, 45, f"BASE: L={p['length']}  W={p['width']}  H={p['height']}", 20)
        factor = min(390 / p["length"], 230 / p["width"])

        def point(x, y):
            return ((60 + x * factor) * scale, (320 - y * factor) * scale)

        d.rectangle(
            (*point(0, p["width"]), *point(p["length"], 0)),
            outline="black",
            width=2 * scale,
        )
        if "bosses" in p:
            for c in p["bosses"]:
                x, y = point(c["x"], c["y"])
                r = c["radius"] * factor * scale
                d.ellipse(
                    (x - r, y - r, x + r, y + r), outline="black", width=2 * scale
                )
            text(
                18, 350, f"BOSSES: OD={2 * p['boss_radius']}  H={p['boss_height']}", 19
            )
            text(18, 376, f"BORES: D={2 * p['bore_radius']}, THROUGH BOSS+BASE", 17)
            start = 406
        else:
            text(18, 350, "HOLES: X, Y, DIAMETER D, DEPTH Z", 18)
            start = 382
        holes = p["holes"] + ([p["extra_hole"]] if "extra_hole" in p else [])
        center_labels = {}
        for i, h in enumerate(holes):
            x, y = point(h["x"], h["y"])
            r = h["radius"] * factor * scale
            d.ellipse((x - r, y - r, x + r, y + r), outline="black", width=2 * scale)
            center_labels.setdefault((h["x"], h["y"]), []).append(
                (chr(65 + i), h["radius"])
            )
            text(
                20,
                start + i * 29,
                f"{chr(65 + i)}: X={h['x']} Y={h['y']} D={2 * h['radius']} Z={h['depth']}",
                18,
            )
        for (cx, cy), labels in center_labels.items():
            x, y = point(cx, cy)
            radius = max(r for _, r in labels) * factor * scale
            d.text(
                (x + radius + 4 * scale, y - 8 * scale),
                "/".join(label for label, _ in labels),
                font=font(15),
                fill="black",
            )
        text(18, 604, "Origin: lower-left; X ->, Y up. Depth from top.", 15)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize((512, 640), Image.Resampling.LANCZOS).save(path)


def generate_complex(output, seeds=48, workers=4, strict_reward=False):
    from functools import partial

    from .factory import generate_dataset

    make_task = partial(make_complex_task, strict_reward=strict_reward)
    tasks = []
    families = ("perforated_plate", "stacked_shaft", "boss_plate", "mixed_boss_plate")
    for seed in range(seeds):
        for family in families[:3]:
            # Text and image conditions for the same latent part stay together.
            tasks.extend(make_task(family, seed, m) for m in ("text", "image"))
    for seed in range(max(8, seeds // 6)):
        for modality in ("text", "image"):
            tasks.append(
                make_task("stacked_shaft", 100000 + seed, modality, "compositional")
            )
            tasks.append(
                make_task("mixed_boss_plate", 100000 + seed, modality, "compositional")
            )
    summary = generate_dataset(
        output,
        seeds=seeds,
        workers=workers,
        task_specs=tasks,
        families=families,
        modes=("create",),
    )
    summary.update(
        benchmark="complex-cad-v4-strict" if strict_reward else "complex-cad-v3",
        supported_training_exports=["train.feedback.jsonl"],
        ood_modalities=["text", "image"],
        reward_mode="strict_success" if strict_reward else "dense_geometry",
    )
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--strict-reward", action="store_true")
    a = p.parse_args()
    s = generate_complex(a.output, a.seeds, a.workers, a.strict_reward)
    print(
        json.dumps(
            {
                k: v
                for k, v in s.items()
                if k not in ("artifact_sha256", "export_sha256", "source_sha256")
            },
            indent=2,
        )
    )
