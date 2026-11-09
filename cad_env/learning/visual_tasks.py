"""Counterfactual visual grounding tasks with private targets and native teachers.

Within a condition group, instruction and initial CAD state are identical.
Only the marker pixels select which editable hole must change.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from cad_env.agent.schema import SCHEMA_VERSION, validate_program


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_visual_group(
    seed,
    *,
    variants=4,
    selected_count=1,
    force_split=None,
    tier="iid",
    target_diameter=12,
    public_diameter=None,
    shuffle_edit_order=False,
    entity_labels=False,
):
    rng = random.Random(f"visual-grounding-v1:{seed}")
    length, width, height = (
        rng.randrange(100, 141, 5),
        rng.randrange(80, 111, 5),
        rng.randrange(8, 15),
    )
    radius = rng.choice([3, 4])
    locations = [
        (
            round(length * x / 4 + rng.randint(-2, 2), 2),
            round(width * y / 4 + rng.randint(-2, 2), 2),
        )
        for x in (1, 2, 3)
        for y in (1, 2, 3)
    ]
    rng.shuffle(locations)
    locations = locations[: rng.randint(max(4, selected_count + 1), 8)]
    # Order is intentionally unrelated to spatial position.
    circles = [{"x": x, "y": y, "radius": radius} for x, y in locations]
    base = {"length": length, "width": width, "height": height, "circles": circles}
    group = canonical_digest(base)[:20]
    bucket = int(group[:8], 16) % 100
    split = force_split or (
        "train" if bucket < 75 else "validation" if bucket < 90 else "test"
    )
    names = {k: k + str(rng.randrange(10, 100)) for k in ("B", "R", "P", "S", "K")}
    initial = validate_program(
        [
            {"op": "create_body", "name": names["B"]},
            {
                "op": "sketch_rectangle",
                "name": names["R"],
                "body": names["B"],
                "length": length,
                "width": width,
            },
            {
                "op": "pad",
                "name": names["P"],
                "body": names["B"],
                "profile": names["R"],
                "length": height,
            },
            {
                "op": "sketch_circles",
                "name": names["S"],
                "body": names["B"],
                "support": names["P"],
                "circles": circles,
            },
            {
                "op": "pocket",
                "name": names["K"],
                "body": names["B"],
                "profile": names["S"],
                "length": height,
            },
        ]
    )
    selections = list(itertools.combinations(range(len(circles)), selected_count))
    rng.shuffle(selections)
    selections = selections[:variants]
    if len(selections) != variants:
        raise ValueError("Too few counterfactual choices")
    angle = (
        rng.choice([0, 90, 180, 270])
        if tier != "render_ood"
        else rng.choice([45, 135, 225, 315])
    )
    style = rng.choice(["red", "blue"]) if tier != "render_ood" else "purple"
    requested = public_diameter or f"{target_diameter:g} mm"
    instruction = f"Resize every hole selected by a colored crosshair in the goal image to diameter {requested}. Use the X/Y arrows to match image positions to current sketch circles. Preserve all other dimensions and the existing feature history. Return one action at a time and finish when every selected hole is correct."
    if public_diameter:
        instruction += " All tool arguments must be in millimeters."
    if entity_labels:
        instruction += " Every hole has a black boxed integer equal to its CURRENT sketch circle index. For boxed label i use parameter diameter_i. Only colored crosshairs indicate the requested holes; black boxes label ALL holes and do not indicate selection."
    tasks = []
    for variant, selection in enumerate(selections):
        reference = copy.deepcopy(initial)
        for i in selection:
            reference[3]["circles"][i]["radius"] = target_diameter / 2
        identity = [group, selected_count, variant, tier]
        if target_diameter != 12:
            identity.append(target_diameter)
        if entity_labels:
            identity.append("entity-labels")
        target_id = canonical_digest(identity)[:24]
        edit_order = list(selection)
        if shuffle_edit_order:
            random.Random(f"teacher-order:{seed}:{variant}").shuffle(edit_order)
        task = {
            "schema_version": SCHEMA_VERSION,
            "task_id": "v-" + target_id,
            "family": "visual_grounding",
            "mode": "edit",
            "instruction": instruction,
            "public_task": {
                "task_id": "cad-task",
                "family": "visual_part",
                "mode": "edit",
                "instruction": instruction,
                "max_steps": 8,
            },
            "parameters": base,
            "initial_actions": copy.deepcopy(initial),
            "reference_actions": reference + [{"op": "finish"}],
            "oracle_actions": [
                {
                    "op": "set_parameter",
                    "target": names["S"],
                    "parameter": f"diameter_{i}",
                    "value": target_diameter,
                }
                for i in edit_order
            ]
            + [{"op": "finish"}],
            "analytic_metrics": {
                "volume": (
                    length * width
                    - math.pi
                    * (
                        (len(circles) - len(selection)) * radius**2
                        + len(selection) * (target_diameter / 2) ** 2
                    )
                )
                * height,
                "bounds": [0, 0, 0, length, width, height],
            },
            "split_group": group,
            "condition_group": group
            + f"-m{selected_count}-{tier}"
            + (f"-d{target_diameter:g}" if target_diameter != 12 else ""),
            "split": split,
            "max_steps": 8,
            "tolerance": 1e-5,
            "require_parametric": True,
            "reward_mode": "strict_success",
            "observation_profile": "compact_v1",
            "goal_only_observation": True,
            "evaluation_tier": tier,
            "goal_spec": {
                **base,
                "selected": list(selection),
                "angle": angle,
                "style": style,
                **({"entity_labels": True} if entity_labels else {}),
            },
        }
        tasks.append(task)
    for i, task in enumerate(tasks):
        peer = tasks[(i + 1) % len(tasks)]
        task["counterfactual_goal_images"] = [f"episodes/{peer['task_id']}/goal.png"]
    return tasks


def render_goal(task, path):
    if task["goal_spec"].get("kind") == "large_composition":
        from .extreme_tasks import render_extreme_goal

        return render_extreme_goal(task, path)
    if task["goal_spec"].get("kind", "marker") != "marker":
        from .complex_tasks import render_complex_goal

        return render_complex_goal(task, path)
    from PIL import Image, ImageDraw, ImageFont

    spec = task["goal_spec"]
    scale = 3
    image = Image.new("RGB", (512 * scale, 512 * scale), "white")
    d = ImageDraw.Draw(image)

    def font(size):
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size * scale
        )

    length, width = spec["length"], spec["width"]
    angle = math.radians(spec["angle"])
    c, s = math.cos(angle), math.sin(angle)
    factor = 300 / max(
        abs(c) * length + abs(s) * width, abs(s) * length + abs(c) * width
    )

    def point(x, y):
        x, y = x - length / 2, y - width / 2
        return (
            (256 + factor * (c * x - s * y)) * scale,
            (264 - factor * (s * x + c * y)) * scale,
        )

    outline = [
        point(0, 0),
        point(length, 0),
        point(length, width),
        point(0, width),
        point(0, 0),
    ]
    d.line(outline, fill="black", width=2 * scale)
    for circle in spec["circles"]:
        x, y = point(circle["x"], circle["y"])
        r = circle["radius"] * factor * scale
        d.ellipse((x - r, y - r, x + r, y + r), outline="black", width=2 * scale)
    if spec.get("entity_labels"):
        centers = [point(h["x"], h["y"]) for h in spec["circles"]]
        keepout = [
            max(h["radius"] * factor + 5, 12) * scale + 3 * scale
            for h in spec["circles"]
        ]
        boxes = []
        for index, (x, y) in enumerate(centers):
            chosen = None
            for distance in (38, 50, 64, 80):
                for dx, dy in (
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                ):
                    norm = math.hypot(dx, dy)
                    cx, cy = (
                        x + distance * scale * dx / norm,
                        y + distance * scale * dy / norm,
                    )
                    box = (
                        cx - 11 * scale,
                        cy - 12 * scale,
                        cx + 11 * scale,
                        cy + 12 * scale,
                    )
                    left, top, right, bottom = box
                    if (
                        left < 12 * scale
                        or right > 500 * scale
                        or top < 80 * scale
                        or bottom > 440 * scale
                    ):
                        continue
                    if any(
                        math.hypot(
                            max(left - hx, 0, hx - right), max(top - hy, 0, hy - bottom)
                        )
                        < radius
                        for (hx, hy), radius in zip(centers, keepout)
                    ):
                        continue
                    if any(
                        left < b[2] + 3 * scale
                        and right > b[0] - 3 * scale
                        and top < b[3] + 3 * scale
                        and bottom > b[1] - 3 * scale
                        for b in boxes
                    ):
                        continue
                    chosen = box
                    break
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(
                    "Cannot place visible entity labels without occluding geometry"
                )
            boxes.append(chosen)
            left, top, right, bottom = chosen
            cx, cy = (left + right) / 2, (top + bottom) / 2
            dx, dy = cx - x, cy - y
            norm = math.hypot(dx, dy)
            radius = spec["circles"][index]["radius"] * factor * scale
            # Draw leaders before colored markers, so no line erases a selection.
            d.line(
                (x + dx / norm * radius, y + dy / norm * radius, cx, cy),
                fill="black",
                width=scale,
            )
            d.rectangle(chosen, fill="white", outline="black", width=scale)
            d.text(
                (left + 5 * scale, top + scale), str(index), font=font(18), fill="black"
            )
    color = {"red": (200, 35, 40), "blue": (25, 85, 190), "purple": (140, 35, 160)}[
        spec["style"]
    ]
    for index in spec["selected"]:
        circle = spec["circles"][index]
        x, y = point(circle["x"], circle["y"])
        r = max(circle["radius"] * factor + 5, 12) * scale
        d.line((x - r, y, x + r, y), fill=color, width=3 * scale)
        d.line((x, y - r, x, y + r), fill=color, width=3 * scale)
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2 * scale)
    # Coordinate arrows share the exact rigid transform of the part.
    origin = point(0, 0)
    for axis, xy in [("X", (length * 0.23, 0)), ("Y", (0, width * 0.23))]:
        end = point(*xy)
        d.line([origin, end], fill=(50, 50, 50), width=2 * scale)
        dx, dy = end[0] - origin[0], end[1] - origin[1]
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        d.polygon(
            [
                end,
                (
                    end[0] - 8 * scale * ux + 4 * scale * uy,
                    end[1] - 8 * scale * uy - 4 * scale * ux,
                ),
                (
                    end[0] - 8 * scale * ux - 4 * scale * uy,
                    end[1] - 8 * scale * uy + 4 * scale * ux,
                ),
            ],
            fill="black",
        )
        d.text(
            (end[0] + 5 * scale, end[1] + 3 * scale), axis, font=font(16), fill="black"
        )
    d.text(
        (24 * scale, 20 * scale),
        "TOP VIEW / EDIT SELECTION",
        font=font(20),
        fill="black",
    )
    d.text(
        (24 * scale, 460 * scale),
        "Crosshair = selected hole; axes define orientation",
        font=font(15),
        fill="black",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((512, 512), Image.Resampling.LANCZOS).save(path)


def generate_visual(
    output,
    groups=128,
    workers=4,
    ood_groups=8,
    vary_diameter=False,
    shuffle_edit_order=False,
    entity_labels=False,
):
    from functools import partial

    from .factory import generate_dataset

    make_group = partial(
        make_visual_group,
        shuffle_edit_order=shuffle_edit_order,
        entity_labels=entity_labels,
    )
    tasks = []
    for seed in range(groups):
        count = 1 if seed % 2 == 0 else 2
        diameter = (
            random.Random(f"visual-diameter:{seed}").choice([10, 12, 14])
            if vary_diameter
            else 12
        )
        tasks.extend(make_group(seed, selected_count=count, target_diameter=diameter))
    for seed in range(ood_groups):
        tasks.extend(
            make_group(
                100000 + seed,
                selected_count=3,
                force_split="test",
                tier="compositional",
            )
        )
        if vary_diameter:
            tasks.extend(
                make_group(
                    300000 + seed,
                    selected_count=1 + seed % 2,
                    force_split="test",
                    tier="value_ood",
                    target_diameter=13,
                )
            )
            tasks.extend(
                make_group(
                    400000 + seed,
                    selected_count=1 + seed % 2,
                    force_split="test",
                    tier="unit_ood",
                    target_diameter=12.7,
                    public_diameter="0.5 inches",
                )
            )
        tasks.extend(
            make_group(
                200000 + seed,
                selected_count=1 + seed % 2,
                force_split="test",
                tier="render_ood",
            )
        )
    summary = generate_dataset(
        output,
        seeds=groups,
        workers=workers,
        preferences=False,
        families=("visual_grounding",),
        modes=("edit",),
        task_specs=tasks,
    )
    summary.update(
        benchmark=(
            "visual-grounding-v4-entities"
            if entity_labels
            else "visual-grounding-v3"
            if shuffle_edit_order
            else "visual-grounding-v2"
            if vary_diameter
            else "visual-grounding-v1"
        ),
        teacher_order="shuffled" if shuffle_edit_order else "ascending",
        entity_labels=entity_labels,
        supported_training_exports=["train.feedback.jsonl"],
        information_control="Identical instruction and initial CAD within each four-way group; only marker pixels differ. Greedy state-only ceiling <=25% per complete group.",
    )
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ood-groups", type=int, default=8)
    parser.add_argument("--vary-diameter", action="store_true")
    parser.add_argument("--shuffle-edit-order", action="store_true")
    parser.add_argument("--entity-labels", action="store_true")
    args = parser.parse_args()
    summary = generate_visual(
        args.output,
        args.groups,
        args.workers,
        args.ood_groups,
        args.vary_diameter,
        args.shuffle_edit_order,
        args.entity_labels,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("artifact_sha256", "export_sha256", "source_sha256")
            },
            indent=2,
        )
    )
