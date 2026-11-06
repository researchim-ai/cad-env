"""Public-image perception plans and deterministic typed-tool execution.

Runtime functions use only public messages/pixels. They never open task files,
goal specifications, teacher actions or grading labels.
"""

import hashlib
import json
import math
import re
import threading
import uuid
from pathlib import Path

from cad_env.agent.schema import validate_program

SYSTEM = """Read the complete public goal and return ONLY one JSON perception plan.
All lengths are millimeters; the plan uses DIAMETERS, not radii. Read every marked
hole or drawing row before returning. Do not output individual CAD actions.
For marked-hole edits, use current sketch indices and include ALL and ONLY marked
holes, sorted by index:
{"kind":"selection","target":"sketch name","parameters":["diameter_i",...],"value":number}
For a plate with coaxial through holes and counterbores plus a separate blind hole:
{"kind":"counterbored_plate","base":[length,width,height],"centers":[[x,y],...],"through_diameter":number,"counterbore_diameter":number,"counterbore_depth":number,"blind":[x,y,diameter,depth]}
Repeated XY centers with a larger diameter and shorter depth describe a counterbore
at that same center. Include every center; do not merge unrelated drilling depths.
For a coaxial hollow stepped shaft with two symmetric bottom-flange mounting holes:
{"kind":"hollow_stacked_shaft","stages":[[diameter,height],...],"bore_diameter":number,"mount_offset":number,"mount_diameter":number}
Stages run bottom to top. The executor creates sketches/pads/pockets, converts
diameters to radii and finishes after the complete predicted plan. It has no target
labels and cannot correct a missing, extra or misread feature.
"""


def public_payload(messages):
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        blocks = (
            message["content"]
            if isinstance(message["content"], list)
            else [{"type": "text", "text": message["content"]}]
        )
        for block in blocks:
            if block.get("type") == "text":
                try:
                    value, _ = json.JSONDecoder().raw_decode(block["text"])
                except ValueError:
                    continue
                if isinstance(value, dict) and "task" in value and "state" in value:
                    return value
    raise ValueError("Missing public CAD state")


def plan_messages(messages, image_count):
    payload = public_payload(messages)
    instruction = (
        payload["task"]["instruction"]
        .replace(
            "Return one action at a time and finish when every selected hole is correct.",
            "",
        )
        .strip()
    )
    sketches = [
        {
            "name": o["name"],
            "circles": [
                {k: c[k] for k in ("index", "x", "y", "radius")} for c in o["circles"]
            ],
        }
        for o in payload["state"].get("objects", [])
        if o.get("circles")
    ]
    text = json.dumps(
        {"instruction": instruction, "current_sketches": sketches}, ensure_ascii=False
    )
    text += "\nFirst image is the complete public goal; second image, when present, is a zoom of the same pixels. Return a complete perception plan."
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
            + [{"type": "image"} for _ in range(image_count)],
        },
    ]


class PublicCropper:
    def __init__(self, output):
        self.output = Path(output)
        self.lock = threading.Lock()

    def images(self, messages, paths):
        if not paths:
            return []
        from PIL import Image

        path = Path(paths[0])
        mode = public_payload(messages)["task"]["mode"]
        # The same deterministic crop is used for all examples in a public mode;
        # there is no target-dependent ROI, OCR or private geometry access.
        with Image.open(path) as im:
            w, h = im.size
            box = (
                (0, round(h * 0.13), w, round(h * 0.88))
                if mode == "edit"
                else (0, round(h * 0.51), w, h)
            )
            key = hashlib.sha256(
                path.read_bytes() + json.dumps(box).encode()
            ).hexdigest()
            target = self.output / (key + ".png")
            with self.lock:
                self.output.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    temp = self.output / (key + ".tmp.png")
                    im.convert("RGB").crop(box).save(temp)
                    temp.replace(target)
        return [str(path), str(target)]


def number(x):
    if (
        isinstance(x, bool)
        or not isinstance(x, (float, int))
        or not math.isfinite(x)
        or not 0 < x <= 10000
    ):
        raise ValueError("Expected positive finite dimension")
    return float(x)


def coordinate(x):
    if (
        isinstance(x, bool)
        or not isinstance(x, (float, int))
        or not math.isfinite(x)
        or abs(x) > 10000
    ):
        raise ValueError("Invalid coordinate")
    return float(x)


def compile_plan(plan, messages):
    if not isinstance(plan, dict):
        raise TypeError("Plan must be an object")
    kind = plan.get("kind")
    if (kind == "selection") != (public_payload(messages)["task"]["mode"] == "edit"):
        raise ValueError("Plan kind does not match public task mode")
    body = "Body"
    actions = []
    if kind == "selection":
        if set(plan) != {"kind", "target", "parameters", "value"}:
            raise ValueError("Unexpected selection fields")
        sketches = {
            o["name"]: o
            for o in public_payload(messages)["state"].get("objects", [])
            if o.get("circles")
        }
        sketch = sketches.get(plan["target"])
        if (
            not sketch
            or not isinstance(plan["parameters"], list)
            or not plan["parameters"]
            or len(plan["parameters"]) > 16
        ):
            raise ValueError("Invalid public sketch/parameter list")
        params = plan["parameters"]
        if any(
            not isinstance(p, str) or not re.fullmatch(r"diameter_[0-9]+", p)
            for p in params
        ) or len(set(params)) != len(params):
            raise ValueError("Invalid or repeated parameter")
        indices = {c["index"] for c in sketch["circles"]}
        if any(int(p.split("_")[-1]) not in indices for p in params):
            raise ValueError("Unknown public circle index")
        actions = [
            {
                "op": "set_parameter",
                "target": plan["target"],
                "parameter": p,
                "value": number(plan["value"]),
            }
            for p in sorted(params, key=lambda p: int(p.split("_")[-1]))
        ]
    elif kind == "counterbored_plate":
        if set(plan) != {
            "kind",
            "base",
            "centers",
            "through_diameter",
            "counterbore_diameter",
            "counterbore_depth",
            "blind",
        }:
            raise ValueError("Unexpected plate fields")
        if (
            not isinstance(plan["base"], list)
            or len(plan["base"]) != 3
            or not isinstance(plan["blind"], list)
            or len(plan["blind"]) != 4
        ):
            raise ValueError("Invalid base/blind tuple")
        length, width, height = map(number, plan["base"])
        centers = plan["centers"]
        if (
            not isinstance(centers, list)
            or not 1 <= len(centers) <= 16
            or any(not isinstance(c, list) or len(c) != 2 for c in centers)
        ):
            raise ValueError("Invalid centers")
        centers = [[coordinate(x), coordinate(y)] for x, y in centers]
        inner = number(plan["through_diameter"]) / 2
        outer = number(plan["counterbore_diameter"]) / 2
        depth = number(plan["counterbore_depth"])
        x, y, diameter, blind_depth = plan["blind"]
        holes = [
            ([{"x": x, "y": y, "radius": inner} for x, y in centers], height),
            ([{"x": x, "y": y, "radius": outer} for x, y in centers], depth),
            (
                [
                    {
                        "x": coordinate(x),
                        "y": coordinate(y),
                        "radius": number(diameter) / 2,
                    }
                ],
                number(blind_depth),
            ),
        ]
        actions = [
            {"op": "create_body", "name": body},
            {
                "op": "sketch_rectangle",
                "name": "BaseSketch",
                "body": body,
                "length": length,
                "width": width,
            },
            {
                "op": "pad",
                "name": "BasePad",
                "body": body,
                "profile": "BaseSketch",
                "length": height,
            },
        ]
        for i, (circles, cut) in enumerate(holes):
            actions += [
                {
                    "op": "sketch_circles",
                    "name": f"DrillSketch{i}",
                    "body": body,
                    "support": "BasePad",
                    "circles": circles,
                },
                {
                    "op": "pocket",
                    "name": f"Cut{i}",
                    "body": body,
                    "profile": f"DrillSketch{i}",
                    "length": cut,
                },
            ]
    elif kind == "hollow_stacked_shaft":
        if set(plan) != {
            "kind",
            "stages",
            "bore_diameter",
            "mount_offset",
            "mount_diameter",
        }:
            raise ValueError("Unexpected shaft fields")
        stages = plan["stages"]
        if (
            not isinstance(stages, list)
            or not 1 <= len(stages) <= 8
            or any(not isinstance(s, list) or len(s) != 2 for s in stages)
        ):
            raise ValueError("Invalid stage list")
        stages = [[number(d), number(h)] for d, h in stages]
        actions = [{"op": "create_body", "name": body}]
        for i, (diameter, height) in enumerate(stages):
            sketch = {
                "op": "sketch_circles",
                "name": f"Profile{i}",
                "body": body,
                "circles": [{"x": 0, "y": 0, "radius": diameter / 2}],
            }
            if i:
                sketch["support"] = f"Level{i - 1}"
            actions += [
                sketch,
                {
                    "op": "pad",
                    "name": f"Level{i}",
                    "body": body,
                    "profile": f"Profile{i}",
                    "length": height,
                },
            ]
        actions += [
            {
                "op": "sketch_circles",
                "name": "BoreSketch",
                "body": body,
                "support": f"Level{len(stages) - 1}",
                "circles": [
                    {"x": 0, "y": 0, "radius": number(plan["bore_diameter"]) / 2}
                ],
            },
            {
                "op": "pocket",
                "name": "AxialBore",
                "body": body,
                "profile": "BoreSketch",
                "length": sum(h for _, h in stages),
            },
            {
                "op": "sketch_circles",
                "name": "MountSketch",
                "body": body,
                "support": "Level0",
                "circles": [
                    {
                        "x": sign * number(plan["mount_offset"]),
                        "y": 0,
                        "radius": number(plan["mount_diameter"]) / 2,
                    }
                    for sign in [-1, 1]
                ],
            },
            {
                "op": "pocket",
                "name": "MountHoles",
                "body": body,
                "profile": "MountSketch",
                "length": stages[0][1],
            },
        ]
    else:
        raise ValueError("Unknown plan kind")
    return validate_program(actions + [{"op": "finish"}])


class PlannedPolicy:
    def __init__(self, model, cropper, logs, seed):
        self.model, self.cropper, self.logs, self.seed = (
            model,
            cropper,
            Path(logs),
            seed,
        )
        self.actions = None

    def __call__(self, messages, image_paths=()):
        if self.actions is None:
            images = self.cropper.images(messages, image_paths)
            prompt = plan_messages(messages, len(images))
            raw = self.model(prompt, images)
            record = {
                "seed": self.seed,
                "raw_plan": raw,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt, sort_keys=True).encode()
                ).hexdigest(),
                "image_sha256": [
                    hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images
                ],
            }
            try:
                self.actions = compile_plan(json.loads(raw), messages)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                record["error"] = str(exc)
                self.actions = [{"op": "invalid_plan"}, {"op": "finish"}]
            self.logs.mkdir(parents=True, exist_ok=True)
            (self.logs / (uuid.uuid4().hex + ".json")).write_text(
                json.dumps(record, indent=2)
            )
        return json.dumps(self.actions.pop(0) if self.actions else {"op": "finish"})
