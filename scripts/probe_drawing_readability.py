import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image

from cad_env.learning.batch_policy import BatchedHFPolicy

parser = argparse.ArgumentParser(
    "Short component probe: drawing-value extraction and JSON compliance, with blank controls"
)
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise ValueError("GPU1 only")
output = Path(args.output)
if output.exists():
    raise FileExistsError("Use a new output to preserve probe evidence")
root = Path(args.dataset)
from cad_env.drawings.readability import grade_response

questions = []
for family in ["heterogeneous_plate", "multi_support_fixture", "terraced_flange_stack"]:
    folder = None
    for manifest in sorted(root.glob("drawings/*/manifest.json")):
        candidate = json.loads((manifest.parent / "source-task.json").read_text())
        if candidate["family"] == family and candidate["difficulty"] == "extreme":
            folder = manifest.parent
            break
    if folder is None:
        raise ValueError("Missing extreme drawing family: " + family)
    task = json.loads((folder / "source-task.json").read_text())
    params = task["parameters"]
    tiles = json.loads((folder / "tiles.json").read_text())["tiles"]
    for index in [0, 3]:
        h = params["cuts"][index]
        row = int("base" in params) + len(params["pads"]) + index
        sheet = 2 + row // 18
        first = (row % 18 // 6) * 6 + 1
        last = min(18, first + 5)
        images = [
            folder / t["image"]
            for t in tiles
            if t["caption"].startswith(f"sheet-{sheet:02d}.png / rows {first}-{last} /")
        ]
        # Last crop may contain fewer than six remaining rows.
        if len(images) != 2:
            images = [
                folder / t["image"]
                for t in tiles
                if t["caption"].startswith(f"sheet-{sheet:02d}.png / rows {first}-")
            ]
        if len(images) != 2:
            raise ValueError("Expected matched position/depth crops")
        question = f"Read feature {h['name']} in the engineering drawing crops. Return ONLY JSON with numeric x, y, diameter, depth, counterbore_diameter, counterbore_depth. Use null for both counterbore fields when the table shows a dash. Diameter means the main hole diameter, not the counterbore."
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Read technical drawing tables carefully. Report only values actually visible in the images. If a value is unreadable, use null.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": question}]
                + [{"type": "image"} for _ in images],
            },
        ]
        expected = {
            "x": h["x"],
            "y": h["y"],
            "diameter": 2 * h["radius"],
            "depth": h["depth"],
            "counterbore_diameter": 2 * h["cb_radius"] if h["cb_radius"] else None,
            "counterbore_depth": h["cb_depth"],
        }
        questions.append(
            {
                "family": family,
                "feature": h["name"],
                "messages": messages,
                "images": images,
                "expected": expected,
            }
        )
policy = BatchedHFPolicy(
    args.model,
    max_context=8192,
    max_pixels=327680,
    max_new_tokens=192,
    temperature=0,
    chat_template_kwargs={"enable_thinking": False},
)
output.parent.mkdir(parents=True, exist_ok=True)
state = {
    "status": "running",
    "scope": "Six component table-reading questions on three synthetic parts; not native CAD task success",
    "answers": [],
}
output.write_text(json.dumps(state, indent=2))
for condition in ["full", "blank"]:
    for q in questions:
        paths = q["images"]
        if condition == "blank":
            blanks = []
            for index, path in enumerate(paths):
                target = Path("/tmp") / f"cad-drawing-blank-{index}.png"
                with Image.open(path) as im:
                    Image.new("RGB", im.size, "white").save(target)
                blanks.append(target)
            paths = blanks
        response = policy.generate_batch([(q["messages"], list(map(str, paths)))])[0]
        try:
            actual = json.loads(response)
            exact = set(actual) == set(q["expected"]) and all(
                (
                    actual[k] is None
                    if value is None
                    else isinstance(actual[k], (int, float))
                    and not isinstance(actual[k], bool)
                    and math.isclose(actual[k], value, abs_tol=1e-6, rel_tol=1e-9)
                )
                for k, value in q["expected"].items()
            )
        except (ValueError, TypeError, KeyError):
            exact = False
        state["answers"].append(
            {
                "condition": condition,
                "family": q["family"],
                "feature": q["feature"],
                "response": response,
                "expected": q["expected"],
                "exact": exact,
                **grade_response(response, q["expected"]),
            }
        )
        output.write_text(json.dumps(state, indent=2))
state.update(
    status="complete",
    full_exact=sum(a["exact"] for a in state["answers"] if a["condition"] == "full"),
    blank_exact=sum(a["exact"] for a in state["answers"] if a["condition"] == "blank"),
    questions_per_condition=len(questions),
)
output.write_text(json.dumps(state, indent=2))

state["field_exact_diagnostic"] = {
    c: sum(a["field_exact_diagnostic"] for a in state["answers"] if a["condition"] == c)
    for c in ["full", "blank"]
}
state["note"] = (
    "Exact primary scores require strict JSON. Literal recovery is a separate diagnostic only, never a CAD reward or agent-action override."
)
output.write_text(json.dumps(state, indent=2))
