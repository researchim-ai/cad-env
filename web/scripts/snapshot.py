"""Export public project facts and TRAIN reference meshes into the static site."""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
source = ROOT / "artifacts/multicad-expanded-freecad"
if not source.exists():
    raise SystemExit(
        "Native source corpus is missing; existing website snapshot preserved."
    )
examples = []
if source.exists():
    rows = [json.loads(l) for l in (source / "manifest.jsonl").read_text().splitlines()]
    for family in ["bolt_flange", "counterbored_plate", "stepped_bore"]:
        row = next(
            r
            for r in rows
            if r["accepted"]
            and r["split"] == "train"
            and r["family"] == family
            and r["mode"] == "create"
        )
        folder = source / "episodes" / row["task_id"]
        task = json.loads((folder / "task.json").read_text())
        target = WEB / "assets" / (family + ".json")
        shutil.copyfile(folder / "reference/mesh.json", target)
        examples.append(
            {
                "family": family,
                "mesh": "assets/" + target.name,
                "actions": task["oracle_actions"],
                "provenance": "Native FreeCAD reference · TRAIN example",
            }
        )
results = []
for engine in ["freecad", "cadquery"]:
    before = (
        ROOT
        / "artifacts/multicad-scale-experiment-v2"
        / ("baseline-" + engine)
        / "evaluation.json"
    )
    after = ROOT / "artifacts/checkpoint500-evaluation" / engine / "evaluation.json"
    if before.exists() and after.exists():
        a, b = json.loads(before.read_text()), json.loads(after.read_text())
        if b["status"] == "complete":
            results.append(
                {
                    "engine": engine,
                    "tasks": len(b["episodes"]),
                    "before": sum(r["task_success"] for r in a["episodes"]),
                    "after": sum(r["task_success"] for r in b["episodes"]),
                    "groups": len({r["split_group"] for r in b["episodes"]}),
                }
            )
retention = []
for name, label, parent in [
    ("retention", "Simple tasks", "vision-leap-c-retention"),
    ("complex", "Complex builds", "vision-leap-c-complex"),
    ("grounding", "Visual selection", "vision-leap-c-grounding"),
    ("entities", "Entity-labelled selection", "vision-leap-entities-candidate"),
]:
    path = ROOT / "artifacts/checkpoint500-batch-evaluation" / name / "evaluation.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report["status"] == "complete":
            before = json.loads(
                (ROOT / "artifacts" / parent / "evaluation.json").read_text()
            )
            retention.append(
                {
                    "name": label,
                    "tasks": len(report["episodes"]),
                    "before": sum(r["task_success"] for r in before["episodes"]),
                    "after": sum(r["task_success"] for r in report["episodes"]),
                    "execution": "Batch ≤4 · 8 episode workers",
                }
            )
state_path = ROOT / "artifacts/checkpoint500-batch-evaluation/status.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
status = (
    "Validation complete; retention uses the batch execution profile."
    if state.get("status") == "complete"
    else "Remaining skill-retention checks are in progress."
)
(WEB / "docs").mkdir(exist_ok=True)
for src, dst in [
    ("MULTICAD.md", "multicad.md"),
    ("GRPO_AND_DATA_AUDIT.md", "training-audit.md"),
    ("BENCHMARK_COVERAGE.md", "benchmark-coverage.md"),
    ("SOLIDCAD.md", "solidcad.md"),
    ("DRAWING_STANDARDS.md", "drawing-standards.md"),
    ("INDUSTRIAL_GENERATOR.md", "industrial-generator.md"),
]:
    shutil.copyfile(ROOT / "docs" / src, WEB / "docs" / dst)
    if dst == "training-audit.md":
        path = WEB / "docs" / dst
        path.write_text(
            "> Historical experiment report. Runtime descriptions and measured scores below refer to those runs. "
            "Current supported backends are FreeCAD and CadQuery; see the project guide and dated updates for the current implementation.\n\n"
            + path.read_text()
        )
data = {
    "updated": datetime.now(timezone(timedelta(hours=5))).isoformat(timespec="minutes"),
    "repository": "https://github.com/researchim-ai/cad-env",
    "examples": examples,
    "results": results,
    "retention": retention,
    "evaluation_status": status,
    "visual_corpora": [
        {"name": "V5 · multi-edit", "episodes": 4352, "physical_groups": 544},
        {"name": "V6 · units", "episodes": 2560, "physical_groups": 160},
    ],
    "training": {
        "checkpoint": 500,
        "rank": 16,
        "parameters": 15400960,
        "vision_parameters": 1024000,
        "base": "Qwen3.5-4B",
        "method": "BF16 LoRA",
    },
}
(WEB / "data.json").write_text(json.dumps(data, indent=2))
print(
    "Exported",
    len(examples),
    "native meshes,",
    len(results),
    "mechanical cohorts and",
    len(retention),
    "retention cohorts",
)
