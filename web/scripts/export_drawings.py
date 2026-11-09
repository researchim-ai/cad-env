"""Publish verified TRAIN drawing examples; never copy fonts or hidden test data."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/industrial-analogues-train-18-v1"
DESTINATION = ROOT / "web/assets/drawings"
FAMILIES = ("bearing_cartridge", "adapter_flange", "alignment_fixture")


def main():
    rows = [
        json.loads(s) for s in (SOURCE / "paired/pairs.jsonl").read_text().splitlines()
    ]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    examples = []
    for family in FAMILIES:
        row = next(
            r for r in rows if r["family"] == family and r["difficulty"] == "medium"
        )
        if row["split"] != "train":
            raise ValueError("Website examples must come from TRAIN")
        record = {
            "family": family,
            "task_id": row["task_id"],
            "split": "train",
            "profiles": {},
        }
        for standard in ("iso", "gost"):
            rep = row["representations"][standard]
            folder = Path(rep["pdf"]).parent
            native = json.loads((folder / "native.json").read_text())
            report = json.loads((folder / "conformance.json").read_text())
            if not (
                native["specification_verified"]
                and native["support_dependencies_verified"]
                and report["format_checks_passed"]
            ):
                raise ValueError("Unverified drawing example")
            if report["can_claim_full_conformance"]:
                raise ValueError("Unexpected full-conformance claim")
            files = {"sheet": folder / "sheet-01.png", "pdf": folder / "drawing.pdf"}
            if family == "bearing_cartridge":
                files["table"] = folder / "vlm/tile-06.png"
            entry = {
                "page_count": rep["page_count"],
                "full_conformance": False,
                "files": {},
            }
            for kind, source in files.items():
                name = f"{family}-{standard}-{kind}{source.suffix}"
                target = DESTINATION / name
                shutil.copyfile(source, target)
                entry["files"][kind] = {
                    "path": name,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            record["profiles"][standard] = entry
        examples.append(record)
    (DESTINATION / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(ROOT)),
                "examples": examples,
                "scope": "Original procedural TRAIN references, not model predictions or certified industrial drawings",
                "font": {
                    "name": "osifont",
                    "copyright": "Zefram Cochrane / hikikomori82 and contributors",
                    "upstream": "https://github.com/hikikomori82/osifont",
                    "commit": "afbc9134f90733eb5eea0a1c70c56a6d5dd5b758",
                    "license": "LGPLv3 with font exception; document uses vector outlines",
                    "font_files_included": False,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Exported three paired TRAIN examples, PDFs and VLM table crops; no font files."
    )


if __name__ == "__main__":
    main()
