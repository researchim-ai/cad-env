"""Audit an existing dataset package before any training or publication."""

import hashlib
import json
from pathlib import Path


def audit_dataset(root):
    root = Path(root).resolve()
    summary = json.loads((root / "summary.json").read_text())
    errors = []
    for relative, digest in {
        **summary.get("export_sha256", {}),
        **summary.get("artifact_sha256", {}),
    }.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append("Missing/unsafe artifact: " + relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append("Hash mismatch: " + relative)
    groups = {}
    counts = {}
    for export in sorted((root / "exports").glob("*.jsonl")):
        split, kind, _ = export.name.split(".")
        rows = [
            json.loads(line) for line in export.read_text().splitlines() if line.strip()
        ]
        counts[split + "." + kind] = len(rows)
        for row in rows:
            groups.setdefault(row["split_group"], set()).add(split)
            serialized = json.dumps(row["prompt"])
            if "reference_actions" in serialized or "oracle_actions" in serialized:
                errors.append("Private answer in prompt: " + row["task_id"])
            if kind == "grpo" and ("completion" in row or "chosen" in row):
                errors.append("GRPO prompt export contains an answer")
            for image in row.get("images", []):
                path = (root / image).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    errors.append("Missing/unsafe image: " + image)
    for group, splits in groups.items():
        if len(splits) != 1:
            errors.append("Split leakage: " + group)
    if counts != summary["counts"]:
        errors.append("Export counts disagree with summary")
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for episode in manifest:
        if episode["accepted"]:
            output = root / "episodes" / episode["task_id"] / "oracle" / "output.FCStd"
            if (
                not output.is_file()
                or hashlib.sha256(output.read_bytes()).hexdigest()
                != episode["artifact_sha256"]
            ):
                errors.append(
                    "Candidate FCStd missing or changed: " + episode["task_id"]
                )
    if (
        len(manifest) != summary["attempted"]
        or sum(row["accepted"] for row in manifest) != summary["accepted"]
    ):
        errors.append("Acceptance counts disagree with manifest")
    return {
        "ok": not errors,
        "errors": errors,
        "episodes": len(manifest),
        "split_groups": len(groups),
        "export_rows": sum(counts.values()),
        "hashes_checked": len(summary.get("export_sha256", {}))
        + len(summary.get("artifact_sha256", {})),
    }
