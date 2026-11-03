"""Immutable image bundles for on-policy recovery SFT without changing base data."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

from .factory import jsonl


def bundle_supervision(rows, output_dir, parent_dataset_digest=None):
    root = Path(output_dir).resolve()
    export = root / "exports/train.feedback.jsonl"
    export.parent.mkdir(parents=True, exist_ok=True)
    assets = {}
    packaged = []
    for source in rows:
        row = copy.deepcopy(source)
        if "images" in row:
            paths = []
            for path in row["images"]:
                path = Path(path).resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                relative = "images/" + digest + ".png"
                destination = root / relative
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(path, destination)
                assets[relative] = digest
                paths.append(relative)
            row["images"] = paths
        packaged.append(row)
    jsonl(export, packaged)
    manifest = {
        "schema_version": "cad-env.supervision-bundle.v1",
        "parent_dataset_digest": parent_dataset_digest,
        "export": "exports/train.feedback.jsonl",
        "export_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
        "assets": assets,
        "rows": len(packaged),
        "task_ids": sorted({row["task_id"] for row in packaged}),
    }
    (root / "supervision.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_bundle(root, source):
    root = Path(root).resolve()
    source = Path(source).resolve()
    manifest = json.loads((root / "supervision.json").read_text())
    if source != root / manifest["export"]:
        raise ValueError("Export does not belong to this supervision bundle")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["export_sha256"]:
        raise ValueError("Supervision export was changed")
    for relative, expected in manifest["assets"].items():
        file = (root / relative).resolve()
        if (
            not file.is_relative_to(root)
            or not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                "Image bundle is missing, modified, or unsafe: " + relative
            )
    return manifest
