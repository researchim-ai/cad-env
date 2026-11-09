"""Reproducible native drawing bundles: FCStd, SVG sheets, vector PDF and PNG."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from .conformance import check_standard
from .profiles import ACTIVE, METADATA, get_profile, validate_metadata
from .sheets import PROFILE, document, frame, make_sheets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolated_env(out):
    env = os.environ.copy()
    for key, name in [
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_RUNTIME_DIR", "runtime"),
    ]:
        folder = out / ".runtime" / name
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(folder)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def export_drawing(
    model,
    output,
    *,
    task=None,
    timeout=180,
    standard="nominal",
    metadata=None,
    require_conformance=False,
):
    from cad_env.agent.backend import FreeCADBackend

    profile = get_profile(standard)
    metadata = validate_metadata(metadata or {})
    model = Path(model).resolve()
    out = Path(output).resolve()
    if model.suffix.lower() not in (".fcstd", ".step", ".stp") or not model.is_file():
        raise ValueError("Existing FCStd or STEP source required")
    if task is not None and (
        not task.get("parameters") or not {"pads", "cuts"} <= set(task["parameters"])
    ):
        raise ValueError(
            "No complete feature-schedule adapter for this task; export unannotated model explicitly instead"
        )
    if task is not None:
        parameters = task["parameters"]
        if set(parameters) - {"base", "pads", "cuts"}:
            raise ValueError("Unsupported drawing specification fields")
        for pad in parameters["pads"]:
            if set(pad) != {"name", "support", "x", "y", "radius", "height"}:
                raise ValueError("Unsupported or incomplete pad specification")
        for hole in parameters["cuts"]:
            if set(hole) != {
                "name",
                "support",
                "x",
                "y",
                "radius",
                "depth",
                "cb_radius",
                "cb_depth",
            }:
                raise ValueError("Unsupported or incomplete drilling specification")
    if profile is not None:
        from .font_cache import require_font

        require_font()
    out.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "preparing",
        "profile": profile.profile_id if profile else PROFILE,
        "drawing_standard": standard,
        "drawing_metadata": metadata,
        "standards_conformance": "not_assessed",
        "scope": "nominal geometry; manufacturing tolerances/material/surface finish not inferred",
        "fully_dimensioned_for_supported_spec": task is not None,
    }

    def persist():
        temp = out / "manifest.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(out / "manifest.json")

    persist()
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    package = Path(__file__).parent
    sources = list(package.glob("*.py")) + list((package / "assets").glob("*"))
    frozen = {str(p.relative_to(package)): sha(p) for p in sources if p.is_file()}
    for name in frozen:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
    if profile is not None:
        from .font_cache import provenance, require_font

        require_font()
        (out / "font-provenance.json").write_text(json.dumps(provenance(), indent=2))
    try:
        source = out / ("source" + model.suffix.lower())
        shutil.copyfile(model, source)
        if task is not None:
            (out / "source-task.json").write_text(json.dumps(task, indent=2))
        identity = task["split_group"] if task else sha(source)[:24]
        title = task["family"] if task else model.stem
        parameters = task["parameters"] if task else {}
        token = ACTIVE.set(profile)
        meta_token = METADATA.set(metadata)
        try:
            (out / "template.svg").write_text(document(frame(identity, title, 1, 1)))
        finally:
            METADATA.reset(meta_token)
            ACTIVE.reset(token)
        request = out / "native-request.json"
        request.write_text(
            json.dumps(
                {
                    "model": str(source),
                    "parameters": parameters,
                    "drawing_standard": standard,
                }
            )
        )
        worker = snapshot / "worker.py"
        bootstrap = out / "run-native.py"
        bootstrap.write_text(
            "import runpy\nrunpy.run_path("
            + repr(str(worker))
            + ')["main"]('
            + repr(str(request))
            + ","
            + repr(str(out / "native.json"))
            + ")\n"
        )
        backend = FreeCADBackend(timeout=timeout)
        env = isolated_env(out)
        with (out / "native.log").open("w") as log:
            subprocess.run(
                [backend.executable, str(bootstrap)],
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        if not (out / "native.json").exists():
            raise RuntimeError("FreeCAD produced no drawing result; inspect native.log")
        native = json.loads((out / "native.json").read_text())
        svgs = []
        for index, svg in enumerate(
            make_sheets(
                native,
                parameters,
                identity,
                title,
                standard=standard,
                metadata=metadata,
            ),
            1,
        ):
            path = out / f"sheet-{index:02d}.svg"
            path.write_text(svg)
            svgs.append(path)
        render = out / "render-request.json"
        render.write_text(
            json.dumps(
                {
                    "svgs": list(map(str, svgs)),
                    "pdf": str(out / "drawing.pdf"),
                    "title": title,
                    "png_width": 2400,
                }
            )
        )
        with (out / "render.log").open("w") as log:
            subprocess.run(
                [
                    "/usr/bin/python3",
                    str(snapshot / "qt_render.py"),
                    str(render),
                ],
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        from .tiles import make_tiles

        tiles = make_tiles(out, parameters, standard=standard)
        if any(
            sha(Path(__file__).parent / name) != digest
            for name, digest in frozen.items()
        ):
            raise RuntimeError("Drawing source changed during export")
        if profile:
            conformance = check_standard(out, standard, metadata)
            (out / "conformance.json").write_text(json.dumps(conformance, indent=2))
            status["standards_conformance"] = conformance["status"]
        status.update(
            status="complete",
            drawing_id=identity,
            title=title,
            units="mm",
            projection="first_angle",
            scale=native["scale"],
            page_count=len(svgs),
            source_sha256=sha(source),
            vlm_tiles=[t["image"] for t in tiles],
            native_spec_check_count=len(native["native_spec_checks"]),
            source_code_sha256=frozen,
            files={
                str(p.relative_to(out)): sha(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
                and ".runtime" not in p.parts
                and (
                    "source_snapshot" in p.parts
                    or p.suffix in (".svg", ".png", ".pdf", ".fcstd", ".FCStd", ".py")
                    or p.name
                    in (
                        "source-task.json",
                        "native.json",
                        "tiles.json",
                        "conformance.json",
                        "font-provenance.json",
                    )
                )
            },
            rebuild_after_model_edit=True,
        )
        persist()
        audit = audit_drawing(out)
        if not audit["ok"]:
            raise RuntimeError("Drawing bundle audit failed: " + str(audit["errors"]))
        if require_conformance and (
            not profile or not conformance["can_claim_full_conformance"]
        ):
            raise ValueError(
                "Full standard conformance is not confirmed; inspect conformance.json. Generated artifacts are retained for review."
            )
        return status
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        persist()
        raise


def audit_drawing(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    errors = []
    if manifest["status"] != "complete":
        errors.append("Incomplete drawing export")
    for relative, digest in manifest.get("files", {}).items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != digest:
            errors.append("Changed/missing/unsafe drawing artifact: " + relative)
    native = json.loads((root / "native.json").read_text())
    if not {"front", "top", "right", "section"} <= set(native.get("views", {})):
        errors.append("Missing model-derived projection/section")
    if (
        manifest.get("fully_dimensioned_for_supported_spec")
        and not native["specification_verified"]
    ):
        errors.append("Feature schedule not verified against native model")
    if manifest.get("fully_dimensioned_for_supported_spec") and not native.get(
        "support_dependencies_verified"
    ):
        errors.append("Native support dependencies were not verified")
    standard = manifest.get("drawing_standard", "nominal")
    profile = get_profile(standard)
    if profile:
        report = check_standard(root, standard, manifest.get("drawing_metadata", {}))
        saved = json.loads((root / "conformance.json").read_text())
        if saved != report or manifest.get("standards_conformance") != report["status"]:
            errors.append(
                "Conformance report does not match actual sheets and metadata"
            )
    elif manifest.get("standards_conformance") != "not_assessed":
        errors.append("Unreviewed standards-conformance claim")
    if manifest.get("projection") != "first_angle" or manifest.get("units") != "mm":
        errors.append("Unsupported or mismatched drawing convention")
    required_dimensions = {
        (d["view"], d["axis"]): d["value_mm"] for d in native.get("dimensions", [])
    }
    bounds = native.get("bounds_mm", [])
    wanted = (
        {
            ("top", 0): bounds[3] - bounds[0],
            ("top", 1): bounds[4] - bounds[1],
            ("front", 1): bounds[5] - bounds[2],
        }
        if len(bounds) == 6
        else {}
    )
    if set(required_dimensions) != set(wanted) or any(
        not math.isclose(required_dimensions[key], value, rel_tol=1e-9, abs_tol=1e-6)
        for key, value in wanted.items()
    ):
        errors.append("Native dimensions disagree with model bounds")
    if "source-task.json" in manifest.get("files", {}):
        task = json.loads((root / "source-task.json").read_text())
        expected = make_sheets(
            native,
            task["parameters"],
            manifest["drawing_id"],
            manifest["title"],
            standard=standard,
            metadata=manifest.get("drawing_metadata", {}),
        )
        for index, svg in enumerate(expected, 1):
            if (root / f"sheet-{index:02d}.svg").read_text() != svg:
                errors.append(
                    "SVG does not match native projection and verified specification"
                )
    return {
        "ok": not errors,
        "errors": errors,
        "files_checked": len(manifest.get("files", {})),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task")
    p.add_argument("--standard", choices=["nominal", "iso", "gost"], default="nominal")
    p.add_argument(
        "--metadata",
        help="JSON file with real document metadata; absent fields remain blank",
    )
    p.add_argument("--require-conformance", action="store_true")
    a = p.parse_args()
    task = json.loads(Path(a.task).read_text()) if a.task else None
    result = export_drawing(
        a.model,
        a.output,
        task=task,
        standard=a.standard,
        metadata=json.loads(Path(a.metadata).read_text()) if a.metadata else None,
        require_conformance=a.require_conformance,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "page_count", "native_spec_check_count", "profile")
            }
        )
    )


if __name__ == "__main__":
    main()
