"""Pair native projections with dimensioned specification sheets, using draft profiles."""

import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image

from cad_env.drawings.export import export_drawing, isolated_env, sha
from cad_env.drawings.font_cache import provenance
from cad_env.drawings.profiles import ACTIVE, METADATA, get_profile
from cad_env.drawings.sheets import document, frame, overview, text


def specification_lines(spec, standard):
    def num(v):
        if abs(v) < 0.0000005:
            v = 0
        return f"{v:.6f}".rstrip("0").rstrip(".")

    def coords(values):
        return "(" + "; ".join(num(v) for v in values) + ")"

    ru = standard == "gost"
    lines = [
        "Единицы: мм. Координаты XYZ; разделитель координат - точка с запятой."
        if ru
        else "Units: mm. World XYZ. Coordinate components separated by semicolons.",
        "Глубина отверстия направлена против нормали N. Глубина цековки отсчитывается от той же плоскости."
        if ru
        else spec["convention"],
    ]
    for i, f in enumerate(spec["forms"], 1):
        label = f"F{i}"
        if f["kind"] == "revolved radial/Z polygon":
            lines.append(
                f"{label} / "
                + (
                    "Сечение r; Z. Вращение вокруг оси "
                    if ru
                    else "Section r; Z. Revolution about axis "
                )
                + coords(f["axis"])
                + f" / {num(f['angle'])} deg"
            )
            lines.extend(
                f"  V{j:02d}: r={num(p[0])}; Z={num(p[1])}"
                for j, p in enumerate(f["points"], 1)
            )
        elif f["kind"] == "rectangular base":
            lines.append(
                f"{label} / "
                + ("Прямоугольная плита" if ru else "Rectangular base")
                + f": L={num(f['length'])}; W={num(f['width'])}; H={num(f['height'])}; XYZ {coords(f['origin'])}"
            )
        elif f["kind"] == "ruled rectangular loft":
            lines.append(
                f"{label} / "
                + (
                    "Переход между прямоугольниками, линейные образующие."
                    if ru
                    else "Ruled transition between rectangular sections."
                )
            )
            for key in ("bottom", "top"):
                lines.append(
                    f"  {key.upper()} / XYZ {coords(f[key + '_origin'])}; L={num(f[key + '_size'][0])}; W={num(f[key + '_size'][1])}"
                )
        else:
            lines.append(
                f"{label} / "
                + (
                    "Профиль XZ; выдавливание по нормали "
                    if ru
                    else "XZ profile; extrude along normal "
                )
                + coords(f["normal"])
                + f"; L={num(f['depth'])}"
            )
            lines.append("  XYZ " + coords(f["origin"]))
            lines.extend(
                f"  V{j:02d}: X={num(p[0])}; Z={num(p[1])}"
                for j, p in enumerate(f["points"], 1)
            )
    lines.append(
        "Отверстия: XYZ - центр входа; N - наружная нормаль; CB - цековка."
        if ru
        else "Holes: XYZ entry center; N outward normal; CB counterbore; D diameter; DEPTH axial depth."
    )
    for i, h in enumerate(spec["holes"], 1):
        lines.append(
            f"H{i:02d} XYZ {coords(h['center'])} N {coords(h['normal'])} D{num(2 * h['radius'])} / DEPTH {num(h['depth'])} CB D{num(2 * h['cb_radius'])} / DEPTH {num(h['cb_depth'])}"
        )
    return [part for line in lines for part in textwrap.wrap(line, 105)]


def export_pair(step, task, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    spec = task["design_spec"]
    result = {}
    for standard in ("iso", "gost"):
        out = root / standard
        out.mkdir()
        views = out / "views"
        export_drawing(step, views, standard=standard)
        native = json.loads((views / "native.json").read_text())
        lines = specification_lines(spec, standard)
        groups = [lines[i : i + 30] for i in range(0, len(lines), 30)]
        total = 1 + len(groups)
        token = ACTIVE.set(get_profile(standard))
        meta = METADATA.set({})
        try:
            sheets = [overview(native, task["split_group"], task["family"], total)]
            for page, rows in enumerate(groups, 2):
                body = frame(task["split_group"], task["family"], page, total, "NTS")
                body += text(
                    24,
                    24,
                    "DIMENSIONAL SPECIFICATION"
                    if standard == "iso"
                    else "ГЕОМЕТРИЯ И РАЗМЕРЫ",
                    4.2,
                )
                for j, value in enumerate(rows):
                    body += text(24, 39 + j * 6.5, value, 3.5)
                sheets.append(document(body))
            for i, svg in enumerate(sheets, 1):
                (out / f"sheet-{i:02d}.svg").write_text(svg)
        finally:
            METADATA.reset(meta)
            ACTIVE.reset(token)
        request = out / "render-request.json"
        request.write_text(
            json.dumps(
                {
                    "svgs": [
                        str((out / f"sheet-{i:02d}.svg").absolute())
                        for i in range(1, total + 1)
                    ],
                    "pdf": str((out / "drawing.pdf").absolute()),
                    "title": task["family"],
                    "png_width": 2400,
                }
            )
        )
        with (out / "render.log").open("w") as log:
            subprocess.run(
                [
                    "/usr/bin/python3",
                    str(Path(__file__).parents[1] / "drawings/qt_render.py"),
                    str(request.absolute()),
                ],
                env=isolated_env(out),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=True,
            )
        # Retain the existing readable native view crops; add every spec row in
        # traceable small crops. No separate text transcription is fed to the VLM.
        images = [
            str(Path("views") / p)
            for p in json.loads((views / "manifest.json").read_text())["vlm_tiles"]
        ]
        (out / "spec-vlm").mkdir()
        records = []
        for page, rows in enumerate(groups, 2):
            with Image.open(out / f"sheet-{page:02d}.png") as im:
                unit = im.width / 420
                for start in range(0, len(rows), 6):
                    count = min(6, len(rows) - start)
                    rect = [21, 33 + start * 6.5, 408, 42 + (start + count - 1) * 6.5]
                    crop = im.crop(tuple(round(v * unit) for v in rect))
                    name = f"spec-vlm/page-{page:02d}-rows-{start:02d}.png"
                    crop.save(out / name)
                    effective = (
                        3.5
                        * unit
                        * min(1, (327680 / (crop.width * crop.height)) ** 0.5)
                    )
                    if effective < 10.5:
                        raise ValueError(
                            "Specification crop is unreadable under VLM pixel budget"
                        )
                    images.append(name)
                    records.append(
                        {
                            "image": name,
                            "source_sheet": page,
                            "rectangle_mm": rect,
                            "min_font_px_at_budget": effective,
                        }
                    )
        (out / "design-spec.json").write_text(json.dumps(spec, indent=2))
        (out / "specification-lines.json").write_text(
            json.dumps(lines, ensure_ascii=False, indent=2)
        )
        (out / "font-provenance.json").write_text(json.dumps(provenance(), indent=2))
        manifest = {
            "status": "complete",
            "standard": standard,
            "split_group": task["split_group"],
            "split": task["split"],
            "source_step_sha256": sha(step),
            "spec_sha256": sha(out / "design-spec.json"),
            "pages": total,
            "vlm_images": images,
            "spec_crops": records,
            "can_claim_full_conformance": False,
            "scope": "Experimental profile framing and native projections plus dimensioned specification appendix; not full manufacturing drawings or full ISO/GOST conformance.",
            "files": {
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and ".runtime" not in p.parts
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result[standard] = {
            "root": str(out),
            "images": [str(out / p) for p in images],
            "pdf": str(out / "drawing.pdf"),
        }
    if sha(root / "iso/design-spec.json") != sha(root / "gost/design-spec.json"):
        raise ValueError("Paired profiles describe different geometry")
    audit_pair(root)
    return result


def audit_pair(root):
    root = Path(root).absolute()
    manifests = []
    for standard in ("iso", "gost"):
        folder = root / standard
        m = json.loads((folder / "manifest.json").read_text())
        if m["standard"] != standard or m["can_claim_full_conformance"]:
            raise ValueError("Incorrect profile/conformance declaration")
        for relative, digest in m["files"].items():
            p = (folder / relative).resolve()
            if (
                not p.is_relative_to(folder.resolve())
                or not p.is_file()
                or sha(p) != digest
            ):
                raise ValueError("Drawing artifact changed or unsafe: " + relative)
        for relative in m["vlm_images"]:
            if relative not in m["files"]:
                raise ValueError("Untracked VLM crop")
        if any(c["min_font_px_at_budget"] < 10.5 for c in m["spec_crops"]):
            raise ValueError("Unreadable specification crop")
        manifests.append(m)
    for key in ("source_step_sha256", "spec_sha256", "split_group", "split"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError("Paired geometry/lineage mismatch: " + key)
    return {
        "ok": True,
        "profiles": 2,
        "scope": "Hash, lineage and crop checks; not full standard conformance.",
    }
