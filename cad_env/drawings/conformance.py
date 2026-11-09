"""Evidence-based, scoped checks. Passing these checks never certifies all standards."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .profiles import get_profile

DATE = "2026-11-09"


def check_standard(root, standard, metadata):
    root = Path(root)
    profile = get_profile(standard)
    checks = []

    def check(code, passed, reference, evidence):
        checks.append(
            {
                "id": code,
                "status": "pass" if passed else "fail",
                "reference": reference,
                "evidence": evidence,
            }
        )

    if profile is None:
        return {
            "requested_standard": standard,
            "status": "not_assessed",
            "can_claim_full_conformance": False,
            "checks": [],
            "pending": ["Legacy nominal profile is not an ISO or ESKD profile."],
        }
    native = json.loads((root / "native.json").read_text())
    sheets = sorted(root.glob("sheet-*.svg"))
    font = root / "source_snapshot/assets/osifont.ttf"
    font_sha = hashlib.sha256(font.read_bytes()).hexdigest() if font.exists() else None
    provenance = root / "font-provenance.json"
    if provenance.exists():
        from .font_cache import FONT_SHA

        value = json.loads(provenance.read_text()).get("font_sha256")
        font_sha = value if value == FONT_SHA else None
    check("pages_present", bool(sheets), "Export integrity", {"pages": len(sheets)})
    rendered = []
    for index, path in enumerate(sheets, 1):
        svg = ET.fromstring(path.read_text())
        check(
            f"page_{index}.size",
            svg.get("width") == "420mm" and svg.get("height") == "297mm",
            profile.references[0][0]
            if standard == "iso"
            else "ГОСТ 2.301-68, таблица 1",
            dict(svg.attrib),
        )
        elements = {e.get("id"): e for e in svg.iter() if e.get("id")}
        border = elements.get("standard-border")
        l, t, r, b = profile.margins
        expected = {"x": l, "y": t, "width": 420 - l - r, "height": 297 - t - b}
        actual = (
            {k: float(border.get(k)) for k in expected} if border is not None else {}
        )
        check(
            f"page_{index}.margins",
            actual == expected,
            "ISO 5457:1999, 4.1"
            if standard == "iso"
            else "ГОСТ Р 2.104-2023, приложение Б",
            actual,
        )
        block_id = (
            "iso-title-block"
            if standard == "iso"
            else "gost-form-1"
            if index == 1
            else "gost-form-2a"
        )
        block = elements.get(block_id)
        size = (
            (180, 50) if standard == "iso" else (185, 55) if index == 1 else (185, 15)
        )
        correct = (
            block is not None
            and (float(block.get("width")), float(block.get("height"))) == size
        )
        check(
            f"page_{index}.title_block_envelope",
            correct,
            "ISO 7200:2004, 6 (width only; height is project layout)"
            if standard == "iso"
            else "ГОСТ Р 2.104-2023, Б.1 / Б.3",
            {"id": block_id, "expected_mm": list(size)},
        )
        if standard == "gost":
            duplicate = elements.get("gost-field-26")
            check(
                f"page_{index}.duplicate_designation",
                duplicate is not None
                and duplicate.get("transform") == "translate(90 19) rotate(180)",
                "ГОСТ Р 2.104-2023, графа 26 / приложение Б",
                {"present": duplicate is not None},
            )
        check(
            f"page_{index}.border_line_width",
            border is not None and float(border.get("stroke-width", 0)) == profile.wide,
            "Profile line-group check; complete line convention review pending",
            {"wide_mm": profile.wide, "thin_mm": profile.thin},
        )
        glyphs = [e for e in svg.iter() if e.get("data-cap-height-mm")]
        rendered.extend(e.get("data-text", "") for e in glyphs)
        check(
            f"page_{index}.lettering_height_and_font",
            bool(glyphs)
            and all(
                float(e.get("data-cap-height-mm")) in (2.5, 3.5, 5, 7, 10)
                and e.get("data-font-sha256") == font_sha
                for e in glyphs
            ),
            "ISO 3098 series / ГОСТ 2.304-81 (height subset only)",
            {"text_runs": len(glyphs), "font_sha256": font_sha},
        )
    check(
        "first_angle_arrangement",
        native["projection"] == "first_angle"
        and native["views"]["top"]["center_mm"][1]
        > native["views"]["front"]["center_mm"][1]
        and native["views"]["right"]["center_mm"][0]
        < native["views"]["front"]["center_mm"][0],
        "ISO 5456-2:1996, 5.1 / ГОСТ 2.305-2008",
        native["projection"],
    )
    check(
        "native_dimension_offset",
        native.get("dimension_offset_mm", 0) >= 10,
        "ГОСТ 2.307-2011, 5.11 (first offset only)"
        if standard == "gost"
        else "Project dimension spacing; full ISO 129-1 review pending",
        native.get("dimension_offset_mm"),
    )
    # Do not infer approval identities, ownership or signatures from a successful render.
    required = (
        ("owner", "issue_date", "creator", "approver", "document_type")
        if standard == "iso"
        else ("owner", "developer", "norm_checker", "material", "registration")
    )
    for field in required:
        value = metadata.get(field)
        check(
            "metadata." + field,
            bool(value),
            "ISO 7200:2004, 5.1-5.3"
            if standard == "iso"
            else "ГОСТ Р 2.104-2023, таблица 1 / приложение А",
            {"provided": bool(value)},
        )
    check(
        "not_synthetic_release",
        not metadata.get("synthetic_example", False),
        "Document provenance",
        {"synthetic_example": metadata.get("synthetic_example", False)},
    )
    check(
        "diameter_notation",
        not any(re.search(r"(?<![A-Za-z0-9_])D[0-9]", value) for value in rendered),
        "ISO 129-1 / ГОСТ 2.307-2011 (symbol subset)",
        {"diameter_symbol_used": any("⌀" in value for value in rendered)},
    )
    check(
        "preferred_scale",
        native["scale"] in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
        "ГОСТ 2.302-68 / ISO 5455 (selected scale subset)",
        native["scale"],
    )
    pending = [
        "Independent clause-by-clause review of glyph shapes, stroke proportions and spacing.",
        "Complete dimensioning/arrow/section conventions and graphical projection-symbol review.",
        "Detailed title-block internal cells/additional fields and document-specific exceptions.",
        "Actual approval/signature/registration evidence and truth of supplied organizational metadata.",
        "Completeness of product requirements, tolerances, material and manufacturing notes where applicable.",
    ]
    return {
        "requested_standard": standard,
        "profile": profile.profile_id,
        "as_of": DATE,
        "status": "nonconforming"
        if any(c["status"] == "fail" for c in checks)
        else "partially_verified",
        "can_claim_full_conformance": False,
        "format_checks_passed": all(
            c["status"] == "pass"
            for c in checks
            if not c["id"].startswith("metadata.")
            and c["id"] != "not_synthetic_release"
        ),
        "checks": checks,
        "pending": pending,
        "scope": "Automated checks cover named formatting subsets only, not an unconditional ISO/GOST declaration.",
        "references": [
            {"standard": r[0], "url": r[1], "scope": r[2]} for r in profile.references
        ],
    }
