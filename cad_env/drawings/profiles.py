"""Versioned drafting targets. Target selection is not a conformance certificate."""

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DraftingProfile:
    key: str
    profile_id: str
    language: str
    margins: tuple
    wide: float
    thin: float
    references: tuple


PROFILES = {
    "iso": DraftingProfile(
        "iso",
        "cad-env.iso-a3.2026.v1",
        "en",
        (20, 10, 10, 10),
        0.7,
        0.35,
        (
            (
                "ISO 5457:1999 + Amd 1:2010",
                "https://www.iso.org/standard/29017.html",
                "sheet layout",
            ),
            (
                "ISO 7200:2004",
                "https://www.iso.org/standard/35446.html",
                "title-block data",
            ),
            (
                "ISO 5456-2:1996",
                "https://www.iso.org/standard/11502.html",
                "first-angle projection",
            ),
            (
                "ISO 128-2:2022",
                "https://www.iso.org/standard/83355.html",
                "line conventions",
            ),
            (
                "ISO 129-1:2018",
                "https://www.iso.org/standard/64007.html",
                "dimension presentation",
            ),
            (
                "ISO 3098 series",
                "https://www.iso.org/standard/65679.html",
                "lettering - full glyph review pending",
            ),
        ),
    ),
    "gost": DraftingProfile(
        "gost",
        "cad-env.gost-r-a3.2026.v1",
        "ru",
        (20, 5, 5, 5),
        0.5,
        0.25,
        (
            (
                "ГОСТ Р 2.104-2023, поправки 2024",
                "https://protect.gost.ru/gost/details/69d422a2-3c0e-4561-a521-0c99f9cf4417",
                "основные надписи",
            ),
            (
                "ГОСТ 2.301-68",
                "https://protect.gost.ru/gost/details/aa681f87-ff49-4115-ad25-169e15d2af06",
                "форматы, действует до перехода 2027",
            ),
            ("ГОСТ 2.302-68", "https://protect.gost.ru/gost", "масштабы"),
            ("ГОСТ 2.303-68", "https://protect.gost.ru/gost", "линии"),
            ("ГОСТ 2.304-81", "https://protect.gost.ru/gost", "шрифты"),
            ("ГОСТ 2.305-2008", "https://protect.gost.ru/gost", "виды и разрезы"),
            ("ГОСТ 2.306-68", "https://protect.gost.ru/gost", "штриховка"),
            (
                "ГОСТ 2.307-2011",
                "https://protect.gost.ru/gost/details/62de9da6-e1a7-4023-955e-7dd9777eaab6",
                "размеры, без досрочного применения редакции 2026",
            ),
        ),
    ),
}
ACTIVE = ContextVar("cad_drafting_profile", default=None)
METADATA = ContextVar("cad_drawing_metadata", default=None)


def get_profile(standard):
    if standard == "nominal":
        return None
    if standard not in PROFILES:
        raise ValueError("standard must be nominal, iso or gost")
    return PROFILES[standard]


def validate_metadata(metadata):
    allowed = {
        "owner",
        "designation",
        "title",
        "issue_date",
        "creator",
        "approver",
        "document_type",
        "status",
        "revision",
        "version",
        "material",
        "mass",
        "developer",
        "checker",
        "norm_checker",
        "approval",
        "inventory_number",
        "registration",
        "synthetic_example",
        "liter",
        "technologist",
    }
    if set(metadata) - allowed:
        raise ValueError(
            "Unknown drawing metadata fields: " + str(sorted(set(metadata) - allowed))
        )
    for key, value in metadata.items():
        if key == "synthetic_example":
            if not isinstance(value, bool):
                raise ValueError("synthetic_example must be boolean")
        elif value is not None and not isinstance(value, str):
            raise ValueError("Metadata values must be strings or null")
    if metadata.get("issue_date"):
        date.fromisoformat(metadata["issue_date"])
    return {k: v for k, v in metadata.items() if v is not None}
