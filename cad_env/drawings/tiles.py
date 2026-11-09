"""Traceable, readable document crops for VLM inputs; never replace raw sheets."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_tiles(root, parameters, *, standard="nominal"):
    technical = standard in ("iso", "gost")
    caption_height = 30 if technical else 42
    root = Path(root)
    folder = root / "vlm"
    folder.mkdir(exist_ok=False)
    records = []
    overview = Image.open(root / "sheet-01.png").convert("RGB")
    unit = overview.width / 420

    def crop(image, rectangle):
        x0, y0, x1, y1 = rectangle
        return image.crop(
            (round(x0 * unit), round(y0 * unit), round(x1 * unit), round(y1 * unit))
        )

    def save(image, title, source, rectangles, minimum_font_mm):
        # Additional caption describes crop identity only, never private labels.
        result = Image.new("RGB", (image.width, image.height + caption_height), "white")
        ImageDraw.Draw(result).text(
            (8, 4 if technical else 7),
            title,
            fill="black",
            font=ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                18 if technical else 22,
            ),
        )
        result.paste(image, (0, caption_height))
        path = folder / f"tile-{len(records) + 1:02d}.png"
        result.save(path)
        effective = (
            minimum_font_mm
            * unit
            * min(1, math.sqrt(327680 / (result.width * result.height)))
        )
        if effective < 10.5:
            raise ValueError(
                "Drawing crop text would be too small under fixed VLM budget"
            )
        records.append(
            {
                "image": str(path.relative_to(root)),
                "source_sheet": source,
                "source_rectangles_mm": rectangles,
                "caption": title,
                "size_px": list(result.size),
                "estimated_min_font_px_at_327680": effective,
            }
        )

    for name, rect in [
        ("RIGHT", [20, 30, 120, 127]),
        ("FRONT", [120, 30, 270, 127]),
        ("TOP", [115, 120, 275, 247]),
        ("SECTION A-A", [278, 120, 405, 240]),
    ]:
        save(
            crop(overview, rect),
            name + " / original overview crop",
            "sheet-01.png",
            [rect],
            3.5 if technical else 3.2,
        )
    count = (
        int("base" in parameters)
        + len(parameters.get("pads", []))
        + len(parameters.get("cuts", []))
    )
    for offset in range(0, count, 18):
        source = f"sheet-{2 + offset // 18:02d}.png"
        with Image.open(root / source) as loaded:
            sheet = loaded.convert("RGB")
        rows = min(18, count - offset)
        for start in range(0, rows, 6):
            end = min(rows, start + 6)
            for kind, columns in [
                ("POSITION", [(23, 179)]),
                ("SIZE / DEPTH", [(46, 87), (179, 407)]),
            ]:
                strips = []
                rectangles = []
                for left, right in columns:
                    header = [left, 50, right, 59]
                    values = [left, 59 + start * 9, right, 59 + end * 9]
                    h, v = crop(sheet, header), crop(sheet, values)
                    strip = Image.new("RGB", (h.width, h.height + v.height), "white")
                    strip.paste(h, (0, 0))
                    strip.paste(v, (0, h.height))
                    strips.append(strip)
                    rectangles += [header, values]
                combined = Image.new(
                    "RGB", (sum(im.width for im in strips), strips[0].height), "white"
                )
                x = 0
                for strip in strips:
                    combined.paste(strip, (x, 0))
                    x += strip.width
                save(
                    combined,
                    f"{source} / rows {start + 1}-{end} / {kind}",
                    source,
                    rectangles,
                    2.5 if technical else 2.7,
                )
    (root / "tiles.json").write_text(
        json.dumps(
            {"pixel_budget": 327680, "raw_sheets_preserved": True, "tiles": records},
            indent=2,
        )
    )
    return records
